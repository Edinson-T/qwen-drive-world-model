#!/usr/bin/env python
"""Build a WOD-E2E validation scene file from the released tfrecords.

The Qwen-Drive repository ships the Waymo benchmark *scene files* as artifacts of
the training pipeline, not as sources, so the reported numbers cannot be
recomputed before such a file exists.  This tool rebuilds the validation one from
the raw ``waymo_open_dataset_end_to_end_camera`` release:

1. scan every ``val_*.tfrecord-*`` shard and index each record by
   ``(scene_token, frame_index)``;
2. keep the frames that carry a human rater label (exactly one per scene);
3. pull the four camera timestamps the model reads -- ``t-1.5s .. t-0s`` at 2 Hz,
   i.e. frame indices ``n-15, n-10, n-5, n`` of the same scene at the source's
   10 Hz rate -- for the front, front-left and front-right cameras;
4. resample the raw 4 Hz ego states to the 10 Hz fields with the repository's own
   ``scripts/interpolate_waymo_trajectory.py``;
5. write one JSONL record per scene plus the extracted JPEG frames.

The output is directly consumable by ``scripts/run_planning.py`` and
``scripts/eval_waymo.py``.  Frames are written as
``<image-root>/<split>/<scene_token>/<CAMERA>/<frame_index>.jpg``, matching the
layout documented in ``docs/data.md``.

Only the validation split is meaningful for RFS: the public test split carries
neither ground-truth futures nor rater labels, so its metrics can only come from
the official leaderboard.  `--split` exists for inspection, not for scoring.

Run it in the released environment (numpy, scipy, Pillow):

    python tools/waymo_e2e/build_scenes.py \
        --data-root /data/waymo_e2e/waymo_open_dataset_end_to_end_camera_v_1_0_0 \
        --output outputs/waymo_val/scenes.jsonl
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

import e2e_proto as wp

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import interpolate_waymo_trajectory as iwt  # noqa: E402  (repo script, not a package)

# Frame indices of the four camera timestamps, relative to the labelled frame.
# The released camera sequence runs at 10 Hz, so 0.5 s of history is five frames.
HISTORY_OFFSETS = (15, 10, 5, 0)
HISTORY_INTERVAL_S = 0.5

# The per-frame resize targets the released metadata carries: history frames stay
# near 320p, the current frame near 720p.
HISTORY_SIZE = (384, 416)
CURRENT_SIZE = (720, 799)

CAMERAS = ("FRONT", "FRONT_LEFT", "FRONT_RIGHT")
VIEW_TAGS = {
    "FRONT": "<FRONT VIEW>",
    "FRONT_LEFT": "<FRONT LEFT VIEW>",
    "FRONT_RIGHT": "<FRONT RIGHT VIEW>",
}

# Raw 4 Hz history acceleration is stored as a velocity delta per raw step.
RAW_HIST_DT_S = 0.25

_FALLBACK_HEADER = (
    "The input images are organized by camera view. Each view contains {num_frames} temporal "
    "frames captured at {interval:g}s intervals (frame 0 at {first_label}, frame {last_frame} is "
    "the current frame at t=0s).\n"
    "1. Historical trajectories (x, y, heading) in the current frame's ego coordinate system. "
    "Positive x points forward, positive y points left, and a positive heading indicates a left "
    "turn\uff1a\n"
)
_FALLBACK_NAV = ("GO STRAIGHT", "TURN LEFT", "TURN RIGHT")
_FALLBACK_LABELS = ("t-1.5s", "t-1.0s", "t-0.5s", "t-0s")


def _prompt_pieces() -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """Reuse the released prompt template so the text matches byte for byte.

    ``qwen_drive.scene`` owns the string the model was trained with; importing it
    keeps this builder from drifting if that template ever changes.  The fallback
    is a verbatim copy for environments where the package is not installed.
    """
    try:
        from qwen_drive.scene import (  # type: ignore
            HISTORY_FRAME_LABELS,
            NAV_COMMANDS,
            _INSTRUCTION_HEADER,
        )

        return _INSTRUCTION_HEADER, tuple(NAV_COMMANDS), tuple(HISTORY_FRAME_LABELS)
    except Exception:
        return _FALLBACK_HEADER, _FALLBACK_NAV, _FALLBACK_LABELS


def instruction_text(hist_traj_10hz: np.ndarray, nav_command: int) -> str:
    """The user-turn text block, identical to ``DrivingScene.instruction()``."""
    header, nav_commands, labels = _prompt_pieces()
    num_frames = len(HISTORY_OFFSETS)
    stride = max(1, (len(hist_traj_10hz) - 1) // max(1, num_frames - 1))
    lines = "".join(
        " -{}: ({:.4f}, {:.4f}, {:.4f});\n".format(label, *hist_traj_10hz[index * stride])
        for index, label in enumerate(labels)
    )
    head = header.format(
        num_frames=num_frames,
        interval=HISTORY_INTERVAL_S,
        first_label=labels[0],
        last_frame=num_frames - 1,
    )
    command = nav_commands[nav_command]
    return f"{head}{lines}2. Active navigation command: [{command}]"


def _intent_command(intent: int) -> tuple[int, list[int]]:
    """Map the raw intent enum to ``(nav_command, driving_command)``."""
    nav = {1: 0, 2: 1, 3: 2}.get(intent, 0)
    # driving_command is one-hot in (left, straight, right, unknown) order.
    one_hot = {2: [1, 0, 0, 0], 1: [0, 1, 0, 0], 3: [0, 0, 1, 0]}.get(intent, [0, 0, 0, 1])
    return nav, one_hot


def _headings_from_xy(xy: np.ndarray) -> np.ndarray:
    """Heading of a trajectory from finite differences, for format completeness.

    The released rater trajectories store positions only; the RFS consumes
    ``[..., :2]``, so this heading is informational.
    """
    deltas = np.diff(xy, axis=0, prepend=xy[:1])
    heading = np.arctan2(deltas[:, 1], deltas[:, 0])
    heading[0] = heading[1] if len(heading) > 1 else 0.0
    return heading


# --------------------------------------------------------------------------- #
# Index
# --------------------------------------------------------------------------- #


def shard_paths(data_root: Path, split: str) -> list[Path]:
    paths = sorted(data_root.glob(f"{split}_*.tfrecord-*"))
    if not paths:
        raise SystemExit(f"no '{split}_*.tfrecord-*' shards found in {data_root}")
    return paths


def build_index(data_root: Path, split: str, log=print) -> dict:
    """Scan every shard and index each record by ``(scene_token, frame_index)``."""
    paths = shard_paths(data_root, split)
    log(f"indexing {len(paths)} {split} shards from {data_root}")
    index: dict[str, dict[str, list]] = {}
    labeled: dict[str, list[int]] = {}
    total = 0
    for shard_id, path in enumerate(paths):
        for offset, length, payload in wp.iter_tfrecords(path):
            total += 1
            meta = wp.scan_metadata(payload)
            if meta is None:
                continue
            index.setdefault(meta.scene_token, {})[str(meta.frame_index)] = [
                shard_id,
                offset,
                length,
            ]
            if meta.has_labels:
                labeled.setdefault(meta.scene_token, []).append(meta.frame_index)
        if shard_id % 20 == 0 or shard_id == len(paths) - 1:
            log(f"  {shard_id + 1}/{len(paths)} shards, {total} records")
    log(f"indexed {total} records, {len(index)} scenes, {len(labeled)} labelled scenes")
    return {
        "split": split,
        "data_root": str(data_root),
        "shards": [path.name for path in paths],
        "index": index,
        "labeled": {token: sorted(frames) for token, frames in labeled.items()},
    }


def save_index(index: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as handle:
        json.dump(index, handle)


def load_index(path: Path) -> dict:
    with gzip.open(path, "rt") as handle:
        return json.load(handle)


# --------------------------------------------------------------------------- #
# Scene assembly
# --------------------------------------------------------------------------- #


def _frame_record(index: dict, shards: list[Path], token: str, frame_index: int) -> dict:
    entry = index["index"][token].get(str(frame_index))
    if entry is None:
        raise KeyError(frame_index)
    shard_id, offset, length = entry
    payload = wp.read_record(shards[shard_id], offset, length)
    return wp.parse_e2ed_frame(payload)


def _select_history(index: dict, token: str, frame_index: int, policy: str) -> list[int] | None:
    """Return the four history frame indices, newest last."""
    available = {int(key) for key in index["index"][token]}
    wanted = [frame_index - offset for offset in HISTORY_OFFSETS]
    if all(index in available for index in wanted):
        return wanted
    if policy == "skip":
        return None
    nearest = []
    for target in wanted:
        nearest.append(min(available, key=lambda value: (abs(value - target), value)))
    return nearest


def load_records(index: dict, shards: list[Path], token: str, frame_index: int, policy: str) -> list:
    """Read the four E2EDFrame records of one scene, oldest camera frame first."""
    history = _select_history(index, token, frame_index, policy)
    if history is None:
        raise KeyError("history frames unavailable")
    return [_frame_record(index, shards, token, value) for value in history]


def _write_frames(scene_dir: Path, token: str, records: list, split: str) -> dict[str, str]:
    """Write the requested camera frames and return the scene-file paths."""
    paths: dict[str, str] = {}
    for record in records:
        for camera in CAMERAS:
            jpeg = record.images.get(camera)
            if jpeg is None:
                raise KeyError(f"{record.context_name} has no {camera} image")
            relative = f"{split}/{token}/{camera}/{record.frame_index}.jpg"
            target = scene_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(jpeg)
            paths[f"{camera}:{record.frame_index}"] = relative
    return paths


def _user_content(frame_paths: dict, records: list, instruction: str) -> list[dict]:
    content: list[dict] = []
    for camera in CAMERAS:
        content.append({"text": VIEW_TAGS[camera]})
        for position, record in enumerate(records):
            content.append({"text": f"frame: {position}"})
            width, height = CURRENT_SIZE if position == len(records) - 1 else HISTORY_SIZE
            content.append(
                {
                    "image": frame_paths[f"{camera}:{record.frame_index}"],
                    "resized_width": width,
                    "resized_height": height,
                }
            )
    content.append({"text": instruction})
    return content


def build_scene(token: str, frame_index: int, records: list) -> dict:
    """Assemble one scene record from its four camera-frame records."""
    current = records[-1]

    hist_traj, hist_vel, hist_acc = iwt.interpolate_history(current.past)
    fut_traj, fut_vel, fut_acc = iwt.interpolate_future(current.future)

    preferences = []
    for pref in current.preferences:
        score = pref.get("preference_score")
        if score is None or score < 0:
            continue
        xy = np.column_stack([pref["pos_x"], pref["pos_y"]])
        trajectory = np.column_stack([xy, _headings_from_xy(xy)])
        preferences.append(
            {
                "pref_traj": trajectory.tolist(),
                "pref_valid_mask": [1] * len(trajectory),
                "preference_score": float(score),
            }
        )

    nav_command, driving_command = _intent_command(current.intent)
    velocity_last = np.array(
        [current.past["vel_x"][-1], current.past["vel_y"][-1]], dtype=np.float64
    )
    acceleration_last = np.array(
        [current.past["accel_x"][-1], current.past["accel_y"][-1]], dtype=np.float64
    ) / RAW_HIST_DT_S

    raw_hist = np.column_stack([current.past["pos_x"], current.past["pos_y"]])
    raw_future = np.column_stack([current.future["pos_x"], current.future["pos_y"]])

    scene_token = token
    record_token = f"{token}-{frame_index}"
    meta_info = {
        "dataset": "WOD_E2E",
        "task_name": "trajectory_prediction",
        "token": record_token,
        "scene_token": scene_token,
        "cam_order": list(CAMERAS),
        "num_history_frames": len(HISTORY_OFFSETS),
        "num_history_frames_raw": len(HISTORY_OFFSETS),
        "num_action_history_frames": len(HISTORY_OFFSETS),
        "history_padding": "none",
        "history_time_offsets": [
            -HISTORY_INTERVAL_S * (len(HISTORY_OFFSETS) - 1 - i)
            for i in range(len(HISTORY_OFFSETS))
        ],
        "image_hz": 1.0 / HISTORY_INTERVAL_S,
        "traj_hz": 4.0,
        "intent": current.intent_name,
        "has_preference_trajectories": bool(preferences),
    }

    def rounded(array: np.ndarray) -> list:
        return np.round(np.asarray(array, dtype=np.float64), 5).tolist()

    trajectory = {
        "hist_traj": [[float(x), float(y), 0.0] for x, y in raw_hist],
        "hist_valid_mask": [1] * len(raw_hist),
        "future_traj": [[float(x), float(y), 0.0] for x, y in raw_future],
        "future_valid_mask": [1] * len(raw_future),
        "ego_status": {
            "in_global_frame": False,
            "ego_pose": [0.0, 0.0, 0.0],
            "ego_velocity": [float(velocity_last[0]), float(velocity_last[1])],
            "ego_acceleration": [float(acceleration_last[0]), float(acceleration_last[1])],
            "driving_command": driving_command,
        },
        "nav_command": nav_command,
        "driving_command": driving_command,
        "hist_hz": 4.0,
        "future_hz": 4.0,
        "hist_traj_10hz": rounded(hist_traj),
        "hist_vel_10hz": rounded(hist_vel),
        "hist_acc_10hz": rounded(hist_acc),
        "hist_valid_mask_10hz": [1] * len(hist_traj),
        "future_traj_10hz": rounded(fut_traj),
        "future_vel_10hz": rounded(fut_vel),
        "future_acc_10hz": rounded(fut_acc),
        "future_valid_mask_10hz": [1] * len(fut_traj),
        "current_state": {
            "ego_velocity": [float(velocity_last[0]), float(velocity_last[1])],
            "ego_acceleration": [float(acceleration_last[0]), float(acceleration_last[1])],
            "ego_yawrate": 0.0,
            "speed": float(np.hypot(velocity_last[0], velocity_last[1])),
        },
        "preference_trajectories": preferences,
        "interp_method_10hz": "cubic_xy_curvature_integrated_10hz_heading_kinematics",
        "traj_10hz_source": "waymo_raw_4hz_interp_acceleration_dt_corrected_curvature_heading",
    }
    # ``benchmarks.py`` prefers the 1.5 s window when present; at 2 Hz the four
    # camera timestamps cover exactly that window, so the two series coincide.
    for name in ("hist_traj", "hist_vel", "hist_acc", "hist_valid_mask"):
        trajectory[f"{name}_1p5s_10hz"] = trajectory[f"{name}_10hz"]

    return {
        "type": "chatml",
        "messages": [
            {"role": "user", "content": []},
            {"role": "assistant", "content": []},
        ],
        "meta_info": meta_info,
        "trajectory": trajectory,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True, type=Path, help="directory holding the tfrecord shards")
    parser.add_argument("--output", required=True, type=Path, help="destination scene file (jsonl)")
    parser.add_argument("--split", default="val", help="shard prefix to read (default: val)")
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="where to write the JPEG frames (default: alongside --output)",
    )
    parser.add_argument("--index", type=Path, default=None, help="index cache path (default: <output>.index.json.gz)")
    parser.add_argument("--rebuild-index", action="store_true", help="ignore an existing index cache")
    parser.add_argument("--limit", type=int, default=None, help="stop after this many scenes")
    parser.add_argument("--tokens", nargs="*", default=None, help="only build these scene tokens")
    parser.add_argument(
        "--missing-history",
        choices=("skip", "nearest"),
        default="skip",
        help="what to do when a scene lacks one of the four history frames (default: skip)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    data_root = args.data_root.resolve()
    output = args.output.resolve()
    image_root = (args.image_root or output.parent).resolve()
    index_path = args.index or output.with_suffix(output.suffix + ".index.json.gz")

    if args.index and args.index.exists() and not args.rebuild_index:
        index = load_index(index_path)
        print(f"loaded index for {len(index['index'])} scenes from {index_path}")
    else:
        index = build_index(data_root, args.split)
        save_index(index, index_path)
        print(f"wrote index to {index_path}")
    shards = [data_root / name for name in index["shards"]]

    tasks = sorted(
        (token, frame)
        for token, frames in index["labeled"].items()
        for frame in frames
        if args.tokens is None or token in set(args.tokens)
    )
    if not tasks:
        raise SystemExit("no labelled scenes selected")
    if args.limit:
        tasks = tasks[: args.limit]

    output.parent.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    with open(output, "w") as handle:
        for position, (token, frame_index) in enumerate(tasks, start=1):
            try:
                records = load_records(index, shards, token, frame_index, args.missing_history)
                record = build_scene(token, frame_index, records)
                paths = _write_frames(image_root, token, records, args.split)
            except KeyError as error:
                skipped += 1
                print(f"  skipping {token}-{frame_index}: {error}")
                continue
            instruction = instruction_text(
                np.asarray(record["trajectory"]["hist_traj_10hz"], dtype=np.float64),
                record["trajectory"]["nav_command"],
            )
            record["messages"][0]["content"] = _user_content(paths, records, instruction)
            handle.write(json.dumps(record) + "\n")
            written += 1
            if position % 50 == 0 or position == len(tasks):
                print(f"  {position}/{len(tasks)} scenes, {skipped} skipped")

    print(f"wrote {written} scenes to {output} ({skipped} skipped)")
    print(f"frames under {image_root}/{args.split}")


if __name__ == "__main__":
    main()
