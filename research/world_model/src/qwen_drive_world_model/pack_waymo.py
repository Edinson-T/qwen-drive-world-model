"""Pack real WOD-E2E camera pairs from t to the 0.5/3/5 second horizons."""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
TOOLS_DIR = REPO_ROOT / "tools" / "waymo_e2e"
ANCHORED_SRC = REPO_ROOT / "research" / "anchored_flow" / "src"
for source in (TOOLS_DIR, ANCHORED_SRC):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

import e2e_proto  # noqa: E402
from qwen_drive_anchored.draw_camera_projection import _parse_calibrations  # noqa: E402
from qwen_drive_anchored.prepare_waymo_perception_frames import _write_frame  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("/mnt/hdd/waymo_e2e/waymo_open_dataset_end_to_end_camera_v_1_0_0"),
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("outputs/waymo_val/scenes.jsonl.index.json.gz"),
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("outputs/visualizations/anchored_flow/paper_figures/base_vs_bestn"),
    )
    parser.add_argument(
        "--all-scenes",
        action="store_true",
        help="use every token in --scenes instead of the paper-figure manifests",
    )
    parser.add_argument(
        "--all-index-scenes",
        action="store_true",
        help="use scene tokens directly from --index, without a scenes.jsonl",
    )
    parser.add_argument("--scenes", type=Path, default=Path("outputs/waymo_val/scenes.jsonl"))
    parser.add_argument(
        "--exclude-pairs",
        type=Path,
        nargs="+",
        default=None,
        help="Skip scene tokens already present in another pairs.jsonl.",
    )
    parser.add_argument(
        "--current-only",
        action="store_true",
        help="Pack only the current camera frame, without future teacher inputs.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera-hz", type=float, default=10.0)
    parser.add_argument(
        "--horizons",
        type=float,
        nargs="+",
        default=(0.5, 1.0, 2.0, 3.0, 5.0),
    )
    parser.add_argument("--extra-per-scene", type=int, default=0)
    parser.add_argument("--val-scenes", type=int, default=3)
    parser.add_argument("--limit-pairs", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _horizon_key(seconds: float) -> str:
    return f"future_{seconds:g}s".replace(".", "p")


def _load_anchor_tokens(manifest_dir: Path) -> list[str]:
    paths = sorted(manifest_dir.glob("base_vs_bestn_*.json"))
    return [json.loads(path.read_text())["token"] for path in paths]


def _load_scene_meta(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            token = str(record["meta_info"]["token"])
            rows[token] = record
    return rows


def _read_record(
    scene_token: str,
    frame_index: int,
    index: dict,
    raw_root: Path,
) -> tuple[dict, bytes]:
    entry = index["index"][scene_token][str(frame_index)]
    shard_id, offset, length = entry
    shard = raw_root / index["shards"][int(shard_id)]
    payload = e2e_proto.read_record(shard, int(offset), int(length))
    return e2e_proto.parse_e2ed_frame(payload), payload


def _trajectory_5s(record: dict) -> np.ndarray:
    future = record.future
    x = np.asarray(future["pos_x"], dtype=np.float64)
    y = np.asarray(future["pos_y"], dtype=np.float64)
    if x.size != y.size or not x.size:
        raise ValueError("record has no future trajectory")
    xy = np.stack([x, y], axis=1)
    heading = np.zeros(len(xy), dtype=np.float64)
    if len(xy) > 1:
        delta = np.diff(xy, axis=0)
        heading[1:] = np.arctan2(delta[:, 1], delta[:, 0])
    future_xyz = np.column_stack([xy, heading])
    return np.concatenate([np.zeros((1, 3), dtype=np.float64), future_xyz], axis=0)


def _driving_command(intent: int) -> list[int]:
    return {2: [1, 0, 0, 0], 1: [0, 1, 0, 0], 3: [0, 0, 1, 0]}.get(
        intent, [0, 0, 0, 1]
    )


def _ego_status(record: dict) -> list[float]:
    past = record.past
    if len(past.get("vel_x", [])) == 0 or len(past.get("vel_y", [])) == 0:
        raise ValueError("record has no ego velocity history")
    vel_x, vel_y = float(past["vel_x"][-1]), float(past["vel_y"][-1])
    accel_x = float(past["accel_x"][-1]) if len(past.get("accel_x", [])) else 0.0
    accel_y = float(past["accel_y"][-1]) if len(past.get("accel_y", [])) else 0.0
    return [vel_x, vel_y, accel_x, accel_y]


def _write_teacher_frame(
    output_dir: Path,
    record: dict,
    payload: bytes,
) -> None:
    calibrations = _parse_calibrations(payload, e2e_proto._iter_fields)
    _write_frame(output_dir, output_dir.name, record.images, calibrations)


def _candidate_start_frames(
    anchor: int,
    available: set[int],
    horizons_frames: list[int],
    extra: int,
) -> list[int]:
    candidates = [anchor]
    for distance in range(1, extra + 1):
        for candidate in (anchor - 10 * distance, anchor + 10 * distance):
            if candidate not in candidates:
                candidates.append(candidate)
    valid = []
    for frame in candidates:
        if all(frame + horizon in available for horizon in horizons_frames):
            valid.append(frame)
    return valid


def main() -> None:
    args = parse_args()
    if args.output.exists() and args.overwrite:
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    with gzip.open(args.index, "rt") as handle:
        index = json.load(handle)
    if args.all_index_scenes:
        scene_meta = {}
        anchors = []
    else:
        scene_meta = _load_scene_meta(args.scenes)
        anchors = sorted(scene_meta) if args.all_scenes else _load_anchor_tokens(args.manifest_dir)
    if args.exclude_pairs is not None:
        existing_tokens = set()
        for pairs_path in args.exclude_pairs:
            with pairs_path.open() as handle:
                for line in handle:
                    if line.strip():
                        existing_tokens.add(str(json.loads(line)["token"]))
        anchors = [token for token in anchors if token not in existing_tokens]
    horizons_frames = [
        int(round(seconds * args.camera_hz)) for seconds in args.horizons
    ]
    required_offsets = [] if args.current_only else horizons_frames
    scene_anchors = {}
    if args.all_index_scenes:
        for scene_token, frames in index["index"].items():
            available = sorted(int(frame) for frame in frames)
            available_set = set(available)
            candidates = [
                frame
                for frame in available
                if all(frame + horizon in available_set for horizon in required_offsets)
            ]
            if candidates:
                scene_anchors[scene_token] = candidates[len(candidates) // 2]
    else:
        for token in anchors:
            scene_token, frame = token.rsplit("-", 1)
            scene_anchors.setdefault(scene_token, int(frame))

    ordered_scenes = sorted(scene_anchors)
    pairs = []
    skipped_scenes = 0
    for scene_token in ordered_scenes:
        available = {int(value) for value in index["index"][scene_token]}
        starts = _candidate_start_frames(
            scene_anchors[scene_token],
            available,
            required_offsets,
            args.extra_per_scene,
        )
        if not starts:
            skipped_scenes += 1
            continue
        for start_frame in starts:
            pair_id = f"{scene_token}-{start_frame}"
            record, payload = _read_record(
                scene_token, start_frame, index, args.raw_root
            )
            pair_dir = args.output / pair_id
            current_dir = pair_dir / "current"
            _write_teacher_frame(current_dir, record, payload)
            future_inputs = {}
            if not args.current_only:
                for seconds, offset in zip(args.horizons, horizons_frames, strict=True):
                    future_frame = start_frame + offset
                    future_record, future_payload = _read_record(
                        scene_token, future_frame, index, args.raw_root
                    )
                    future_dir = pair_dir / _horizon_key(seconds)
                    _write_teacher_frame(future_dir, future_record, future_payload)
                    future_inputs[str(seconds)] = str(future_dir.resolve())

            trajectory = _trajectory_5s(record)
            relative_poses = {}
            trajectory_windows = {}
            for seconds in args.horizons:
                point_index = int(round(seconds / 0.25))
                point_index = min(point_index, len(trajectory) - 1)
                relative_poses[str(seconds)] = trajectory[point_index].tolist()
                trajectory_windows[str(seconds)] = trajectory[: point_index + 1].tolist()
            source_token = f"{scene_token}-{start_frame}"
            meta = scene_meta.get(source_token, {})
            ego = _ego_status(record)
            pairs.append(
                {
                    "pair_id": pair_id,
                    "token": source_token,
                    "scene_token": scene_token,
                    "t_frame": start_frame,
                    "split": "train",
                    "nav_command": {0: 0, 1: 0, 2: 1, 3: 2}.get(record.intent, 0),
                    "intent": e2e_proto.INTENT_NAMES.get(record.intent, "UNKNOWN"),
                    "ego_velocity": ego[:2],
                    "ego_acceleration": ego[2:],
                    "driving_command": _driving_command(record.intent),
                    "teacher_inputs": {
                        "current": str(current_dir.resolve()),
                        **future_inputs,
                    },
                    "horizons": list(args.horizons),
                    "trajectory_5s": trajectory.tolist(),
                    "trajectory_windows": trajectory_windows,
                    "relative_poses": relative_poses,
                    "source_scene_token": str(meta.get("meta_info", {}).get("scene_token", scene_token)),
                }
            )
            if len(pairs) % 100 == 0:
                print(
                    f"packed {len(pairs)} pairs from {len(ordered_scenes)} candidate scenes",
                    flush=True,
                )
            if args.limit_pairs is not None and len(pairs) >= args.limit_pairs:
                break
        if args.limit_pairs is not None and len(pairs) >= args.limit_pairs:
            break

    included_scenes = sorted({pair["scene_token"] for pair in pairs})
    val_scenes = (
        set(included_scenes[-args.val_scenes :]) if args.val_scenes else set()
    )
    for pair in pairs:
        pair["split"] = "val" if pair["scene_token"] in val_scenes else "train"

    with (args.output / "pairs.jsonl").open("w") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair) + "\n")
    summary = {
        "pairs": len(pairs),
        "scenes": len({pair["scene_token"] for pair in pairs}),
        "candidate_scenes": len(ordered_scenes),
        "skipped_scenes": skipped_scenes,
        "train_pairs": sum(pair["split"] == "train" for pair in pairs),
        "val_pairs": sum(pair["split"] == "val" for pair in pairs),
        "horizons": list(args.horizons),
        "camera_hz": args.camera_hz,
        "current_only": args.current_only,
        "frame_offsets": dict(zip(map(str, args.horizons), horizons_frames, strict=True)),
        "coordinate_protocol": "future ego-frame prediction; full 5 s trajectory is an action condition",
        "raw_root": str(args.raw_root),
        "index": str(args.index),
    }
    (args.output / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.output / 'pairs.jsonl'}")


if __name__ == "__main__":
    main()
