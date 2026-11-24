"""Build QD-compatible WM windows directly from Waymo Motion scenarios."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import struct
import time

import numpy as np

from .motion_gt import (
    HEIGHT_BANDS,
    MODEL_HW,
    MOTION_OBJECT_CLASS_NAMES,
    QD_HW,
    QD_RESOLUTION,
    QD_X_RANGE,
    QD_Y_RANGE,
    compress_gt_rasters,
    ego_pose_from_state,
    global_to_ego,
    rasterize_boxes,
    rasterize_map_features,
    scenario_map_features,
    track_boxes,
    wrap_angle,
)
from .perception_gt import GT_MAP_CLASS_NAMES
from .protos import womd_pb2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/mnt/hdd/waymo_motion_v1_3_1/training"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-scenarios", type=int, default=4000)
    parser.add_argument("--anchors-per-scenario", type=int, default=8)
    parser.add_argument("--val-every", type=int, default=10)
    parser.add_argument("--shard-size", type=int, default=32)
    parser.add_argument("--max-boxes", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _iter_tfrecords(path: Path):
    with path.open("rb") as handle:
        while True:
            header = handle.read(8)
            if len(header) < 8:
                return
            length = struct.unpack("<Q", header)[0]
            handle.read(4)
            payload = handle.read(length)
            handle.read(4)
            if len(payload) < length:
                return
            yield payload


def _horizon_key(seconds: float) -> str:
    return f"{float(seconds):g}s".replace(".", "p")


def _split_for_scenario(index: int, val_every: int) -> str:
    return "val" if val_every > 0 and index % val_every == 0 else "train"


def _sdc(scenario: womd_pb2.Scenario):
    index = int(scenario.sdc_track_index)
    if not 0 <= index < len(scenario.tracks):
        return None
    return scenario.tracks[index]


def _anchor_indices(
    scenario: womd_pb2.Scenario,
    offsets: list[int],
    count: int,
) -> list[int]:
    max_anchor = len(scenario.timestamps_seconds) - 1 - max(offsets)
    valid = [
        index
        for index in range(max_anchor + 1)
        if scenario.tracks[scenario.sdc_track_index].states[index].valid
        and all(
            scenario.tracks[scenario.sdc_track_index].states[index + offset].valid
            for offset in offsets
        )
    ]
    if not valid:
        return []
    selected = np.linspace(0, len(valid) - 1, min(count, len(valid)), dtype=int)
    return [valid[index] for index in sorted(set(selected.tolist()))]


def _trajectory(
    sdc,
    anchor: int,
    steps: int,
) -> np.ndarray:
    anchor_state = sdc.states[anchor]
    anchor_pose = ego_pose_from_state(anchor_state)
    states = []
    for step in range(anchor, anchor + steps):
        state = sdc.states[step]
        position = global_to_ego(
            np.asarray([[state.center_x, state.center_y, 0.0]]),
            anchor_pose,
        )[0]
        states.append(
            [
                position[0],
                position[1],
                float(wrap_angle(float(state.heading) - anchor_pose[2])),
            ]
        )
    return np.asarray(states, dtype=np.float32)


def _ego_status(sdc, anchor: int, trajectory: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    heading = float(sdc.states[anchor].heading)
    cosine, sine = math.cos(heading), math.sin(heading)
    velocity_global = np.asarray(
        [sdc.states[anchor].velocity_x, sdc.states[anchor].velocity_y],
        dtype=np.float64,
    )
    velocity = np.asarray(
        [
            cosine * velocity_global[0] + sine * velocity_global[1],
            -sine * velocity_global[0] + cosine * velocity_global[1],
        ],
        dtype=np.float32,
    )
    next_state = sdc.states[min(anchor + 1, len(sdc.states) - 1)]
    next_velocity_global = np.asarray(
        [next_state.velocity_x, next_state.velocity_y],
        dtype=np.float64,
    )
    next_velocity = np.asarray(
        [
            cosine * next_velocity_global[0] + sine * next_velocity_global[1],
            -sine * next_velocity_global[0] + cosine * next_velocity_global[1],
        ],
        dtype=np.float32,
    )
    acceleration = (next_velocity - velocity) * 10.0
    yaw = float(trajectory[-1, 2])
    if yaw > 0.17:
        command, nav = np.asarray([1.0, 0.0, 0.0, 0.0], np.float32), 2
    elif yaw < -0.17:
        command, nav = np.asarray([0.0, 0.0, 1.0, 0.0], np.float32), 1
    else:
        command, nav = np.asarray([0.0, 1.0, 0.0, 0.0], np.float32), 0
    return velocity, acceleration, command, nav


def _write_shard(path: Path, rows: list[dict[str, np.ndarray]]) -> None:
    keys = {"z_t", "map_t", "occ_t", "height_t", "height_t_highres", "boxes_t", "box_labels_t", "box_ids_t", "box_counts_t"}
    for horizon in rows[0]["horizons"]:
        keys.update(
            {
                f"z_{horizon}",
                f"map_{horizon}",
                f"occ_{horizon}",
                f"boxes_{horizon}",
                f"box_labels_{horizon}",
                f"box_ids_{horizon}",
                f"box_counts_{horizon}",
            }
        )
    arrays = {key: np.stack([row[key] for row in rows]) for key in keys}
    arrays.update(
        {
            "tokens": np.asarray([row["token"] for row in rows]),
            "scenario_ids": np.asarray([row["scenario_id"] for row in rows]),
            "split": np.asarray([row["split"] for row in rows]),
            "t_frames": np.asarray([row["t_frame"] for row in rows], np.int32),
            "nav_command": np.asarray([row["nav_command"] for row in rows], np.int64),
            "ego_velocity": np.stack([row["ego_velocity"] for row in rows]),
            "ego_acceleration": np.stack([row["ego_acceleration"] for row in rows]),
            "driving_command": np.stack([row["driving_command"] for row in rows]),
            "trajectory_5s": np.stack([row["trajectory_5s"] for row in rows]),
            "relative_poses": np.stack([row["relative_poses"] for row in rows]),
        }
    )
    np.savez_compressed(path, **arrays)


def main() -> None:
    args = parse_args()
    if args.output.exists() and args.overwrite:
        import shutil

        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    horizons = (0.5, 1.0, 2.0, 3.0, 5.0)
    horizon_keys = [_horizon_key(value) for value in horizons]
    offsets = [int(round(value * 10.0)) for value in horizons]
    trajectory_steps = offsets[-1] + 1
    files = sorted(args.data_dir.glob("*.tfrecord-*"))
    if not files:
        raise SystemExit(f"no Motion TFRecords under {args.data_dir}")

    rows = []
    shards = []
    scenario_count = 0
    sample_count = 0
    train_count = 0
    val_count = 0
    started = time.perf_counter()
    stop = False
    for file_path in files:
        for payload in _iter_tfrecords(file_path):
            scenario = womd_pb2.Scenario()
            scenario.ParseFromString(payload)
            if not scenario.scenario_id:
                continue
            sdc = _sdc(scenario)
            if sdc is None:
                continue
            map_features = scenario_map_features(scenario)
            anchors = _anchor_indices(scenario, offsets, args.anchors_per_scenario)
            if not map_features or not anchors:
                continue
            split = _split_for_scenario(scenario_count, args.val_every)
            raster_cache = {}

            def frame_raster(step: int):
                if step not in raster_cache:
                    pose = ego_pose_from_state(sdc.states[step])
                    semantic = rasterize_map_features(map_features, pose)
                    boxes, labels, track_ids = track_boxes(
                        scenario,
                        step,
                        pose,
                        ego_center_z=float(sdc.states[step].center_z),
                        max_boxes=args.max_boxes,
                    )
                    occupancy, height_slices = rasterize_boxes(boxes, labels)
                    wm_input, target = compress_gt_rasters(
                        semantic,
                        occupancy,
                        height_slices,
                        MODEL_HW,
                    )
                    raster_cache[step] = {
                        "map": semantic,
                        "occ": occupancy,
                        "height_highres": height_slices,
                        "z_input": wm_input,
                        "z_target": target,
                        "boxes": boxes,
                        "labels": labels,
                        "ids": track_ids,
                    }
                return raster_cache[step]

            for anchor in anchors:
                current = frame_raster(anchor)
                trajectory = _trajectory(sdc, anchor, trajectory_steps)
                velocity, acceleration, command, nav = _ego_status(sdc, anchor, trajectory)
                anchor_pose = ego_pose_from_state(sdc.states[anchor])
                relative_poses = []
                for offset in offsets:
                    future_pose = ego_pose_from_state(sdc.states[anchor + offset])
                    relative = global_to_ego(
                        np.asarray([[future_pose[0], future_pose[1], 0.0]]),
                        anchor_pose,
                    )[0]
                    relative_poses.append(
                        [relative[0], relative[1], wrap_angle(float(future_pose[2] - anchor_pose[2]))]
                    )
                token = f"{scenario.scenario_id}-{anchor:03d}"
                row = {
                    "token": token,
                    "scenario_id": scenario.scenario_id,
                    "split": split,
                    "t_frame": anchor,
                    "nav_command": nav,
                    "ego_velocity": velocity,
                    "ego_acceleration": acceleration,
                    "driving_command": command,
                    "trajectory_5s": trajectory,
                    "relative_poses": np.asarray(relative_poses, np.float32),
                    "horizons": horizon_keys,
                    "z_t": current["z_input"],
                    "map_t": current["map"],
                    "occ_t": current["occ"],
                    "height_t": current["z_input"][9:],
                    "height_t_highres": current["height_highres"],
                    "boxes_t": current["boxes"],
                    "box_labels_t": current["labels"],
                    "box_ids_t": current["ids"],
                    "box_counts_t": np.asarray((current["labels"] >= 0).sum(), np.int16),
                }
                for key, offset in zip(horizon_keys, offsets, strict=True):
                    future = frame_raster(anchor + offset)
                    row[f"z_{key}"] = future["z_target"]
                    row[f"map_{key}"] = future["map"]
                    row[f"occ_{key}"] = future["occ"]
                    row[f"boxes_{key}"] = future["boxes"]
                    row[f"box_labels_{key}"] = future["labels"]
                    row[f"box_ids_{key}"] = future["ids"]
                    row[f"box_counts_{key}"] = np.asarray(
                        (future["labels"] >= 0).sum(), np.int16
                    )
                rows.append(row)
                sample_count += 1
                if split == "train":
                    train_count += 1
                else:
                    val_count += 1
                if len(rows) >= args.shard_size:
                    path = args.output / f"pilot-{len(shards):05d}.npz"
                    _write_shard(path, rows)
                    shards.append(
                        {
                            "path": str(path.resolve()),
                            "samples": len(rows),
                            "train": sum(item["split"] == "train" for item in rows),
                            "val": sum(item["split"] == "val" for item in rows),
                        }
                    )
                    rows = []

            scenario_count += 1
            if scenario_count % 100 == 0:
                print(
                    f"scenarios={scenario_count} samples={sample_count} "
                    f"elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )
            if scenario_count >= args.max_scenarios:
                stop = True
                break
        if stop:
            break

    if rows:
        path = args.output / f"pilot-{len(shards):05d}.npz"
        _write_shard(path, rows)
        shards.append(
            {
                "path": str(path.resolve()),
                "samples": len(rows),
                "train": sum(item["split"] == "train" for item in rows),
                "val": sum(item["split"] == "val" for item in rows),
            }
        )

    metadata = {
        "source": "Waymo Open Motion Dataset v1.3.1",
        "source_dir": str(args.data_dir.resolve()),
        "scenarios": scenario_count,
        "samples": sample_count,
        "train_samples": train_count,
        "val_samples": val_count,
        "anchors_per_scenario": args.anchors_per_scenario,
        "val_every": args.val_every,
        "horizons": list(horizons),
        "frame_rate": 10.0,
        "bev_x_range": list(QD_X_RANGE),
        "bev_y_range": list(QD_Y_RANGE),
        "bev_resolution": QD_RESOLUTION,
        "bev_hw": list(QD_HW),
        "model_hw": list(MODEL_HW),
        "height_bands": [list(band) for band in HEIGHT_BANDS],
        "gt_map_classes": list(GT_MAP_CLASS_NAMES),
        "object_classes": list(MOTION_OBJECT_CLASS_NAMES),
        "coordinate_protocol": "future ego-frame prediction",
        "action_protocol": "logged SDC trajectory at 10 Hz",
        "shards": shards,
    }
    (args.output / "dataset.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        f"wrote scenarios={scenario_count} samples={sample_count} "
        f"train={train_count} val={val_count} to {args.output}"
    )


if __name__ == "__main__":
    main()

