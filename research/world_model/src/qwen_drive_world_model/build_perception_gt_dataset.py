"""Build action-conditioned BEV world-model windows from Waymo Perception.

Unlike the WOD-E2E packer, this path keeps the official LiDAR labels, map
features and ego poses. It produces 200x400 semantic/occupancy targets and
compact 9-channel BEV tensors for direct GT world-model training.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

from lidar_supervision.waymo_perception import iter_tfrecords

from .perception_gt import (
    BEV_HW,
    GT_MAP_CLASS_NAMES,
    OBJECT_CLASS_NAMES,
    PerceptionGtFrame,
    downsample_gt_bev,
    parse_gt_frame,
    rasterize_boxes,
    rasterize_map_features,
    select_boxes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/home/fractal/datasets/waymo_perception"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--horizons",
        type=float,
        nargs="+",
        default=(0.5, 1.0, 2.0, 3.0, 5.0),
    )
    parser.add_argument("--frame-rate", type=float, default=10.0)
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--limit-segments", type=int, default=None)
    parser.add_argument("--anchors-per-segment", type=int, default=40)
    parser.add_argument("--val-scenes", type=int, default=4)
    parser.add_argument("--model-hw", type=int, nargs=2, default=(50, 100))
    parser.add_argument("--max-boxes", type=int, default=64)
    parser.add_argument("--shard-size", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _horizon_key(seconds: float) -> str:
    return f"{float(seconds):g}s".replace(".", "p")


def _load_segment(path: Path, max_frames: int) -> tuple[list[PerceptionGtFrame], list]:
    frames = []
    map_features = []
    for payload in iter_tfrecords(path):
        frame = parse_gt_frame(payload)
        if not map_features and frame.map_features:
            map_features = frame.map_features
        frames.append(frame)
        if len(frames) >= max_frames:
            break
    if not frames:
        raise ValueError(f"{path} has no frames")
    if not map_features:
        raise ValueError(f"{path} has no map features in its first {len(frames)} frames")
    return frames, map_features


def _relative_pose(current: np.ndarray, future: np.ndarray) -> np.ndarray:
    transform = np.linalg.inv(current) @ future
    yaw = float(np.arctan2(transform[1, 0], transform[0, 0]))
    return np.asarray([transform[0, 3], transform[1, 3], yaw], dtype=np.float32)


def _trajectory(frames: list[PerceptionGtFrame], anchor: int, count: int) -> np.ndarray:
    values = np.zeros((count, 3), dtype=np.float32)
    for offset in range(count):
        values[offset] = _relative_pose(
            frames[anchor].ego_to_global,
            frames[anchor + offset].ego_to_global,
        )
    return values


def _driving_command(trajectory: np.ndarray) -> tuple[int, list[float]]:
    yaw = float(trajectory[-1, 2])
    if yaw > 0.17:
        return 2, [1.0, 0.0, 0.0, 0.0]
    if yaw < -0.17:
        return 1, [0.0, 0.0, 1.0, 0.0]
    return 0, [0.0, 1.0, 0.0, 0.0]


def _ego_status(
    trajectory: np.ndarray,
    frame_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    dt = 1.0 / frame_rate
    velocity = (trajectory[1, :2] - trajectory[0, :2]) / dt
    if len(trajectory) > 2:
        previous = (trajectory[1, :2] - trajectory[0, :2]) / dt
        current = (trajectory[2, :2] - trajectory[1, :2]) / dt
        acceleration = (current - previous) / dt
    else:
        acceleration = np.zeros(2, dtype=np.float32)
    return velocity.astype(np.float32), acceleration.astype(np.float32)


def _frame_raster(
    frame: PerceptionGtFrame,
    map_features: list,
    model_hw: tuple[int, int],
    max_boxes: int,
) -> dict[str, np.ndarray]:
    semantic = rasterize_map_features(map_features, frame.ego_to_global)
    occupancy = rasterize_boxes(frame.boxes)
    boxes, labels = select_boxes(frame.boxes, frame.box_labels, max_boxes=max_boxes)
    return {
        "map": semantic,
        "occ": occupancy,
        "z": downsample_gt_bev(semantic, occupancy, model_hw),
        "boxes": boxes,
        "box_labels": labels,
        "box_count": np.asarray((labels >= 0).sum(), dtype=np.int16),
    }


def _write_shard(path: Path, rows: list[dict[str, np.ndarray]]) -> None:
    keys = [
        "z_t",
        "map_t",
        "occ_t",
        "boxes_t",
        "box_labels_t",
        "box_counts_t",
    ]
    for horizon in rows[0]["horizons"]:
        keys.extend(
            [
                f"z_{horizon}",
                f"map_{horizon}",
                f"occ_{horizon}",
                f"boxes_{horizon}",
                f"box_labels_{horizon}",
                f"box_counts_{horizon}",
            ]
        )
    arrays = {key: np.stack([row[key] for row in rows]) for key in keys}
    arrays.update(
        {
            "tokens": np.asarray([row["token"] for row in rows]),
            "pair_ids": np.asarray([row["pair_id"] for row in rows]),
            "scene_tokens": np.asarray([row["scene_token"] for row in rows]),
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
    data_dir = args.data_dir
    output = args.output
    if output.exists() and args.overwrite:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    segment_paths = sorted(data_dir.glob("segment-*.tfrecord*"))
    if not segment_paths:
        raise SystemExit(f"no Waymo Perception tfrecords under {data_dir}")
    if args.limit_segments is not None:
        segment_paths = segment_paths[: args.limit_segments]
    val_count = min(max(args.val_scenes, 0), len(segment_paths))
    val_start = len(segment_paths) - val_count if val_count else len(segment_paths)
    val_scene_tokens = []

    frame_offsets = [
        int(round(seconds * args.frame_rate)) for seconds in args.horizons
    ]
    trajectory_count = int(round(max(args.horizons) * args.frame_rate)) + 1
    model_hw = tuple(int(value) for value in args.model_hw)
    shards = []
    buffer: list[dict[str, np.ndarray]] = []
    total_samples = 0
    train_samples = 0
    val_samples = 0
    started = time.perf_counter()

    for segment_number, path in enumerate(segment_paths, start=1):
        frames, map_features = _load_segment(path, args.max_frames)
        scene_token = frames[0].context_name
        segment_index = segment_number - 1
        split = "val" if segment_index >= val_start else "train"
        if split == "val":
            val_scene_tokens.append(scene_token)
        valid_anchors = [
            index
            for index in range(len(frames))
            if index + max(frame_offsets) < len(frames)
        ]
        if not valid_anchors:
            print(f"{path.name}: no 5 s windows")
            continue
        count = min(args.anchors_per_segment, len(valid_anchors))
        selected = np.linspace(0, len(valid_anchors) - 1, count, dtype=int)
        anchors = [valid_anchors[index] for index in sorted(set(selected.tolist()))]
        raster_cache: dict[int, dict[str, np.ndarray]] = {}

        def raster(index: int) -> dict[str, np.ndarray]:
            if index not in raster_cache:
                raster_cache[index] = _frame_raster(
                    frames[index],
                    map_features,
                    model_hw,
                    args.max_boxes,
                )
            return raster_cache[index]

        for anchor in anchors:
            current = raster(anchor)
            trajectory = _trajectory(frames, anchor, trajectory_count)
            velocity, acceleration = _ego_status(trajectory, args.frame_rate)
            nav_command, command = _driving_command(trajectory)
            relative_poses = np.stack(
                [_relative_pose(frames[anchor].ego_to_global, frames[anchor + offset].ego_to_global)
                 for offset in frame_offsets]
            )
            row = {
                "token": f"{scene_token}-{anchor:03d}",
                "pair_id": f"{scene_token}-{anchor:03d}",
                "scene_token": scene_token,
                "split": split,
                "t_frame": anchor,
                "nav_command": nav_command,
                "ego_velocity": velocity,
                "ego_acceleration": acceleration,
                "driving_command": np.asarray(command, np.float32),
                "trajectory_5s": trajectory,
                "relative_poses": relative_poses,
                "horizons": [_horizon_key(seconds) for seconds in args.horizons],
                "z_t": current["z"],
                "map_t": current["map"],
                "occ_t": current["occ"],
                "boxes_t": current["boxes"],
                "box_labels_t": current["box_labels"],
                "box_counts_t": current["box_count"],
            }
            for key, seconds, offset in zip(
                row["horizons"], args.horizons, frame_offsets, strict=True
            ):
                future = raster(anchor + offset)
                row[f"z_{key}"] = future["z"]
                row[f"map_{key}"] = future["map"]
                row[f"occ_{key}"] = future["occ"]
                row[f"boxes_{key}"] = future["boxes"]
                row[f"box_labels_{key}"] = future["box_labels"]
                row[f"box_counts_{key}"] = future["box_count"]
            buffer.append(row)
            total_samples += 1
            if split == "train":
                train_samples += 1
            else:
                val_samples += 1
            if len(buffer) >= args.shard_size:
                shard_path = output / f"pilot-{len(shards):05d}.npz"
                _write_shard(shard_path, buffer)
                shards.append(
                    {
                        "path": str(shard_path.resolve()),
                        "samples": len(buffer),
                        "train": sum(row["split"] == "train" for row in buffer),
                        "val": sum(row["split"] == "val" for row in buffer),
                    }
                )
                buffer = []

        print(
            f"{segment_number}/{len(segment_paths)} {scene_token} "
            f"frames={len(frames)} anchors={len(anchors)} split={split} "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )

    if buffer:
        shard_path = output / f"pilot-{len(shards):05d}.npz"
        _write_shard(shard_path, buffer)
        shards.append(
            {
                "path": str(shard_path.resolve()),
                "samples": len(buffer),
                "train": sum(row["split"] == "train" for row in buffer),
                "val": sum(row["split"] == "val" for row in buffer),
            }
        )

    metadata = {
        "samples": total_samples,
        "train_samples": train_samples,
        "val_samples": val_samples,
        "scenes": len({path.stem for path in segment_paths}),
        "val_scenes": val_scene_tokens,
        "horizons": list(args.horizons),
        "frame_rate": args.frame_rate,
        "bev_hw": list(BEV_HW),
        "model_hw": list(model_hw),
        "gt_map_classes": list(GT_MAP_CLASS_NAMES),
        "object_classes": list(OBJECT_CLASS_NAMES),
        "coordinate_protocol": "future ego-frame prediction",
        "source": "Waymo Open Dataset Perception",
        "supervision": "3D LiDAR labels, map features, and ego poses",
        "shards": shards,
    }
    (output / "dataset.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        f"wrote {total_samples} samples "
        f"({train_samples} train / {val_samples} val) to {output}"
    )


if __name__ == "__main__":
    main()

