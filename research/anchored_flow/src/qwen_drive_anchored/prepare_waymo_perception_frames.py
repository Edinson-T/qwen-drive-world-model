"""Pack WOD-E2E current frames for the Qwen-Drive perception head."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

from .draw_camera_projection import _parse_calibrations


REPO_ROOT = Path(__file__).resolve().parents[4]
TOOLS_DIR = REPO_ROOT / "tools" / "waymo_e2e"

# Match the released nuPlan prompt order: front, then clockwise around the ring.
CAMERAS = (
    ("FRONT", "<FRONT VIEW>"),
    ("FRONT_RIGHT", "<FRONT RIGHT VIEW>"),
    ("SIDE_RIGHT", "<RIGHT VIEW>"),
    ("REAR_RIGHT", "<BACK RIGHT VIEW>"),
    ("REAR", "<BACK VIEW>"),
    ("REAR_LEFT", "<BACK LEFT VIEW>"),
    ("SIDE_LEFT", "<LEFT VIEW>"),
    ("FRONT_LEFT", "<FRONT LEFT VIEW>"),
)
CAMERA_IDS = {
    "FRONT": 1,
    "FRONT_LEFT": 2,
    "FRONT_RIGHT": 3,
    "SIDE_LEFT": 4,
    "SIDE_RIGHT": 5,
    "REAR_LEFT": 6,
    "REAR": 7,
    "REAR_RIGHT": 8,
}

# Waymo camera coordinates use x-forward, y-left, z-up. Qwen-Drive expects
# OpenCV optical coordinates (x-right, y-down, z-forward) in
# ``sensor2lidar_rotation``. This matrix maps a Waymo camera vector to optical.
WAYMO_CAMERA_TO_OPENCV = np.asarray(
    [
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _intrinsic_matrix(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            [values[0], 0.0, values[2]],
            [0.0, values[1], values[3]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def optical_camera_to_vehicle(camera_to_vehicle: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert Waymo camera extrinsics to Qwen-Drive's optical convention."""
    transform = np.asarray(camera_to_vehicle, dtype=np.float32)
    # If R maps Waymo camera -> vehicle, then R @ C.T maps optical -> vehicle.
    rotation = transform[:3, :3] @ WAYMO_CAMERA_TO_OPENCV.T
    translation = transform[:3, 3]
    return rotation, translation


def _write_frame(
    output_dir: Path,
    token: str,
    images: dict[str, bytes],
    calibrations: dict[int, tuple[np.ndarray, np.ndarray]],
) -> None:
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    content: list[dict] = []
    intrinsics = []
    rotations = []
    translations = []
    for camera, view_tag in CAMERAS:
        if camera not in images:
            raise ValueError(f"{token}: raw record is missing {camera}")
        camera_id = CAMERA_IDS[camera]
        if camera_id not in calibrations:
            raise ValueError(f"{token}: raw record is missing {camera} calibration")
        intrinsic, camera_to_vehicle = calibrations[camera_id]
        rotation, translation = optical_camera_to_vehicle(camera_to_vehicle)
        intrinsics.append(_intrinsic_matrix(intrinsic))
        rotations.append(rotation)
        translations.append(translation)
        (image_dir / f"{camera}.jpg").write_bytes(images[camera])
        content.extend(({"text": view_tag}, {"image": camera}))
    content.append({"text": "Analyze the current driving scene."})

    frame = {
        "dataset_type": "nuplan",
        "cam_order": [camera for camera, _ in CAMERAS],
        "content": content,
    }
    (output_dir / "frame.json").write_text(json.dumps(frame, indent=2) + "\n")
    np.savez_compressed(
        output_dir / "calib.npz",
        cam_intrinsic=np.stack(intrinsics),
        sensor2lidar_rotation=np.stack(rotations),
        sensor2lidar_translation=np.stack(translations),
        lidar2ego=np.eye(4, dtype=np.float32),
    )
    # PerceptionFrame requires the file, but inference does not consume GT.
    np.savez_compressed(
        output_dir / "gt.npz",
        boxes=np.empty((0, 9), dtype=np.float32),
    )


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(TOOLS_DIR))
    import e2e_proto

    manifests = sorted(args.manifests.glob("base_vs_bestn_*.json"))
    if not manifests:
        raise SystemExit(f"no base-vs-best-of-N manifests under {args.manifests}")
    tokens = [json.loads(path.read_text())["token"] for path in manifests]

    with gzip.open(args.index, "rt") as handle:
        index = json.load(handle)
    shards = [args.raw_root / name for name in index["shards"]]

    args.output.mkdir(parents=True, exist_ok=True)
    for position, token in enumerate(tokens, start=1):
        scene_token, frame_index = token.rsplit("-", 1)
        shard_id, offset, length = index["index"][scene_token][frame_index]
        payload = e2e_proto.read_record(
            shards[int(shard_id)],
            int(offset),
            int(length),
        )
        record = e2e_proto.parse_e2ed_frame(payload)
        calibrations = _parse_calibrations(payload, e2e_proto._iter_fields)
        output_dir = args.output / token
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_frame(output_dir, token, record.images, calibrations)
        print(f"{position}/{len(tokens)} {token}", flush=True)

    print(f"wrote {len(tokens)} perception frames to {args.output}")


if __name__ == "__main__":
    main()
