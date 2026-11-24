"""Loader for the self-contained demo perception frames."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["Camera", "DemoFrame", "load_frames"]


@dataclass
class Camera:
    name: str
    image_path: Path
    intrinsic: np.ndarray
    sensor2lidar_rotation: np.ndarray
    sensor2lidar_translation: np.ndarray


@dataclass
class DemoFrame:
    path: Path
    token: str
    dataset_type: str
    cameras: list[Camera]
    lidar: np.ndarray
    lidar2ego: np.ndarray

    def image_size(self, index: int) -> tuple[int, int]:
        from PIL import Image

        with Image.open(self.cameras[index].image_path) as image:
            return image.size

    def pairs(self):
        """Yield ``(frame, camera_index)`` for every camera in the frame."""
        for index in range(len(self.cameras)):
            yield self, index


def _image_path(root: Path, entry: str) -> Path:
    candidate = root / entry if "/" in entry else root / "images" / f"{entry}.jpg"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"image not found for entry {entry!r} under {root}")


def load_frame(path: str | Path) -> DemoFrame:
    root = Path(path)
    meta = json.loads((root / "frame.json").read_text())
    calib = np.load(root / "calib.npz")
    cam_order = list(meta["cam_order"])

    image_entries = [item["image"] for item in meta["content"] if "image" in item]
    if len(image_entries) != len(cam_order):
        raise ValueError(f"{root}: {len(image_entries)} images but {len(cam_order)} cameras")

    cameras = [
        Camera(
            name=name,
            image_path=_image_path(root, entry),
            intrinsic=calib["cam_intrinsic"][index],
            sensor2lidar_rotation=calib["sensor2lidar_rotation"][index],
            sensor2lidar_translation=calib["sensor2lidar_translation"][index],
        )
        for index, (name, entry) in enumerate(zip(cam_order, image_entries))
    ]
    lidar_path = root / "lidar.npy"
    lidar = np.load(lidar_path) if lidar_path.exists() else np.zeros((0, 3), dtype=np.float32)

    return DemoFrame(
        path=root,
        token=root.name,
        dataset_type=meta.get("dataset_type", "unknown"),
        cameras=cameras,
        lidar=lidar,
        lidar2ego=calib["lidar2ego"],
    )


def load_frames(root: str | Path, limit: int | None = None) -> list[DemoFrame]:
    frames = []
    for path in sorted(Path(root).iterdir()):
        if path.is_dir() and (path / "frame.json").exists():
            frames.append(load_frame(path))
            if limit is not None and len(frames) >= limit:
                break
    if not frames:
        raise FileNotFoundError(f"no frame directories with frame.json under {root}")
    return frames
