"""Projection helpers, standalone and dependency-free (numpy only).

Mirrors ``qwen_drive_perception/geometry.py::build_lidar2img`` so this sandbox
does not import the Qwen-Drive packages (and therefore never loads the model
weights). If the upstream formula changes, update both.
"""

from __future__ import annotations

import numpy as np

__all__ = ["build_lidar2img", "project_lidar_to_depth"]


def build_lidar2img(intrinsic, sensor2lidar_rotation, sensor2lidar_translation) -> np.ndarray:
    """Compose the 4x4 lidar -> image matrix of one camera (pre-resize)."""
    lidar2cam_r = np.linalg.inv(np.asarray(sensor2lidar_rotation, dtype=np.float64))
    lidar2cam_t = np.asarray(sensor2lidar_translation, dtype=np.float64) @ lidar2cam_r.T
    lidar2cam_rt = np.eye(4, dtype=np.float64)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t
    viewpad = np.eye(4, dtype=np.float64)
    viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = np.asarray(intrinsic, dtype=np.float64)
    return viewpad @ lidar2cam_rt.T


def project_lidar_to_depth(
    lidar_points: np.ndarray,
    lidar2img: np.ndarray,
    image_size: tuple[int, int],
    out_size: tuple[int, int] | None = None,
    min_depth: float = 1.0,
    max_depth: float = 80.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project a lidar sweep and z-buffer it into a per-pixel depth map.

    ``image_size`` is the native ``(width, height)`` the calibration belongs to.
    ``out_size`` is the target ``(width, height)``; points are splatted straight
    into that grid, so no resampling of the depth map is involved.

    Returns ``(depth, mask, u, v, z)`` where ``u, v, z`` are the per-point
    projection at native resolution (useful for overlays).
    """
    points = np.asarray(lidar_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"lidar points must be [N,>=3], got {points.shape}")
    native_w, native_h = int(image_size[0]), int(image_size[1])
    out_w, out_h = (native_w, native_h) if out_size is None else (int(out_size[0]), int(out_size[1]))

    homogeneous = np.concatenate([points[:, :3], np.ones((len(points), 1))], axis=1)
    projected = homogeneous @ np.asarray(lidar2img, dtype=np.float64).T
    z = projected[:, 2]
    safe_z = np.maximum(z, 1e-6)
    u = projected[:, 0] / safe_z
    v = projected[:, 1] / safe_z

    valid = (
        (z >= min_depth)
        & (z <= max_depth)
        & (u >= 0.0)
        & (u < native_w)
        & (v >= 0.0)
        & (v < native_h)
    )

    # Splat directly into the target grid: z-buffer keeps the nearest return.
    ut = u[valid] * (out_w / native_w)
    vt = v[valid] * (out_h / native_h)
    ui = np.clip(np.floor(ut).astype(np.int64), 0, out_w - 1)
    vi = np.clip(np.floor(vt).astype(np.int64), 0, out_h - 1)

    depth = np.full((out_h, out_w), np.inf, dtype=np.float32)
    np.minimum.at(depth, (vi, ui), z[valid].astype(np.float32))
    mask = np.isfinite(depth)
    depth[~mask] = 0.0
    return depth, mask, u, v, z
