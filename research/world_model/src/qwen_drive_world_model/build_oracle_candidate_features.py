"""Warp logged future QD BEV into each candidate's future ego frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--oracle-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _warp(candidate_z: np.ndarray, log_pose: np.ndarray, candidate_pose: np.ndarray):
    z = torch.from_numpy(candidate_z.astype(np.float32))
    x = torch.linspace(-30 + 60 / 64, 30 - 60 / 64, 32)
    y = torch.linspace(-15 + 30 / 32, 15 - 30 / 32, 16)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    points = torch.stack([xx, yy], dim=-1)
    dx, dy, yaw = candidate_pose
    lx, ly, lyaw = log_pose
    cosine, sine = np.cos(yaw), np.sin(yaw)
    wx = cosine * points[..., 0] - sine * points[..., 1] + dx
    wy = sine * points[..., 0] + cosine * points[..., 1] + dy
    cosine, sine = np.cos(lyaw), np.sin(lyaw)
    px = cosine * (wx - lx) + sine * (wy - ly)
    py = -sine * (wx - lx) + cosine * (wy - ly)
    grid = torch.stack([px / 30, py / 15], dim=-1).unsqueeze(0)
    sampled = F.grid_sample(
        z.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )[0]
    valid = (px.abs() <= 30) & (py.abs() <= 15)
    return sampled.numpy().astype(np.float16), valid.numpy()


def main() -> None:
    args = parse_args()
    candidates = np.load(args.candidates)
    available = {
        path.stem: path for path in args.oracle_cache.glob("*.npz")
    }
    indices = [
        index
        for index, token in enumerate(candidates["tokens"])
        if str(token) in available
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    future_dir = args.output / "future"
    future_dir.mkdir(exist_ok=True)
    subset = {
        key: candidates[key][indices]
        for key in candidates.files
    }
    np.savez_compressed(args.output / "candidates.npz", **subset)
    for row, index in enumerate(indices, start=1):
        token = str(candidates["tokens"][index])
        with np.load(available[token]) as payload:
            actual = payload["future_z"]
            logged_poses = payload["relative_poses"]
        candidate_poses = np.stack(
            [candidates["trajectories"][index, :, point] for point in (1, 3, 7, 11, 19)],
            axis=1,
        )
        warped = []
        masks = []
        for horizon in range(5):
            candidate_horizon = []
            candidate_mask = []
            for candidate_index in range(candidate_poses.shape[0]):
                value, mask = _warp(
                    actual[horizon],
                    logged_poses[horizon],
                    candidate_poses[candidate_index, horizon],
                )
                candidate_horizon.append(value)
                candidate_mask.append(mask)
            warped.append(candidate_horizon)
            masks.append(candidate_mask)
        np.savez_compressed(
            future_dir / f"{token}.npz",
            future_z=np.asarray(warped, dtype=np.float16).transpose(1, 0, 2, 3, 4),
            valid=np.asarray(masks, dtype=bool).transpose(1, 0, 2, 3),
            relative_poses=candidate_poses.astype(np.float32),
        )
        print(f"{row}/{len(indices)} {token}", flush=True)
    (args.output / "summary.json").write_text(
        json.dumps({"scenes": len(indices)}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
