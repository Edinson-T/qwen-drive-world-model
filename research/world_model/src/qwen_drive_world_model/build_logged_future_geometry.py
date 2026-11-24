"""Build explicit logged-future occupancy geometry for candidate ranking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .build_selector_features import _occupancy_from_qd


KEY_BY_HORIZON = ("occ_0p5s", "occ_1s", "occ_2s", "occ_3s", "occ_5s")
POINT_INDEX = (1, 3, 7, 11, 19)
FEATURE_NAMES = (
    "footprint_mean_occupancy",
    "footprint_max_occupancy",
    "nearest_obstacle_distance_normalized",
    "front_clearance_normalized",
    "occupied_cells_within_10m_per_1000",
    "occupied_fraction",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--pair-cache", type=Path, required=True)
    parser.add_argument("--oracle-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _warp_occupancy(occupancy: np.ndarray, log_pose, candidate_pose):
    value = torch.from_numpy(occupancy.astype(np.float32))[None, None]
    x = torch.linspace(-30 + 0.6 / 2, 30 - 0.6 / 2, 100)
    y = torch.linspace(-15 + 0.6 / 2, 15 - 0.6 / 2, 50)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    dx, dy, yaw = candidate_pose
    lx, ly, lyaw = log_pose
    cosine, sine = np.cos(yaw), np.sin(yaw)
    wx = cosine * xx - sine * yy + dx
    wy = sine * xx + cosine * yy + dy
    cosine, sine = np.cos(lyaw), np.sin(lyaw)
    px = cosine * (wx - lx) + sine * (wy - ly)
    py = -sine * (wx - lx) + cosine * (wy - ly)
    grid = torch.stack([px / 30, py / 15], -1)[None]
    sampled = F.grid_sample(
        value, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )[0, 0]
    valid = (px.abs() <= 30) & (py.abs() <= 15)
    return sampled.numpy(), valid.numpy()


def _geometry(occupancy: np.ndarray, threshold: float = 0.5) -> list[float]:
    x = np.linspace(-30 + 0.3, 30 - 0.3, 100)[None].repeat(50, 0)
    y = np.linspace(-15 + 0.3, 15 - 0.3, 50)[:, None].repeat(100, 1)
    footprint = (np.abs(x) <= 2.4) & (np.abs(y) <= 1.0)
    occupied = occupancy >= threshold
    footprint_values = occupancy[footprint]
    distances = np.hypot(x[occupied], y[occupied])
    nearest = float(distances.min()) if len(distances) else 50.0
    near = occupied & (np.hypot(x, y) <= 10.0)
    front = occupied & (x > 0) & (np.abs(y) <= 2.5)
    front_clearance = float(x[front].min()) if front.any() else 30.0
    return [
        float(footprint_values.mean()),
        float(footprint_values.max()),
        min(nearest, 50.0) / 50.0,
        min(front_clearance, 30.0) / 30.0,
        float(near.sum()) / 1000.0,
        float(occupied.sum()) / occupancy.size,
    ]


def main() -> None:
    args = parse_args()
    candidates = np.load(args.candidates)
    oracle_paths = {path.stem: path for path in args.oracle_cache.glob("*.npz")}
    indices = []
    for index, token in enumerate(candidates["tokens"]):
        token = str(token)
        if token in oracle_paths and (args.pair_cache / f"{token}.npz").exists():
            indices.append(index)
    subset = {key: candidates[key][indices] for key in candidates.files}
    features = []
    valid_fraction = []
    for row, index in enumerate(indices, start=1):
        token = str(candidates["tokens"][index])
        with np.load(oracle_paths[token]) as payload:
            logged_poses = payload["relative_poses"]
        with np.load(args.pair_cache / f"{token}.npz") as payload:
            occupancy = np.stack(
                [_occupancy_from_qd(payload[key])[0] for key in KEY_BY_HORIZON]
            )
        candidate_poses = np.stack(
            [
                candidates["trajectories"][index, :, point]
                for point in POINT_INDEX
            ],
            axis=1,
        )
        scene_features = []
        scene_valid = []
        for candidate_index in range(candidate_poses.shape[0]):
            candidate_features = []
            candidate_valid = []
            for horizon in range(5):
                warped, valid = _warp_occupancy(
                    occupancy[horizon],
                    logged_poses[horizon],
                    candidate_poses[candidate_index, horizon],
                )
                candidate_features.extend(_geometry(warped))
                candidate_valid.append(float(valid.mean()))
            scene_features.append(candidate_features)
            scene_valid.append(candidate_valid)
        features.append(scene_features)
        valid_fraction.append(scene_valid)
        print(f"{row}/{len(indices)} {token}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "candidates.npz", **subset)
    np.savez_compressed(
        args.output / "geometry.npz",
        geometry=np.asarray(features, dtype=np.float32),
        valid_fraction=np.asarray(valid_fraction, dtype=np.float32),
    )
    (args.output / "summary.json").write_text(
        json.dumps(
            {"scenes": len(indices), "candidates": subset["trajectories"].shape[1]},
            indent=2,
        )
        + "\n"
    )
    (args.output / "feature_schema.json").write_text(
        json.dumps(
            {
                "features_per_horizon": len(FEATURE_NAMES),
                "dimensions": len(FEATURE_NAMES) * 5,
                "order": [
                    f"{name}@{horizon}"
                    for horizon in ("0.5s", "1s", "2s", "3s", "5s")
                    for name in FEATURE_NAMES
                ],
                "units_and_missing_rules": {
                    "footprint_mean_occupancy": "probability [0,1]; 0 when no cells",
                    "footprint_max_occupancy": "probability [0,1]; 0 when no cells",
                    "nearest_obstacle_distance_normalized": "distance / 50m clipped [0,1]; 1 when none",
                    "front_clearance_normalized": "forward clearance / 30m clipped [0,1]; 1 when none",
                    "occupied_cells_within_10m_per_1000": "occupied cell count / 1000",
                    "occupied_fraction": "occupied cells / all grid cells",
                },
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
