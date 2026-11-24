"""Build predicted-future occupancy geometry with the logged-geometry schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .build_logged_future_geometry import _geometry
from .build_selector_features import (
    _downsample_map,
    _load_independent,
    _occupancy_from_qd,
)


FEATURE_NAMES = (
    "footprint_mean_occupancy",
    "footprint_max_occupancy",
    "nearest_obstacle_distance_normalized",
    "front_clearance_normalized",
    "occupied_cells_within_10m_per_1000",
    "occupied_fraction",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--bev-cache", type=Path, required=True)
    parser.add_argument("--h0p5-run", type=Path, required=True)
    parser.add_argument("--h1-run", type=Path, required=True)
    parser.add_argument("--h2-run", type=Path, required=True)
    parser.add_argument("--h3-run", type=Path, required=True)
    parser.add_argument("--h5-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs=5,
        default=(0.5, 0.5, 0.5, 0.5, 0.5),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    candidates = np.load(args.data / "candidates.npz")
    models = [
        _load_independent(path, device)
        for path in (
            args.h0p5_run,
            args.h1_run,
            args.h2_run,
            args.h3_run,
            args.h5_run,
        )
    ]
    geometry = []
    for scene_index, token in enumerate(candidates["tokens"], start=1):
        token = str(token)
        with np.load(args.bev_cache / f"{token}.npz") as payload:
            map_low = _downsample_map(payload["map_t"])
            occupancy, height = _occupancy_from_qd(payload["occ_t"])
            latent = payload["z_t"]
        current_bev = np.concatenate([map_low, occupancy[None], height])
        batch = candidates["trajectories"][scene_index - 1].shape[0]
        model_input = (
            latent
            if models[0].config.input_channels == latent.shape[0]
            else current_bev
        )
        z = torch.from_numpy(model_input)[None].repeat(batch, 1, 1, 1).to(device)
        trajectory = torch.from_numpy(
            candidates["trajectories"][scene_index - 1].astype(np.float32)
        ).to(device)
        ego = torch.zeros((batch, 8), dtype=torch.float32, device=device)
        ego[:, 0] = float(candidates["initial_speed"][scene_index - 1])
        scene_geometry = []
        with torch.inference_mode():
            outputs = [model(z, trajectory, ego) for model in models]
        for candidate_index in range(batch):
            row = []
            for horizon_index, output in enumerate(outputs):
                probability = output["occupancy_logits"][candidate_index].sigmoid().cpu().numpy()
                if probability.shape != (50, 100):
                    probability = (
                        torch.nn.functional.interpolate(
                            torch.from_numpy(probability)[None, None],
                            size=(50, 100),
                            mode="bilinear",
                            align_corners=False,
                        )[0, 0]
                        .numpy()
                    )
                row.extend(_geometry(probability, args.thresholds[horizon_index]))
            scene_geometry.append(row)
        geometry.append(scene_geometry)
        print(f"{scene_index}/{len(candidates['tokens'])} {token}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "candidates.npz", **{key: candidates[key] for key in candidates.files})
    np.savez_compressed(
        args.output / "geometry.npz",
        geometry=np.asarray(geometry, dtype=np.float32),
        valid_fraction=np.ones(
            (len(geometry), candidates["trajectories"].shape[1], 5),
            dtype=np.float32,
        ),
    )
    schema = {
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
    }
    (args.output / "feature_schema.json").write_text(
        json.dumps(schema, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
