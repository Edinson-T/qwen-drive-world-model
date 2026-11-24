"""Cache QD-domain action-conditioned future BEV latents for reward training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .model import BEVWorldModel, WorldModelConfig


HORIZONS = (0.5, 1.0, 2.0, 3.0, 5.0)
HORIZON_INDICES = (1, 3, 7, 11, 19)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--bev-cache", type=Path, required=True)
    parser.add_argument("--h0p5-run", type=Path, required=True)
    parser.add_argument("--h1-run", type=Path, required=True)
    parser.add_argument("--h2-run", type=Path, required=True)
    parser.add_argument("--h3-run", type=Path, required=True)
    parser.add_argument("--h5-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit-scenes", type=int, default=None)
    return parser.parse_args()


def _load_model(path: Path, device: torch.device) -> BEVWorldModel:
    checkpoint = torch.load(path / "best.pt", map_location=device, weights_only=False)
    model = BEVWorldModel(WorldModelConfig(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    candidates = np.load(args.candidates)
    models = [
        _load_model(path, device)
        for path in (
            args.h0p5_run,
            args.h1_run,
            args.h2_run,
            args.h3_run,
            args.h5_run,
        )
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    scene_count = (
        len(candidates["tokens"])
        if args.limit_scenes is None
        else min(args.limit_scenes, len(candidates["tokens"]))
    )
    for scene_index, token in enumerate(candidates["tokens"][:scene_count], start=1):
        token = str(token)
        with np.load(args.bev_cache / f"{token}.npz") as payload:
            current_z = torch.from_numpy(payload["z_t"]).float()
        trajectories = torch.from_numpy(
            candidates["trajectories"][scene_index - 1].astype(np.float32)
        )
        candidate_count = trajectories.shape[0]
        batch_z = current_z[None].repeat(candidate_count, 1, 1, 1).to(device)
        batch_trajectory = trajectories.to(device)
        ego = torch.zeros(
            (candidate_count, 8),
            dtype=torch.float32,
            device=device,
        )
        ego[:, 0] = float(candidates["initial_speed"][scene_index - 1])
        poses = torch.from_numpy(
            np.stack(
                [
                    candidates["trajectories"][scene_index - 1, :, index]
                    for index in HORIZON_INDICES
                ],
                axis=1,
            ).astype(np.float32)
        )
        future = []
        with torch.inference_mode():
            for model in models:
                future.append(
                    model(batch_z, batch_trajectory, ego)
                    .cpu()
                    .numpy()
                    .astype(np.float16)
                )
        np.savez_compressed(
            args.output / f"{token}.npz",
            future_z=np.stack(future, axis=1),
            relative_poses=poses.numpy(),
        )
        print(f"{scene_index}/{scene_count} {token}", flush=True)

    summary = {
        "scenes": scene_count,
        "candidates": int(candidates["trajectories"].shape[1]),
        "horizons": list(HORIZONS),
        "future_z_shape": [
            int(candidates["trajectories"].shape[1]),
            len(HORIZONS),
            128,
            16,
            32,
        ],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
