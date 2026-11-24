"""Build trajectory-only features for fixed RL direct+anchor candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .build_selector_features import _trajectory_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache = np.load(args.candidates)
    rows = []
    for scene_index in range(len(cache["tokens"])):
        scene_rows = []
        for candidate_index in range(cache["trajectories"].shape[1]):
            scene_rows.append(
                _trajectory_features(
                    cache["trajectories"][scene_index, candidate_index],
                    int(cache["anchor_ids"][scene_index, candidate_index]),
                    float(cache["initial_speed"][scene_index]),
                )
            )
        rows.append(scene_rows)
    base = np.asarray(rows, dtype=np.float32)

    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / "features.npz",
        tokens=cache["tokens"],
        scene_tokens=cache["scene_tokens"],
        split=cache["split"],
        candidate_rfs=cache["candidate_rfs"],
        anchor_ids=cache["anchor_ids"],
        base=base,
    )
    summary = {
        "scenes": int(base.shape[0]),
        "records": int(base.shape[0]),
        "scene_groups": len(set(cache["scene_tokens"].tolist())),
        "candidates": int(base.shape[1]),
        "feature_dim": int(base.shape[2]),
        "split_counts": {
            name: int((cache["split"] == name).sum())
            for name in ("train", "val", "test")
        },
        "scene_split_counts": {
            name: len(
                {
                    scene_token
                    for scene_token, split in zip(
                        cache["scene_tokens"], cache["split"]
                    )
                    if split == name
                }
            )
            for name in ("train", "val", "test")
        },
        "mean_candidate_rfs": float(cache["candidate_rfs"].mean()),
        "mean_oracle_rfs": float(cache["candidate_rfs"].max(1).mean()),
        "mean_first_candidate_rfs": float(cache["candidate_rfs"][:, 0].mean()),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
