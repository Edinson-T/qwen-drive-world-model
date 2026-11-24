"""Fit final pairwise-linear selectors over fixed candidate features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .train_selector import MODES, _mode_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--baseline-index", type=int, default=None)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    return parser.parse_args()


def fit_pairwise_linear(
    values: np.ndarray,
    targets: np.ndarray,
    alpha: float,
) -> dict[str, object]:
    mean = values.mean((0, 1), keepdims=True)
    std = values.std((0, 1), keepdims=True).clip(min=1e-5)
    centered = (values - mean) / std
    centered -= centered.mean(2, keepdims=True)
    centered_target = targets - targets.mean(1, keepdims=True)

    difference_rows = []
    difference_targets = []
    difference_weights = []
    for scene_index in range(centered.shape[0]):
        for low_index in range(centered.shape[1] - 1):
            for high_index in range(low_index + 1, centered.shape[1]):
                target_difference = (
                    centered_target[scene_index, high_index]
                    - centered_target[scene_index, low_index]
                )
                if abs(target_difference) < 1e-8:
                    continue
                if target_difference > 0:
                    first, second = low_index, high_index
                else:
                    first, second = high_index, low_index
                difference_rows.append(
                    centered[scene_index, second]
                    - centered[scene_index, first]
                )
                difference_targets.append(abs(target_difference))
                difference_weights.append(abs(target_difference))

    design = np.asarray(difference_rows, dtype=np.float64)
    pair_targets = np.asarray(difference_targets, dtype=np.float64)
    pair_weights = np.asarray(difference_weights, dtype=np.float64)
    weights = np.linalg.solve(
        design.T @ (pair_weights[:, None] * design)
        + alpha * np.eye(design.shape[1]),
        design.T @ (pair_weights * pair_targets),
    )
    return {
        "feature_mean": mean,
        "feature_std": std,
        "weights": weights,
        "alpha": alpha,
        "input_dim": int(values.shape[-1]),
        "pairs": int(len(design)),
    }


def main() -> None:
    args = parse_args()
    data = np.load(args.features)
    targets = np.asarray(data["candidate_rfs"], dtype=np.float64)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {
        "scenes": int(targets.shape[0]),
        "scene_groups": len(set(data["scene_tokens"].tolist())),
        "candidates": int(targets.shape[1]),
        "alpha": args.alpha,
        "baseline_index": args.baseline_index,
        "margin": args.margin,
        "note": "Weights are fit on all validation candidates; report performance from nested CV.",
        "modes": {},
    }
    for mode in args.modes:
        values = np.asarray(_mode_features(data, mode), dtype=np.float64)
        checkpoint = fit_pairwise_linear(values, targets, args.alpha)
        checkpoint["baseline_index"] = args.baseline_index
        checkpoint["margin"] = args.margin
        np.savez_compressed(args.output / f"{mode}.npz", **checkpoint)
        summary["modes"][mode] = {
            "input_dim": checkpoint["input_dim"],
            "pairs": checkpoint["pairs"],
        }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
