"""Evaluate fixed-candidate linear selectors with scene-group cross-validation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .train_selector import MODES, _mode_features, _metrics


ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
MARGINS = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--objective",
        choices=("regression", "pairwise"),
        default="regression",
    )
    parser.add_argument(
        "--baseline-index",
        type=int,
        default=None,
        help="Keep this candidate unless an alternative exceeds it by a margin.",
    )
    parser.add_argument("--margins", type=float, nargs="+", default=list(MARGINS))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    return parser.parse_args()


def _scene_folds(scene_tokens: np.ndarray, fold_count: int, seed: int) -> np.ndarray:
    scenes = np.asarray(sorted(set(scene_tokens.tolist())))
    order = np.random.default_rng(seed).permutation(len(scenes))
    fold_by_scene = {
        scenes[int(scene_index)]: fold_index % fold_count
        for fold_index, scene_index in enumerate(order)
    }
    return np.asarray([fold_by_scene[scene] for scene in scene_tokens], dtype=np.int64)


def _fit_predict(
    values: np.ndarray,
    targets: np.ndarray,
    train_mask: np.ndarray,
    predict_mask: np.ndarray,
    alpha: float,
    objective: str,
) -> np.ndarray:
    mean = values[train_mask].mean((0, 1), keepdims=True)
    std = values[train_mask].std((0, 1), keepdims=True).clip(min=1e-5)
    normalized = (values - mean) / std
    centered = normalized - normalized.mean(2, keepdims=True)
    centered_target = targets - targets.mean(1, keepdims=True)
    if objective == "regression":
        x_train = centered[train_mask].reshape(-1, centered.shape[-1])
        y_train = centered_target[train_mask].reshape(-1)
        weights = np.linalg.solve(
            x_train.T @ x_train + alpha * np.eye(x_train.shape[1]),
            x_train.T @ y_train,
        )
    else:
        difference_rows = []
        difference_targets = []
        difference_weights = []
        for scene_index in np.flatnonzero(train_mask):
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
        x_train = np.asarray(difference_rows)
        y_train = np.asarray(difference_targets)
        sample_weights = np.asarray(difference_weights)
        weights = np.linalg.solve(
            x_train.T @ (sample_weights[:, None] * x_train)
            + alpha * np.eye(x_train.shape[1]),
            x_train.T @ (sample_weights * y_train),
        )
    return centered[predict_mask] @ weights


def _bootstrap_interval(
    differences: np.ndarray,
    *,
    seed: int,
    samples: int = 20000,
) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), (samples, len(differences)))
    means = differences[indices].mean(1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def _policy_selection(
    scores: np.ndarray,
    baseline_index: int | None,
    margin: float,
) -> np.ndarray:
    if baseline_index is None:
        return scores.argmax(1)
    alternative_scores = np.delete(scores, baseline_index, axis=1)
    alternative_indices = np.delete(np.arange(scores.shape[1]), baseline_index)
    best_alternative = alternative_scores.argmax(1)
    alternative_values = alternative_scores[
        np.arange(len(alternative_scores)),
        best_alternative,
    ]
    switch = alternative_values > scores[:, baseline_index] + margin
    return np.where(
        switch,
        alternative_indices[best_alternative],
        baseline_index,
    )


def _metrics_from_selection(
    selected: np.ndarray,
    targets: np.ndarray,
    baseline_index: int | None,
) -> dict[str, float]:
    baseline_index = 0 if baseline_index is None else baseline_index
    selected_rfs = targets[np.arange(len(targets)), selected]
    oracle = targets.max(1)
    best = targets.argmax(1)
    return {
        "top1_rfs": float(selected_rfs.mean()),
        "mean_candidate_rfs": float(targets.mean(1).mean()),
        "oracle_rfs": float(oracle.mean()),
        "rfs_gap": float((oracle - selected_rfs).mean()),
        "first_candidate_rfs": float(targets[:, baseline_index].mean()),
        "top_oracle_accuracy": float((selected == best).mean()),
    }


def main() -> None:
    args = parse_args()
    data = np.load(args.features)
    targets = np.asarray(data["candidate_rfs"], dtype=np.float64)
    scene_tokens = np.asarray(data["scene_tokens"])
    folds = _scene_folds(scene_tokens, args.folds, args.seed)
    report = {
        "protocol": f"scene-group nested cross-validation ({args.objective})",
        "objective": args.objective,
        "folds": args.folds,
        "scenes": len(set(scene_tokens.tolist())),
        "candidates": int(targets.shape[1]),
        "baseline_index": args.baseline_index,
        "margins": args.margins,
        "fold_counts": {
            str(fold): int((folds == fold).sum()) for fold in range(args.folds)
        },
        "alphas": list(ALPHAS),
        "modes": {},
    }
    predictions = {}
    prediction_scores = {}
    for mode in args.modes:
        values = np.asarray(_mode_features(data, mode), dtype=np.float64)
        out_of_fold_scores = np.empty(targets.shape, dtype=np.float64)
        out_of_fold_selected = np.empty(len(targets), dtype=np.int64)
        selected_alphas = []
        selected_margins = []
        for outer_fold in range(args.folds):
            outer_train = folds != outer_fold
            outer_test = folds == outer_fold
            inner_folds = [fold for fold in range(args.folds) if fold != outer_fold]
            alpha_scores = []
            for alpha in ALPHAS:
                predictions_by_fold = []
                targets_by_fold = []
                for inner_fold in inner_folds:
                    inner_train = outer_train & (folds != inner_fold)
                    inner_val = folds == inner_fold
                    prediction = _fit_predict(
                        values,
                        targets,
                        inner_train,
                        inner_val,
                        alpha,
                        args.objective,
                    )
                    predictions_by_fold.append(prediction)
                    targets_by_fold.append(targets[inner_val])
                metrics = _metrics(
                    np.concatenate(predictions_by_fold, axis=0),
                    np.concatenate(targets_by_fold, axis=0),
                )
                alpha_scores.append(metrics["top1_rfs"])
            selected_alpha = ALPHAS[int(np.argmax(alpha_scores))]
            selected_alphas.append(selected_alpha)
            if args.baseline_index is None:
                selected_margin = 0.0
            else:
                inner_scores = []
                inner_targets = []
                for inner_fold in inner_folds:
                    inner_train = outer_train & (folds != inner_fold)
                    inner_val = folds == inner_fold
                    inner_scores.append(
                        _fit_predict(
                            values,
                            targets,
                            inner_train,
                            inner_val,
                            selected_alpha,
                            args.objective,
                        )
                    )
                    inner_targets.append(targets[inner_val])
                margin_targets = np.concatenate(inner_targets, axis=0)
                margin_scores = np.concatenate(inner_scores, axis=0)
                margin_values = []
                for margin in args.margins:
                    inner_selected = _policy_selection(
                        margin_scores,
                        args.baseline_index,
                        margin,
                    )
                    inner_metrics = _metrics_from_selection(
                        inner_selected,
                        margin_targets,
                        args.baseline_index,
                    )
                    margin_values.append(inner_metrics["top1_rfs"])
                selected_margin = args.margins[int(np.argmax(margin_values))]
            selected_margins.append(selected_margin)
            outer_scores = _fit_predict(
                values,
                targets,
                outer_train,
                outer_test,
                selected_alpha,
                args.objective,
            )
            out_of_fold_scores[outer_test] = outer_scores
            out_of_fold_selected[outer_test] = _policy_selection(
                outer_scores,
                args.baseline_index,
                selected_margin,
            )
        overall = _metrics_from_selection(
            out_of_fold_selected,
            targets,
            args.baseline_index,
        )
        baseline_index = 0 if args.baseline_index is None else args.baseline_index
        selected = out_of_fold_selected
        differences = (
            targets[np.arange(len(targets)), selected]
            - targets[:, baseline_index]
        )
        overall["ci95_vs_first_candidate"] = _bootstrap_interval(
            differences,
            seed=args.seed,
        )
        overall["changed_from_first_candidate"] = int(
            (selected != baseline_index).sum()
        )
        overall["wins_against_first_candidate"] = int((differences > 0).sum())
        overall["losses_against_first_candidate"] = int((differences < 0).sum())
        report["modes"][mode] = {
            "feature_dim": int(values.shape[-1]),
            "selected_alphas": selected_alphas,
            "selected_margins": selected_margins,
            "metrics": overall,
        }
        predictions[mode] = selected
        prediction_scores[mode] = out_of_fold_scores

    if "shared" in predictions and "base" in predictions:
        shared_rfs = targets[np.arange(len(targets)), predictions["shared"]]
        base_rfs = targets[np.arange(len(targets)), predictions["base"]]
        shared_minus_base = shared_rfs - base_rfs
        report["shared_minus_base"] = {
            "mean_rfs": float(shared_minus_base.mean()),
            "ci95": _bootstrap_interval(shared_minus_base, seed=args.seed + 1),
            "wins": int((shared_minus_base > 0).sum()),
            "losses": int((shared_minus_base < 0).sum()),
            "ties": int((shared_minus_base == 0).sum()),
        }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    np.savez_compressed(
        args.output / "predictions.npz",
        folds=folds,
        tokens=data["tokens"],
        scene_tokens=data["scene_tokens"],
        candidate_rfs=targets,
        **predictions,
        **{
            f"{mode}_scores": prediction_scores[mode]
            for mode in args.modes
        },
    )
    with (args.output / "per_scene.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "token",
                "scene_token",
                "fold",
                "first_candidate_rfs",
                "oracle_rfs",
                *args.modes,
            ]
        )
        for scene_index in range(len(targets)):
            row = [
                str(data["tokens"][scene_index]),
                str(scene_tokens[scene_index]),
                int(folds[scene_index]),
                float(targets[scene_index, 0]),
                float(targets[scene_index].max()),
            ]
            row.extend(
                float(targets[scene_index, predictions[mode][scene_index]])
                for mode in args.modes
            )
            writer.writerow(row)

    lines = [
        f"# Scene-group cross-validation: {args.objective}",
        "",
        "| Selector | Top-1 RFS | First candidate | Random@20 | Oracle | 95% CI vs first |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in args.modes:
        item = report["modes"][mode]["metrics"]
        interval = item["ci95_vs_first_candidate"]
        lines.append(
            f"| {mode} | {item['top1_rfs']:.4f} | "
            f"{item['first_candidate_rfs']:.4f} | "
            f"{item['mean_candidate_rfs']:.4f} | "
            f"{item['oracle_rfs']:.4f} | "
            f"[{interval[0]:+.4f}, {interval[1]:+.4f}] |"
        )
    if "shared_minus_base" in report:
        comparison = report["shared_minus_base"]
        interval = comparison["ci95"]
        lines += [
            "",
            f"Shared minus base: `{comparison['mean_rfs']:+.4f}` RFS, "
            f"95% CI `[{interval[0]:+.4f}, {interval[1]:+.4f}]`, "
            f"{comparison['wins']} wins / {comparison['losses']} losses / "
            f"{comparison['ties']} ties.",
        ]
    (args.output / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
