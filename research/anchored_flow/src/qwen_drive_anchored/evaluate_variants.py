"""Compare Gaussian, anchored, and anchored-LoRA candidate sets."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from .anchors import AnchorBank, normalize_futures

__all__ = ["main", "parse_args"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="NAME=PREDICTIONS_JSONL; repeat once per method",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--anchors", default=None)
    return parser.parse_args(argv)


def _load(path: str | Path) -> dict[str, dict]:
    rows = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            rows[str(record["token"])] = record
    return rows


def _resample(trajectory: np.ndarray, count: int = 20) -> np.ndarray:
    trajectory = np.asarray(trajectory, dtype=np.float64)
    if len(trajectory) == count:
        return trajectory
    source = np.linspace(0.0, 1.0, len(trajectory))
    target = np.linspace(0.0, 1.0, count)
    out = np.zeros((count, trajectory.shape[1]), dtype=np.float64)
    for channel in range(trajectory.shape[1]):
        out[:, channel] = np.interp(target, source, trajectory[:, channel])
    return out


def _modes(points: np.ndarray, threshold: float = 1.0) -> int:
    centres = []
    for point in points:
        if not any(np.linalg.norm(point - centre) < threshold for centre in centres):
            centres.append(point)
    return max(1, len(centres))


def _metrics(record: dict, bank: AnchorBank | None) -> dict:
    trajectories = np.asarray(record["trajectories"], dtype=np.float64)
    if trajectories.ndim != 3 or trajectories.shape[-1] != 3:
        raise ValueError("trajectories must have shape [N, T, 3]")
    xy = trajectories[..., :2]
    pairs = [
        np.linalg.norm(xy[i] - xy[j], axis=-1).mean()
        for i in range(len(xy))
        for j in range(i + 1, len(xy))
    ]
    pair_fde = [
        np.linalg.norm(xy[i, -1] - xy[j, -1])
        for i in range(len(xy))
        for j in range(i + 1, len(xy))
    ]
    endpoint_std = float(np.linalg.norm(xy[:, -1] - xy[:, -1].mean(0), axis=-1).mean())
    mean_scale = float(np.linalg.norm(xy[:, -1], axis=-1).mean())
    out = {
        "candidates": int(len(trajectories)),
        "modes_1m": _modes(xy[:, -1]),
        "pairwise_ade": float(np.mean(pairs)),
        "pairwise_fde": float(np.mean(pair_fde)),
        "endpoint_std": endpoint_std,
        "diversity_ratio": float(np.mean(pair_fde) / max(mean_scale, 1e-6)),
        "all_within_1m": int(len(pairs) > 0 and max(pair_fde) < 1.0),
    }

    future = record.get("future_trajectory")
    gt_errors = None
    if future is not None:
        gt = _resample(np.asarray(future, dtype=np.float64))
        gt_errors = [
            np.linalg.norm(_resample(trajectory)[:, :2] - gt[:, :2], axis=-1).mean()
            for trajectory in trajectories
        ]
        out["first_ade_gt"] = float(gt_errors[0])
        out["mean_ade_gt"] = float(np.mean(gt_errors))
        out["oracle_ade_gt"] = float(np.min(gt_errors))

    preferences = record.get("preference_trajectories") or []
    scores = record.get("preference_scores") or []
    if preferences and scores and len(preferences) == len(scores):
        best = _resample(
            np.asarray(preferences[int(np.argmax(scores))], dtype=np.float64)
        )
        pref_errors = [
            np.linalg.norm(_resample(trajectory)[:, :2] - best[:, :2], axis=-1).mean()
            for trajectory in trajectories
        ]
        out["first_ade_pref"] = float(pref_errors[0])
        out["mean_ade_pref"] = float(np.mean(pref_errors))
        out["oracle_ade_pref"] = float(np.min(pref_errors))
        out["headroom_pref"] = float(np.mean(pref_errors) - np.min(pref_errors))
        out["headroom_ratio"] = float(
            out["headroom_pref"] / max(np.mean(pref_errors), 1e-6)
        )
        scores = record.get("candidate_scores")
        if scores is not None and len(scores) == len(trajectories):
            selected = int(record.get(
                "selected_index",
                int(np.argmax(scores)),
            ))
            out["selected_ade_pref"] = float(pref_errors[selected])

    scores = record.get("candidate_scores")
    if (
        gt_errors is not None
        and scores is not None
        and len(scores) == len(trajectories)
    ):
        selected = int(record.get(
            "selected_index",
            int(np.argmax(scores)),
        ))
        out["selected_ade_gt"] = float(gt_errors[selected])

    anchor_ids = np.asarray(record.get("anchor_ids", []), dtype=np.int64)
    if (
        bank is not None
        and len(anchor_ids) == len(trajectories)
        and np.all(anchor_ids >= 0)
    ):
        normalized = normalize_futures(trajectories.astype(np.float32), bank.scale)
        final_ids = np.asarray(
            [bank.nearest_xy(path) for path in normalized],
            dtype=np.int64,
        )
        out["anchor_retention"] = float(np.mean(final_ids == anchor_ids))
        out["final_anchor_modes"] = int(len(set(final_ids.tolist())))
        out["_start_anchor_ids"] = anchor_ids
        out["_final_anchor_ids"] = final_ids
    return out


def _parse_runs(items: list[str]) -> dict[str, Path]:
    runs = {}
    for item in items:
        name, separator, path = item.partition("=")
        if not separator or not name or not path:
            raise ValueError(f"invalid --run {item!r}; expected NAME=PATH")
        runs[name] = Path(path)
    return runs


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    mpl_config = output / ".mplconfig"
    mpl_config.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bank = AnchorBank.load(args.anchors) if args.anchors else None
    runs = _parse_runs(args.run)
    records = {name: _load(path) for name, path in runs.items()}
    token_sets = [set(rows) for rows in records.values()]
    common_tokens = sorted(set.intersection(*token_sets))
    if not common_tokens:
        raise SystemExit("the prediction files have no common tokens")

    results: dict[str, list[dict]] = {}
    for name, rows in records.items():
        results[name] = [
            {"token": token, **_metrics(rows[token], bank)}
            for token in common_tokens
        ]

    summary = {}
    scalar_keys = [
        "candidates",
        "modes_1m",
        "pairwise_ade",
        "pairwise_fde",
        "endpoint_std",
        "diversity_ratio",
        "all_within_1m",
        "first_ade_gt",
        "mean_ade_gt",
        "oracle_ade_gt",
        "first_ade_pref",
        "mean_ade_pref",
        "oracle_ade_pref",
        "selected_ade_pref",
        "selected_ade_gt",
        "headroom_pref",
        "headroom_ratio",
        "anchor_retention",
        "final_anchor_modes",
    ]
    for name, values in results.items():
        summary[name] = {
            key: float(np.mean([row[key] for row in values if key in row]))
            for key in scalar_keys
            if any(key in row for row in values)
        }

    (output / "summary.json").write_text(
        json.dumps({"tokens": len(common_tokens), "runs": summary}, indent=2)
    )
    with (output / "per_scene.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        keys = ["run", "token", *scalar_keys]
        writer.writerow(keys)
        for name, values in results.items():
            for row in values:
                writer.writerow([name, *[row.get(key, "") for key in keys[1:]]])

    names = list(results)

    def values(key: str) -> list[float]:
        return [summary[name].get(key, float("nan")) for name in names]

    plots = [
        ("modes_1m", "Modes @ 1 m", "higher"),
        ("pairwise_fde", "Pairwise FDE [m]", "higher"),
        ("headroom_ratio", "Selection headroom", "higher"),
        ("all_within_1m", "All within 1 m", "lower"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(17, 5))
    for axis, (key, title, _) in zip(axes, plots):
        axis.bar(names, values(key))
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "summary.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for axis, key, title in zip(
        axes,
        ("modes_1m", "headroom_ratio"),
        ("Per-scene modes @ 1 m", "Per-scene selection headroom"),
    ):
        axis.boxplot(
            [[row[key] for row in results[name] if key in row] for name in names],
            tick_labels=names,
            showmeans=True,
        )
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "per_scene_distributions.png", dpi=180)
    plt.close(fig)

    if bank is not None:
        for name, values in results.items():
            if not values or "_start_anchor_ids" not in values[0]:
                continue
            matrix = np.zeros((bank.num_anchors, bank.num_anchors), dtype=np.float64)
            for row in values:
                np.add.at(
                    matrix,
                    (row["_start_anchor_ids"], row["_final_anchor_ids"]),
                    1,
                )
            row_sums = matrix.sum(axis=1, keepdims=True)
            normalized = np.divide(
                matrix,
                row_sums,
                out=np.zeros_like(matrix),
                where=row_sums > 0,
            )
            fig, axis = plt.subplots(figsize=(10, 8))
            image = axis.imshow(normalized, cmap="viridis", vmin=0, vmax=1)
            axis.set_xlabel("final nearest anchor")
            axis.set_ylabel("initial anchor")
            axis.set_title(f"Anchor transition matrix: {name}")
            axis.set_xticks(range(bank.num_anchors))
            axis.set_yticks(range(bank.num_anchors))
            fig.colorbar(image, ax=axis, label="fraction of candidates")
            fig.tight_layout()
            safe_name = "".join(character if character.isalnum() else "_" for character in name)
            fig.savefig(output / f"anchor_transition_{safe_name}.png", dpi=180)
            plt.close(fig)
            np.savetxt(output / f"anchor_transition_{safe_name}.csv", matrix, delimiter=",")

    lines = [
        "# Candidate evaluation",
        "",
        f"Common validation tokens: {len(common_tokens)}",
        "",
        "| Method | N | modes@1m | Pair FDE | Headroom | Selected pref ADE | Oracle pref ADE | Anchor retention |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in names:
        row = summary[name]
        lines.append(
            f"| {name} | {row.get('candidates', float('nan')):.0f} "
            f"| {row.get('modes_1m', float('nan')):.3f} "
            f"| {row.get('pairwise_fde', float('nan')):.3f} "
            f"| {row.get('headroom_ratio', float('nan')):.3f} "
            f"| {row.get('selected_ade_pref', float('nan')):.3f} "
            f"| {row.get('oracle_ade_pref', float('nan')):.3f} "
            f"| {row.get('anchor_retention', float('nan')):.3f} |"
        )
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote evaluation to {output}")


if __name__ == "__main__":
    main()

