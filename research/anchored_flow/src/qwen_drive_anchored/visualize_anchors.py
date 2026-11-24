"""Visualize anchor geometry and train/validation anchor coverage."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .anchors import AnchorBank, normalize_futures

__all__ = ["main", "parse_args"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchors", required=True)
    parser.add_argument("--train-scenes", required=True)
    parser.add_argument("--val-scenes", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument(
        "--scale",
        type=float,
        nargs=3,
        default=(165.0, 25.0, 1.5703125),
    )
    return parser.parse_args(argv)


def _load_futures(path: str | Path, limit: int | None) -> np.ndarray:
    futures = []
    with open(path) as handle:
        for line in handle:
            if limit is not None and len(futures) >= limit:
                break
            future = json.loads(line).get("trajectory", {}).get("future_traj_10hz")
            if future is None or len(future) < 2:
                continue
            array = np.asarray(future, dtype=np.float32)
            if array.shape[-1] == 3:
                futures.append(array)
    if not futures:
        raise ValueError(f"no future trajectories found in {path}")
    return np.stack(futures)


def _nearest(
    bank: AnchorBank, trajectories: np.ndarray, scale: np.ndarray, chunk: int = 2048
) -> tuple[np.ndarray, np.ndarray]:
    normalized = normalize_futures(trajectories, scale)
    query = normalized.reshape(len(normalized), -1).astype(np.float64)
    centres = bank.normalized.reshape(bank.num_anchors, -1).astype(np.float64)
    ids = np.empty(len(query), dtype=np.int64)
    distances = np.empty(len(query), dtype=np.float64)
    for start in range(0, len(query), chunk):
        block = query[start : start + chunk]
        values = np.linalg.norm(block[:, None] - centres[None], axis=-1)
        ids[start : start + len(block)] = values.argmin(axis=1)
        distances[start : start + len(block)] = values.min(axis=1)
    return ids, distances


def _coverage(distances: np.ndarray, thresholds: list[float]) -> dict[str, float]:
    return {
        f"{threshold:g}": float(np.mean(distances <= threshold))
        for threshold in thresholds
    }


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

    bank = AnchorBank.load(args.anchors)
    scale = np.asarray(args.scale, dtype=np.float32)
    train = _load_futures(args.train_scenes, args.limit_train)
    val = _load_futures(args.val_scenes, args.limit_val)
    train_ids, train_distances = _nearest(bank, train, scale)
    val_ids, val_distances = _nearest(bank, val, scale)

    colours = plt.get_cmap("tab20")(np.linspace(0, 1, bank.num_anchors))
    xy = bank.physical[:, :, :2]

    fig, axis = plt.subplots(figsize=(14, 10))
    for index, trajectory in enumerate(xy):
        axis.plot(
            trajectory[:, 1],
            trajectory[:, 0],
            color=colours[index],
            linewidth=2.2,
            label=f"A{index:02d} n={bank.counts[index]}",
        )
        axis.scatter(trajectory[-1, 1], trajectory[-1, 0], color=colours[index], s=28)
        axis.annotate(
            f"A{index:02d}",
            (trajectory[-1, 1], trajectory[-1, 0]),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=8,
        )
    axis.scatter([0], [0], marker="s", color="black", s=55, label="ego")
    axis.set_aspect("equal")
    axis.grid(alpha=0.25)
    axis.set_xlabel("lateral y [m]")
    axis.set_ylabel("longitudinal x [m]")
    axis.set_title("20 training-set K-Means trajectory anchors")
    axis.legend(ncol=2, fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(output / "anchors_overlay.png", dpi=180)
    plt.close(fig)

    endpoints = xy[:, -1]
    sizes = 30 + 260 * np.sqrt(bank.counts / max(1, bank.counts.max()))
    fig, axis = plt.subplots(figsize=(10, 8))
    axis.scatter(
        endpoints[:, 1],
        endpoints[:, 0],
        c=np.arange(bank.num_anchors),
        cmap="tab20",
        s=sizes,
        edgecolors="black",
        linewidths=0.5,
    )
    for index, point in enumerate(endpoints):
        axis.annotate(
            f"A{index:02d}",
            (point[1], point[0]),
            xytext=(5, 3),
            textcoords="offset points",
            fontsize=8,
        )
    axis.scatter([0], [0], marker="s", color="red", s=60, label="ego")
    axis.axhline(0, color="gray", linewidth=0.7)
    axis.axvline(0, color="gray", linewidth=0.7)
    axis.set_aspect("equal")
    axis.grid(alpha=0.25)
    axis.set_xlabel("lateral endpoint y [m]")
    axis.set_ylabel("longitudinal endpoint x [m]")
    axis.set_title("Anchor endpoints; marker size is sqrt(training count)")
    fig.tight_layout()
    fig.savefig(output / "anchor_endpoints.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 6))
    order = np.argsort(bank.counts)
    axis.barh([f"A{i:02d}" for i in order], bank.counts[order], color=colours[order])
    axis.set_xlabel("training scenes")
    axis.set_title("Anchor usage before anchor-balanced sampling")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "anchor_counts.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(4, 5, figsize=(15, 11), sharex=True, sharey=True)
    for index, axis in enumerate(axes.flat):
        axis.plot(xy[index, :, 1], xy[index, :, 0], color=colours[index], linewidth=2)
        axis.scatter([0], [0], marker="s", color="red", s=18)
        axis.scatter(xy[index, -1, 1], xy[index, -1, 0], color="black", s=12)
        axis.set_title(f"A{index:02d}  n={bank.counts[index]}", fontsize=9)
        axis.grid(alpha=0.2)
        axis.set_aspect("equal")
    for axis in axes[-1]:
        axis.set_xlabel("y [m]")
    for axis in axes[:, 0]:
        axis.set_ylabel("x [m]")
    fig.suptitle("Individual anchor modes")
    fig.tight_layout()
    fig.savefig(output / "anchor_shape_grid.png", dpi=160)
    plt.close(fig)

    upper = float(np.quantile(np.concatenate([train_distances, val_distances]), 0.99))
    bins = np.linspace(0.0, max(upper, 1.0), 50)
    fig, axis = plt.subplots(figsize=(10, 6))
    axis.hist(train_distances, bins=bins, density=True, alpha=0.55, label="training")
    axis.hist(val_distances, bins=bins, density=True, alpha=0.55, label="validation")
    axis.set_yscale("log")
    axis.set_xlabel("distance to nearest anchor in normalized trajectory space")
    axis.set_ylabel("density (log scale)")
    axis.set_title("Train/validation nearest-anchor distances")
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "nearest_anchor_distance.png", dpi=180)
    plt.close(fig)

    thresholds = np.linspace(0, max(upper, 1.0), 200)
    train_coverage = [np.mean(train_distances <= value) for value in thresholds]
    val_coverage = [np.mean(val_distances <= value) for value in thresholds]
    fig, axis = plt.subplots(figsize=(9, 6))
    axis.plot(thresholds, train_coverage, linewidth=2, label="training")
    axis.plot(thresholds, val_coverage, linewidth=2, label="validation")
    axis.set_xlabel("distance threshold in normalized trajectory space")
    axis.set_ylabel("fraction covered by an anchor")
    axis.set_title("Anchor-set coverage")
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "anchor_coverage.png", dpi=180)
    plt.close(fig)

    summary = {
        "num_anchors": bank.num_anchors,
        "anchor_counts": bank.counts.tolist(),
        "train_scenes": int(len(train)),
        "val_scenes": int(len(val)),
        "train_nearest_distance": {
            "mean": float(train_distances.mean()),
            "median": float(np.median(train_distances)),
            "p95": float(np.quantile(train_distances, 0.95)),
        },
        "val_nearest_distance": {
            "mean": float(val_distances.mean()),
            "median": float(np.median(val_distances)),
            "p95": float(np.quantile(val_distances, 0.95)),
        },
        "train_coverage": _coverage(train_distances, [0.25, 0.5, 1.0, 2.0, 5.0]),
        "val_coverage": _coverage(val_distances, [0.25, 0.5, 1.0, 2.0, 5.0]),
        "val_anchor_usage": np.bincount(val_ids, minlength=bank.num_anchors).tolist(),
    }
    (output / "anchor_diagnostics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote visualizations to {output}")


if __name__ == "__main__":
    main()

