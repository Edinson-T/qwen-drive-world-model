"""Plot the N sampled trajectories of selected scenes in BEV.

Left panel: full 5 s horizon. Right panel: zoom on the endpoints, where the
samples are usually only centimetres apart.

Usage:
    python tools/visualize_candidates.py \
        --predictions outputs/waymo_val/sft_6sample_stratified.jsonl \
        --indices 10 16 17 1 --out outputs/visualizations/waymo/candidates
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def resample(trajectory: np.ndarray, count: int = 20) -> np.ndarray:
    trajectory = np.asarray(trajectory, dtype=np.float64)
    if len(trajectory) == count:
        return trajectory
    source = np.linspace(0.0, 1.0, len(trajectory))
    target = np.linspace(0.0, 1.0, count)
    out = np.zeros((count, trajectory.shape[1]))
    out[:, :2] = np.stack(
        [np.interp(target, source, trajectory[:, i]) for i in range(2)], axis=1
    )
    return out


def scene_metrics(record: dict) -> tuple[float, float]:
    trajectories = np.asarray(record["trajectories"], dtype=np.float64)
    xy = trajectories[..., :2]
    fde = np.mean(
        [
            np.linalg.norm(xy[i, -1] - xy[j, -1])
            for i in range(len(xy))
            for j in range(i + 1, len(xy))
        ]
    )
    headroom = float("nan")
    preferences = record.get("preference_trajectories") or []
    scores = record.get("preference_scores") or []
    if preferences and scores:
        best = resample(np.asarray(preferences[int(np.argmax(scores))], dtype=np.float64), 20)
        errors = [
            np.linalg.norm(resample(t, 20)[:, :2] - best[:, :2], axis=-1).mean()
            for t in trajectories
        ]
        headroom = float(np.mean(errors) - np.min(errors))
    return float(fde), headroom


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--indices", type=int, nargs="+", required=True)
    parser.add_argument("--out", default="outputs/visualizations/waymo/candidates")
    parser.add_argument("--zoom", type=float, default=2.0, help="half-width of the endpoint zoom (m)")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = [json.loads(line) for line in Path(args.predictions).read_text().splitlines() if line.strip()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    colours = plt.get_cmap("tab10").colors

    for index in args.indices:
        record = records[index]
        trajectories = np.asarray(record["trajectories"], dtype=np.float64)
        ground_truth = record.get("future_trajectory")
        preferences = record.get("preference_trajectories") or []
        scores = record.get("preference_scores") or []
        fde, headroom = scene_metrics(record)

        figure, (full, zoom) = plt.subplots(1, 2, figsize=(12, 7))
        for axis in (full, zoom):
            for i, trajectory in enumerate(trajectories):
                axis.plot(
                    trajectory[:, 1],
                    trajectory[:, 0],
                    color=colours[i % len(colours)],
                    linewidth=1.5,
                    alpha=0.85,
                    label=f"sample {i}" if axis is full else None,
                )
                axis.scatter(
                    trajectory[-1, 1],
                    trajectory[-1, 0],
                    color=colours[i % len(colours)],
                    s=22,
                    zorder=6,
                )
            if ground_truth is not None:
                gt = np.asarray(ground_truth)
                axis.plot(gt[:, 1], gt[:, 0], "k--", linewidth=2, label="ground truth")
            if preferences and scores:
                best = np.asarray(preferences[int(np.argmax(scores))], dtype=np.float64)
                axis.plot(best[:, 1], best[:, 0], "--", color="green", linewidth=2, label="best rated")
            axis.scatter([0], [0], marker="s", color="red", s=60, zorder=7, label="ego")
            axis.set_aspect("equal")
            axis.grid(alpha=0.3)
            axis.set_xlabel("lateral y [m] (left +)")
            axis.set_ylabel("longitudinal x [m]")

        endpoints = trajectories[:, -1, :2]
        centre = endpoints.mean(0)
        for axis, half in ((full, 0.5), (zoom, args.zoom)):
            if axis is full:
                all_y = np.concatenate([trajectories[:, :, 1].ravel(), [0.0]])
                centre_y = 0.5 * (all_y.min() + all_y.max())
                span = max(3.0, 0.55 * (all_y.max() - all_y.min()))
                axis.set_xlim(centre_y - span, centre_y + span)
                axis.set_ylim(-3, trajectories[:, :, 0].max() + 3)
            else:
                axis.set_xlim(centre[1] - half, centre[1] + half)
                axis.set_ylim(centre[0] - half, centre[0] + half)

        full.legend(loc="upper left", fontsize=8)
        figure.suptitle(
            f"{record.get('token', '')}   pairwise FDE={fde:.3f} m   "
            f"headroom={headroom:.3f} m",
            fontsize=10,
        )
        figure.tight_layout()
        name = record.get("token", str(index)).replace("/", "_")
        figure.savefig(out / f"{index:03d}_{name}.png", dpi=150)
        plt.close(figure)
        print(f"{index:3d} {record.get('token','')}  FDE {fde:.3f}  headroom {headroom:.3f}")

    print(f"wrote {len(args.indices)} figures to {out}")


if __name__ == "__main__":
    main()
