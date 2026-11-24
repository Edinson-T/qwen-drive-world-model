"""Measure whether the N trajectories sampled per scene are diverse enough to select from.

Usage:
    python tools/analyze_candidates.py --predictions outputs/waymo_val/sft_6sample.jsonl
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


def greedy_modes(points: np.ndarray, threshold: float = 1.0) -> int:
    centres = []
    for point in points:
        if not any(np.linalg.norm(point - centre) < threshold for centre in centres):
            centres.append(point)
    return max(1, len(centres))


def scene_metrics(record: dict) -> dict | None:
    trajectories = np.asarray(record.get("trajectories", []), dtype=np.float64)
    if trajectories.ndim != 3 or len(trajectories) < 2:
        return None
    n, steps, _ = trajectories.shape
    xy = trajectories[..., :2]

    pairwise = [
        np.linalg.norm(xy[i] - xy[j], axis=-1).mean()
        for i in range(n)
        for j in range(i + 1, n)
    ]
    final = [
        np.linalg.norm(xy[i, -1] - xy[j, -1])
        for i in range(n)
        for j in range(i + 1, n)
    ]
    scale = float(np.linalg.norm(xy[:, -1], axis=-1).mean())
    endpoint_std = float(np.linalg.norm(xy[:, -1] - xy[:, -1].mean(0), axis=-1).mean())
    metrics = {
        "samples": int(n),
        "pairwise_ade": float(np.mean(pairwise)),
        "pairwise_fde": float(np.mean(final)),
        "min_pairwise_fde": float(np.min(final)),
        "endpoint_std": endpoint_std,
        "scale": scale,
        "diversity_ratio": float(np.mean(final) / max(scale, 1e-6)),
        "modes_1m": greedy_modes(xy[:, -1], 1.0),
        "lateral_std_2s": float(np.std(xy[:, min(20, steps - 1), 1])),
        "lateral_std_5s": float(np.std(xy[:, -1, 1])),
    }

    ground_truth = record.get("future_trajectory")
    if ground_truth is not None:
        gt = resample(np.asarray(ground_truth, dtype=np.float64), 20)
        errors = [np.linalg.norm(resample(t, 20)[:, :2] - gt[:, :2], axis=-1).mean() for t in trajectories]
        metrics["first_ade_gt"] = float(errors[0])
        metrics["oracle_ade_gt"] = float(np.min(errors))
        metrics["mean_ade_gt"] = float(np.mean(errors))

    preferences = record.get("preference_trajectories") or []
    scores = record.get("preference_scores") or []
    if preferences and scores and len(preferences) == len(scores):
        best = resample(np.asarray(preferences[int(np.argmax(scores))], dtype=np.float64), 20)
        pref_errors = [
            np.linalg.norm(resample(t, 20)[:, :2] - best[:, :2], axis=-1).mean()
            for t in trajectories
        ]
        pairwise_matrix = np.linalg.norm(
            xy[:, None, :, :] - xy[None, :, :, :], axis=-1
        ).mean(-1)
        medoid = int(np.argmin(pairwise_matrix.sum(1)))
        metrics.update(
            {
                "first_ade_pref": float(pref_errors[0]),
                "medoid_ade_pref": float(pref_errors[medoid]),
                "mean_ade_pref": float(np.mean(pref_errors)),
                "oracle_ade_pref": float(np.min(pref_errors)),
            }
        )
        headroom = metrics["mean_ade_pref"] - metrics["oracle_ade_pref"]
        metrics["headroom_pref"] = float(headroom)
        metrics["headroom_ratio"] = float(headroom / max(metrics["mean_ade_pref"], 1e-6))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    rows = []
    for line in Path(args.predictions).read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        metrics = scene_metrics(record)
        if metrics is not None:
            metrics["token"] = record.get("token", "")
            rows.append(metrics)
    if not rows:
        raise SystemExit("no scenes with >=2 trajectories")

    def mean(key):
        values = [r[key] for r in rows if key in r]
        return float(np.mean(values)) if values else float("nan")

    summary = {
        "scenes": len(rows),
        "samples_per_scene": rows[0]["samples"],
        "pairwise_ade": mean("pairwise_ade"),
        "pairwise_fde": mean("pairwise_fde"),
        "min_pairwise_fde": mean("min_pairwise_fde"),
        "endpoint_std": mean("endpoint_std"),
        "diversity_ratio": mean("diversity_ratio"),
        "modes_1m": mean("modes_1m"),
        "lateral_std_2s": mean("lateral_std_2s"),
        "lateral_std_5s": mean("lateral_std_5s"),
        "first_ade_gt": mean("first_ade_gt"),
        "oracle_ade_gt": mean("oracle_ade_gt"),
        "mean_ade_gt": mean("mean_ade_gt"),
        "first_ade_pref": mean("first_ade_pref"),
        "medoid_ade_pref": mean("medoid_ade_pref"),
        "mean_ade_pref": mean("mean_ade_pref"),
        "oracle_ade_pref": mean("oracle_ade_pref"),
        "headroom_pref": mean("headroom_pref"),
        "headroom_ratio": mean("headroom_ratio"),
        "low_diversity_scenes_pct": float(
            100 * np.mean([r["diversity_ratio"] < 0.03 for r in rows])
        ),
        "meaningful_headroom_scenes_pct": float(
            100 * np.mean([r.get("headroom_ratio", 0) > 0.10 for r in rows])
        ),
        "all_samples_within_1m_pct": float(
            100 * np.mean([r["modes_1m"] == 1 for r in rows])
        ),
    }

    print(f"scenes: {len(rows)}  samples/scene: {rows[0]['samples']}")
    print(f"pairwise ADE (m):        {summary['pairwise_ade']:.3f}")
    print(f"pairwise FDE (m):        {summary['pairwise_fde']:.3f}")
    print(f"min pairwise FDE (m):    {summary['min_pairwise_fde']:.3f}")
    print(f"endpoint std (m):        {summary['endpoint_std']:.3f}")
    print(f"diversity ratio:         {summary['diversity_ratio']:.3f}")
    print(f"modes @1m:               {summary['modes_1m']:.2f}")
    print(f"lateral std @2s/5s (m):  {summary['lateral_std_2s']:.3f} / {summary['lateral_std_5s']:.3f}")
    if "mean_ade_pref" in rows[0]:
        print(f"pref ADE first/mean/oracle: {summary['first_ade_pref']:.3f} / "
              f"{summary['mean_ade_pref']:.3f} / {summary['oracle_ade_pref']:.3f}")
        print(f"selection headroom:      {summary['headroom_pref']:.3f} m "
              f"({100 * summary['headroom_ratio']:.1f}%)")
    print(f"low-diversity scenes:    {summary['low_diversity_scenes_pct']:.0f}%")
    print(f"headroom >10% scenes:    {summary['meaningful_headroom_scenes_pct']:.0f}%")
    print(f"all samples within 1m:   {summary['all_samples_within_1m_pct']:.0f}%")

    output = args.output or str(Path(args.predictions).with_suffix(".diversity.json"))
    Path(output).write_text(json.dumps({"summary": summary, "scenes": rows}, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
