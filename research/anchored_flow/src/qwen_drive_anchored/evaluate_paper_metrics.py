"""Evaluate candidate sets with WOD-E2E and DiffusionDriveV2 metrics.

This intentionally excludes project-specific diagnostics such as ``modes@1m``
and ADE-based selection headroom. The reported quantities are:

* WOD-E2E RFS for the deployed/first candidate;
* WOD-E2E ADE for that same candidate;
* DiffusionDriveV2 ``Div`` on the generated candidate set;
* mean per-candidate official RFS;
* Oracle RFS, the offline maximum over candidates;
* RFS gap, defined as Oracle RFS minus deployed RFS.

Oracle RFS and the RFS gap use ground-truth rater labels and are analysis-only.
They must never be used to select a trajectory at deployment time.

Example:

    WAYMO_OPEN_DATASET_SRC="$PWD/third_party/waymo-open-dataset/src" \
    PYTHONPATH="$PWD/src:$PWD/research/anchored_flow/src" \
    python -m qwen_drive_anchored.evaluate_paper_metrics \
      --run QD-SFT=outputs/waymo_val/sft_16sample_fullval.jsonl \
      --run QD-RL=outputs/waymo_val/rl_16sample_fullval.jsonl \
      --output outputs/anchored_flow/paper_metrics
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

from qwen_drive.metrics import waymo_displacement_metrics
from qwen_drive.trajectory import resample_to_4hz


METRICS = (
    "RFS",
    "ADE",
    "DDv2_Div",
    "DDv2_Div_at_16",
    "Mean candidate RFS",
    "Oracle RFS",
    "RFS gap",
)


def _load_predictions(path: Path) -> dict[str, dict]:
    records = {}
    with path.open() as handle:
        for line in handle:
            if not line.endswith("\n"):
                break
            record = json.loads(line)
            if record.get("future_trajectory") is None:
                continue
            records[record["token"]] = record
    if not records:
        raise ValueError(f"{path} contains no scorable records")
    return records


def _resample_candidates(record: dict) -> np.ndarray:
    trajectories = np.asarray(record["trajectories"], dtype=np.float64)
    if trajectories.ndim != 3 or trajectories.shape[-1] < 2:
        raise ValueError(f"{record['token']}: trajectories must have shape [N, T, C]")
    return np.stack(
        [resample_to_4hz(path, 20, official_grid=True) for path in trajectories]
    )


def _selected_index(record: dict, candidates: np.ndarray) -> int:
    scores = record.get("candidate_scores")
    if scores is not None and len(scores) == len(candidates):
        return int(np.argmax(np.asarray(scores, dtype=np.float64)))
    selected = record.get("selected_index")
    if selected is not None:
        selected = int(selected)
        if 0 <= selected < len(candidates):
            return selected
    return 0


def _ddv2_div(candidates: np.ndarray) -> float:
    """DiffusionDriveV2 Eq. 12--13, averaged over waypoints and scenes.

    ``candidates`` has shape [N, T, C]. For each waypoint, pairwise Euclidean
    distances are divided by the mean candidate distance from the ego origin,
    clipped to one, and then averaged over waypoints.
    """
    xy = np.asarray(candidates, dtype=np.float64)[..., :2]
    count = xy.shape[0]
    if count < 2:
        return 0.0

    i, j = np.triu_indices(count, k=1)
    pairwise = np.linalg.norm(xy[i] - xy[j], axis=-1)
    raw = pairwise.mean(axis=0)
    scale = 1e-6 + np.linalg.norm(xy, axis=-1).mean(axis=0)
    return float(np.clip(raw / scale, 0.0, 1.0).mean())


def _rater_feedback(paths: np.ndarray, references: list, labels: list, speeds: np.ndarray) -> np.ndarray:
    source = os.environ.get("WAYMO_OPEN_DATASET_SRC")
    if source and source not in sys.path:
        sys.path.insert(0, source)
    from waymo_open_dataset.metrics.python.rater_feedback_utils import (
        get_rater_feedback_score,
    )

    # The official function expects a probability distribution over candidates
    # and returns its weighted mean. Query one candidate at a time with a
    # one-hot probability so every candidate's official RFS remains available.
    per_candidate = []
    for index in range(paths.shape[1]):
        result = get_rater_feedback_score(
            inference_trajectories=np.asarray(
                paths[:, index : index + 1], dtype=np.float64
            )[..., :2],
            inference_probs=np.ones((paths.shape[0], 1), dtype=np.float64),
            rater_specified_trajectories=references,
            rater_feedback_labels=labels,
            init_speed=np.asarray(speeds, dtype=np.float64),
            frequency=4,
            length_seconds=5,
        )
        per_candidate.append(np.asarray(result["rater_feedback_score"], dtype=np.float64))
    return np.stack(per_candidate, axis=1)


def _read_run(path: Path) -> dict[str, dict]:
    return _load_predictions(path)


def _parse_run(items: list[str]) -> dict[str, Path]:
    runs = {}
    for item in items:
        name, separator, path = item.partition("=")
        if not separator or not name or not path:
            raise ValueError(f"invalid --run {item!r}; expected NAME=PATH")
        runs[name] = Path(path)
    return runs


def _evaluate_run(
    rows: list[dict],
    deployed_rows: list[dict],
    *,
    include_div_at_16: bool,
) -> tuple[list[dict], np.ndarray]:
    candidates = np.stack([_resample_candidates(row) for row in rows])
    references = [
        [np.asarray(path, dtype=np.float64)[:20, :2] for path in row["preference_trajectories"]]
        for row in rows
    ]
    labels = [np.asarray(row["preference_scores"], dtype=np.float64) for row in rows]
    speeds = np.asarray([row["initial_speed"] for row in rows], dtype=np.float64)
    candidate_rfs = _rater_feedback(candidates, references, labels, speeds)

    oracle_rfs = candidate_rfs.max(axis=1)
    mean_candidate_rfs = candidate_rfs.mean(axis=1)

    deployed_candidates = np.stack(
        [_resample_candidates(row) for row in deployed_rows]
    )
    deployed_indices = np.asarray(
        [
            _selected_index(row, paths)
            for row, paths in zip(deployed_rows, deployed_candidates, strict=True)
        ]
    )
    deployed_selected = deployed_candidates[
        np.arange(len(deployed_rows)),
        deployed_indices,
    ]
    deployed_rfs = _rater_feedback(
        deployed_selected[:, np.newaxis],
        references,
        labels,
        speeds,
    )[:, 0]
    deployed_ground_truth = np.stack(
        [
            resample_to_4hz(
                np.asarray(row["future_trajectory"]),
                20,
                official_grid=True,
            )
            for row in deployed_rows
        ]
    )
    selected_ade = np.asarray(
        [
            waymo_displacement_metrics(
                deployed_selected[index : index + 1],
                deployed_ground_truth[index : index + 1],
            )["ADE"]
            for index in range(len(rows))
        ]
    )

    scene_rows = []
    for index, (row, ade) in enumerate(zip(rows, selected_ade, strict=True)):
        metrics = {
            "RFS": float(deployed_rfs[index]),
            "ADE": float(ade),
            "DDv2_Div": 100.0 * _ddv2_div(candidates[index]),
            "Mean candidate RFS": float(mean_candidate_rfs[index]),
            "Oracle RFS": float(oracle_rfs[index]),
            "RFS gap": float(oracle_rfs[index] - deployed_rfs[index]),
            "selected_index": int(deployed_indices[index]),
        }
        if include_div_at_16 and len(candidates[index]) >= 16:
            metrics["DDv2_Div_at_16"] = 100.0 * _ddv2_div(candidates[index, :16])
        scene_rows.append({"token": row["token"], **metrics})
    return scene_rows, candidate_rfs


def _summary(rows: list[dict]) -> dict[str, float]:
    return {
        key: float(np.mean([row[key] for row in rows if key in row]))
        for key in METRICS
        if any(key in row for row in rows)
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="NAME=predictions.jsonl")
    parser.add_argument(
        "--deployed",
        action="append",
        default=[],
        help="optional NAME=deployed_predictions.jsonl; defaults to the matching --run",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    runs = _parse_run(args.run)
    records = {name: _read_run(path) for name, path in runs.items()}
    deployed_paths = _parse_run(args.deployed)
    unknown_deployed = set(deployed_paths) - set(records)
    if unknown_deployed:
        raise ValueError(f"unknown --deployed runs: {sorted(unknown_deployed)}")
    deployed_records = {
        name: _read_run(path) for name, path in deployed_paths.items()
    }
    common_tokens = sorted(set.intersection(*(set(rows) for rows in records.values())))
    if not common_tokens:
        raise SystemExit("the prediction files have no common tokens")

    summaries = {}
    per_scene = []
    candidate_counts = {}
    for name, run_records in records.items():
        rows = [run_records[token] for token in common_tokens]
        counts = {len(row["trajectories"]) for row in rows}
        if len(counts) != 1:
            raise ValueError(f"{name} has inconsistent candidate counts: {sorted(counts)}")
        candidate_counts[name] = counts.pop()
        deployment_rows = [
            deployed_records[name][token] if name in deployed_records else run_records[token]
            for token in common_tokens
        ]
        scene_rows, _ = _evaluate_run(
            rows,
            deployment_rows,
            include_div_at_16=True,
        )
        summaries[name] = _summary(scene_rows)
        per_scene.extend({"run": name, **row} for row in scene_rows)

    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "scenes": len(common_tokens),
        "selection": "deployed candidate_scores argmax when present, otherwise first candidate",
        "div": "DiffusionDriveV2 Eq. 12-13, reported as percent",
        "oracle_warning": "Oracle RFS and RFS gap use validation labels and are analysis-only",
        "candidate_counts": candidate_counts,
        "runs": summaries,
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")

    with (args.output / "per_scene.csv").open("w", newline="") as handle:
        fields = ["run", "token", "RFS", "ADE", "DDv2_Div", "DDv2_Div_at_16",
                  "Mean candidate RFS", "Oracle RFS", "RFS gap", "selected_index"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_scene)

    lines = [
        "# WOD-E2E and DiffusionDriveV2 metrics",
        "",
        f"Common validation scenes: {len(common_tokens)}",
        "",
        "| Method | N | RFS↑ | ADE↓ | DDv2 Div↑ | Div@16↑ | Mean candidate RFS↑ | Oracle RFS↑ | RFS gap↓ |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, values in summaries.items():
        lines.append(
            f"| {name} | {candidate_counts[name]} | {values['RFS']:.4f} | {values['ADE']:.4f} "
            f"| {values['DDv2_Div']:.2f}% | {values.get('DDv2_Div_at_16', float('nan')):.2f}% "
            f"| {values['Mean candidate RFS']:.4f} | {values['Oracle RFS']:.4f} "
            f"| {values['RFS gap']:.4f} |"
        )
    lines += [
        "",
        "`Oracle RFS` and `RFS gap` use validation labels and are not deployable.",
        "`Div@16` compares the first 16 candidates so that native 16- and 20-candidate sets are comparable.",
    ]
    (args.output / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote evaluation to {args.output}")


if __name__ == "__main__":
    main()
