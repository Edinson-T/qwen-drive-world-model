"""Build fixed RL direct+anchor candidate cache with official per-candidate RFS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from qwen_drive_anchored.evaluate_paper_metrics import _rater_feedback, _resample_candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument(
        "--bev-cache",
        type=Path,
        default=None,
        help="Optional current-BEV cache used to filter candidate records.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_tokens = (
        {path.stem for path in args.bev_cache.glob("*.npz")}
        if args.bev_cache is not None
        else None
    )
    records = []
    with args.candidates.open() as handle:
        for line in handle:
            if not line.endswith("\n"):
                break
            record = json.loads(line)
            if cache_tokens is not None and record["token"] not in cache_tokens:
                continue
            if not record.get("preference_trajectories"):
                continue
            records.append(record)
    if not records:
        raise SystemExit("no candidate records overlap the current-BEV cache")

    records.sort(key=lambda item: item["token"])
    candidates = np.stack(
        [_resample_candidates(record) for record in records]
    ).astype(np.float32)
    references = [
        [
            np.asarray(path, dtype=np.float64)[:20, :2]
            for path in record["preference_trajectories"]
        ]
        for record in records
    ]
    labels = [
        np.asarray(record["preference_scores"], dtype=np.float64)
        for record in records
    ]
    speeds = np.asarray([record["initial_speed"] for record in records], dtype=np.float64)
    candidate_rfs = _rater_feedback(candidates, references, labels, speeds)
    scene_tokens = sorted({record["scene_token"] for record in records})
    train_end = int(round(len(scene_tokens) * args.train_fraction))
    val_end = train_end + int(round(len(scene_tokens) * args.val_fraction))
    scene_split = {
        token: (
            "train"
            if index < train_end
            else "val"
            if index < val_end
            else "test"
        )
        for index, token in enumerate(scene_tokens)
    }
    split = np.asarray(
        [scene_split[record["scene_token"]] for record in records],
        dtype="U5",
    )

    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / "candidates.npz",
        tokens=np.asarray([record["token"] for record in records]),
        scene_tokens=np.asarray([record["scene_token"] for record in records]),
        split=split,
        trajectories=candidates,
        candidate_rfs=candidate_rfs.astype(np.float32),
        anchor_ids=np.stack(
            [np.asarray(record["anchor_ids"], dtype=np.int32) for record in records]
        ),
        initial_speed=np.asarray([record["initial_speed"] for record in records], np.float32),
        future_trajectory=np.stack(
            [np.asarray(record["future_trajectory"], np.float32) for record in records]
        ),
        preference_scores=np.stack(labels).astype(np.float32)
        if all(len(value) == len(labels[0]) for value in labels)
        else np.asarray(labels, dtype=object),
    )
    summary = {
        "records": len(records),
        "scene_groups": len(scene_tokens),
        "candidates_per_scene": int(candidates.shape[1]),
        "trajectory_points": int(candidates.shape[2]),
        "split_counts": {
            name: int((split == name).sum()) for name in ("train", "val", "test")
        },
        "mean_candidate_rfs": float(candidate_rfs.mean()),
        "mean_oracle_rfs": float(candidate_rfs.max(1).mean()),
        "mean_first_candidate_rfs": float(candidate_rfs[:, 0].mean()),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

