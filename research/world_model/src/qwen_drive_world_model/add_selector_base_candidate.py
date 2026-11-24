"""Add the frozen RL base trajectory as a fixed selector candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from qwen_drive_anchored.evaluate_paper_metrics import (
    _rater_feedback,
    _resample_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--base-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_records(path: Path) -> dict[str, dict]:
    records = {}
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            records[str(record["token"])] = record
    return records


def main() -> None:
    args = parse_args()
    cache = np.load(args.candidates)
    base_records = _load_records(args.base_predictions)
    ordered_base = []
    for token in cache["tokens"]:
        token = str(token)
        if token not in base_records:
            raise SystemExit(f"missing RL base record for {token}")
        ordered_base.append(base_records[token])
    base_trajectories = np.stack(
        [_resample_candidates(record) for record in ordered_base]
    ).astype(np.float32)
    references = [
        [
            np.asarray(path, dtype=np.float64)[:20, :2]
            for path in record["preference_trajectories"]
        ]
        for record in ordered_base
    ]
    labels = [
        np.asarray(record["preference_scores"], dtype=np.float64)
        for record in ordered_base
    ]
    speeds = np.asarray(
        [record["initial_speed"] for record in ordered_base],
        dtype=np.float64,
    )
    base_rfs = _rater_feedback(base_trajectories, references, labels, speeds)
    trajectories = np.concatenate(
        [base_trajectories, cache["trajectories"]],
        axis=1,
    )
    candidate_rfs = np.concatenate(
        [base_rfs, cache["candidate_rfs"]],
        axis=1,
    ).astype(np.float32)
    anchor_ids = np.concatenate(
        [
            np.full((len(cache["tokens"]), 1), -1, dtype=np.int32),
            cache["anchor_ids"],
        ],
        axis=1,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        tokens=cache["tokens"],
        scene_tokens=cache["scene_tokens"],
        split=cache["split"],
        trajectories=trajectories,
        candidate_rfs=candidate_rfs,
        anchor_ids=anchor_ids,
        initial_speed=cache["initial_speed"],
        future_trajectory=cache["future_trajectory"],
        preference_scores=cache["preference_scores"],
    )
    summary = {
        "records": len(cache["tokens"]),
        "scene_groups": len(set(cache["scene_tokens"].tolist())),
        "candidates_per_scene": int(trajectories.shape[1]),
        "rl_base_rfs": float(base_rfs[:, 0].mean()),
        "anchor_first_candidate_rfs": float(cache["candidate_rfs"][:, 0].mean()),
        "mean_candidate_rfs": float(candidate_rfs.mean()),
        "oracle_rfs": float(candidate_rfs.max(1).mean()),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
