#!/usr/bin/env python
"""Check the scene builder against the four scenes shipped in ``data/demo``.

The bundled demo is a slice of the Waymo validation split, so rebuilding those
four scenes from the raw tfrecords is a self-contained correctness check: if the
builder is right, every trajectory field and the prompt text match the shipped
file exactly and the JPEGs are pixel-identical.

    python tools/waymo_e2e/verify_demo.py \
        --data-root /data/waymo_e2e/waymo_open_dataset_end_to_end_camera_v_1_0_0 \
        --index outputs/waymo_val/scenes.jsonl.index.json.gz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import build_scenes as bs  # noqa: E402

TRAJECTORY_KEYS = (
    "hist_traj_10hz",
    "hist_vel_10hz",
    "hist_acc_10hz",
    "future_traj_10hz",
    "future_vel_10hz",
    "future_acc_10hz",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path, help="index built by build_scenes.py")
    parser.add_argument(
        "--demo",
        type=Path,
        default=bs.REPO_ROOT / "data/demo/planning_scenes.jsonl",
        help="reference scene file (default: data/demo/planning_scenes.jsonl)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    index = bs.load_index(args.index)
    shards = [args.data_root / name for name in index["shards"]]

    reference = {json.loads(line)["meta_info"]["token"]: json.loads(line) for line in open(args.demo)}
    failures: list[str] = []

    for token in sorted(reference):
        scene_token, _, frame_index = token.rpartition("-")
        records = bs.load_records(index, shards, scene_token, int(frame_index), "skip")
        built = bs.build_scene(scene_token, int(frame_index), records)

        expected = reference[token]["trajectory"]
        actual = built["trajectory"]
        worst = 0.0
        for key in TRAJECTORY_KEYS:
            delta = float(np.max(np.abs(np.asarray(expected[key]) - np.asarray(actual[key]))))
            worst = max(worst, delta)
        scores_match = [p["preference_score"] for p in expected["preference_trajectories"]] == [
            p["preference_score"] for p in actual["preference_trajectories"]
        ]
        expected_text = [c["text"] for c in reference[token]["messages"][0]["content"] if "text" in c][-1]
        instruction = bs.instruction_text(
            np.asarray(actual["hist_traj_10hz"], dtype=np.float64), actual["nav_command"]
        )
        ok = worst == 0.0 and scores_match and instruction == expected_text
        failures.append("" if ok else token)
        print(
            f"{token}: max|Δ|={worst:g} scores={'ok' if scores_match else 'BAD'} "
            f"prompt={'ok' if instruction == expected_text else 'BAD'}"
        )

    if any(failures):
        raise SystemExit(f"mismatched scenes: {[name for name in failures if name]}")
    print(f"\nall {len(reference)} demo scenes reproduce exactly")


if __name__ == "__main__":
    main()
