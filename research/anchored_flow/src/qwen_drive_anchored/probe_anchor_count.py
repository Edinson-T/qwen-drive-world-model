"""Zero-training anchor-bank and candidate-count probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .anchors import AnchorBank
from .data import speed_bin_indices

__all__ = ["main", "parse_args"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="scene-base source JSONL")
    parser.add_argument("--scenes", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--bank",
        action="append",
        required=True,
        help="LABEL=PATH:NUM_CANDIDATES",
    )
    parser.add_argument("--scale", type=float, default=0.045)
    return parser.parse_args(argv)


def _load_scene_meta(path: str | Path) -> dict[str, tuple[int, int]]:
    rows = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        trajectory = record["trajectory"]
        token = str(record["meta_info"]["token"])
        nav = int(trajectory.get("nav_command", 0))
        speed = float(
            trajectory.get("current_state", {}).get(
                "speed",
                np.hypot(*trajectory["ego_status"]["ego_velocity"][:2]),
            )
        )
        rows[token] = (nav, int(speed_bin_indices(np.asarray([speed]))[0]))
    return rows


def _select_anchor_ids(
    bank: AnchorBank,
    count: int,
    nav: int,
    speed_bin: int,
) -> np.ndarray:
    if count >= bank.num_anchors:
        return np.arange(bank.num_anchors, dtype=np.int64)
    if bank.nav_commands is None or bank.speed_bins is None:
        return bank.candidate_ids(count)
    order = sorted(
        range(bank.num_anchors),
        key=lambda index: (
            int(bank.nav_commands[index]) != nav,
            abs(int(bank.speed_bins[index]) - speed_bin),
            index,
        ),
    )
    return np.asarray(order[:count], dtype=np.int64)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    base_rows = [
        json.loads(line)
        for line in Path(args.predictions).read_text().splitlines()
        if line.strip()
    ]
    scene_meta = _load_scene_meta(args.scenes)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    banks = []
    for item in args.bank:
        label, separator, rest = item.partition("=")
        path, separator2, count_text = rest.rpartition(":")
        if not separator or not separator2 or not label or not path:
            raise ValueError(f"invalid --bank {item!r}; expected LABEL=PATH:N")
        banks.append((label, AnchorBank.load(path), int(count_text)))

    for label, bank, count in banks:
        if count <= 0:
            raise ValueError(f"candidate count must be positive for {label}")
        center = bank.physical.mean(axis=0)
        path = output / f"{label}.jsonl"
        with path.open("w") as handle:
            for record in base_rows:
                token = str(record["token"])
                if token not in scene_meta:
                    continue
                nav, speed_bin = scene_meta[token]
                base = np.asarray(record["trajectories"], dtype=np.float64).mean(axis=0)
                ids = _select_anchor_ids(bank, count, nav, speed_bin)
                trajectories = base[None] + args.scale * (
                    bank.physical[ids] - center[None]
                )
                row = dict(record)
                row["trajectories"] = trajectories.tolist()
                row["anchor_ids"] = ids.tolist()
                row["candidate_scores"] = None
                row["selected_index"] = None
                handle.write(json.dumps(row) + "\n")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
