"""Build strided WOD-E2E training scenes, including unlabelled future windows.

The benchmark builder is intentionally restricted to frames with human preference
labels. That is appropriate for validation, but training also uses future ego
trajectories at frames without preference annotations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
TOOLS_DIR = REPO_ROOT / "tools" / "waymo_e2e"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import build_scenes as scenes  # noqa: E402

__all__ = ["main", "parse_args"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--index", type=Path, default=None)
    parser.add_argument("--split", default="training")
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=10,
        help="keep every Nth source frame; 10 gives approximately 1 Hz",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument(
        "--missing-history",
        choices=("skip", "nearest"),
        default="skip",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.frame_stride <= 0:
        raise SystemExit("--frame-stride must be positive")
    data_root = args.data_root.resolve()
    output = args.output.resolve()
    image_root = args.image_root.resolve()
    index_path = args.index or output.with_suffix(output.suffix + ".index.json.gz")

    if index_path.exists() and not args.rebuild_index:
        index = scenes.load_index(index_path)
        print(f"loaded index for {len(index['index'])} scenes from {index_path}", flush=True)
    else:
        index = scenes.build_index(data_root, args.split, log=lambda text: print(text, flush=True))
        scenes.save_index(index, index_path)
        print(f"wrote index to {index_path}", flush=True)

    tasks = []
    for token, frames in index["index"].items():
        numeric = sorted(int(frame) for frame in frames)
        tasks.extend(
            (token, frame)
            for frame in numeric
            if frame % args.frame_stride == 0
        )
    tasks.sort()
    if args.limit is not None:
        tasks = tasks[: args.limit]
    if not tasks:
        raise SystemExit("no training frames selected")

    shards = [data_root / name for name in index["shards"]]
    output.parent.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    with output.open("w") as handle:
        for position, (token, frame_index) in enumerate(tasks, start=1):
            try:
                records = scenes.load_records(
                    index,
                    shards,
                    token,
                    frame_index,
                    args.missing_history,
                )
                current = records[-1]
                if len(current.future.get("pos_x", [])) < 2:
                    raise ValueError("future trajectory is unavailable")
                record = scenes.build_scene(token, frame_index, records)
                paths = scenes._write_frames(image_root, token, records, args.split)
                instruction = scenes.instruction_text(
                    np.asarray(record["trajectory"]["hist_traj_10hz"], dtype=np.float64),
                    record["trajectory"]["nav_command"],
                )
                record["messages"][0]["content"] = scenes._user_content(
                    paths, records, instruction
                )
            except (KeyError, ValueError, IndexError) as error:
                skipped += 1
                if skipped <= 20:
                    print(f"skip {token}-{frame_index}: {error}", flush=True)
                continue
            handle.write(json.dumps(record) + "\n")
            written += 1
            if position % 100 == 0 or position == len(tasks):
                print(
                    f"{position}/{len(tasks)} selected frames, "
                    f"{written} written, {skipped} skipped",
                    flush=True,
                )
    print(f"wrote {written} scenes to {output}; skipped {skipped}", flush=True)


if __name__ == "__main__":
    main()

