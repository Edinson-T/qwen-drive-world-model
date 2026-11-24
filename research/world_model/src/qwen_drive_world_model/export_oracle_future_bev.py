"""Export QD BEV from logged future camera frames for oracle experiments."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from .export_latents import DTYPES, _infer_frame, _load_models, _load_pairs


HORIZONS = ("0.5", "1.0", "2.0", "3.0", "5.0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vlm", type=Path, default=Path("Qwen-Drive-1.0-4B"))
    parser.add_argument(
        "--perception",
        type=Path,
        default=Path("Qwen-Drive-1.0-4B/perception"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPES))
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        choices=("sdpa", "flash_attention_2"),
    )
    parser.add_argument("--latent-height", type=int, default=16)
    parser.add_argument("--latent-width", type=int, default=32)
    parser.add_argument("--latent-channels", type=int, default=128)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--existing-pair-cache",
        type=Path,
        default=None,
        help="Reuse future z_* fields from an existing teacher pair cache.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    skipped = []
    for path in args.pairs:
        for row in _load_pairs(path):
            inputs = row.get("teacher_inputs", {})
            if all(
                key in inputs and Path(inputs[key]).exists()
                for key in HORIZONS
            ):
                rows.append(row)
            else:
                skipped.append(row["pair_id"])
    rows.sort(key=lambda row: row["pair_id"])
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(
        "TORCH_EXTENSIONS_DIR",
        str((args.output / ".torch_extensions").resolve()),
    )
    model, processor, device = _load_models(args)
    latent_size = (args.latent_height, args.latent_width)
    for index, row in enumerate(rows, start=1):
        path = args.output / f"{row['pair_id']}.npz"
        if path.exists() and not args.force:
            print(f"{index}/{len(rows)} {row['pair_id']} cached", flush=True)
            continue
        existing = (
            args.existing_pair_cache / f"{row['pair_id']}.npz"
            if args.existing_pair_cache is not None
            else None
        )
        if existing is not None and existing.exists():
            keys = ("z_0p5s", "z_1s", "z_2s", "z_3s", "z_5s")
            with np.load(existing) as payload:
                if all(key in payload.files for key in keys):
                    np.savez_compressed(
                        path,
                        future_z=np.stack(
                            [payload[key] for key in keys]
                        ).astype(np.float16),
                        relative_poses=np.asarray(
                            [
                                row["relative_poses"][horizon]
                                for horizon in HORIZONS
                            ],
                            dtype=np.float32,
                        ),
                    )
                    print(
                        f"{index}/{len(rows)} {row['pair_id']} reused",
                        flush=True,
                    )
                    continue
        future = []
        for horizon in HORIZONS:
            output = _infer_frame(
                model,
                processor,
                Path(row["teacher_inputs"][horizon]),
                device=device,
                latent_size=latent_size,
                latent_channels=args.latent_channels,
            )
            future.append(np.asarray(output["bev_embed"], dtype=np.float16))
        np.savez_compressed(
            path,
            future_z=np.stack(future),
            relative_poses=np.asarray(
                [row["relative_poses"][horizon] for horizon in HORIZONS],
                dtype=np.float32,
            ),
        )
        print(f"{index}/{len(rows)} {row['pair_id']}", flush=True)
    summary = {
        "pairs": len(rows),
        "skipped": skipped,
        "horizons": HORIZONS,
        "future_z_shape": [5, args.latent_channels, *latent_size],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
