"""Export frozen-QD current BEV caches for fixed-candidate selector scenes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from .export_latents import (
    DTYPES,
    _infer_frame,
    _load_models,
    _load_pairs,
)


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for path in args.pairs:
        rows.extend(_load_pairs(path))
    rows.sort(key=lambda row: row["pair_id"])
    if not rows:
        raise SystemExit("no pairs to export")
    if len({row["pair_id"] for row in rows}) != len(rows):
        raise SystemExit("duplicate pair_id across input pairs files")

    args.output.mkdir(parents=True, exist_ok=True)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise SystemExit(
            "QD Perception current-BEV export requires a CUDA GPU."
        )
    os.environ.setdefault(
        "TORCH_EXTENSIONS_DIR",
        str((args.output / ".torch_extensions").resolve()),
    )
    model, processor, device = _load_models(args)
    latent_size = (args.latent_height, args.latent_width)
    for index, pair in enumerate(rows, start=1):
        cache_path = args.output / f"{pair['pair_id']}.npz"
        if cache_path.exists() and not args.force:
            print(f"{index}/{len(rows)} {pair['pair_id']} cached", flush=True)
            continue
        current = _infer_frame(
            model,
            processor,
            Path(pair["teacher_inputs"]["current"]),
            device=device,
            latent_size=latent_size,
            latent_channels=args.latent_channels,
        )
        np.savez_compressed(
            cache_path,
            z_t=np.asarray(current["bev_embed"], dtype=np.float16),
            f_t=np.asarray(current["bev_feat"], dtype=np.float16),
            map_t=np.asarray(current["map"], dtype=np.int8),
            occ_t=np.asarray(current["occ"], dtype=np.int8),
            boxes_t=np.asarray(current["boxes"], dtype=np.float32),
            scores_t=np.asarray(current["scores"], dtype=np.float32),
            labels_t=np.asarray(current["labels"], dtype=np.int64),
        )
        print(f"{index}/{len(rows)} {pair['pair_id']}", flush=True)

    summary = {
        "pairs": len(rows),
        "scenes": len({row["scene_token"] for row in rows}),
        "cached": len(list(args.output.glob("*.npz"))),
        "device": str(device),
        "latent_size": list(latent_size),
        "latent_channels": args.latent_channels,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
