"""Export compact, frozen-QD-Perception BEV latents for packed Waymo pairs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from qwen_drive import QwenDriveForPlanning  # noqa: E402
from qwen_drive_perception import QwenDrivePerception  # noqa: E402
from qwen_drive_perception.dataset import PerceptionFrame, PerceptionProcessor  # noqa: E402


DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vlm", type=Path, default=Path("Qwen-Drive-1.0-4B"))
    parser.add_argument(
        "--perception",
        type=Path,
        default=Path("Qwen-Drive-1.0-4B/perception"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPES))
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        choices=("sdpa", "flash_attention_2"),
    )
    parser.add_argument("--latent-height", type=int, default=16)
    parser.add_argument("--latent-width", type=int, default=32)
    parser.add_argument("--latent-channels", type=int, default=128)
    parser.add_argument("--shard-size", type=int, default=32)
    parser.add_argument("--limit-pairs", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="merge completed pair caches without loading the VLM",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _horizon_key(seconds: str | float) -> str:
    return f"{float(seconds):g}s".replace(".", "p")


def _load_pairs(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_models(args: argparse.Namespace):
    from transformers import AutoTokenizer

    dtype = DTYPES[args.dtype]
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    holder = QwenDriveForPlanning.from_pretrained(
        args.vlm,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
    )
    vlm = holder.vlm.to(device)
    del holder.planning_expert
    model = QwenDrivePerception.from_pretrained(
        args.perception,
        dtype=dtype,
    ).to(device).eval()
    processor = PerceptionProcessor(AutoTokenizer.from_pretrained(args.vlm))
    model.attach(vlm.eval(), processor)
    return model, processor, device


def _infer_frame(
    model,
    processor,
    frame_dir: Path,
    *,
    device: str,
    latent_size: tuple[int, int],
    latent_channels: int,
) -> dict:
    frame = PerceptionFrame(frame_dir)
    inputs, img_metas = processor(frame, device=device)
    with torch.inference_mode():
        return model.infer(
            inputs,
            img_metas,
            return_latent=True,
            latent_size=latent_size,
            latent_channels=latent_channels,
        )


def _cache_pair(
    model,
    processor,
    pair: dict,
    *,
    device: str,
    latent_size: tuple[int, int],
    latent_channels: int,
) -> dict[str, np.ndarray]:
    outputs = {}
    current = _infer_frame(
        model,
        processor,
        Path(pair["teacher_inputs"]["current"]),
        device=device,
        latent_size=latent_size,
        latent_channels=latent_channels,
    )
    outputs["z_t"] = np.asarray(current["bev_embed"], dtype=np.float16)
    outputs["f_t"] = np.asarray(current["bev_feat"], dtype=np.float16)
    outputs["map_t"] = np.asarray(current["map"], dtype=np.int8)
    outputs["occ_t"] = np.asarray(current["occ"], dtype=np.int8)
    outputs["boxes_t"] = np.asarray(current["boxes"], dtype=np.float32)
    outputs["scores_t"] = np.asarray(current["scores"], dtype=np.float32)
    outputs["labels_t"] = np.asarray(current["labels"], dtype=np.int64)
    for seconds in pair["horizons"]:
        key = _horizon_key(seconds)
        future = _infer_frame(
            model,
            processor,
            Path(pair["teacher_inputs"][str(seconds)]),
            device=device,
            latent_size=latent_size,
            latent_channels=latent_channels,
        )
        outputs[f"z_{key}"] = np.asarray(future["bev_embed"], dtype=np.float16)
        outputs[f"f_{key}"] = np.asarray(future["bev_feat"], dtype=np.float16)
        outputs[f"map_{key}"] = np.asarray(future["map"], dtype=np.int8)
        outputs[f"occ_{key}"] = np.asarray(future["occ"], dtype=np.int8)
        outputs[f"boxes_{key}"] = np.asarray(future["boxes"], dtype=np.float32)
        outputs[f"scores_{key}"] = np.asarray(future["scores"], dtype=np.float32)
        outputs[f"labels_{key}"] = np.asarray(future["labels"], dtype=np.int64)
    return outputs


def _stack(values: list[np.ndarray]) -> np.ndarray:
    return np.stack(values, axis=0)


def _write_shards(
    rows: list[dict],
    cached: list[dict],
    output: Path,
    *,
    shard_size: int,
    latent_size: tuple[int, int],
    latent_channels: int,
    vlm_path: Path,
    perception_path: Path,
) -> list[dict]:
    output.mkdir(parents=True, exist_ok=True)
    horizons = list(rows[0]["horizons"])
    keys = ["z_t", "f_t", "map_t", "occ_t"]
    for seconds in horizons:
        suffix = _horizon_key(seconds)
        keys.extend([f"z_{suffix}", f"f_{suffix}", f"map_{suffix}", f"occ_{suffix}"])

    shards = []
    for shard_index, start in enumerate(range(0, len(rows), shard_size)):
        end = min(start + shard_size, len(rows))
        shard_rows = rows[start:end]
        shard_cached = cached[start:end]
        arrays = {
            key: _stack([item[key] for item in shard_cached])
            for key in keys
        }
        arrays.update(
            {
                "tokens": np.asarray([row["token"] for row in shard_rows]),
                "pair_ids": np.asarray([row["pair_id"] for row in shard_rows]),
                "scene_tokens": np.asarray([row["scene_token"] for row in shard_rows]),
                "split": np.asarray([row["split"] for row in shard_rows]),
                "t_frames": np.asarray([row["t_frame"] for row in shard_rows], dtype=np.int32),
                "nav_command": np.asarray([row["nav_command"] for row in shard_rows], dtype=np.int64),
                "ego_velocity": np.asarray([row["ego_velocity"] for row in shard_rows], dtype=np.float32),
                "ego_acceleration": np.asarray(
                    [row["ego_acceleration"] for row in shard_rows], dtype=np.float32
                ),
                "driving_command": np.asarray(
                    [row["driving_command"] for row in shard_rows], dtype=np.float32
                ),
                "trajectory_5s": np.asarray(
                    [row["trajectory_5s"] for row in shard_rows], dtype=np.float32
                ),
                "relative_poses": np.asarray(
                    [
                        [row["relative_poses"][str(seconds)] for seconds in horizons]
                        for row in shard_rows
                    ],
                    dtype=np.float32,
                ),
            }
        )
        for seconds in horizons:
            suffix = _horizon_key(seconds)
            windows = [row["trajectory_windows"][str(seconds)] for row in shard_rows]
            width = max(len(window) for window in windows)
            padded = np.zeros((len(windows), width, 3), dtype=np.float32)
            valid = np.zeros((len(windows), width), dtype=bool)
            for index, window in enumerate(windows):
                padded[index, : len(window)] = window
                valid[index, : len(window)] = True
            arrays[f"traj_{suffix}"] = padded
            arrays[f"traj_valid_{suffix}"] = valid

        shard_path = output / f"pilot-{shard_index:05d}.npz"
        np.savez_compressed(shard_path, **arrays)
        shards.append(
            {
                "path": str(shard_path.resolve()),
                "samples": len(shard_rows),
                "train": int(sum(row["split"] == "train" for row in shard_rows)),
                "val": int(sum(row["split"] == "val" for row in shard_rows)),
            }
        )
        print(f"wrote {shard_path} ({len(shard_rows)} samples)")

    meta = {
        "samples": len(rows),
        "horizons": horizons,
        "latent_size": list(latent_size),
        "latent_channels": latent_channels,
        "coordinate_protocol": "future ego-frame prediction",
        "teacher": "frozen QwenDrivePerception",
        "teacher_checkpoint": str(perception_path.resolve()),
        "vlm_checkpoint": str(vlm_path.resolve()),
        "shards": shards,
    }
    (output / "dataset.json").write_text(json.dumps(meta, indent=2) + "\n")
    return shards


def main() -> None:
    args = parse_args()
    rows = _load_pairs(args.pairs)
    if args.limit_pairs is not None:
        rows = rows[: args.limit_pairs]
    if not rows:
        raise SystemExit("no pairs to export")
    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit(
            f"--shard-index must be in [0,{args.num_shards}), got {args.shard_index}"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output / "pair_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    latent_size = (args.latent_height, args.latent_width)
    if args.aggregate_only:
        cached = []
        missing = []
        for pair in rows:
            cache_path = cache_dir / f"{pair['pair_id']}.npz"
            if not cache_path.exists():
                missing.append(str(cache_path))
                continue
            with np.load(cache_path) as payload:
                cached.append({key: payload[key] for key in payload.files})
        if missing:
            raise SystemExit(
                f"cannot aggregate: {len(missing)} pair caches are missing; "
                f"first missing: {missing[0]}"
            )
        _write_shards(
            rows,
            cached,
            args.output,
            shard_size=args.shard_size,
            latent_size=latent_size,
            latent_channels=args.latent_channels,
            vlm_path=args.vlm,
            perception_path=args.perception,
        )
        return

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cpu":
        raise SystemExit(
            "QD Perception uses the CUDA-only qwen_drive_voxel_pool operator for "
            "UVTR view transformation. A working NVIDIA driver/GPU is required for "
            "real latent export; the world-model training and tests remain CPU-runnable."
        )
    extension_shard = args.output / ".torch_extensions" / f"shard_{args.shard_index}"
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(extension_shard.resolve()))

    shard_rows = rows[args.shard_index :: args.num_shards]
    if not shard_rows:
        raise SystemExit("this shard has no pairs")
    print(
        f"device={device} pairs={len(shard_rows)} "
        f"shard={args.shard_index}/{args.num_shards}"
    )
    model, processor, device = _load_models(args)

    for index, pair in enumerate(shard_rows, start=1):
        cache_path = cache_dir / f"{pair['pair_id']}.npz"
        if cache_path.exists() and not args.force:
            print(f"{index}/{len(shard_rows)} {pair['pair_id']} cached")
            continue
        item = _cache_pair(
            model,
            processor,
            pair,
            device=device,
            latent_size=latent_size,
            latent_channels=args.latent_channels,
        )
        np.savez_compressed(cache_path, **item)
        z = item["z_t"]
        print(f"{index}/{len(shard_rows)} {pair['pair_id']} z={tuple(z.shape)}")

    print(
        f"shard {args.shard_index}/{args.num_shards} complete; "
        "run with --aggregate-only after all shards finish"
    )
    if args.num_shards == 1:
        cached = []
        for pair in rows:
            with np.load(cache_dir / f"{pair['pair_id']}.npz") as payload:
                cached.append({key: payload[key] for key in payload.files})
        _write_shards(
            rows,
            cached,
            args.output,
            shard_size=args.shard_size,
            latent_size=latent_size,
            latent_channels=args.latent_channels,
            vlm_path=args.vlm,
            perception_path=args.perception,
        )


if __name__ == "__main__":
    main()
