"""Build a global scene/frame index for WOD-E2E tfrecord shards."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
TOOLS_DIR = REPO_ROOT / "tools" / "waymo_e2e"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import e2e_proto  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/mnt/hdd/waymo_e2e/waymo_open_dataset_end_to_end_camera_v_1_0_0"),
    )
    parser.add_argument("--split", default="training")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("outputs/waymo_train/index_cache"),
    )
    parser.add_argument(
        "--cached-only",
        action="store_true",
        help="combine completed per-shard caches without scanning new shards",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _scan_shard(
    item: tuple[int, str, str, str],
) -> tuple[int, dict[str, dict[str, list[int]]], int, bool]:
    shard_id, path, split, cache_root = item
    cache_dir = Path(cache_root)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{split}-{shard_id:05d}.json.gz"
    if cache_path.exists():
        with gzip.open(cache_path, "rt") as handle:
            payload = json.load(handle)
        return shard_id, payload["records"], int(payload["count"]), True
    records: dict[str, dict[str, list[int]]] = {}
    count = 0
    for offset, length, payload in e2e_proto.iter_tfrecords(path):
        meta = e2e_proto.scan_metadata(payload)
        if meta is None:
            continue
        records.setdefault(meta.scene_token, {})[str(meta.frame_index)] = [
            shard_id,
            offset,
            length,
        ]
        count += 1
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with gzip.open(temporary, "wt") as handle:
        json.dump({"records": records, "count": count}, handle)
    temporary.replace(cache_path)
    return shard_id, records, count, False


def main() -> None:
    args = parse_args()
    shards = sorted(args.data_root.glob(f"{args.split}_*.tfrecord-*"))
    if not shards:
        raise SystemExit(f"no {args.split}_*.tfrecord-* shards under {args.data_root}")
    if args.max_shards is not None:
        shards = shards[: args.max_shards]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    if args.cached_only:
        cached_paths = sorted(args.cache_dir.glob(f"{args.split}-*.json.gz"))
        if not cached_paths:
            raise SystemExit(f"no cached shard indexes under {args.cache_dir}")
        index = {}
        total = 0
        cached_shard_ids = []
        for path in cached_paths:
            shard_id = int(path.stem.rsplit("-", 1)[-1].split(".")[0])
            cached_shard_ids.append(shard_id)
            with gzip.open(path, "rt") as handle:
                payload = json.load(handle)
            for scene, frames in payload["records"].items():
                index.setdefault(scene, {}).update(frames)
            total += int(payload["count"])
        payload = {
            "split": args.split,
            "data_root": str(args.data_root),
            "shards": [path.name for path in shards],
            "cached_shards": cached_shard_ids,
            "index": index,
        }
        with gzip.open(args.output, "wt") as handle:
            json.dump(payload, handle)
        print(
            f"combined {len(cached_paths)} cached shards, {total} records, "
            f"{len(index)} scenes; wrote {args.output}",
            flush=True,
        )
        return
    index: dict[str, dict[str, list[int]]] = {}
    total = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _scan_shard,
                (shard_id, str(path), args.split, str(args.cache_dir)),
            ): shard_id
            for shard_id, path in enumerate(shards)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            shard_id, records, count, cached = future.result()
            for scene, frames in records.items():
                index.setdefault(scene, {}).update(frames)
            total += count
            print(
                f"{completed}/{len(shards)} shard {shard_id}: "
                f"{count} records, {len(index)} scenes total"
                f"{' [cached]' if cached else ''}",
                flush=True,
            )
    payload = {
        "split": args.split,
        "data_root": str(args.data_root),
        "shards": [path.name for path in shards],
        "index": index,
    }
    with gzip.open(args.output, "wt") as handle:
        json.dump(payload, handle)
    print(
        f"indexed {total} records, {len(index)} scenes; wrote {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
