"""Convert Waymo Perception tfrecords into compact training shards.

Each output ``<segment>.npz`` holds, for the chosen cameras:
    rgb    uint8   [N, 3, H, W]
    depth  float16 [N, 1, H/4, W/4]
    mask   bool    [N, 1, H/4, W/4]
    cams   int32   [N]
    frames int32   [N]

Usage:
    python -m lidar_supervision.prepare_waymo \
        --data /home/fractal/datasets/waymo_perception \
        --out  /home/fractal/datasets/waymo_perception_targets \
        --image-size 256 144 --cameras 1 2 3
"""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .waymo_perception import CAMERA_NAMES, depth_for_camera, iter_tfrecords, parse_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="directory with *.tfrecord segments")
    parser.add_argument("--out", required=True)
    parser.add_argument("--image-size", type=int, nargs=2, default=(256, 144), metavar=("W", "H"))
    parser.add_argument("--cameras", type=int, nargs="+", default=(1, 2, 3), help="1=FRONT, 2=FRONT_LEFT, 3=FRONT_RIGHT")
    parser.add_argument("--max-frames", type=int, default=200, help="frames per segment")
    parser.add_argument("--min-coverage", type=float, default=0.05, help="skip samples below this mask fraction")
    parser.add_argument("--min-depth", type=float, default=1.0)
    parser.add_argument("--max-depth", type=float, default=80.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    width, height = args.image_size
    target = (width // 4, height // 4)

    shards = sorted(data_dir.glob("*.tfrecord*"))
    if not shards:
        raise SystemExit(f"no tfrecords under {data_dir}")

    manifest = []
    for shard in shards:
        segment = shard.name.split("_")[0].replace("segment-", "")
        rgb, depth, mask, cams, frames = [], [], [], [], []
        started = time.perf_counter()
        for record_index, payload in enumerate(iter_tfrecords(shard)):
            if record_index >= args.max_frames:
                break
            frame = parse_frame(payload)
            if frame.range_image.size == 0:
                continue
            for camera_id in args.cameras:
                jpeg = frame.cameras.get(camera_id)
                if jpeg is None:
                    continue
                image = Image.open(io.BytesIO(jpeg)).convert("RGB")
                native = image.size
                target_depth, target_mask = depth_for_camera(
                    frame, camera_id, native, target, args.min_depth, args.max_depth
                )
                if target_mask.mean() < args.min_coverage:
                    continue
                image = TF.resize(image, [height, width], interpolation=InterpolationMode.BICUBIC)
                rgb.append(np.asarray(image, dtype=np.uint8).transpose(2, 0, 1))
                depth.append(target_depth[None])
                mask.append(target_mask[None])
                cams.append(camera_id)
                frames.append(frame.frame_index)

        if not rgb:
            print(f"{shard.name}: no usable frames")
            continue
        np.savez_compressed(
            out_dir / f"{segment}.npz",
            rgb=np.stack(rgb),
            depth=np.stack(depth).astype(np.float16),
            mask=np.stack(mask),
            cams=np.asarray(cams, dtype=np.int32),
            frames=np.asarray(frames, dtype=np.int32),
        )
        coverage = float(np.mean([m.mean() for m in mask]))
        manifest.append(
            {
                "segment": segment,
                "shard": shard.name,
                "samples": len(rgb),
                "mean_coverage": round(coverage, 4),
            }
        )
        print(
            f"{segment[:12]}  frames={len(rgb) // len(args.cameras):4d}  samples={len(rgb):5d}  "
            f"coverage={coverage:.2f}  {time.perf_counter() - started:.0f}s"
        )

    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "image_size": [width, height],
                "target_size": list(target),
                "cameras": list(args.cameras),
                "segments": manifest,
            },
            indent=2,
        )
    )
    print(f"\nwrote {len(manifest)} shards to {out_dir}")


if __name__ == "__main__":
    main()
