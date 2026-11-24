"""Render LiDAR projections on the demo images and report coverage.

Usage:
    python -m lidar_supervision.verify_targets \
        --data data/demo/perception --out outputs/lidar_debug
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .data import load_frames
from .geometry import build_lidar2img, project_lidar_to_depth


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/demo/perception")
    parser.add_argument("--out", default="outputs/lidar_debug")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-points", type=int, default=40000, help="subsample for the overlay")
    parser.add_argument("--min-depth", type=float, default=1.0)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--target-stride", type=int, default=4, help="depth-map stride (1 = native pixels)")
    parser.add_argument("--point-radius", type=int, default=1, help="overlay dot radius in pixels")
    parser.add_argument("--no-depth-map", action="store_true", help="skip the colour-coded depth map")
    return parser.parse_args()


def _depth_image(depth: np.ndarray, mask: np.ndarray, maximum: float, size: tuple[int, int]) -> Image.Image:
    """Near = red, far = blue (same convention as the point overlay)."""
    from matplotlib import cm

    normalised = np.clip(depth / maximum, 0.0, 1.0)
    rgb = (cm.turbo_r(normalised)[..., :3] * 255).astype(np.uint8)
    rgb[~mask] = 0
    image = Image.fromarray(rgb).resize(size, Image.NEAREST)
    draw = ImageDraw.Draw(image)
    draw.rectangle([8, 8, 250, 40], fill=(0, 0, 0))
    draw.text((14, 14), "depth: red = near, blue = far", fill=(255, 255, 255))
    return image


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frames = load_frames(args.data, limit=args.max_frames)

    for frame in frames:
        print(f"\n{frame.token}  [{frame.dataset_type}]  {len(frame.cameras)} cameras  "
              f"{len(frame.lidar)} lidar points")
        for index, camera in enumerate(frame.cameras):
            width, height = frame.image_size(index)
            lidar2img = build_lidar2img(
                camera.intrinsic, camera.sensor2lidar_rotation, camera.sensor2lidar_translation
            )
            depth, mask, u, v, z = project_lidar_to_depth(
                frame.lidar,
                lidar2img,
                image_size=(width, height),
                out_size=(max(1, width // args.target_stride), max(1, height // args.target_stride)),
                min_depth=args.min_depth,
                max_depth=args.max_depth,
            )
            coverage = 100.0 * mask.mean()
            if mask.any():
                values = depth[mask]
                print(f"  {camera.name:<20} {width}x{height}  coverage {coverage:5.2f}%  "
                      f"depth min/med/max {values.min():5.1f}/{np.median(values):5.1f}/{values.max():5.1f} m")
            else:
                print(f"  {camera.name:<20} {width}x{height}  coverage  0.00%  <-- PROJECTION BROKEN")

            if not args.no_depth_map:
                _depth_image(depth, mask, args.max_depth, (width, height)).save(
                    out / f"{frame.token}_{camera.name}_depth.png"
                )

            image = Image.open(camera.image_path).convert("RGB")
            draw = ImageDraw.Draw(image)
            valid = (z >= args.min_depth) & (z <= args.max_depth) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
            idx = np.flatnonzero(valid)
            if len(idx) > args.max_points:
                idx = idx[np.linspace(0, len(idx) - 1, args.max_points).astype(np.int64)]
            for point in idx:
                shade = min(1.0, z[point] / args.max_depth)
                colour = (int(255 * (1 - shade)), 40, int(255 * shade))
                r = args.point_radius
                draw.ellipse([u[point] - r, v[point] - r, u[point] + r, v[point] + r], fill=colour)
            image.save(out / f"{frame.token}_{camera.name}.png")

    print(f"\nwrote overlays to {out}")


if __name__ == "__main__":
    main()
