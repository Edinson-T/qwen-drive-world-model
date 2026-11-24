"""Train the tiny depth head on the demo frames (pipeline smoke test only).

Examples:
    python -m lidar_supervision.train_geometry --smoke
    python -m lidar_supervision.train_geometry --steps 300 --image-size 256 144
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .data import DemoFrame, load_frames
from .geometry import build_lidar2img, project_lidar_to_depth
from .losses import abs_rel, delta1, masked_log_l1
from .model import DepthHead


class PairDataset(Dataset):
    """One sample per (frame, camera) pair, with a LiDAR depth target."""

    def __init__(self, frames: list[DemoFrame], image_size, min_depth: float, max_depth: float):
        self.pairs = [pair for frame in frames for pair in frame.pairs()]
        self.width, self.height = image_size
        self.min_depth, self.max_depth = min_depth, max_depth

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index):
        frame, cam_index = self.pairs[index]
        camera = frame.cameras[cam_index]
        image = Image.open(camera.image_path).convert("RGB")
        native_w, native_h = image.size
        image = TF.resize(
            image, [self.height, self.width], interpolation=InterpolationMode.BICUBIC
        )
        pixels = TF.pil_to_tensor(image).float().div_(255.0).sub_(0.5).div_(0.5)

        lidar2img = build_lidar2img(
            camera.intrinsic, camera.sensor2lidar_rotation, camera.sensor2lidar_translation
        )
        depth, mask, _, _, _ = project_lidar_to_depth(
            frame.lidar,
            lidar2img,
            image_size=(native_w, native_h),
            out_size=(self.width // 4, self.height // 4),
            min_depth=self.min_depth,
            max_depth=self.max_depth,
        )
        return pixels, torch.from_numpy(depth)[None], torch.from_numpy(mask)[None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/demo/perception")
    parser.add_argument("--out", default="outputs/lidar_smoke")
    parser.add_argument("--image-size", type=int, nargs=2, default=(256, 144), metavar=("W", "H"))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-depth", type=float, default=1.0)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--amp", action="store_true", help="bf16 autocast on CUDA")
    parser.add_argument("--smoke", action="store_true", help="random tensors only; no data")
    parser.add_argument(
        "--overfit",
        type=int,
        default=0,
        metavar="N",
        help="train repeatedly on the first N samples (pipeline check, not a real run)",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    width, height = args.image_size
    if width % 4 or height % 4:
        raise SystemExit("--image-size must be divisible by 4 (the head predicts at 1/4 resolution)")

    model = DepthHead().to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    use_amp = args.amp and device == "cuda"

    if args.smoke:
        images = torch.rand(args.batch_size, 3, height, width, device=device)
        target = torch.rand(args.batch_size, 1, height // 4, width // 4, device=device) * 40 + 5
        mask = torch.ones_like(target, dtype=torch.bool)
        dataset = None
        num_frames = None
    else:
        frames = load_frames(args.data)
        num_frames = len(frames)
        dataset = PairDataset(frames, (width, height), args.min_depth, args.max_depth)
        if args.overfit > 0:
            dataset = torch.utils.data.Subset(dataset, list(range(min(args.overfit, len(dataset)))))
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=args.overfit == 0,
            drop_last=False,
            num_workers=0,
        )
        print(f"device={device}  frames={len(frames)}  samples={len(dataset)}  "
              f"image={width}x{height}  steps={args.steps}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    step = 0
    started = time.perf_counter()
    last = {"step": 0, "loss": None, "absrel": None, "delta1": None, "grad_norm": None}

    while step < args.steps:
        for batch in ([None] if args.smoke else loader):
            if args.smoke:
                images_b, target_b, mask_b = images, target, mask
            else:
                images_b, target_b, mask_b = (tensor.to(device) for tensor in batch)
            if mask_b.sum() == 0:
                continue

            model.train()
            optimiser.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
                prediction = model(images_b)
                loss = masked_log_l1(prediction.float(), target_b, mask_b)
            loss.backward()
            grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            optimiser.step()

            step += 1
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                with torch.no_grad():
                    model.eval()
                    prediction = model(images_b)
                    metrics = (
                        abs_rel(prediction, target_b, mask_b),
                        delta1(prediction, target_b, mask_b),
                    )
                memory = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0.0
                elapsed = time.perf_counter() - started
                last = {
                    "step": step,
                    "loss": float(loss.item()),
                    "absrel": metrics[0],
                    "delta1": metrics[1],
                    "grad_norm": grad_norm,
                }
                print(f"step {step:5d}  loss {loss.item():.4f}  grad {grad_norm:.2f}  "
                      f"absrel {metrics[0]:.3f}  d1.25 {metrics[1]:.3f}  "
                      f"mem {memory:.1f}G  {elapsed:.1f}s")
            if step >= args.steps:
                break

    # Store the run configuration alongside the weights so a checkpoint is
    # self-describing: image size, target range, data path, steps, final metrics.
    config = {
        "argv": list(sys.argv),
        "args": vars(args),
        "image_size": [width, height],
        "target_size": [width // 4, height // 4],
        "final": last,
        "frames": num_frames,
        "samples": None if dataset is None else len(dataset),
        "device": device,
    }
    torch.save(
        {
            "model": model.state_dict(),
            "width": width,
            "height": height,
            "config": config,
            "final": last,
        },
        out / "head.pt",
    )
    (out / "train_config.json").write_text(json.dumps(config, indent=2, default=str))
    print(f"saved {out / 'head.pt'}")
    print(f"saved {out / 'train_config.json'}")


if __name__ == "__main__":
    main()
