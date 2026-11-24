"""Train the depth head on prepared Waymo Perception shards (segment-level split).

Usage:
    python -m lidar_supervision.train_waymo \
        --data /home/fractal/datasets/waymo_perception_targets \
        --out outputs/waymo_geom --val-segments 5 --steps 3000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .losses import abs_rel, delta1, masked_log_l1
from .model import DepthHead


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="directory written by prepare_waymo")
    parser.add_argument("--out", default="outputs/waymo_geom")
    parser.add_argument("--val-segments", type=int, default=5)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


class Shards:
    """Lazily loaded per-segment arrays."""

    def __init__(self, directory: Path):
        self.manifest = json.loads((directory / "manifest.json").read_text())
        self.entries = self.manifest["segments"]
        self.directory = directory
        self._cache: dict[str, dict] = {}
        self._index: list[tuple[str, int]] = []
        for entry in self.entries:
            arrays = self.load(entry["segment"])
            for i in range(len(arrays["cams"])):
                self._index.append((entry["segment"], i))

    def load(self, segment: str) -> dict:
        if segment not in self._cache:
            payload = np.load(self.directory / f"{segment}.npz")
            self._cache[segment] = {k: payload[k] for k in payload.files}
        return self._cache[segment]

    @property
    def segments(self) -> list[str]:
        return [entry["segment"] for entry in self.entries]

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int):
        segment, i = self._index[index]
        arrays = self.load(segment)
        return arrays["rgb"][i], arrays["depth"][i], arrays["mask"][i]


def evaluate(model, shards: Shards, segments: list[str], device: str, batch_size: int, constant: float):
    model.eval()
    rows = {}
    with torch.no_grad():
        for segment in segments:
            arrays = shards.load(segment)
            rgb = torch.from_numpy(arrays["rgb"]).float().div_(255.0)
            depth = torch.from_numpy(arrays["depth"]).float()
            mask = torch.from_numpy(arrays["mask"]).bool()
            abs_rels, delta1s = [], []
            base_abs, base_d1 = [], []
            for start in range(0, len(rgb), batch_size):
                x = rgb[start : start + batch_size].to(device)
                d = depth[start : start + batch_size].to(device)
                m = mask[start : start + batch_size].to(device)
                prediction = model(x)
                abs_rels.append(abs_rel(prediction, d, m))
                delta1s.append(delta1(prediction, d, m))
                baseline = torch.full_like(d, constant)
                base_abs.append(abs_rel(baseline, d, m))
                base_d1.append(delta1(baseline, d, m))
            rows[segment] = (
                float(np.mean(abs_rels)),
                float(np.mean(delta1s)),
                float(np.mean(base_abs)),
                float(np.mean(base_d1)),
                len(rgb),
            )
    return rows


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

    shards = Shards(Path(args.data))
    segments = sorted(shards.segments)
    val_segments = segments[-args.val_segments :]
    train_segments = segments[: -args.val_segments]
    train_indices = [i for i, (s, _) in enumerate(shards._index) if s in set(train_segments)]
    print(f"segments: {len(train_segments)} train / {len(val_segments)} val")
    print(f"samples:  {len(train_indices)} train / {len(shards) - len(train_indices)} val")

    # Global constant baseline, measured on the training split.
    depth_sum, pixel_count = 0.0, 0
    for segment in train_segments:
        arrays = shards.load(segment)
        mask = arrays["mask"]
        depth_sum += float((arrays["depth"].astype(np.float32) * mask).sum())
        pixel_count += int(mask.sum())
    constant = depth_sum / max(pixel_count, 1)
    print(f"constant-depth baseline: {constant:.2f} m")

    model = DepthHead().to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    use_amp = args.amp and device == "cuda"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    step = 0
    last = {}
    while step < args.steps:
        order = np.random.permutation(train_indices)
        for start in range(0, len(order), args.batch_size):
            batch = [shards[i] for i in order[start : start + args.batch_size]]
            rgb = torch.from_numpy(np.stack([b[0] for b in batch])).float().div_(255.0).to(device)
            depth = torch.from_numpy(np.stack([b[1] for b in batch])).float().to(device)
            mask = torch.from_numpy(np.stack([b[2] for b in batch])).bool().to(device)
            if mask.sum() == 0:
                continue

            model.train()
            optimiser.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
                prediction = model(rgb)
                loss = masked_log_l1(prediction.float(), depth, mask)
            loss.backward()
            grad = float(nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            optimiser.step()
            step += 1

            if step == 1 or step % args.log_every == 0:
                with torch.no_grad():
                    model.eval()
                    p = model(rgb)
                    print(
                        f"step {step:5d}  loss {loss.item():.4f}  absrel {abs_rel(p, depth, mask):.3f}  "
                        f"d1 {delta1(p, depth, mask):.3f}  grad {grad:.2f}  "
                        f"{time.perf_counter() - started:.0f}s"
                    )
            if step % args.eval_every == 0:
                rows = evaluate(model, shards, val_segments, device, args.batch_size, constant)
                mean_abs = float(np.mean([r[0] for r in rows.values()]))
                mean_d1 = float(np.mean([r[1] for r in rows.values()]))
                mean_base = float(np.mean([r[2] for r in rows.values()]))
                print(f"  VAL step {step}: absrel {mean_abs:.3f}  d1.25 {mean_d1:.3f}  "
                      f"(constant baseline absrel {mean_base:.3f})")
            if step >= args.steps:
                break

    rows = evaluate(model, shards, val_segments, device, args.batch_size, constant)
    summary = {
        "argv": list(sys.argv),
        "train_segments": train_segments,
        "val_segments": val_segments,
        "constant_baseline_m": constant,
        "val": {
            s: {"absrel": r[0], "delta1": r[1], "baseline_absrel": r[2],
                "baseline_delta1": r[3], "samples": r[4]}
            for s, r in rows.items()
        },
        "mean_absrel": float(np.mean([r[0] for r in rows.values()])),
        "mean_delta1": float(np.mean([r[1] for r in rows.values()])),
    }
    print("\nper-segment validation:")
    for segment, (a, d, ba, bd, n) in rows.items():
        print(f"  {segment[:12]}  n={n:4d}  absrel {a:.3f} (base {ba:.3f})  d1.25 {d:.3f} (base {bd:.3f})")
    mean_base = float(np.mean([r[2] for r in rows.values()]))
    print(f"\nmean absrel {summary['mean_absrel']:.3f}  mean d1.25 {summary['mean_delta1']:.3f}  "
          f"baseline absrel {mean_base:.3f}")
    summary["mean_baseline_absrel"] = mean_base

    torch.save({"model": model.state_dict(), "config": summary}, out / "head.pt")
    (out / "results.json").write_text(json.dumps(summary, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
