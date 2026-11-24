"""Train a lightweight latent-to-BEV-map decoder on frozen world-model outputs."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import WorldWindowDataset, horizon_key
from .map_decoder import LatentMapDecoder
from .model import BEVWorldModel, WorldModelConfig
from .train import EarlyStopping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--val-data-dir", type=Path, required=True)
    parser.add_argument("--wm-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-horizon", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _move(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _load_world_model(path: Path, device: torch.device) -> BEVWorldModel:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = WorldModelConfig(**checkpoint["config"])
    model = BEVWorldModel(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _class_weights(dataset: WorldWindowDataset, horizon: float, device: torch.device) -> torch.Tensor:
    target = dataset.arrays[f"map_{horizon_key(horizon)}"].astype(np.int64)
    counts = np.bincount(target.ravel(), minlength=6).astype(np.float64)
    frequency = counts / max(counts.sum(), 1.0)
    weights = 1.0 / np.sqrt(frequency + 1e-6)
    weights /= weights.mean()
    weights = np.clip(weights, 0.25, 10.0)
    print("map class counts:", counts.astype(np.int64).tolist())
    print("map class weights:", np.round(weights, 3).tolist())
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _confusion(prediction: np.ndarray, target: np.ndarray, num_classes: int = 6) -> np.ndarray:
    values = target.reshape(-1).astype(np.int64) * num_classes + prediction.reshape(-1).astype(
        np.int64
    )
    return np.bincount(values, minlength=num_classes * num_classes).reshape(
        num_classes, num_classes
    )


def _segmentation_metrics(confusion: np.ndarray) -> dict[str, float]:
    diagonal = np.diag(confusion).astype(np.float64)
    union = confusion.sum(axis=0) + confusion.sum(axis=1) - diagonal
    iou = np.divide(diagonal, union, out=np.zeros_like(diagonal), where=union > 0)
    present = confusion.sum(axis=1) > 0
    return {
        "pixel_accuracy": float(diagonal.sum() / max(confusion.sum(), 1)),
        "miou": float(iou[present].mean()) if present.any() else 0.0,
        "iou_background": float(iou[0]),
        "iou_driveable": float(iou[1]),
        "iou_road_line": float(iou[2]),
        "iou_road_edge": float(iou[3]),
        "iou_crosswalk": float(iou[4]),
        "iou_walkway": float(iou[5]),
    }


@torch.no_grad()
def evaluate(
    world_model: BEVWorldModel,
    decoder: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_weights: torch.Tensor,
) -> dict[str, float]:
    decoder.eval()
    total_loss = []
    confusion = np.zeros((6, 6), dtype=np.int64)
    for raw_batch in loader:
        batch = _move(raw_batch, device)
        latent = world_model(batch["z_t"], batch["trajectory_5s"], batch["ego_status"])
        logits = decoder(latent)
        target = batch["map_future"].long()
        total_loss.append(
            float(
                nn.functional.cross_entropy(
                    logits,
                    target,
                    weight=class_weights,
                )
            )
        )
        prediction = logits.argmax(1).cpu().numpy()
        confusion += _confusion(prediction, target.cpu().numpy())
    metrics = _segmentation_metrics(confusion)
    metrics["loss"] = float(np.mean(total_loss))
    metrics["confusion"] = confusion.tolist()
    return metrics


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = _device(args.device)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    train_dataset = WorldWindowDataset(
        args.data_dir,
        horizon=args.target_horizon,
        split="train",
        include_maps=True,
    )
    val_dataset = WorldWindowDataset(
        args.val_data_dir,
        horizon=args.target_horizon,
        split="val",
        include_maps=True,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    world_model = _load_world_model(args.wm_checkpoint, device)
    input_channels = int(train_dataset.arrays["z_t"].shape[1])
    decoder = LatentMapDecoder(
        in_channels=input_channels,
        hidden=args.hidden,
        output_hw=tuple(
            val_dataset.arrays[f"map_{horizon_key(args.target_horizon)}"].shape[-2:]
        ),
    ).to(device)
    class_weights = _class_weights(train_dataset, args.target_horizon, device)
    optimiser = torch.optim.AdamW(
        decoder.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    stopper = EarlyStopping(
        args.early_stop_patience,
        args.early_stop_min_delta,
    )
    history = []
    best_metrics = None
    step = 0
    started = time.perf_counter()
    stop_reason = "max_steps"
    while step < args.steps:
        decoder.train()
        for raw_batch in train_loader:
            batch = _move(raw_batch, device)
            with torch.no_grad():
                latent = world_model(
                    batch["z_t"],
                    batch["trajectory_5s"],
                    batch["ego_status"],
                )
            logits = decoder(latent)
            target = batch["map_future"].long()
            loss = nn.functional.cross_entropy(logits, target, weight=class_weights)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            optimiser.step()
            step += 1
            if step == 1 or step % args.log_every == 0:
                print(
                    f"step {step:4d} map loss {float(loss.detach()):.5f} "
                    f"grad {float(grad_norm):.3f} {time.perf_counter() - started:.1f}s",
                    flush=True,
                )
            if step % args.eval_every == 0 or step >= args.steps:
                metrics = evaluate(world_model, decoder, val_loader, device, class_weights)
                row = {
                    "step": step,
                    "train_loss": float(loss.detach()),
                    **metrics,
                    "elapsed_seconds": time.perf_counter() - started,
                    "cuda_max_memory_mb": (
                        torch.cuda.max_memory_allocated() / 2**20
                        if device.type == "cuda"
                        else 0.0
                    ),
                }
                history.append(row)
                print("  " + json.dumps({k: v for k, v in row.items() if k != "confusion"}), flush=True)
                checkpoint = {
                    "decoder": decoder.state_dict(),
                    "args": {
                        key: str(value) if isinstance(value, Path) else value
                        for key, value in vars(args).items()
                    },
                    "class_weights": class_weights.cpu().tolist(),
                    "step": step,
                }
                torch.save(checkpoint, output / "latest.pt")
                if stopper.update(metrics["loss"], step):
                    best_metrics = metrics
                    torch.save(checkpoint, output / "best.pt")
                if stopper.should_stop:
                    stop_reason = "early_stopping"
                    print(
                        f"early stopping at step {step}: best step {stopper.best_step}",
                        flush=True,
                    )
                    break
            if step >= args.steps:
                stop_reason = "max_steps"
                break
        if stopper.should_stop or step >= args.steps:
            break

    summary = {
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "samples": {"train": len(train_dataset), "val": len(val_dataset)},
        "best_step": stopper.best_step,
        "stop_reason": stop_reason,
        "best_val_loss": stopper.best,
        "best_metrics": best_metrics,
        "class_weights": class_weights.cpu().tolist(),
        "history": history,
    }
    (output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {output / 'metrics.json'}")


if __name__ == "__main__":
    main()
