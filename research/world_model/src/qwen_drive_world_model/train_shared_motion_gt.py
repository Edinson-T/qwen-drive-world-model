"""Train one shared Motion WM across all five horizons."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .gt_model import ObjectHeadConfig, bev_gt_loss, object_query_loss, segmentation_metrics
from .model import WorldModelConfig
from .motion_multi_data import MotionMultiHorizonDataset
from .shared_horizon_model import SharedHorizonWorldModel
from .train import EarlyStopping
from .train_perception_gt import _box_detection_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--test-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=Path("outputs/world_model/motion_gt_v1/train_h1/best.pt"),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--token-height", type=int, default=8)
    parser.add_argument("--token-width", type=int, default=8)
    parser.add_argument("--num-queries", type=int, default=48)
    parser.add_argument("--object-loss-weight", type=float, default=0.2)
    parser.add_argument(
        "--object-class-weights",
        type=float,
        nargs=5,
        default=(1.0, 3.0, 3.0, 3.0, 0.25),
    )
    parser.add_argument("--object-focal-gamma", type=float, default=0.0)
    parser.add_argument("--occupancy-weight", type=float, default=0.5)
    parser.add_argument("--occupancy-positive-weight", type=float, default=10.0)
    parser.add_argument(
        "--occupancy-horizon-weights",
        type=float,
        nargs=5,
        default=(1.0, 1.0, 1.0, 1.0, 1.0),
    )
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
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


def _class_weights(
    dataset: MotionMultiHorizonDataset,
    device: torch.device,
) -> torch.Tensor:
    maps = []
    for source in dataset.indices:
        for key in dataset.keys:
            maps.append(dataset.arrays[f"z_{key}"][int(source), :8].argmax(0))
    counts = np.bincount(np.stack(maps).ravel(), minlength=8).astype(np.float64)
    frequency = counts / max(counts.sum(), 1.0)
    weights = 1.0 / np.sqrt(frequency + 1e-6)
    weights /= weights.mean()
    weights = np.clip(weights, 0.25, 10.0)
    print("map class counts:", counts.astype(np.int64).tolist())
    print("map class weights:", np.round(weights, 3).tolist())
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _warm_start(
    model: SharedHorizonWorldModel,
    checkpoint_path: Path,
    device: torch.device,
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    allowed = {
        "horizon_embedding.weight",
        "pose_projection.0.weight",
        "pose_projection.0.bias",
        "pose_projection.2.weight",
        "pose_projection.2.bias",
        "condition_mlp.0.weight",
        "condition_mlp.0.bias",
        "condition_mlp.2.weight",
        "condition_mlp.2.bias",
    }
    disallowed = sorted(set(missing) - allowed)
    if disallowed or unexpected:
        raise RuntimeError(
            f"unsafe shared warm start: missing={disallowed}, unexpected={sorted(unexpected)}"
        )
    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "missing_keys": sorted(missing),
        "unexpected_keys": sorted(unexpected),
        "new_modules_zero_initialized": [
            "horizon_embedding",
            "pose_projection",
            "condition_mlp",
        ],
    }


def _loss(
    output: dict[str, torch.Tensor],
    batch: dict,
    class_weights: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    total = output["raster"].new_zeros(())
    map_values = []
    occupancy_values = []
    box_values = []
    horizon_losses = []
    for horizon in range(output["raster"].shape[1]):
        map_loss, map_parts = bev_gt_loss(
            {
                "map_logits": output["map_logits"][:, horizon],
                "occupancy_logits": output["occupancy_logits"][:, horizon],
            },
            batch["target_maps"][:, horizon],
            batch["target_occupancy"][:, horizon],
            class_weights,
            occupancy_weight=args.occupancy_weight,
            occupancy_positive_weight=args.occupancy_positive_weight,
        )
        object_loss, _ = object_query_loss(
            output["boxes"][:, horizon],
            output["box_logits"][:, horizon],
            batch["box_targets"][:, horizon],
            batch["box_labels"][:, horizon],
            batch["box_counts"][:, horizon],
            class_weights=output["raster"].new_tensor(args.object_class_weights),
            focal_gamma=args.object_focal_gamma,
        )
        horizon_loss = map_loss + args.object_loss_weight * object_loss
        horizon_losses.append(horizon_loss)
        map_values.append(map_parts["map_loss"])
        occupancy_values.append(map_parts["occupancy_loss"])
        box_values.append(object_loss.detach())
    horizon_count = output["raster"].shape[1]
    weights = output["raster"].new_tensor(args.occupancy_horizon_weights)
    total = (
        torch.stack(horizon_losses) * weights
    ).sum() / weights.sum()
    return total, {
        "map_loss": torch.stack(map_values).mean(),
        "occupancy_loss": torch.stack(occupancy_values).mean(),
        "object_loss": torch.stack(box_values).mean(),
    }


@torch.no_grad()
def evaluate(
    model: SharedHorizonWorldModel,
    loader: DataLoader,
    device: torch.device,
    class_weights: torch.Tensor,
    args: argparse.Namespace,
) -> dict:
    model.eval()
    loss_values = []
    horizon_values = {
        horizon: {
            "map_miou": [],
            "map_pixel_accuracy": [],
            "occupancy_iou": [],
            "box_precision": [],
            "box_recall": [],
            "box_center_mae": [],
        }
        for horizon in range(5)
    }
    for raw_batch in loader:
        batch = _move(raw_batch, device)
        output = model(
            batch["z_t"],
            batch["trajectory_5s"],
            batch["ego_status"],
            batch["relative_poses"],
        )
        loss, _ = _loss(output, batch, class_weights, args)
        loss_values.append(float(loss))
        for horizon in range(5):
            metrics = segmentation_metrics(
                output["map_logits"][:, horizon],
                batch["target_maps"][:, horizon],
                output["occupancy_logits"][:, horizon],
                batch["target_occupancy"][:, horizon],
            )
            true_count, predicted_count, match_count, errors = _box_detection_counts(
                output["boxes"][:, horizon],
                output["box_logits"][:, horizon],
                batch["box_targets"][:, horizon],
                batch["box_labels"][:, horizon],
                batch["box_counts"][:, horizon],
            )
            row = horizon_values[horizon]
            row["map_miou"].append(metrics["map_miou"])
            row["map_pixel_accuracy"].append(metrics["map_pixel_accuracy"])
            row["occupancy_iou"].append(metrics["occupancy_iou"])
            row["box_precision"].append(match_count / max(predicted_count, 1))
            row["box_recall"].append(match_count / max(true_count, 1))
            if errors:
                row["box_center_mae"].append(float(np.mean(errors)))
    result = {
        "loss": float(np.mean(loss_values)),
        "horizons": {},
    }
    for horizon, values in horizon_values.items():
        result["horizons"][str(horizon)] = {
            key: float(np.mean(items)) if items else math.nan
            for key, items in values.items()
        }
    return result


def _latency(model, batch: dict, device: torch.device, repeats: int = 10) -> float:
    model.eval()
    inputs = (
        batch["z_t"].to(device),
        batch["trajectory_5s"].to(device),
        batch["ego_status"].to(device),
        batch["relative_poses"].to(device),
    )
    with torch.no_grad():
        for _ in range(3):
            model(*inputs)
        start = time.perf_counter()
        for _ in range(repeats):
            model(*inputs)
    return (time.perf_counter() - start) / repeats


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = _device(args.device)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str((output_dir / ".mplconfig").resolve()))

    train_dataset = MotionMultiHorizonDataset(args.data_dir, split="train")
    val_dataset = MotionMultiHorizonDataset(args.data_dir, split="val")
    test_dataset = MotionMultiHorizonDataset(args.test_data_dir, split="val")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    config = WorldModelConfig(
        input_channels=13,
        base_channels=9,
        output_channels=9,
        hidden_size=args.hidden_size,
        token_hw=(args.token_height, args.token_width),
        target_hw=(50, 100),
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    )
    object_config = ObjectHeadConfig(num_queries=args.num_queries, num_classes=4)
    model = SharedHorizonWorldModel(config, object_config, num_horizons=5).to(device)
    warm_start = _warm_start(model, args.init_checkpoint, device)
    class_weights = _class_weights(train_dataset, device)
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    stopper = EarlyStopping(args.early_stop_patience, args.early_stop_min_delta)
    history = []
    best = math.inf
    stop_reason = "max_steps"
    started = time.perf_counter()
    step = 0
    while step < args.steps:
        model.train()
        for raw_batch in train_loader:
            batch = _move(raw_batch, device)
            output = model(
                batch["z_t"],
                batch["trajectory_5s"],
                batch["ego_status"],
                batch["relative_poses"],
            )
            loss, parts = _loss(output, batch, class_weights, args)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            step += 1
            if step == 1 or step % args.log_every == 0:
                print(
                    f"step {step:4d} loss {float(loss.detach()):.5f} "
                    f"map {float(parts['map_loss']):.5f} "
                    f"occ {float(parts['occupancy_loss']):.5f} "
                    f"box {float(parts['object_loss']):.5f} "
                    f"grad {float(grad_norm):.3f} "
                    f"{time.perf_counter() - started:.1f}s",
                    flush=True,
                )
            if step % args.eval_every == 0 or step >= args.steps:
                metrics = evaluate(model, val_loader, device, class_weights, args)
                row = {
                    "step": step,
                    "train_loss": float(loss.detach()),
                    **{f"val_{key}": value for key, value in metrics.items() if key != "horizons"},
                    "val_horizons": metrics["horizons"],
                    "elapsed_seconds": time.perf_counter() - started,
                }
                history.append(row)
                print("  " + json.dumps(row, sort_keys=True), flush=True)
                checkpoint = {
                    "model": model.state_dict(),
                    "config": asdict(config),
                    "object_config": asdict(object_config),
                    "args": {
                        key: str(value) if isinstance(value, Path) else value
                        for key, value in vars(args).items()
                    },
                    "step": step,
                    "class_weights": class_weights.cpu().tolist(),
                    "warm_start": warm_start,
                }
                torch.save(checkpoint, output_dir / "latest.pt")
                if stopper.update(metrics["loss"], step):
                    best = metrics["loss"]
                    torch.save(checkpoint, output_dir / "best.pt")
                if stopper.should_stop:
                    stop_reason = "early_stopping"
                    break
            if step >= args.steps:
                break
        if stopper.should_stop or step >= args.steps:
            break

    checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_metrics = evaluate(model, test_loader, device, class_weights, args)
    sample_batch = next(iter(test_loader))
    shared_latency = _latency(model, sample_batch, device)
    summary = {
        "config": asdict(config),
        "object_config": asdict(object_config),
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "samples": {
            "train": len(train_dataset),
            "val": len(val_dataset),
            "test": len(test_dataset),
        },
        "warm_start": warm_start,
        "history": history,
        "best_val_loss": best,
        "best_step": stopper.best_step,
        "test_metrics": test_metrics,
        "shared_latency_seconds": shared_latency,
        "peak_cuda_memory_mb": (
            torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0.0
        ),
        "stop_reason": stop_reason,
        "rfs_note": "Motion does not provide WOD-E2E RFS",
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()

