"""Mini-batch training and diagnostics for the QD BEV world-model prototype."""

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
from torch.utils.data import DataLoader, Subset

from .data import WorldWindowDataset, overfit_view
from .model import BEVWorldModel, WorldModelConfig, latent_loss, normalize_latent
from .visualize import (
    plot_action_sensitivity,
    plot_copy_quad,
    plot_quad,
    plot_training_curve,
)


class EarlyStopping:
    """Stop after a fixed number of validation checks without improvement."""

    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        if patience < 0:
            raise ValueError("early-stop patience must be non-negative")
        if min_delta < 0:
            raise ValueError("early-stop min-delta must be non-negative")
        self.patience = patience
        self.min_delta = min_delta
        self.best = math.inf
        self.best_step: int | None = None
        self.bad_checks = 0
        self.should_stop = False

    def update(self, value: float, step: int) -> bool:
        improved = value < self.best - self.min_delta
        if improved:
            self.best = value
            self.best_step = step
            self.bad_checks = 0
        else:
            self.bad_checks += 1
            if self.patience > 0 and self.bad_checks >= self.patience:
                self.should_stop = True
        return improved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--val-data-dir",
        type=Path,
        default=None,
        help="optional separate dataset directory for validation",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-horizon", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--cosine-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--overfit-scenes", type=int, default=2)
    parser.add_argument("--overfit-samples", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=5,
        help="stop after this many validation checks without improvement; 0 disables",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=1e-4,
        help="minimum validation-loss decrease counted as an improvement",
    )
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--token-height", type=int, default=8)
    parser.add_argument("--token-width", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def _device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _infer_channels(dataset: WorldWindowDataset) -> int:
    source = int(dataset.indices[0])
    return int(dataset.arrays["z_t"][source].shape[0])


def _infer_target_hw(dataset: WorldWindowDataset) -> tuple[int, int]:
    source = int(dataset.indices[0])
    shape = dataset.arrays[dataset.target_key][source].shape
    return int(shape[1]), int(shape[2])


def _make_loader(indices: list[int] | np.ndarray, dataset, batch_size: int, *, shuffle: bool, workers: int):
    subset = Subset(dataset, list(map(int, indices)))
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=False,
    )


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


@torch.no_grad()
def evaluate(model: BEVWorldModel, loader: DataLoader, device: torch.device, cosine_weight: float):
    model.eval()
    numbers = {
        "loss": [],
        "smooth_l1": [],
        "cosine": [],
        "raw_l1": [],
        "copy_loss": [],
        "copy_smooth_l1": [],
        "copy_cosine": [],
        "copy_raw_l1": [],
    }
    kept = []
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        prediction = model(batch["z_t"], batch["trajectory_5s"], batch["ego_status"])
        loss, parts = latent_loss(
            prediction,
            batch["z_future"],
            cosine_weight=cosine_weight,
        )
        copy = BEVWorldModel.copy_current(
            batch["z_t"],
            (batch["z_future"].shape[-2], batch["z_future"].shape[-1]),
        )
        copy_loss, copy_parts = latent_loss(
            copy,
            batch["z_future"],
            cosine_weight=cosine_weight,
        )
        numbers["loss"].append(float(loss))
        for key, value in parts.items():
            numbers[key].append(float(value))
        numbers["copy_loss"].append(float(copy_loss))
        for key, value in copy_parts.items():
            numbers[f"copy_{key}"].append(float(value))
        if len(kept) < 8:
            for index in range(min(batch["z_t"].shape[0], 8 - len(kept))):
                kept.append(
                    {
                        "current": batch["z_t"][index].float().cpu().numpy(),
                        "target": batch["z_future"][index].float().cpu().numpy(),
                        "prediction": prediction[index].float().cpu().numpy(),
                        "copy": copy[index].float().cpu().numpy(),
                        "trajectory": batch["trajectory_5s"][index].float().cpu().numpy(),
                        "ego_status": batch["ego_status"][index].float().cpu().numpy(),
                        "token": batch["token"][index],
                        "pair_id": batch["pair_id"][index],
                    }
                )
    metrics = {key: float(np.mean(values)) if values else math.nan for key, values in numbers.items()}
    return metrics, kept


def _action_variants(trajectory: torch.Tensor) -> dict[str, torch.Tensor]:
    left = trajectory.clone()
    right = trajectory.clone()
    if left.shape[1] > 1:
        left[:, 1:, 1] += 2.0
        right[:, 1:, 1] -= 2.0
    return {"straight": trajectory, "left": left, "right": right}


@torch.no_grad()
def action_sensitivity(model: BEVWorldModel, sample: dict, device: torch.device) -> float:
    model.eval()
    z_t = torch.from_numpy(sample["current"])[None].float().to(device)
    trajectory = torch.from_numpy(sample["trajectory"])[None].float().to(device)
    ego = torch.from_numpy(sample["ego_status"])[None].float().to(device)
    predictions = [
        model(z_t, value, ego)[0].float().cpu().numpy()
        for value in _action_variants(trajectory).values()
    ]
    distances = []
    for left in range(len(predictions)):
        for right in range(left + 1, len(predictions)):
            distances.append(float(np.mean(np.abs(predictions[left] - predictions[right]))))
    return float(np.mean(distances)) if distances else 0.0


@torch.no_grad()
def save_diagnostics(
    model: BEVWorldModel,
    samples: list[dict],
    output_dir: Path,
    *,
    horizon: float,
    device: torch.device,
) -> dict:
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    metrics = []
    for index, sample in enumerate(samples[:4]):
        stem = f"{index:02d}_{sample['pair_id']}"
        target = sample["target"]
        prediction = sample["prediction"]
        current = sample["current"]
        copy = sample["copy"]
        pred_norm = normalize_latent(torch.from_numpy(prediction)[None]).numpy()[0]
        copy_norm = normalize_latent(torch.from_numpy(copy)[None]).numpy()[0]
        target_norm = normalize_latent(torch.from_numpy(target)[None]).numpy()[0]
        mean_error = float(np.mean(np.abs(pred_norm - target_norm)))
        copy_error = float(np.mean(np.abs(copy_norm - target_norm)))
        raw_mean_error = float(np.mean(np.abs(prediction - target)))
        raw_copy_error = float(np.mean(np.abs(copy - target)))
        title = (
            f"{sample['token']} | t+{horizon:g}s | "
            f"WM norm-MAE {mean_error:.5f} | copy norm-MAE {copy_error:.5f}"
        )
        plot_quad(current, target, prediction, vis_dir / f"quad_{stem}.png", title=title)
        plot_copy_quad(current, target, prediction, vis_dir / f"copy_{stem}.png", title=title)

        z_t = torch.from_numpy(current)[None].float().to(device)
        trajectory = torch.from_numpy(sample["trajectory"])[None].float().to(device)
        ego = torch.from_numpy(sample["ego_status"])[None].float().to(device)
        action_predictions = {
            name: model(z_t, value, ego)[0].float().cpu().numpy()
            for name, value in _action_variants(trajectory).items()
        }
        plot_action_sensitivity(
            current,
            action_predictions,
            vis_dir / f"actions_{stem}.png",
            title=f"{sample['token']} | action sensitivity",
        )
        metrics.append(
            {
                "token": sample["token"],
                "pair_id": sample["pair_id"],
                "wm_mae": mean_error,
                "copy_mae": copy_error,
                "wm_raw_mae": raw_mean_error,
                "copy_raw_mae": raw_copy_error,
                "action_pairwise_distance": action_sensitivity(model, sample, device),
            }
        )
    return {"samples": metrics}


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = _device(args.device)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str((output_dir / ".mplconfig").resolve()))

    train_dataset = WorldWindowDataset(args.data_dir, horizon=args.target_horizon, split="train")
    validation_root = args.val_data_dir or args.data_dir
    try:
        val_dataset = WorldWindowDataset(
            validation_root,
            horizon=args.target_horizon,
            split="val",
        )
    except ValueError:
        val_dataset = train_dataset
        val_indices = np.arange(len(val_dataset))
    else:
        val_indices = np.arange(len(val_dataset))
    if args.overfit:
        overfit_indices = overfit_view(
            train_dataset,
            args.overfit_scenes,
            args.overfit_samples,
        )
        train_loader = _make_loader(
            overfit_indices,
            train_dataset,
            min(args.batch_size, max(1, len(overfit_indices))),
            shuffle=True,
            workers=args.num_workers,
        )
        val_loader = _make_loader(
            overfit_indices,
            train_dataset,
            min(args.batch_size, max(1, len(overfit_indices))),
            shuffle=False,
            workers=0,
        )
    else:
        train_loader = _make_loader(
            np.arange(len(train_dataset)),
            train_dataset,
            args.batch_size,
            shuffle=True,
            workers=args.num_workers,
        )
        val_loader = _make_loader(
            val_indices,
            val_dataset,
            args.batch_size,
            shuffle=False,
            workers=0,
        )

    config = WorldModelConfig(
        input_channels=_infer_channels(train_dataset),
        hidden_size=args.hidden_size,
        token_hw=(args.token_height, args.token_width),
        target_hw=_infer_target_hw(train_dataset),
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    )
    model = BEVWorldModel(config).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    history = []
    best_val = math.inf
    early_stopping = EarlyStopping(
        args.early_stop_patience,
        args.early_stop_min_delta,
    )
    stop_reason = "max_steps"
    start = time.perf_counter()
    step = 0
    while step < args.steps:
        model.train()
        for raw_batch in train_loader:
            batch = _move_batch(raw_batch, device)
            optimiser.zero_grad(set_to_none=True)
            prediction = model(batch["z_t"], batch["trajectory_5s"], batch["ego_status"])
            loss, parts = latent_loss(
                prediction,
                batch["z_future"],
                cosine_weight=args.cosine_weight,
            )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            step += 1

            if step == 1 or step % args.log_every == 0:
                elapsed = time.perf_counter() - start
                print(
                    f"step {step:4d} loss {float(loss.detach()):.6f} "
                    f"smooth {float(parts['smooth_l1']):.6f} "
                    f"cos {float(parts['cosine']):.4f} "
                    f"grad {float(grad_norm):.3f} {elapsed:.1f}s",
                    flush=True,
                )

            if step % args.eval_every == 0 or step >= args.steps:
                val_metrics, samples = evaluate(
                    model,
                    val_loader,
                    device,
                    args.cosine_weight,
                )
                sensitivity = (
                    action_sensitivity(model, samples[0], device) if samples else 0.0
                )
                row = {
                    "step": step,
                    "train_loss": float(loss.detach()),
                    "train_smooth_l1": float(parts["smooth_l1"]),
                    "train_cosine": float(parts["cosine"]),
                    "val_loss": val_metrics["loss"],
                    "val_smooth_l1": val_metrics["smooth_l1"],
                    "val_cosine": val_metrics["cosine"],
                    "copy_loss": val_metrics["copy_loss"],
                    "copy_smooth_l1": val_metrics["copy_smooth_l1"],
                    "copy_cosine": val_metrics["copy_cosine"],
                    "action_pairwise_distance": sensitivity,
                    "elapsed_seconds": time.perf_counter() - start,
                    "cuda_max_memory_mb": (
                        torch.cuda.max_memory_allocated() / 2**20
                        if device.type == "cuda"
                        else 0.0
                    ),
                }
                history.append(row)
                print("  " + json.dumps(row, sort_keys=True), flush=True)
                checkpoint = {
                    "model": model.state_dict(),
                    "config": asdict(config),
                    "args": vars(args),
                    "step": step,
                }
                torch.save(checkpoint, output_dir / "latest.pt")
                improved = early_stopping.update(val_metrics["loss"], step)
                if improved:
                    best_val = val_metrics["loss"]
                    torch.save(checkpoint, output_dir / "best.pt")

                diagnostics = save_diagnostics(
                    model,
                    samples,
                    output_dir,
                    horizon=args.target_horizon,
                    device=device,
                )
                (output_dir / "diagnostics.json").write_text(
                    json.dumps(diagnostics, indent=2) + "\n"
                )
                if early_stopping.should_stop:
                    stop_reason = "early_stopping"
                    print(
                        f"early stopping at step {step}: no improvement for "
                        f"{early_stopping.bad_checks} validation checks "
                        f"(best step {early_stopping.best_step})",
                        flush=True,
                    )
                    break
            if step >= args.steps:
                stop_reason = "max_steps"
                break
            if early_stopping.should_stop:
                break
        if early_stopping.should_stop or step >= args.steps:
            break

    plot_training_curve(history, output_dir / "training_curves.png")
    best_diagnostics = None
    best_path = output_dir / "best.pt"
    if best_path.exists():
        best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(best_checkpoint["model"])
        best_metrics, best_samples = evaluate(
            model,
            val_loader,
            device,
            args.cosine_weight,
        )
        best_samples_diagnostics = save_diagnostics(
            model,
            best_samples,
            output_dir,
            horizon=args.target_horizon,
            device=device,
        )
        best_diagnostics = {
            "metrics": best_metrics,
            "samples": best_samples_diagnostics["samples"],
        }
        (output_dir / "best_diagnostics.json").write_text(
            json.dumps(best_diagnostics, indent=2) + "\n"
        )
    summary = {
        "config": asdict(config),
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "samples": {
            "train": len(train_loader.dataset),
            "val": len(val_loader.dataset),
        },
        "history": history,
        "best_val_loss": best_val,
        "best_step": early_stopping.best_step,
        "stop_reason": stop_reason,
        "stopped_early": stop_reason == "early_stopping",
        "bad_validation_checks": early_stopping.bad_checks,
        "early_stop_patience": args.early_stop_patience,
        "best_diagnostics": best_diagnostics,
        "coordinate_protocol": "future ego-frame prediction",
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
