"""Train the action-conditioned GT BEV world model on Waymo Motion."""

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
from torch import nn
from torch.utils.data import DataLoader

from .gt_model import GTBEVWorldModel, ObjectHeadConfig
from .model import WorldModelConfig
from .motion_data import MotionWorldDataset
from .motion_gt import MODEL_HW, rasterize_boxes
from .perception_gt import GT_MAP_CLASS_NAMES, GT_MAP_PALETTE
from .train import EarlyStopping
from .train_perception_gt import _loss, _map_weights, evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--test-data-dir",
        type=Path,
        default=None,
        help="optional official Motion validation split, used only after training",
    )
    parser.add_argument("--target-horizon", type=float, default=1.0)
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
    parser.add_argument("--occupancy-weight", type=float, default=0.5)
    parser.add_argument("--occupancy-positive-weight", type=float, default=10.0)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=Path("outputs/world_model/perception_gt_v1/train_h1_v2/best.pt"),
    )
    parser.add_argument("--from-scratch", action="store_true")
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


def _warm_start(
    model: GTBEVWorldModel,
    checkpoint_path: Path,
    device: torch.device,
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    old_config = WorldModelConfig(**checkpoint["config"])
    if old_config.input_channels != 9:
        raise ValueError(f"expected 9-channel Perception checkpoint, got {old_config.input_channels}")
    old_state = checkpoint["model"]
    missing, unexpected = model.load_state_dict(old_state, strict=False)
    allowed_missing = {
        "backbone.bridge.height_projection.weight",
        "backbone.bridge.height_projection.bias",
    }
    disallowed_missing = sorted(set(missing) - allowed_missing)
    if disallowed_missing or unexpected:
        raise RuntimeError(
            f"unsafe warm start: missing={disallowed_missing}, unexpected={sorted(unexpected)}"
        )

    # Motion class 3 is "other", while the Perception checkpoint's class 3 was
    # "sign". Preserve shared box-regression weights but reinitialize the
    # semantics-bearing classification output.
    class_head = model.object_head.class_head[-1]
    nn.init.xavier_uniform_(class_head.weight)
    nn.init.zeros_(class_head.bias)
    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "missing_keys": sorted(missing),
        "unexpected_keys": sorted(unexpected),
        "reset_class_head": "object_head.class_head[-1]",
        "reason": "Motion TYPE_OTHER replaced Perception sign class semantics",
    }


def _height_comparison(
    gt_height: np.ndarray,
    predicted_height: np.ndarray,
) -> dict[str, float]:
    prediction = predicted_height > 0.5
    target = gt_height > 0.5
    ious = []
    for index in range(target.shape[0]):
        union = (prediction[index] | target[index]).sum()
        intersection = (prediction[index] & target[index]).sum()
        ious.append(float(intersection) / float(union) if union else 1.0)
    return {
        **{
            f"predicted_box_height{index + 1}_iou": value
            for index, value in enumerate(ious)
        },
        "predicted_box_height_mean_iou": float(np.mean(ious)),
    }


def _boxes_to_height(
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray | None = None,
    *,
    threshold: float = 0.3,
) -> np.ndarray:
    keep = labels >= 0
    if scores is not None:
        keep &= scores >= threshold
    boxes = boxes[keep]
    labels = labels[keep]
    if len(boxes):
        boxes = boxes.copy()
    else:
        return np.zeros((4, *MODEL_HW), dtype=np.uint8)
    _, height = rasterize_boxes(boxes[:, :9], labels)
    out_h, out_w = MODEL_HW
    return height.reshape(4, out_h, 4, out_w, 4).max(axis=(2, 4))


@torch.no_grad()
def _validation_with_height(
    model: GTBEVWorldModel,
    loader: DataLoader,
    device: torch.device,
    class_weights: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[dict[str, float], list[dict]]:
    # The shared evaluator already covers map/occupancy/box metrics. Height
    # slices are an input-only representation, so add their direct reconstruction
    # check separately rather than pretending the model predicts future height.
    metrics, samples = evaluate(
        model,
        loader,
        device,
        class_weights,
        occupancy_weight=args.occupancy_weight,
        occupancy_positive_weight=args.occupancy_positive_weight,
        object_weight=args.object_loss_weight,
    )
    height_values = []
    for raw_batch in loader:
        batch = _move(raw_batch, device)
        output = model(batch["z_t"], batch["trajectory_5s"], batch["ego_status"])
        scores = output["box_logits"][..., :-1].softmax(-1)
        predicted_scores, predicted_classes = scores.max(-1)
        for index in range(batch["z_t"].shape[0]):
            target_height = _boxes_to_height(
                batch["box_targets"][index].cpu().numpy(),
                batch["box_labels"][index].cpu().numpy(),
            )
            predicted_height = _boxes_to_height(
                output["boxes"][index].cpu().numpy(),
                predicted_classes[index].cpu().numpy(),
                predicted_scores[index].cpu().numpy(),
            )
            height_values.append(_height_comparison(target_height, predicted_height))
    for index in range(4):
        metrics[f"predicted_box_height{index + 1}_iou"] = float(
            np.mean(
                [
                    value[f"predicted_box_height{index + 1}_iou"]
                    for value in height_values
                ]
            )
        )
    metrics["predicted_box_height_mean_iou"] = float(
        np.mean([value["predicted_box_height_mean_iou"] for value in height_values])
    )
    return metrics, samples


def _save_height_inputs(samples: list[dict], output_dir: Path) -> None:
    if not samples:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sample = samples[0]
    height = sample.get("height_t")
    if height is None:
        return
    height = np.repeat(
        np.repeat(height, 4, axis=1),
        4,
        axis=2,
    )
    figure, axes = plt.subplots(1, 5, figsize=(16, 3.2))
    for index in range(4):
        axes[index].imshow(height[index], cmap="viridis", vmin=0, vmax=1)
        axes[index].set_title(f"height band {index + 1}")
        axes[index].axis("off")
    axes[4].imshow(height.max(0), cmap="magma", vmin=0, vmax=1)
    axes[4].set_title("max projection")
    axes[4].axis("off")
    figure.suptitle(f"{sample['token']} | current height occupancy input")
    figure.tight_layout()
    figure.savefig(output_dir / "height_inputs.png", dpi=180)
    plt.close(figure)

    voxel_values = sample.get("height_t_highres")
    if voxel_values is not None:
        voxel_values = voxel_values[:, ::8, ::8] > 0.5
        z, y, x = np.nonzero(voxel_values)
        figure = plt.figure(figsize=(7, 6))
        axis = figure.add_subplot(111, projection="3d")
        axis.scatter(x * 0.15 - 30.0, y * 0.15 - 15.0, z, c=z, cmap="viridis", s=8)
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.set_zlabel("height band")
        axis.set_title("Current box-derived 3D occupancy")
        figure.tight_layout()
        figure.savefig(output_dir / "height_voxels_3d.png", dpi=180)
        plt.close(figure)


def _draw_motion_boxes(axis, boxes: np.ndarray, color: str, linewidth: float = 1.3) -> None:
    from matplotlib.patches import Polygon

    for box in boxes:
        x, y, _, length, width, _, yaw = box[:7]
        cosine, sine = np.cos(yaw), np.sin(yaw)
        local = np.asarray(
            [
                [length / 2, width / 2],
                [length / 2, -width / 2],
                [-length / 2, -width / 2],
                [-length / 2, width / 2],
            ]
        )
        footprint = np.asarray(
            [[x + cosine * px - sine * py, y + sine * px + cosine * py] for px, py in local]
        )
        columns = (footprint[:, 0] + 30.0) / 0.15
        rows = (footprint[:, 1] + 15.0) / 0.15
        axis.add_patch(
            Polygon(
                np.column_stack([columns, rows]),
                closed=True,
                fill=False,
                edgecolor=color,
                linewidth=linewidth,
                zorder=8,
            )
        )


def _save_motion_visualizations(samples: list[dict], output_dir: Path, horizon: float) -> None:
    if not samples:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    palette = np.asarray(GT_MAP_PALETTE, dtype=np.uint8)
    for index, sample in enumerate(samples[:4]):
        target_map = np.repeat(
            np.repeat(sample["map_target"], 4, axis=0),
            4,
            axis=1,
        )
        predicted_map = np.repeat(
            np.repeat(sample["map_prediction"], 4, axis=0),
            4,
            axis=1,
        )
        target_occ = np.repeat(
            np.repeat(sample["occupancy_target"], 4, axis=0),
            4,
            axis=1,
        )
        predicted_occ = np.repeat(
            np.repeat(sample["occupancy_prediction"], 4, axis=0),
            4,
            axis=1,
        )
        figure, axes = plt.subplots(2, 2, figsize=(13, 7.5))
        axes[0, 0].imshow(palette[target_map], interpolation="nearest")
        axes[0, 0].set_title("Motion map GT")
        axes[0, 1].imshow(palette[predicted_map], interpolation="nearest")
        axes[0, 1].set_title("GT WM prediction")
        axes[1, 0].imshow(target_occ, cmap="Greens", vmin=0, vmax=1)
        _draw_motion_boxes(
            axes[1, 0],
            sample["boxes_target"][: sample["box_count"]],
            "#2e7d32",
            1.8,
        )
        axes[1, 0].set_title(f"GT occupancy + boxes ({sample['box_count']})")
        prediction_scores = sample["box_scores"][:, :-1].max(-1)
        keep = np.flatnonzero(prediction_scores >= 0.3)
        keep = keep[np.argsort(prediction_scores[keep])[-30:]]
        axes[1, 1].imshow(predicted_occ, cmap="Oranges", vmin=0, vmax=1)
        _draw_motion_boxes(
            axes[1, 1],
            sample["boxes_prediction"][keep],
            "#c62828",
            1.2,
        )
        axes[1, 1].set_title(f"Predicted occupancy + boxes ({len(keep)})")
        for axis in axes.flat:
            axis.set_xlim(0, 400)
            axis.set_ylim(200, 0)
            axis.set_aspect("equal")
            axis.axis("off")
        figure.suptitle(f"{sample['token']} | Motion t+{horizon:g}s")
        figure.tight_layout()
        figure.savefig(output_dir / f"motion_wm_{index:02d}_{sample['pair_id']}.png", dpi=180)
        plt.close(figure)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = _device(args.device)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str((output / ".mplconfig").resolve()))

    train_dataset = MotionWorldDataset(
        args.data_dir,
        horizon=args.target_horizon,
        split="train",
        include_highres=True,
    )
    val_dataset = MotionWorldDataset(
        args.data_dir,
        horizon=args.target_horizon,
        split="val",
        include_highres=True,
    )
    test_dataset = (
        MotionWorldDataset(
            args.test_data_dir,
            horizon=args.target_horizon,
            split="val",
            include_highres=True,
        )
        if args.test_data_dir is not None
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    channels, height, width = train_dataset.arrays["z_t"].shape[1:]
    if (int(channels), int(height), int(width)) != (13, 50, 100):
        raise ValueError(f"expected [13,50,100] Motion input, got {channels,height,width}")
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
    model = GTBEVWorldModel(config, object_config).to(device)
    warm_start = None
    if not args.from_scratch:
        warm_start = _warm_start(model, args.init_checkpoint, device)
    class_weights = _map_weights(train_dataset, args.target_horizon, device)
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
            prediction = model(batch["z_t"], batch["trajectory_5s"], batch["ego_status"])
            loss, parts = _loss(
                prediction,
                batch,
                class_weights,
                occupancy_weight=args.occupancy_weight,
                occupancy_positive_weight=args.occupancy_positive_weight,
                object_weight=args.object_loss_weight,
            )
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
                metrics, samples = _validation_with_height(
                    model,
                    val_loader,
                    device,
                    class_weights,
                    args,
                )
                row = {
                    "step": step,
                    "train_loss": float(loss.detach()),
                    **{f"val_{key}": value for key, value in metrics.items()},
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
                torch.save(checkpoint, output / "latest.pt")
                if stopper.update(metrics["loss"], step):
                    best = metrics["loss"]
                    torch.save(checkpoint, output / "best.pt")
                _save_motion_visualizations(samples, output / "visualizations", args.target_horizon)
                _save_height_inputs(samples, output / "visualizations")
                if stopper.should_stop:
                    stop_reason = "early_stopping"
                    break
            if step >= args.steps:
                break
        if stopper.should_stop or step >= args.steps:
            break

    best_metrics = None
    test_metrics = None
    if (output / "best.pt").exists():
        checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        best_metrics, best_samples = _validation_with_height(
            model,
            val_loader,
            device,
            class_weights,
            args,
        )
        _save_motion_visualizations(
            best_samples,
            output / "best_visualizations",
            args.target_horizon,
        )
        _save_height_inputs(best_samples, output / "best_visualizations")
        if test_dataset is not None:
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
            )
            test_metrics, test_samples = _validation_with_height(
                model,
                test_loader,
                device,
                class_weights,
                args,
            )
            _save_motion_visualizations(
                test_samples,
                output / "test_visualizations",
                args.target_horizon,
            )
            _save_height_inputs(test_samples, output / "test_visualizations")

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
            "test": len(test_dataset) if test_dataset is not None else 0,
        },
        "warm_start": warm_start,
        "history": history,
        "best_val_loss": best,
        "best_step": stopper.best_step,
        "best_metrics": best_metrics,
        "test_metrics": test_metrics,
        "stop_reason": stop_reason,
        "map_class_names": list(GT_MAP_CLASS_NAMES),
        "coordinate_protocol": "future ego-frame prediction",
        "action_protocol": "logged Motion SDC trajectory",
        "rfs_note": "Motion does not provide WOD-E2E RFS",
    }
    (output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {output / 'metrics.json'}")


if __name__ == "__main__":
    main()

