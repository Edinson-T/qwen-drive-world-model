"""Train the GT BEV world model directly on Waymo Perception labels."""

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

from .data import horizon_key
from .gt_model import (
    GTBEVWorldModel,
    ObjectHeadConfig,
    bev_gt_loss,
    object_query_loss,
    segmentation_metrics,
)
from .model import WorldModelConfig
from .perception_data import PerceptionWorldDataset
from .perception_gt import GT_MAP_CLASS_NAMES, GT_MAP_PALETTE
from .train import EarlyStopping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--val-data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-horizon", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=500)
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
    parser.add_argument("--map-loss-weight", type=float, default=1.0)
    parser.add_argument("--occupancy-positive-weight", type=float, default=10.0)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--use-highres-targets",
        action="store_true",
        help="Use map_*/occ_* arrays as targets instead of latent channels.",
    )
    parser.add_argument(
        "--allow-missing-boxes",
        action="store_true",
        help="Disable box targets when the dataset does not contain boxes.",
    )
    parser.add_argument("--target-height", type=int, default=None)
    parser.add_argument("--target-width", type=int, default=None)
    parser.add_argument("--progressive-decoder", action="store_true")
    parser.add_argument("--input-key", choices=("z_t", "f_t"), default="z_t")
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


def _map_weights(
    dataset: PerceptionWorldDataset,
    horizon: float,
    device: torch.device,
    *,
    use_highres_targets: bool,
):
    key = horizon_key(horizon)
    if use_highres_targets:
        maps = dataset.arrays[f"map_{key}"][:, ::8, ::8]
    else:
        maps = dataset.arrays[f"z_{key}"][:, :8].argmax(1)
    counts = np.bincount(maps.ravel(), minlength=8).astype(np.float64)
    frequency = counts / max(counts.sum(), 1.0)
    weights = 1.0 / np.sqrt(frequency + 1e-6)
    weights /= weights.mean()
    weights = np.clip(weights, 0.25, 10.0)
    print("map class counts:", counts.astype(np.int64).tolist())
    print("map class weights:", np.round(weights, 3).tolist())
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _loss(
    output: dict[str, torch.Tensor],
    batch: dict,
    class_weights: torch.Tensor,
    *,
    occupancy_weight: float,
    map_loss_weight: float,
    occupancy_positive_weight: float,
    object_weight: float,
):
    bev_loss, bev_parts = bev_gt_loss(
        output,
        batch["target_map"],
        batch["target_occupancy"],
        class_weights,
        occupancy_weight=occupancy_weight,
        occupancy_positive_weight=occupancy_positive_weight,
    )
    if object_weight > 0:
        object_loss, object_parts = object_query_loss(
            output["boxes"],
            output["box_logits"],
            batch["box_targets"],
            batch["box_labels"],
            batch["box_count"],
        )
    else:
        object_loss = output["raster"].sum() * 0.0
        object_parts = {
            "object_classification": object_loss.detach(),
            "object_center_l1": object_loss.detach(),
            "object_size_l1": object_loss.detach(),
            "object_yaw_l1": object_loss.detach(),
            "object_regression": object_loss.detach(),
        }
    total = (
        map_loss_weight * bev_parts["map_loss"]
        + occupancy_weight * bev_parts["occupancy_loss"]
        + object_weight * object_loss
    )
    return total, {
        **bev_parts,
        **object_parts,
        "object_loss": object_loss.detach(),
    }


def _box_detection_counts(
    boxes: torch.Tensor,
    logits: torch.Tensor,
    target_boxes: torch.Tensor,
    target_labels: torch.Tensor,
    box_counts: torch.Tensor,
    *,
    score_threshold: float = 0.3,
    match_distance: float = 2.0,
) -> tuple[int, int, int, list[float]]:
    total_true = 0
    total_predicted = 0
    total_matched = 0
    matched_errors = []
    scores, classes = logits[..., :-1].max(-1)
    for batch_index in range(boxes.shape[0]):
        count = int(box_counts[batch_index])
        true_boxes = target_boxes[batch_index, :count].cpu().numpy()
        true_labels = target_labels[batch_index, :count].cpu().numpy()
        keep = torch.nonzero(
            scores[batch_index] >= score_threshold,
            as_tuple=False,
        ).flatten()
        keep = keep[torch.argsort(scores[batch_index, keep], descending=True)]
        predicted_boxes = boxes[batch_index, keep].cpu().numpy()
        predicted_labels = classes[batch_index, keep].cpu().numpy()
        total_true += count
        total_predicted += len(keep)
        used = np.zeros(count, dtype=bool)
        for predicted_box, predicted_label in zip(predicted_boxes, predicted_labels, strict=True):
            distance = np.linalg.norm(true_boxes[:, :2] - predicted_box[:2], axis=1)
            candidates = np.flatnonzero(
                (~used) & (true_labels == predicted_label) & (distance <= match_distance)
            )
            if not len(candidates):
                continue
            selected = candidates[np.argmin(distance[candidates])]
            used[selected] = True
            total_matched += 1
            matched_errors.append(float(distance[selected]))
    return total_true, total_predicted, total_matched, matched_errors


@torch.no_grad()
def evaluate(
    model: GTBEVWorldModel,
    loader: DataLoader,
    device: torch.device,
    class_weights: torch.Tensor,
    *,
    occupancy_weight: float,
    map_loss_weight: float,
    occupancy_positive_weight: float,
    object_weight: float,
) -> tuple[dict[str, float], list[dict]]:
    model.eval()
    values: dict[str, list[float]] = {}
    samples = []
    total_objects = 0
    total_predictions = 0
    total_matches = 0
    all_match_errors = []
    for raw_batch in loader:
        batch = _move(raw_batch, device)
        output = model(batch["z_t"], batch["trajectory_5s"], batch["ego_status"])
        loss, parts = _loss(
            output,
            batch,
            class_weights,
            occupancy_weight=occupancy_weight,
            map_loss_weight=map_loss_weight,
            occupancy_positive_weight=occupancy_positive_weight,
            object_weight=object_weight,
        )
        metrics = segmentation_metrics(
            output["map_logits"],
            batch["target_map"],
            output["occupancy_logits"],
            batch["target_occupancy"],
        )
        parts = {**parts, **metrics, "loss": loss.detach()}
        true_count, predicted_count, match_count, match_errors = _box_detection_counts(
            output["boxes"],
            output["box_logits"],
            batch["box_targets"],
            batch["box_labels"],
            batch["box_count"],
        )
        total_objects += true_count
        total_predictions += predicted_count
        total_matches += match_count
        all_match_errors.extend(match_errors)
        for key, value in parts.items():
            values.setdefault(key, []).append(float(value))
        if len(samples) < 8:
            for index in range(min(batch["z_t"].shape[0], 8 - len(samples))):
                samples.append(
                    {
                        "map_prediction": output["map_logits"][index].argmax(0).cpu().numpy(),
                        "map_target": batch["target_map"][index].cpu().numpy(),
                        "occupancy_prediction": output["occupancy_logits"][index].sigmoid().cpu().numpy(),
                        "occupancy_target": batch["target_occupancy"][index].cpu().numpy(),
                        "boxes_prediction": output["boxes"][index].cpu().numpy(),
                        "box_scores": output["box_logits"][index].softmax(-1).cpu().numpy(),
                        "boxes_target": batch["box_targets"][index].cpu().numpy(),
                        "box_labels_target": batch["box_labels"][index].cpu().numpy(),
                        "box_count": int(batch["box_count"][index]),
                        "token": batch["token"][index],
                        "pair_id": (
                            batch["pair_id"][index]
                            if "pair_id" in batch
                            else batch["token"][index]
                        ),
                    }
                )
                if "height_t" in batch:
                    samples[-1]["height_t"] = (
                        batch["height_t"][index].float().cpu().numpy()
                    )
                if "height_t_highres" in batch:
                    samples[-1]["height_t_highres"] = (
                        batch["height_t_highres"][index].float().cpu().numpy()
                    )
    metrics = {key: float(np.mean(item)) if item else math.nan for key, item in values.items()}
    metrics["box_precision"] = total_matches / max(total_predictions, 1)
    metrics["box_recall"] = total_matches / max(total_objects, 1)
    metrics["box_matched_center_mae"] = (
        float(np.mean(all_match_errors)) if all_match_errors else math.nan
    )
    return metrics, samples


def _draw_boxes(axis, boxes: np.ndarray, color: str, linewidth: float = 1.3) -> None:
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
        columns = (15.0 - footprint[:, 1]) / 0.15
        rows = (30.0 - footprint[:, 0]) / 0.15
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


def save_visualizations(
    samples: list[dict],
    output_dir: Path,
    horizon: float,
    device: torch.device,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    palette = np.asarray(GT_MAP_PALETTE, dtype=np.uint8)
    for index, sample in enumerate(samples[:4]):
        map_height, map_width = sample["map_target"].shape
        scale_y = max(1, 200 // map_height)
        scale_x = max(1, 400 // map_width)

        def highres(values: np.ndarray) -> np.ndarray:
            return np.repeat(
                np.repeat(np.asarray(values), scale_y, axis=0),
                scale_x,
                axis=1,
            )

        figure, axes = plt.subplots(2, 2, figsize=(13, 7.5))
        target_rgb = palette[highres(sample["map_target"])]
        prediction_rgb = palette[highres(sample["map_prediction"])]
        axes[0, 0].imshow(target_rgb, interpolation="nearest")
        axes[0, 0].set_title("LiDAR/map GT")
        axes[0, 1].imshow(prediction_rgb, interpolation="nearest")
        axes[0, 1].set_title("GT WM prediction")

        axes[1, 0].imshow(highres(sample["occupancy_target"]), cmap="Greens", vmin=0, vmax=1)
        _draw_boxes(axes[1, 0], sample["boxes_target"][: sample["box_count"]], "#2e7d32", 1.8)
        axes[1, 0].set_title(f"GT occupancy + boxes ({sample['box_count']})")

        axes[1, 1].imshow(highres(sample["occupancy_prediction"]), cmap="Oranges", vmin=0, vmax=1)
        scores = sample["box_scores"][:, :-1].max(-1)
        keep = np.flatnonzero(scores >= 0.3)
        keep = keep[np.argsort(scores[keep])[-30:]]
        _draw_boxes(axes[1, 1], sample["boxes_prediction"][keep], "#c62828", 1.2)
        axes[1, 1].set_title(f"Predicted occupancy + boxes ({len(keep)})")
        for axis in axes.flat:
            axis.set_xlim(0, 400)
            axis.set_ylim(200, 0)
            axis.set_aspect("equal")
            axis.axis("off")
        figure.suptitle(
            f"{sample['token']} | t+{horizon:g}s | future ego frame"
        )
        figure.tight_layout()
        figure.savefig(
            output_dir / f"gt_wm_{index:02d}_{sample['pair_id']}.png",
            dpi=180,
        )
        plt.close(figure)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = _device(args.device)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str((output / ".mplconfig").resolve()))

    train_dataset = PerceptionWorldDataset(
        args.data_dir,
        horizon=args.target_horizon,
        split="train",
        include_highres=args.use_highres_targets,
        use_highres_targets=args.use_highres_targets,
        allow_missing_boxes=args.allow_missing_boxes,
        target_hw=(
            None
            if args.target_height is None or args.target_width is None
            else (args.target_height, args.target_width)
        ),
        input_key=args.input_key,
    )
    validation_root = args.val_data_dir or args.data_dir
    try:
        val_dataset = PerceptionWorldDataset(
            validation_root,
            horizon=args.target_horizon,
            split="val",
            include_highres=args.use_highres_targets,
            use_highres_targets=args.use_highres_targets,
            allow_missing_boxes=args.allow_missing_boxes,
            target_hw=(
                None
                if args.target_height is None or args.target_width is None
                else (args.target_height, args.target_width)
            ),
            input_key=args.input_key,
        )
    except ValueError:
        val_dataset = train_dataset
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
    channels, height, width = train_dataset.arrays[args.input_key].shape[1:]
    output_hw = train_dataset.target_hw or (height, width)
    config = WorldModelConfig(
        input_channels=int(channels),
        hidden_size=args.hidden_size,
        token_hw=(args.token_height, args.token_width),
        target_hw=output_hw,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        progressive_decoder=args.progressive_decoder,
    )
    object_config = ObjectHeadConfig(num_queries=args.num_queries)
    model = GTBEVWorldModel(config, object_config).to(device)
    class_weights = _map_weights(
        train_dataset,
        args.target_horizon,
        device,
        use_highres_targets=args.use_highres_targets,
    )
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
            model_output = model(batch["z_t"], batch["trajectory_5s"], batch["ego_status"])
            loss, parts = _loss(
                model_output,
                batch,
                class_weights,
                occupancy_weight=args.occupancy_weight,
                map_loss_weight=args.map_loss_weight,
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
                metrics, samples = evaluate(
                    model,
                    val_loader,
                    device,
                    class_weights,
                    occupancy_weight=args.occupancy_weight,
                    map_loss_weight=args.map_loss_weight,
                    occupancy_positive_weight=args.occupancy_positive_weight,
                    object_weight=args.object_loss_weight,
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
                }
                torch.save(checkpoint, output / "latest.pt")
                if stopper.update(metrics["loss"], step):
                    best = metrics["loss"]
                    torch.save(checkpoint, output / "best.pt")
                save_visualizations(
                    samples,
                    output / "visualizations",
                    args.target_horizon,
                    device,
                )
                if stopper.should_stop:
                    stop_reason = "early_stopping"
                    break
            if step >= args.steps:
                break
        if stopper.should_stop or step >= args.steps:
            break

    best_metrics = None
    best_path = output / "best.pt"
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        best_metrics, best_samples = evaluate(
            model,
            val_loader,
            device,
            class_weights,
            occupancy_weight=args.occupancy_weight,
            map_loss_weight=args.map_loss_weight,
            occupancy_positive_weight=args.occupancy_positive_weight,
            object_weight=args.object_loss_weight,
        )
        save_visualizations(
            best_samples,
            output / "best_visualizations",
            args.target_horizon,
            device,
        )

    summary = {
        "config": asdict(config),
        "object_config": asdict(object_config),
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "samples": {"train": len(train_dataset), "val": len(val_dataset)},
        "history": history,
        "best_val_loss": best,
        "best_step": stopper.best_step,
        "best_metrics": best_metrics,
        "stop_reason": stop_reason,
        "map_class_names": list(GT_MAP_CLASS_NAMES),
        "coordinate_protocol": "future ego-frame prediction",
        "supervision": "Waymo Perception LiDAR labels + map features",
    }
    (output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {output / 'metrics.json'}")


if __name__ == "__main__":
    main()

