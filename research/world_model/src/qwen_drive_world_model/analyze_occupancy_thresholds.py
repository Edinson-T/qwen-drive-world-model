"""Compare shared/independent occupancy PR curves on Motion validation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader

from .gt_model import GTBEVWorldModel, ObjectHeadConfig
from .model import WorldModelConfig
from .motion_multi_data import MotionMultiHorizonDataset
from .shared_horizon_model import SharedHorizonWorldModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--shared-run", type=Path, required=True)
    parser.add_argument("--h0p5-run", type=Path, required=True)
    parser.add_argument("--h1-run", type=Path, required=True)
    parser.add_argument("--h2-run", type=Path, required=True)
    parser.add_argument("--h3-run", type=Path, required=True)
    parser.add_argument("--h5-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corridor-half-width-m", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_shared(path: Path, device: torch.device) -> SharedHorizonWorldModel:
    checkpoint = torch.load(path / "best.pt", map_location=device, weights_only=False)
    model = SharedHorizonWorldModel(
        WorldModelConfig(**checkpoint["config"]),
        ObjectHeadConfig(**checkpoint["object_config"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval()


def _load_independent(path: Path, device: torch.device) -> GTBEVWorldModel:
    checkpoint = torch.load(path / "best.pt", map_location=device, weights_only=False)
    model = GTBEVWorldModel(
        WorldModelConfig(**checkpoint["config"]),
        ObjectHeadConfig(**checkpoint["object_config"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval()


def _trajectory_in_target_frame(trajectory: np.ndarray, relative_pose: np.ndarray) -> np.ndarray:
    dx, dy, yaw = np.asarray(relative_pose, dtype=np.float64)
    cosine, sine = np.cos(yaw), np.sin(yaw)
    x, y = trajectory[:, 0] - dx, trajectory[:, 1] - dy
    return np.column_stack(
        [
            cosine * x + sine * y,
            -sine * x + cosine * y,
        ]
    )


def _corridor_mask(
    trajectory: np.ndarray,
    horizon_index: int,
    half_width_m: float,
) -> np.ndarray:
    steps = (5, 10, 20, 30, 50)[horizon_index] + 1
    points = trajectory[:steps]
    columns = (points[:, 0] + 30.0) / 0.6
    rows = (points[:, 1] + 15.0) / 0.6
    canvas = Image.new("L", (100, 50), color=0)
    width = max(1, int(round((2.0 * half_width_m) / 0.6)))
    ImageDraw.Draw(canvas).line(
        [tuple(map(float, point)) for point in np.column_stack([columns, rows])],
        fill=1,
        width=width,
        joint="curve",
    )
    return np.asarray(canvas, dtype=bool)


def _metrics_at_threshold(
    probability: np.ndarray,
    target: np.ndarray,
    corridor: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    prediction = probability >= threshold
    truth = target > 0.5
    true_positive = np.logical_and(prediction, truth).sum()
    predicted = prediction.sum()
    positive = truth.sum()
    union = np.logical_or(prediction, truth).sum()
    corridor_truth = np.logical_and(truth, corridor)
    corridor_true_positive = np.logical_and(prediction, corridor_truth).sum()
    return {
        "threshold": float(threshold),
        "precision": float(true_positive / predicted) if predicted else 1.0,
        "recall": float(true_positive / positive) if positive else 1.0,
        "iou": float(true_positive / union) if union else 1.0,
        "corridor_recall": (
            float(corridor_true_positive / corridor_truth.sum())
            if corridor_truth.any()
            else 1.0
        ),
    }


def main() -> None:
    args = parse_args()
    device = _device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    dataset = MotionMultiHorizonDataset(args.data_dir, split="val")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    shared = _load_shared(args.shared_run, device)
    independent_runs = [
        args.h0p5_run,
        args.h1_run,
        args.h2_run,
        args.h3_run,
        args.h5_run,
    ]
    independent = [_load_independent(path, device) for path in independent_runs]
    thresholds = np.linspace(0.05, 0.95, 19)
    probabilities = {
        "shared": [[] for _ in range(5)],
        "independent": [[] for _ in range(5)],
    }
    targets = [[] for _ in range(5)]
    corridors = [[] for _ in range(5)]
    with torch.no_grad():
        for batch in loader:
            z = batch["z_t"].to(device)
            trajectory = batch["trajectory_5s"].to(device)
            ego = batch["ego_status"].to(device)
            poses = batch["relative_poses"].to(device)
            shared_output = shared(z, trajectory, ego, poses)
            independent_outputs = [
                model(z, trajectory, ego) for model in independent
            ]
            for horizon in range(5):
                probabilities["shared"][horizon].append(
                    shared_output["occupancy_logits"][:, horizon].sigmoid().cpu().numpy()
                )
                probabilities["independent"][horizon].append(
                    independent_outputs[horizon]["occupancy_logits"].sigmoid().cpu().numpy()
                )
                targets[horizon].append(batch["target_occupancy"][:, horizon].numpy())
                corridors[horizon].extend(
                    [
                        _corridor_mask(
                            batch["trajectory_5s"][index].numpy(),
                            horizon,
                            args.corridor_half_width_m,
                        )
                        for index in range(z.shape[0])
                    ]
                )

    rows = []
    summary = {"horizons": {}, "corridor_half_width_m": args.corridor_half_width_m}
    for horizon, seconds in enumerate((0.5, 1.0, 2.0, 3.0, 5.0)):
        target = np.concatenate(targets[horizon])
        corridor = np.stack(corridors[horizon])
        horizon_result = {"seconds": seconds}
        for method in ("independent", "shared"):
            probability = np.concatenate(probabilities[method][horizon])
            metrics = [
                _metrics_at_threshold(probability, target, corridor, threshold)
                for threshold in thresholds
            ]
            best = max(metrics, key=lambda row: row["iou"])
            curve = sorted(metrics, key=lambda row: row["recall"])
            pr_area = float(
                np.trapezoid(
                    [row["precision"] for row in curve],
                    [row["recall"] for row in curve],
                )
            )
            horizon_result[method] = {"best_iou": best, "pr_area": pr_area}
            for row in metrics:
                rows.append({"method": method, "horizon": seconds, **row})
        summary["horizons"][str(seconds)] = horizon_result

    with (args.output / "threshold_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method", "horizon", "threshold", "precision", "recall", "iou", "corridor_recall"],
        )
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 5, figsize=(20, 4))
    for horizon, seconds in enumerate((0.5, 1.0, 2.0, 3.0, 5.0)):
        axis = axes[horizon]
        for method, color in (("independent", "#1f77b4"), ("shared", "#d62728")):
            selected = [
                row
                for row in rows
                if row["method"] == method and row["horizon"] == seconds
            ]
            axis.plot(
                [row["recall"] for row in selected],
                [row["precision"] for row in selected],
                marker="o",
                markersize=3,
                color=color,
                label=method,
            )
        axis.set_title(f"{seconds:g}s")
        axis.set_xlabel("recall")
        axis.set_ylabel("precision")
        axis.grid(alpha=0.25)
    axes[-1].legend()
    figure.tight_layout()
    figure.savefig(args.output / "occupancy_pr_curves.png", dpi=180)

    (args.output / "summary.md").write_text(
        "# Occupancy threshold analysis\n\n"
        + "\n".join(
            f"- {seconds:g}s: shared best IoU "
            f"{summary['horizons'][str(seconds)]['shared']['best_iou']['iou']:.4f}, "
            f"independent best IoU "
            f"{summary['horizons'][str(seconds)]['independent']['best_iou']['iou']:.4f}"
            for seconds in (0.5, 1.0, 2.0, 3.0, 5.0)
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

