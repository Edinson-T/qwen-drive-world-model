"""Generate five 3-camera + current/1s/3s BEV map comparison figures."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from qwen_drive_perception import geometry
from qwen_drive_perception.configuration_perception import (
    DET_BOX_COLORS,
    DET_CLASS_NAMES,
    MAP_CLASS_NAMES,
    MAP_PALETTE,
)

from .data import WorldWindowDataset
from .map_decoder import LatentMapDecoder
from .model import BEVWorldModel, WorldModelConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-data-dir", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--annotations-cache", type=Path, required=True)
    parser.add_argument("--h1-run", type=Path, required=True)
    parser.add_argument("--h1-decoder", type=Path, required=True)
    parser.add_argument("--h3-run", type=Path, required=True)
    parser.add_argument("--h3-decoder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-figures", type=int, default=5)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _device(value: str) -> torch.device:
    return torch.device(
        value if value != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )


def _load_world_model(path: Path, device: torch.device) -> BEVWorldModel:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = WorldModelConfig(**checkpoint["config"])
    model = BEVWorldModel(config).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval()


def _load_decoder(path: Path, input_channels: int, output_hw: tuple[int, int], device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    hidden = int(checkpoint["args"].get("hidden", 128))
    decoder = LatentMapDecoder(
        in_channels=input_channels,
        hidden=hidden,
        output_hw=output_hw,
    ).to(device)
    decoder.load_state_dict(checkpoint["decoder"])
    return decoder.eval()


def _map_rgb(label_map: np.ndarray) -> np.ndarray:
    palette = np.asarray(MAP_PALETTE, dtype=np.uint8)
    values = np.clip(np.asarray(label_map, dtype=np.int64), 0, len(palette) - 1)
    return palette[values].transpose(1, 0, 2)[::-1, ::-1]


def _iou(prediction: np.ndarray, target: np.ndarray, class_id: int) -> float:
    pred = prediction == class_id
    truth = target == class_id
    union = np.logical_or(pred, truth).sum()
    return float(np.logical_and(pred, truth).sum() / union) if union else 1.0


def _miou(prediction: np.ndarray, target: np.ndarray, num_classes: int = 6) -> float:
    return float(np.mean([_iou(prediction, target, index) for index in range(num_classes)]))


def _load_pairs(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open() as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[row["token"]] = row
    return rows


def _xy_to_map_pixels(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xy = np.asarray(xy, dtype=np.float64)
    return (15.0 - xy[:, 1]) / 0.15, (30.0 - xy[:, 0]) / 0.15


def _trajectory_in_frame(
    trajectory: np.ndarray,
    relative_pose: np.ndarray | None,
) -> np.ndarray:
    trajectory = np.asarray(trajectory, dtype=np.float64)
    if relative_pose is None:
        return trajectory[:, :2]
    dx, dy, yaw = np.asarray(relative_pose, dtype=np.float64)
    cosine, sine = np.cos(yaw), np.sin(yaw)
    rotated = np.column_stack(
        [
            cosine * (trajectory[:, 0] - dx) + sine * (trajectory[:, 1] - dy),
            -sine * (trajectory[:, 0] - dx) + cosine * (trajectory[:, 1] - dy),
        ]
    )
    return rotated


def _draw_boxes(
    axis,
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    score_threshold: float = 0.3,
    max_boxes: int = 30,
) -> int:
    if not len(boxes):
        return 0
    from matplotlib.patches import Polygon

    keep = np.flatnonzero(scores >= score_threshold)
    if len(keep) > max_boxes:
        keep = keep[np.argsort(scores[keep])[-max_boxes:]]
    corners = geometry.box_corners(torch.from_numpy(boxes[keep].astype(np.float32)))
    labelled = set(int(index) for index in np.argsort(scores[keep])[::-1][:6])
    for box_index, (footprint, label) in enumerate(
        zip(np.asarray(corners)[:, :4, :2], labels[keep], strict=True)
    ):
        columns, rows = _xy_to_map_pixels(footprint)
        class_name = DET_CLASS_NAMES[int(np.clip(label, 0, len(DET_CLASS_NAMES) - 1))]
        color = np.asarray(DET_BOX_COLORS[class_name], dtype=np.float64) / 255.0
        vertices = np.column_stack([columns, rows])
        axis.add_patch(
            Polygon(
                vertices,
                closed=True,
                facecolor="none",
                edgecolor="white",
                linewidth=3.2,
                alpha=0.9,
                zorder=7,
            )
        )
        axis.add_patch(
            Polygon(
                vertices,
                closed=True,
                facecolor=color,
                edgecolor=color,
                linewidth=1.7,
                alpha=0.34,
                zorder=8,
            )
        )
        axis.plot(
            np.append(columns, columns[0]),
            np.append(rows, rows[0]),
            color=color,
            linewidth=1.5,
            alpha=1.0,
            zorder=9,
        )
        if box_index in labelled:
            abbreviation = {
                "vehicle": "V",
                "czone_sign": "S",
                "bicycle": "B",
                "generic_object": "G",
                "pedestrian": "P",
                "traffic_cone": "C",
                "barrier": "R",
            }[class_name]
            axis.text(
                float(np.mean(columns)),
                float(np.mean(rows)),
                f"{abbreviation}{scores[keep[box_index]]:.2f}".replace("0.", "."),
                color="black",
                fontsize=5.2,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=11,
                clip_on=True,
                bbox={
                    "boxstyle": "round,pad=0.12",
                    "facecolor": "white",
                    "edgecolor": color,
                    "linewidth": 0.45,
                    "alpha": 0.82,
                },
            )
    return int(len(keep))


def _draw_trajectory(axis, trajectory_xy: np.ndarray) -> None:
    if len(trajectory_xy) < 2:
        return
    columns, rows = _xy_to_map_pixels(trajectory_xy)
    axis.plot(columns, rows, color="white", linewidth=5.0, alpha=0.95, zorder=10)
    axis.plot(columns, rows, color="#9d19ff", linewidth=2.8, zorder=11)
    point_interval = max(1, (len(columns) - 1) // 20)
    marker_indices = [seconds * 4 for seconds in (1, 2, 3, 5) if seconds * 4 < len(columns)]
    axis.scatter(
        columns[marker_indices],
        rows[marker_indices],
        s=24,
        facecolor="white",
        edgecolor="#9d19ff",
        linewidth=1.4,
        zorder=12,
    )
    for seconds, point_index in zip((1, 2, 3, 5), marker_indices, strict=False):
        axis.annotate(
            f"{seconds}s",
            (columns[point_index], rows[point_index]),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=5.5,
            color="#4d006b",
            fontweight="bold",
            zorder=13,
        )
    axis.scatter(
        [columns[0]],
        [rows[0]],
        s=32,
        marker="o",
        facecolor="#9d19ff",
        edgecolor="white",
        linewidth=1.3,
        zorder=13,
    )
    end = min(len(columns) - 1, max(1, point_interval))
    axis.annotate(
        "",
        xy=(columns[-1], rows[-1]),
        xytext=(columns[-1 - end], rows[-1 - end]),
        arrowprops={
            "arrowstyle": "-|>",
            "color": "#9d19ff",
            "lw": 2.2,
            "mutation_scale": 12,
            "shrinkA": 0,
            "shrinkB": 0,
        },
        zorder=13,
    )


def _draw_ego(axis) -> None:
    corners = np.asarray(
        [[-1.5, -0.9], [1.9, -0.9], [2.8, -0.6], [2.8, 0.6], [1.9, 0.9], [-1.5, 0.9]]
    )
    columns, rows = _xy_to_map_pixels(corners)
    axis.fill(columns, rows, facecolor="white", edgecolor="black", linewidth=1.0, zorder=9)


def _infer_map(world_model, decoder, sample: dict, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        z_t = sample["z_t"].float().unsqueeze(0).to(device)
        trajectory = sample["trajectory_5s"].float().unsqueeze(0).to(device)
        ego = sample["ego_status"].float().unsqueeze(0).to(device)
        latent = world_model(z_t, trajectory, ego)
        return decoder(latent).argmax(1)[0].cpu().numpy()


def main() -> None:
    args = parse_args()
    device = _device(args.device)
    os.environ.setdefault("MPLCONFIGDIR", str((args.output / ".mplconfig").resolve()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    h1_data = WorldWindowDataset(
        args.val_data_dir,
        horizon=1.0,
        split="val",
        include_maps=True,
    )
    h3_data = WorldWindowDataset(
        args.val_data_dir,
        horizon=3.0,
        split="val",
        include_maps=True,
    )
    indices = np.linspace(0, len(h1_data) - 1, args.num_figures, dtype=int)
    indices = sorted(set(int(index) for index in indices))
    h1_sample = h1_data[indices[0]]
    input_channels = int(h1_sample["z_t"].shape[0])
    output_hw = tuple(int(value) for value in h1_sample["map_future"].shape)

    h1_wm = _load_world_model(args.h1_run / "best.pt", device)
    h3_wm = _load_world_model(args.h3_run / "best.pt", device)
    h1_decoder = _load_decoder(args.h1_decoder / "best.pt", input_channels, output_hw, device)
    h3_decoder = _load_decoder(args.h3_decoder / "best.pt", input_channels, output_hw, device)
    pairs = _load_pairs(args.pairs)

    args.output.mkdir(parents=True, exist_ok=True)
    summary = []
    legend = [
        Patch(
            facecolor=np.asarray(MAP_PALETTE[index]) / 255.0,
            label=MAP_CLASS_NAMES[index],
        )
        for index in range(len(MAP_CLASS_NAMES))
    ]
    for figure_index, sample_index in enumerate(indices, start=1):
        sample1 = h1_data[sample_index]
        sample3 = h3_data[sample_index]
        if sample1["token"] != sample3["token"]:
            raise RuntimeError("1s and 3s validation datasets are not aligned")
        pair = pairs[str(sample1["token"])]
        annotation_path = args.annotations_cache / f"{sample1['pair_id']}.npz"
        with np.load(annotation_path) as annotation:
            annotations = {key: annotation[key] for key in annotation.files}
        current_dir = Path(pair["teacher_inputs"]["current"]) / "images"
        cameras = [("Front left", "FRONT_LEFT.jpg"), ("Front", "FRONT.jpg"), ("Front right", "FRONT_RIGHT.jpg")]

        prediction1 = _infer_map(h1_wm, h1_decoder, sample1, device)
        prediction3 = _infer_map(h3_wm, h3_decoder, sample3, device)
        target1 = np.asarray(sample1["map_future"])
        target3 = np.asarray(sample3["map_future"])
        current_map = np.asarray(sample1["map_t"])
        miou1 = _miou(prediction1, target1)
        miou3 = _miou(prediction3, target3)

        figure = plt.figure(figsize=(20, 13.8))
        outer = figure.add_gridspec(
            2,
            1,
            height_ratios=(0.50, 1.50),
            left=0.035,
            right=0.965,
            top=0.925,
            bottom=0.065,
            hspace=0.22,
        )
        top = outer[0].subgridspec(1, 3, wspace=0.035)
        bottom = outer[1].subgridspec(2, 3, wspace=0.08, hspace=0.24)
        for column, (title, filename) in enumerate(cameras):
            axis = figure.add_subplot(top[0, column])
            axis.imshow(Image.open(current_dir / filename).convert("RGB"))
            axis.set_title(title, fontsize=10)
            axis.axis("off")
        trajectory = sample1["trajectory_5s"].numpy()
        relative_pose = sample1["relative_pose"].numpy()
        panels = [
            (current_map, "Current BEV map", "t", None),
            (
                prediction1,
                f"1s WM prediction\nteacher mIoU {miou1:.3f}",
                "1s",
                relative_pose[1],
            ),
            (target1, "1s QD teacher BEV map", "1s", relative_pose[1]),
            (
                prediction3,
                f"3s WM prediction\nteacher mIoU {miou3:.3f}",
                "3s",
                relative_pose[3],
            ),
            (target3, "3s QD teacher BEV map", "3s", relative_pose[3]),
        ]
        panel_cells = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
        for column, (label_map, title, annotation_key, panel_pose) in enumerate(panels):
            axis = figure.add_subplot(bottom[panel_cells[column]])
            axis.imshow(_map_rgb(label_map), interpolation="nearest")
            box_count = _draw_boxes(
                axis,
                annotations[f"boxes_{annotation_key}"],
                annotations[f"scores_{annotation_key}"],
                annotations[f"labels_{annotation_key}"],
            )
            _draw_trajectory(axis, _trajectory_in_frame(trajectory, panel_pose))
            _draw_ego(axis)
            axis.set_xlim(0, label_map.shape[1])
            axis.set_ylim(label_map.shape[0], 0)
            axis.set_aspect("equal", adjustable="box")
            axis.set_title(
                f"{title}\n{box_count} detections (score >= 0.30)",
                fontsize=9.5,
                linespacing=1.25,
            )
            axis.axis("off")
        guide_axis = figure.add_subplot(bottom[1, 2])
        guide_axis.axis("off")
        guide_axis.set_xlim(0, 1)
        guide_axis.set_ylim(0, 1)
        guide_axis.text(
            0.02,
            0.94,
            "How to read the BEV panels",
            fontsize=11,
            fontweight="bold",
            va="top",
            transform=guide_axis.transAxes,
        )
        guide_axis.text(
            0.02,
            0.84,
            "x forward: up    |    y left: left\n"
            "Map crop: 30 m ahead, 15 m each side, 0.15 m/pixel",
            fontsize=8.5,
            color="#444444",
            va="top",
            linespacing=1.6,
            transform=guide_axis.transAxes,
        )
        guide_axis.text(
            0.02,
            0.68,
            "Action trajectory",
            fontsize=9,
            fontweight="bold",
            va="top",
            color="#9d19ff",
            transform=guide_axis.transAxes,
        )
        guide_axis.plot(
            [0.04, 0.34],
            [0.59, 0.59],
            transform=guide_axis.transAxes,
            color="white",
            linewidth=5,
            solid_capstyle="round",
        )
        guide_axis.plot(
            [0.04, 0.34],
            [0.59, 0.59],
            transform=guide_axis.transAxes,
            color="#9d19ff",
            linewidth=2.8,
            solid_capstyle="round",
        )
        guide_axis.scatter(
            [0.04, 0.34],
            [0.59, 0.59],
            transform=guide_axis.transAxes,
            s=25,
            facecolor="white",
            edgecolor="#9d19ff",
            linewidth=1.3,
            clip_on=False,
        )
        guide_axis.text(
            0.38,
            0.59,
            "logged 5 s trajectory in each panel's ego frame",
            fontsize=8,
            va="center",
            transform=guide_axis.transAxes,
        )
        guide_axis.text(
            0.02,
            0.48,
            "Detection boxes",
            fontsize=9,
            fontweight="bold",
            va="top",
            transform=guide_axis.transAxes,
        )
        guide_axis.text(
            0.02,
            0.40,
            "Colored footprint = frozen QD Perception detection.\n"
            "V vehicle  S sign  B bicycle  G object\n"
            "P pedestrian  C cone  R barrier\n"
            "Top six labels show abbreviated class + confidence.",
            fontsize=8,
            color="#333333",
            va="top",
            linespacing=1.55,
            transform=guide_axis.transAxes,
        )
        guide_axis.text(
            0.02,
            0.19,
            "Prediction / teacher target",
            fontsize=9,
            fontweight="bold",
            va="top",
            transform=guide_axis.transAxes,
        )
        guide_axis.text(
            0.02,
            0.11,
            "Teacher maps are frozen camera-perception pseudo-targets,\n"
            "not LiDAR ground truth. Boxes are frozen perception detections.",
            fontsize=8,
            color="#333333",
            va="top",
            linespacing=1.55,
            transform=guide_axis.transAxes,
        )
        figure.suptitle(
            f"{sample1['token']} | QD teacher comparison (no LiDAR GT)",
            fontsize=13,
        )
        figure.legend(
            handles=legend,
            loc="lower center",
            ncol=6,
            fontsize=8,
            frameon=False,
            bbox_to_anchor=(0.5, 0.028),
        )
        figure.text(
            0.5,
            0.004,
            "All detection boxes and maps here come from frozen camera-only QD Perception. "
            "The E2E caches contain no LiDAR ground truth.",
            ha="center",
            fontsize=8.5,
            color="#444444",
        )
        output_path = args.output / f"bev_compare_{figure_index:02d}_{sample1['pair_id']}.png"
        figure.savefig(output_path, dpi=180)
        plt.close(figure)
        summary.append(
            {
                "token": sample1["token"],
                "pair_id": sample1["pair_id"],
                "miou_1s": miou1,
                "miou_3s": miou3,
                "image": str(output_path.resolve()),
            }
        )
        print(f"{figure_index}/{len(indices)} wrote {output_path}")

    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "reference": "frozen QD Perception camera-only maps",
                "lidar_ground_truth": False,
                "figures": summary,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
