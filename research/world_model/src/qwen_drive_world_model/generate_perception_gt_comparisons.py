"""Generate five Waymo Perception GT 3-camera / 1s-3s-5s BEV comparisons."""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from lidar_supervision.waymo_perception import iter_tfrecords, parse_frame

from .gt_model import GTBEVWorldModel, ObjectHeadConfig
from .model import WorldModelConfig
from .perception_data import PerceptionWorldDataset
from .perception_gt import (
    GT_MAP_CLASS_NAMES,
    GT_MAP_PALETTE,
    OBJECT_CLASS_NAMES,
)


OBJECT_COLORS = {
    "vehicle": "#ff8a00",
    "pedestrian": "#eb3b5a",
    "cyclist": "#20bf6b",
    "sign": "#9b51e0",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=Path("/home/fractal/datasets/waymo_perception"),
    )
    parser.add_argument("--h1-run", type=Path, required=True)
    parser.add_argument("--h3-run", type=Path, required=True)
    parser.add_argument("--h5-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-figures", type=int, default=5)
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _checkpoint_path(path: Path) -> Path:
    return path / "best.pt" if path.is_dir() else path


def _load_model(path: Path, device: torch.device) -> GTBEVWorldModel:
    checkpoint = torch.load(
        _checkpoint_path(path),
        map_location=device,
        weights_only=False,
    )
    config = WorldModelConfig(**checkpoint["config"])
    object_config = ObjectHeadConfig(**checkpoint["object_config"])
    model = GTBEVWorldModel(config, object_config).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval()


def _selected_camera_frames(
    raw_root: Path,
    samples: list[dict],
) -> dict[tuple[str, int], dict[int, bytes]]:
    wanted: dict[str, set[int]] = {}
    for sample in samples:
        wanted.setdefault(sample["scene_token"], set()).add(int(sample["t_frame"]))
    result = {}
    for scene, frames in wanted.items():
        paths = sorted(raw_root.glob(f"segment-{scene}*.tfrecord*"))
        if not paths:
            raise FileNotFoundError(f"no raw Perception segment for {scene}")
        for frame_index, payload in enumerate(iter_tfrecords(paths[0])):
            if frame_index not in frames:
                continue
            frame = parse_frame(payload)
            result[(scene, frame_index)] = frame.cameras
            if len(result) == sum(len(values) for values in wanted.values()):
                return result
    return result


def _map_rgb(labels: np.ndarray) -> np.ndarray:
    palette = np.asarray(GT_MAP_PALETTE, dtype=np.uint8)
    return palette[np.asarray(labels, dtype=np.int64)]


def _highres(values: np.ndarray, output_hw: tuple[int, int]) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(values))[None, None].float()
    tensor = F.interpolate(tensor, size=output_hw, mode="nearest")[0, 0]
    return tensor.numpy()


def _trajectory_in_target_frame(trajectory: np.ndarray, relative_pose: np.ndarray) -> np.ndarray:
    dx, dy, yaw = np.asarray(relative_pose, dtype=np.float64)
    cosine, sine = np.cos(yaw), np.sin(yaw)
    x = trajectory[:, 0] - dx
    y = trajectory[:, 1] - dy
    return np.column_stack(
        [
            cosine * x + sine * y,
            -sine * x + cosine * y,
        ]
    )


def _xy_to_pixels(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return np.stack(
        [
            (15.0 - points[:, 1]) / 0.15,
            (30.0 - points[:, 0]) / 0.15,
        ],
        axis=1,
    )


def _box_footprint(box: np.ndarray) -> np.ndarray:
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
    return np.asarray(
        [[x + cosine * px - sine * py, y + sine * px + cosine * py] for px, py in local]
    )


def _draw_boxes(
    axis,
    boxes: np.ndarray,
    colors: list[str],
    *,
    linewidth: float = 1.2,
) -> None:
    from matplotlib.patches import Polygon

    for box, color in zip(boxes, colors, strict=True):
        pixels = _xy_to_pixels(_box_footprint(box))
        axis.add_patch(
            Polygon(
                pixels,
                closed=True,
                fill=False,
                edgecolor="white",
                linewidth=linewidth + 1.5,
                zorder=8,
            )
        )
        axis.add_patch(
            Polygon(
                pixels,
                closed=True,
                fill=False,
                edgecolor=color,
                linewidth=linewidth,
                zorder=9,
            )
        )


def _draw_trajectory(axis, trajectory: np.ndarray) -> None:
    pixels = _xy_to_pixels(trajectory)
    axis.plot(pixels[:, 0], pixels[:, 1], color="white", linewidth=4.5, zorder=10)
    axis.plot(pixels[:, 0], pixels[:, 1], color="#9d19ff", linewidth=2.5, zorder=11)
    for seconds in (1, 2, 3, 5):
        index = min(seconds * 10, len(pixels) - 1)
        axis.scatter(
            pixels[index, 0],
            pixels[index, 1],
            s=18,
            facecolor="white",
            edgecolor="#9d19ff",
            linewidth=1.1,
            zorder=12,
        )


def _draw_ego(axis) -> None:
    from matplotlib.patches import Polygon

    corners = np.asarray(
        [[-1.4, -0.9], [1.8, -0.9], [2.7, -0.6], [2.7, 0.6], [1.8, 0.9], [-1.4, 0.9]]
    )
    axis.add_patch(
        Polygon(
            _xy_to_pixels(corners),
            closed=True,
            facecolor="white",
            edgecolor="black",
            linewidth=0.9,
            zorder=13,
        )
    )


def _map_metrics(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    miou = []
    for class_id in range(len(GT_MAP_CLASS_NAMES)):
        predicted = prediction == class_id
        truth = target == class_id
        union = np.logical_or(predicted, truth).sum()
        if union:
            miou.append(np.logical_and(predicted, truth).sum() / union)
    return float(np.mean(miou)) if miou else 0.0, float((prediction == target).mean())


def _occupancy_iou(prediction: np.ndarray, target: np.ndarray) -> float:
    predicted = prediction > 0.5
    truth = target > 0.5
    union = np.logical_or(predicted, truth).sum()
    return float(np.logical_and(predicted, truth).sum() / union) if union else 1.0


def _occupancy_rgba(values: np.ndarray, rgb: tuple[int, int, int], alpha: float) -> np.ndarray:
    rgba = np.zeros((*values.shape, 4), dtype=np.float32)
    rgba[..., :3] = np.asarray(rgb, dtype=np.float32) / 255.0
    rgba[..., 3] = np.clip(np.asarray(values, dtype=np.float32), 0, 1) * alpha
    return rgba


def main() -> None:
    args = parse_args()
    device = _device(args.device)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str((output / ".mplconfig").resolve()))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    horizons = (1.0, 3.0, 5.0)
    runs = {1.0: args.h1_run, 3.0: args.h3_run, 5.0: args.h5_run}
    datasets = {
        horizon: PerceptionWorldDataset(
            args.data_dir,
            horizon=horizon,
            split="val",
            include_highres=True,
        )
        for horizon in horizons
    }
    models = {horizon: _load_model(runs[horizon], device) for horizon in horizons}
    figure_indices = np.linspace(
        0,
        len(datasets[1.0]) - 1,
        args.num_figures,
        dtype=int,
    )
    figure_indices = sorted(set(int(index) for index in figure_indices))
    selected = [datasets[1.0][index] for index in figure_indices]
    camera_frames = _selected_camera_frames(
        args.raw_data_dir,
        [{"scene_token": sample["scene_token"], "t_frame": sample["t_frame"]} for sample in selected],
    )

    legend = [
        Patch(
            facecolor=np.asarray(color, dtype=np.float64) / 255.0,
            label=name.replace("_", " "),
        )
        for name, color in zip(GT_MAP_CLASS_NAMES, GT_MAP_PALETTE, strict=True)
    ]
    legend.extend(
        [
            Patch(facecolor=(0.12, 0.55, 0.24, 0.3), label="GT LiDAR occupancy"),
            Patch(facecolor=(0.95, 0.35, 0.1, 0.3), label="predicted occupancy"),
            Patch(edgecolor="#c62828", facecolor="none", label="predicted box"),
            Patch(edgecolor="#2e7d32", facecolor="none", label="GT LiDAR box"),
            Patch(edgecolor="#9d19ff", facecolor="none", label="logged 5 s trajectory"),
        ]
    )
    summary = []

    for figure_index, logical_index in enumerate(figure_indices, start=1):
        references = [datasets[horizon][logical_index] for horizon in horizons]
        token = str(references[0]["token"])
        if any(str(reference["token"]) != token for reference in references):
            raise RuntimeError("horizon datasets are not aligned")

        figure = plt.figure(figsize=(20, 16.5))
        outer = figure.add_gridspec(
            2,
            1,
            height_ratios=(0.42, 1.58),
            left=0.035,
            right=0.965,
            top=0.935,
            bottom=0.085,
            hspace=0.14,
        )
        camera_grid = outer[0].subgridspec(1, 3, wspace=0.035)
        bev_grid = outer[1].subgridspec(3, 2, wspace=0.045, hspace=0.18)
        cameras = camera_frames[(references[0]["scene_token"], int(references[0]["t_frame"]))]
        for column, (camera_id, title) in enumerate(
            [(2, "Front left"), (1, "Front"), (3, "Front right")]
        ):
            axis = figure.add_subplot(camera_grid[0, column])
            axis.imshow(Image.open(io.BytesIO(cameras[camera_id])).convert("RGB"))
            axis.set_title(title, fontsize=10)
            axis.axis("off")

        horizon_rows = []
        for row, horizon in enumerate(horizons):
            sample = references[row]
            with torch.no_grad():
                prediction = models[horizon](
                    sample["z_t"][None].to(device),
                    sample["trajectory_5s"][None].to(device),
                    sample["ego_status"][None].to(device),
                )
            output_hw = tuple(int(value) for value in sample["map_future"].shape)
            predicted_map = _highres(
                prediction["map_logits"][0].argmax(0).cpu().numpy(),
                output_hw,
            ).astype(np.int64)
            target_map = sample["map_future"].numpy()
            predicted_occupancy = _highres(
                prediction["occupancy_logits"][0].sigmoid().cpu().numpy(),
                output_hw,
            )
            target_occupancy = sample["occupancy_future"].numpy()
            scores = prediction["box_logits"][0, :, :-1].softmax(-1)
            predicted_scores, predicted_classes = scores.max(-1)
            keep = torch.nonzero(
                predicted_scores >= args.score_threshold,
                as_tuple=False,
            ).flatten()
            keep = keep[torch.argsort(predicted_scores[keep], descending=True)[:30]]
            predicted_boxes = prediction["boxes"][0, keep].cpu().numpy()
            predicted_labels = predicted_classes[keep].cpu().numpy()
            target_count = int(sample["box_count"])
            target_boxes = sample["box_targets"][:target_count].numpy()
            target_labels = sample["box_labels"][:target_count].numpy()
            trajectory = _trajectory_in_target_frame(
                sample["trajectory_5s"].numpy(),
                sample["relative_pose"].numpy(),
            )
            map_miou, map_accuracy = _map_metrics(predicted_map, target_map)
            occupation_iou = _occupancy_iou(predicted_occupancy, target_occupancy)

            panels = [
                (
                    predicted_map,
                    predicted_occupancy,
                    predicted_boxes,
                    [OBJECT_COLORS[OBJECT_CLASS_NAMES[label]] for label in predicted_labels],
                    f"{horizon:g}s prediction | map mIoU {map_miou:.3f} | "
                    f"occ IoU {occupation_iou:.3f}",
                    True,
                ),
                (
                    target_map,
                    target_occupancy,
                    target_boxes,
                    ["#2e7d32"] * len(target_boxes),
                    f"{horizon:g}s GT | map pixel acc {map_accuracy:.3f} | "
                    f"{target_count} LiDAR boxes",
                    False,
                ),
            ]
            for column, (
                map_labels,
                occupancy,
                boxes,
                colors,
                title,
                is_prediction,
            ) in enumerate(panels):
                axis = figure.add_subplot(bev_grid[row, column])
                axis.imshow(_map_rgb(map_labels), interpolation="nearest")
                if is_prediction:
                    axis.imshow(
                        _occupancy_rgba(occupancy, (242, 89, 26), 0.32),
                        interpolation="bilinear",
                    )
                else:
                    axis.imshow(
                        _occupancy_rgba(occupancy, (31, 140, 61), 0.34),
                        interpolation="nearest",
                    )
                _draw_boxes(axis, boxes, colors)
                _draw_trajectory(axis, trajectory)
                _draw_ego(axis)
                axis.set_xlim(0, output_hw[1])
                axis.set_ylim(output_hw[0], 0)
                axis.set_aspect("equal", adjustable="box")
                axis.set_title(title, fontsize=9.2)
                axis.axis("off")
            horizon_rows.append(
                {
                    "horizon": horizon,
                    "map_miou": map_miou,
                    "map_pixel_accuracy": map_accuracy,
                    "occupancy_iou": occupation_iou,
                    "predicted_boxes": int(len(predicted_boxes)),
                    "gt_boxes": target_count,
                }
            )

        figure.suptitle(
            f"{token} | Waymo Perception GT world model | "
            "3 cameras, 1 s / 3 s / 5 s",
            fontsize=13,
        )
        figure.legend(
            handles=legend,
            loc="lower center",
            ncol=7,
            fontsize=7.5,
            frameon=False,
            bbox_to_anchor=(0.5, 0.028),
        )
        figure.text(
            0.5,
            0.004,
            "GT: official Waymo map features + 3D LiDAR labels. "
            "BEV crop: x forward (up), y left; 30 m ahead, 15 m each side.",
            ha="center",
            fontsize=8.2,
            color="#444444",
        )
        output_path = output / f"perception_gt_compare_{figure_index:02d}_{token}.png"
        figure.savefig(output_path, dpi=180)
        plt.close(figure)
        summary.append(
            {
                "token": token,
                "pair_id": references[0]["pair_id"],
                "image": str(output_path.resolve()),
                "horizons": horizon_rows,
            }
        )
        print(f"{figure_index}/{len(figure_indices)} wrote {output_path}")

    (output / "summary.json").write_text(
        json.dumps(
            {
                "source": "Waymo Open Dataset Perception",
                "reference": "map features + 3D LiDAR labels",
                "cameras": ["FRONT_LEFT", "FRONT", "FRONT_RIGHT"],
                "horizons": list(horizons),
                "figures": summary,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

