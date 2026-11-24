"""Generate five Motion scenes with prediction/GT pairs at five horizons."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .gt_model import GTBEVWorldModel, ObjectHeadConfig
from .model import WorldModelConfig
from .motion_data import MotionWorldDataset
from .motion_gt import MOTION_OBJECT_CLASS_NAMES
from .perception_gt import GT_MAP_CLASS_NAMES, GT_MAP_PALETTE


OBJECT_COLORS = {
    "vehicle": "#ff8a00",
    "pedestrian": "#eb3b5a",
    "cyclist": "#20bf6b",
    "other": "#7b2cbf",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--h0p5-run", type=Path, required=True)
    parser.add_argument("--h1-run", type=Path, required=True)
    parser.add_argument("--h2-run", type=Path, required=True)
    parser.add_argument("--h3-run", type=Path, required=True)
    parser.add_argument("--h5-run", type=Path, required=True)
    parser.add_argument("--num-figures", type=int, default=5)
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model(path: Path, device: torch.device) -> GTBEVWorldModel:
    checkpoint = torch.load(path / "best.pt", map_location=device, weights_only=False)
    model = GTBEVWorldModel(
        WorldModelConfig(**checkpoint["config"]),
        ObjectHeadConfig(**checkpoint["object_config"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval()


def _highres(values: np.ndarray, output_hw: tuple[int, int]) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(values))[None, None].float()
    return F.interpolate(tensor, size=output_hw, mode="nearest")[0, 0].numpy()


def _map_rgb(labels: np.ndarray) -> np.ndarray:
    palette = np.asarray(GT_MAP_PALETTE, dtype=np.uint8)
    raw = palette[np.asarray(labels, dtype=np.int64)]
    # Motion stores rows as y and columns as x. Rotate for forward-up display.
    return raw.transpose(1, 0, 2)[::-1, ::-1]


def _display_pixels(points: np.ndarray) -> np.ndarray:
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
    labels: np.ndarray,
) -> None:
    from matplotlib.patches import Polygon

    for box, label in zip(boxes, labels, strict=True):
        color = (
            "#2e7d32"
            if int(label) < 0
            else OBJECT_COLORS[MOTION_OBJECT_CLASS_NAMES[int(label)]]
        )
        pixels = _display_pixels(_box_footprint(box))
        axis.add_patch(
            Polygon(
                pixels,
                closed=True,
                fill=False,
                edgecolor="white",
                linewidth=3.0,
                zorder=8,
            )
        )
        axis.add_patch(
            Polygon(
                pixels,
                closed=True,
                fill=False,
                edgecolor=color,
                linewidth=1.5,
                zorder=9,
            )
        )


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


def _draw_trajectory(axis, trajectory: np.ndarray) -> None:
    pixels = _display_pixels(trajectory)
    axis.plot(pixels[:, 0], pixels[:, 1], color="white", linewidth=5, zorder=10)
    axis.plot(pixels[:, 0], pixels[:, 1], color="#9d19ff", linewidth=2.7, zorder=11)
    for seconds in (1, 2, 3, 5):
        index = min(seconds * 10, len(pixels) - 1)
        axis.scatter(
            pixels[index, 0],
            pixels[index, 1],
            s=20,
            facecolor="white",
            edgecolor="#9d19ff",
            linewidth=1.2,
            zorder=12,
        )


def _draw_ego(axis) -> None:
    from matplotlib.patches import Polygon

    corners = np.asarray(
        [[-1.4, -0.9], [1.8, -0.9], [2.7, -0.6], [2.7, 0.6], [1.8, 0.9], [-1.4, 0.9]]
    )
    axis.add_patch(
        Polygon(
            _display_pixels(corners),
            closed=True,
            facecolor="white",
            edgecolor="black",
            linewidth=0.9,
            zorder=13,
        )
    )


def _map_metrics(prediction: np.ndarray, target: np.ndarray) -> float:
    values = []
    for class_id in range(len(GT_MAP_CLASS_NAMES)):
        predicted = prediction == class_id
        truth = target == class_id
        union = np.logical_or(predicted, truth).sum()
        if union:
            values.append(np.logical_and(predicted, truth).sum() / union)
    return float(np.mean(values))


def _occupancy_iou(prediction: np.ndarray, target: np.ndarray) -> float:
    predicted = prediction > 0.5
    truth = target > 0.5
    union = np.logical_or(predicted, truth).sum()
    return float(np.logical_and(predicted, truth).sum() / union) if union else 1.0


def _rgba(values: np.ndarray, rgb: tuple[int, int, int], alpha: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    rgba = np.empty((*values.shape, 4), dtype=np.float32)
    rgba[..., :3] = np.asarray(rgb, dtype=np.float32) / 255.0
    rgba[..., 3] = np.clip(values, 0, 1) * alpha
    return rgba


def main() -> None:
    args = parse_args()
    device = _device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str((args.output / ".mplconfig").resolve()))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    horizons = (0.5, 1.0, 2.0, 3.0, 5.0)
    runs = {
        0.5: args.h0p5_run,
        1.0: args.h1_run,
        2.0: args.h2_run,
        3.0: args.h3_run,
        5.0: args.h5_run,
    }
    datasets = {
        horizon: MotionWorldDataset(
            args.data_dir,
            horizon=horizon,
            split="val",
            include_highres=True,
        )
        for horizon in horizons
    }
    models = {horizon: _load_model(runs[horizon], device) for horizon in horizons}

    reference = datasets[1.0]
    source_indices = reference.indices
    scenario_ids = reference.arrays["scenario_ids"][source_indices]
    unique_scenarios = list(dict.fromkeys(str(value) for value in scenario_ids))
    chosen = np.linspace(0, len(unique_scenarios) - 1, args.num_figures, dtype=int)
    chosen_scenarios = [unique_scenarios[int(index)] for index in chosen]
    token_lookup = {
        horizon: {
            str(token): int(index)
            for token, index in zip(
                dataset.arrays["tokens"][dataset.indices],
                dataset.indices,
                strict=True,
            )
        }
        for horizon, dataset in datasets.items()
    }
    first_token_by_scenario = {}
    for token, scenario in zip(
        reference.arrays["tokens"][source_indices],
        scenario_ids,
        strict=True,
    ):
        first_token_by_scenario.setdefault(str(scenario), str(token))

    legend = [
        Patch(
            facecolor=np.asarray(color, dtype=np.float64) / 255.0,
            label=name.replace("_", " "),
        )
        for name, color in zip(GT_MAP_CLASS_NAMES, GT_MAP_PALETTE, strict=True)
    ]
    legend.extend(
        [
            Patch(facecolor=(0.12, 0.55, 0.24, 0.35), label="GT occupancy"),
            Patch(facecolor=(0.95, 0.35, 0.1, 0.35), label="predicted occupancy"),
            Patch(edgecolor="#2e7d32", facecolor="none", label="GT box"),
            Patch(edgecolor="#c62828", facecolor="none", label="predicted box"),
            Patch(edgecolor="#9d19ff", facecolor="none", label="logged trajectory"),
        ]
    )
    summary = []

    for figure_index, scenario_id in enumerate(chosen_scenarios, start=1):
        token = first_token_by_scenario[scenario_id]
        figure = plt.figure(figsize=(20, 12.5))
        grid = figure.add_gridspec(
            3,
            6,
            left=0.02,
            right=0.98,
            top=0.9,
            bottom=0.09,
            wspace=0.035,
            hspace=0.18,
            height_ratios=(0.8, 1.0, 1.0),
        )
        current = datasets[1.0][token_lookup[1.0][token]]
        current_map = current["map_t"].numpy()
        current_occ = current["occ_t"].numpy()
        current_height = current["height_t_highres"].numpy()
        current_boxes = current["boxes_t"][: current["box_count_t"]].numpy()
        current_labels = current["box_labels_t"][: current["box_count_t"]].numpy()
        top_band = np.where(
            current_height.max(0) > 0.5,
            current_height.argmax(0) + 1,
            0,
        )
        top_panels = [
            (
                _map_rgb(current_map),
                None,
                "Current GT map + boxes",
            ),
            (
                _map_rgb(current_map),
                _rgba(current_occ, (31, 140, 61), 0.5).transpose(1, 0, 2)[::-1, ::-1],
                "Current GT occupancy + boxes",
            ),
            (
                None,
                None,
                "Current highest occupied height band",
            ),
        ]
        for column, (image, overlay, title) in enumerate(top_panels):
            axis = figure.add_subplot(grid[0, column * 2 : column * 2 + 2])
            if column < 2:
                axis.imshow(image, interpolation="nearest")
                if overlay is not None:
                    axis.imshow(overlay, interpolation="nearest")
            else:
                top_display = top_band.T[::-1, ::-1]
                axis.imshow(top_display, cmap="viridis", vmin=0, vmax=4)
            _draw_boxes(axis, current_boxes, current_labels)
            _draw_ego(axis)
            axis.set_xlim(0, 200)
            axis.set_ylim(400, 0)
            axis.set_aspect("equal", adjustable="box")
            axis.set_title(title, fontsize=9)
            axis.axis("off")

        horizon_cells = {
            0.5: (1, 0),
            1.0: (1, 2),
            2.0: (1, 4),
            3.0: (2, 0),
            5.0: (2, 2),
        }
        horizon_rows = []
        for row, horizon in enumerate(horizons):
            dataset = datasets[horizon]
            sample = dataset[token_lookup[horizon][token]]
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
            probabilities = prediction["box_logits"][0, :, :-1].softmax(-1)
            scores, classes = probabilities.max(-1)
            keep = torch.nonzero(scores >= args.score_threshold, as_tuple=False).flatten()
            keep = keep[torch.argsort(scores[keep], descending=True)[:20]]
            predicted_boxes = prediction["boxes"][0, keep].cpu().numpy()
            predicted_labels = classes[keep].cpu().numpy()
            count = int(sample["box_count"])
            target_boxes = sample["box_targets"][:count].numpy()
            target_labels = sample["box_labels"][:count].numpy()
            trajectory = _trajectory_in_target_frame(
                sample["trajectory_5s"].numpy(),
                sample["relative_pose"].numpy(),
            )
            map_miou = _map_metrics(predicted_map, target_map)
            occupancy_iou = _occupancy_iou(predicted_occupancy, target_occupancy)
            panels = [
                (
                    predicted_map,
                    _rgba(predicted_occupancy, (242, 89, 26), 0.35),
                    predicted_boxes,
                    predicted_labels,
                    f"{horizon:g}s prediction | map mIoU {map_miou:.3f} | "
                    f"occ IoU {occupancy_iou:.3f} | {len(predicted_boxes)} boxes",
                ),
                (
                    target_map,
                    _rgba(target_occupancy, (31, 140, 61), 0.38),
                    target_boxes,
                    target_labels,
                    f"{horizon:g}s GT | {count} boxes",
                ),
            ]
            for column, (map_labels, occupancy_rgba, boxes, labels, title) in enumerate(panels):
                grid_row, grid_column = horizon_cells[horizon]
                axis = figure.add_subplot(grid[grid_row, grid_column + column])
                axis.imshow(_map_rgb(map_labels), interpolation="nearest")
                axis.imshow(
                    occupancy_rgba.transpose(1, 0, 2)[::-1, ::-1],
                    interpolation="nearest",
                )
                _draw_boxes(axis, boxes, labels)
                _draw_trajectory(axis, trajectory)
                _draw_ego(axis)
                axis.set_xlim(0, 200)
                axis.set_ylim(400, 0)
                axis.set_aspect("equal", adjustable="box")
                axis.set_title(title, fontsize=7.2, linespacing=1.15)
                axis.axis("off")
            horizon_rows.append(
                {
                    "horizon": horizon,
                    "map_miou": map_miou,
                    "occupancy_iou": occupancy_iou,
                    "predicted_boxes": int(len(predicted_boxes)),
                    "gt_boxes": count,
                }
            )

        info_axis = figure.add_subplot(grid[2, 4:6])
        info_axis.axis("off")
        info_axis.text(
            0.02,
            0.95,
            "Motion prediction summary",
            fontsize=10,
            fontweight="bold",
            va="top",
        )
        info_axis.text(
            0.02,
            0.84,
            "\n".join(
                f"{item['horizon']:g}s   map {item['map_miou']:.3f}   "
                f"occ {item['occupancy_iou']:.3f}   "
                f"boxes {item['predicted_boxes']}/{item['gt_boxes']}"
                for item in horizon_rows
            ),
            fontsize=8.3,
            family="monospace",
            va="top",
            linespacing=1.6,
        )
        info_axis.text(
            0.02,
            0.25,
            "Waymo Motion contains no camera images.\n"
            "Top row shows current GT BEV context instead.\n"
            "Every prediction/GT pair uses the same logged anchor.",
            fontsize=8,
            color="#444444",
            va="top",
            linespacing=1.5,
        )

        figure.suptitle(
            f"{scenario_id} | Motion prediction vs GT | 0.5 / 1 / 2 / 3 / 5 s",
            fontsize=13,
        )
        figure.legend(
            handles=legend,
            loc="lower center",
            ncol=7,
            fontsize=7.4,
            frameon=False,
            bbox_to_anchor=(0.5, 0.018),
        )
        figure.text(
            0.5,
            0.001,
            "BEV: x forward (up), y left. Occupancy is box-derived, not LiDAR free-space.",
            ha="center",
            fontsize=8,
            color="#444444",
        )
        path = args.output / f"motion_compare_{figure_index:02d}_{token}.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        summary.append(
            {
                "scenario_id": scenario_id,
                "token": token,
                "image": str(path.resolve()),
                "horizons": horizon_rows,
            }
        )
        print(f"{figure_index}/{len(chosen_scenarios)} wrote {path}")

    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "source": "Waymo Motion v1.3.1 official validation held-out test",
                "scenarios": chosen_scenarios,
                "horizons": list(horizons),
                "figures": summary,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

