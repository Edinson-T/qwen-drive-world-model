"""Generate five Motion shared-WM prediction/GT comparison figures."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from .gt_model import ObjectHeadConfig, class_aware_nms_indices
from .model import WorldModelConfig
from .motion_multi_data import MotionMultiHorizonDataset
from .perception_gt import GT_MAP_CLASS_NAMES, GT_MAP_PALETTE
from .shared_horizon_model import SharedHorizonWorldModel
from .generate_motion_gt_comparisons import (
    _display_pixels,
    _draw_boxes,
    _draw_ego,
    _draw_trajectory,
    _highres,
    _map_metrics,
    _map_rgb,
    _occupancy_iou,
    _rgba,
    _trajectory_in_target_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--shared-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-figures", type=int, default=5)
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--occupancy-threshold", type=float, default=0.7)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    device = _device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str((args.output / ".mplconfig").resolve()))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    dataset = MotionMultiHorizonDataset(
        args.data_dir,
        split="val",
        include_highres=True,
    )
    checkpoint = torch.load(
        args.shared_run / "best.pt",
        map_location=device,
        weights_only=False,
    )
    model = SharedHorizonWorldModel(
        WorldModelConfig(**checkpoint["config"]),
        ObjectHeadConfig(**checkpoint["object_config"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    horizons = (0.5, 1.0, 2.0, 3.0, 5.0)
    scenario_ids = dataset.arrays["scenario_ids"][dataset.indices]
    unique_scenarios = list(dict.fromkeys(str(value) for value in scenario_ids))
    chosen = np.linspace(0, len(unique_scenarios) - 1, args.num_figures, dtype=int)
    chosen_scenarios = [unique_scenarios[int(index)] for index in chosen]
    first_indices = {}
    for source in dataset.indices:
        scenario = str(dataset.arrays["scenario_ids"][source])
        first_indices.setdefault(scenario, int(source))

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
        sample = dataset[dataset.indices.tolist().index(first_indices[scenario_id])]
        with torch.no_grad():
            output = model(
                sample["z_t"][None].to(device),
                sample["trajectory_5s"][None].to(device),
                sample["ego_status"][None].to(device),
                sample["relative_poses"][None].to(device),
            )
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
        current_map = sample["map_t"].numpy()
        current_occ = sample["occ_t"].numpy()
        current_height = sample["height_t_highres"].numpy()
        current_boxes = sample["boxes_t"][: sample["box_count_t"]].numpy()
        current_labels = sample["box_labels_t"][: sample["box_count_t"]].numpy()
        top_band = np.where(
            current_height.max(0) > 0.5,
            current_height.argmax(0) + 1,
            0,
        )
        top_panels = (
            (_map_rgb(current_map), None, "Current GT map + boxes"),
            (
                _map_rgb(current_map),
                _rgba(current_occ, (31, 140, 61), 0.5).transpose(1, 0, 2)[::-1, ::-1],
                "Current GT occupancy + boxes",
            ),
            (None, None, "Current highest occupied height band"),
        )
        for column, (image, overlay, title) in enumerate(top_panels):
            axis = figure.add_subplot(grid[0, column * 2 : column * 2 + 2])
            if column < 2:
                axis.imshow(image, interpolation="nearest")
                if overlay is not None:
                    axis.imshow(overlay, interpolation="nearest")
            else:
                axis.imshow(
                    top_band.T[::-1, ::-1],
                    cmap="viridis",
                    vmin=0,
                    vmax=4,
                )
            _draw_boxes(axis, current_boxes, current_labels)
            _draw_ego(axis)
            axis.set_xlim(0, 200)
            axis.set_ylim(400, 0)
            axis.set_aspect("equal")
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
        for horizon_index, horizon in enumerate(horizons):
            target_map = sample["map_future"][horizon_index].numpy()
            target_occ = sample["occupancy_future"][horizon_index].numpy()
            predicted_map = _highres(
                output["map_logits"][0, horizon_index].argmax(0).cpu().numpy(),
                target_map.shape,
            ).astype(np.int64)
            predicted_occ = _highres(
                output["occupancy_logits"][0, horizon_index].sigmoid().cpu().numpy(),
                target_occ.shape,
            )
            scores = output["box_logits"][0, horizon_index, :, :-1].softmax(-1)
            score_values, classes = scores.max(-1)
            keep = class_aware_nms_indices(
                output["boxes"][0, horizon_index : horizon_index + 1],
                output["box_logits"][0, horizon_index : horizon_index + 1],
                score_threshold=args.score_threshold,
                iou_threshold=args.nms_iou,
            )[0]
            predicted_boxes = output["boxes"][0, horizon_index, keep].cpu().numpy()
            predicted_labels = classes[keep].cpu().numpy()
            target_count = int(sample["box_counts"][horizon_index])
            target_boxes = sample["box_targets"][horizon_index, :target_count].numpy()
            target_labels = sample["box_labels"][horizon_index, :target_count].numpy()
            trajectory = _trajectory_in_target_frame(
                sample["trajectory_5s"].numpy(),
                sample["relative_poses"][horizon_index].numpy(),
            )
            map_miou = _map_metrics(predicted_map, target_map)
            occupancy_iou = _occupancy_iou(
                predicted_occ >= args.occupancy_threshold,
                target_occ,
            )
            panels = (
                (
                    predicted_map,
                    _rgba(
                        (predicted_occ >= args.occupancy_threshold).astype(np.float32),
                        (242, 89, 26),
                        0.4,
                    ),
                    predicted_boxes,
                    predicted_labels,
                    f"{horizon:g}s shared prediction | map {map_miou:.3f} | "
                    f"occ@{args.occupancy_threshold:g} {occupancy_iou:.3f} | "
                    f"{len(predicted_boxes)} boxes after NMS",
                ),
                (
                    target_map,
                    _rgba(target_occ, (31, 140, 61), 0.4),
                    target_boxes,
                    target_labels,
                    f"{horizon:g}s GT | {target_count} boxes",
                ),
            )
            grid_row, grid_column = horizon_cells[horizon]
            for column, (map_labels, occ_rgba, boxes, labels, title) in enumerate(panels):
                axis = figure.add_subplot(grid[grid_row, grid_column + column])
                axis.imshow(_map_rgb(map_labels), interpolation="nearest")
                axis.imshow(
                    occ_rgba.transpose(1, 0, 2)[::-1, ::-1],
                    interpolation="nearest",
                )
                _draw_boxes(axis, boxes, labels)
                _draw_trajectory(axis, trajectory)
                _draw_ego(axis)
                axis.set_xlim(0, 200)
                axis.set_ylim(400, 0)
                axis.set_aspect("equal")
                axis.set_title(title, fontsize=7.2)
                axis.axis("off")
            horizon_rows.append(
                {
                    "horizon": horizon,
                    "map_miou": map_miou,
                    "occupancy_iou": occupancy_iou,
                    "predicted_boxes": int(len(predicted_boxes)),
                    "gt_boxes": target_count,
                }
            )

        info_axis = figure.add_subplot(grid[2, 4:6])
        info_axis.axis("off")
        info_axis.text(0.02, 0.95, "Shared WM summary", fontsize=10, fontweight="bold", va="top")
        info_axis.text(
            0.02,
            0.84,
            "\n".join(
                f"{row['horizon']:g}s map {row['map_miou']:.3f} "
                f"occ {row['occupancy_iou']:.3f} "
                f"boxes {row['predicted_boxes']}/{row['gt_boxes']}"
                for row in horizon_rows
            ),
            fontsize=8.3,
            family="monospace",
            va="top",
            linespacing=1.6,
        )
        info_axis.text(
            0.02,
            0.3,
            "Motion has no camera images.\n"
            f"Predicted occupancy uses fixed threshold {args.occupancy_threshold:g}.\n"
            "Prediction and GT share the same scenario anchor.",
            fontsize=8,
            color="#444444",
            va="top",
            linespacing=1.5,
        )
        figure.suptitle(
            f"{scenario_id} | shared WM prediction vs GT | 0.5 / 1 / 2 / 3 / 5s",
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
        path = args.output / f"motion_shared_compare_{figure_index:02d}_{scenario_id}.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        summary.append(
            {
                "scenario_id": scenario_id,
                "token": sample["token"],
                "image": str(path.resolve()),
                "horizons": horizon_rows,
            }
        )
        print(f"{figure_index}/{args.num_figures} wrote {path}")

    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "model": "shared occupancy-weighted WM",
                "checkpoint": str((args.shared_run / "best.pt").resolve()),
                "occupancy_threshold": args.occupancy_threshold,
                "figures": summary,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

