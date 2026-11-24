"""Draw one real scene comparing the RL base with the best-of-N anchor path."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from qwen_drive.trajectory import resample_to_4hz
from qwen_drive_perception import geometry
from qwen_drive_perception.configuration_perception import (
    DET_BOX_COLORS,
    DET_CLASS_NAMES,
    MAP_PALETTE,
)

from .evaluate_paper_metrics import _rater_feedback
from .paper_figures import (
    _camera_paths,
    _draw_camera_axis,
    _find_scene_record,
    _load_predictions,
    _resample_xy,
)


def _trajectory_points(record: dict, index: int) -> np.ndarray:
    return np.asarray(record["trajectories"], dtype=np.float64)[index, :, :2]


def _pad_preference(path: np.ndarray) -> np.ndarray:
    path = np.asarray(path, dtype=np.float64)[:20, :2]
    if len(path) >= 20:
        return path
    return np.concatenate([path, np.repeat(path[-1:], 20 - len(path), axis=0)])


def _rfs_for_base_and_candidates(base_record: dict, candidate_record: dict) -> tuple[float, np.ndarray]:
    base = resample_to_4hz(
        np.asarray(base_record["trajectories"][0], dtype=np.float64),
        20,
        official_grid=True,
    )
    candidates = np.stack(
        [
            resample_to_4hz(np.asarray(path, dtype=np.float64), 20, official_grid=True)
            for path in candidate_record["trajectories"]
        ]
    )
    references = [
        [
            np.asarray(path, dtype=np.float64)[:20, :2]
            for path in candidate_record["preference_trajectories"]
        ]
    ]
    labels = [
        np.asarray(candidate_record["preference_scores"], dtype=np.float64)
    ]
    speeds = np.asarray([candidate_record["initial_speed"]], dtype=np.float64)
    base_rfs = _rater_feedback(base[None, None], references, labels, speeds)[0, 0]
    candidate_rfs = _rater_feedback(candidates[None], references, labels, speeds)[0]
    return float(base_rfs), candidate_rfs


def _render_qd_bev(
    axis,
    prediction: dict[str, np.ndarray],
    *,
    score_threshold: float,
    base_trajectory: np.ndarray,
    candidate_trajectory: np.ndarray,
    candidate_label: str,
) -> dict[str, float]:
    """Draw QD's current BEV map with paths, accepted boxes and the ego vehicle."""
    map_labels = np.clip(
        np.asarray(prediction["map"], dtype=np.int64),
        0,
        len(MAP_PALETTE) - 1,
    )
    palette = np.asarray(MAP_PALETTE, dtype=np.uint8)
    image = palette[map_labels].transpose(1, 0, 2)[::-1, ::-1]
    height, width = image.shape[:2]
    axis.imshow(image, interpolation="nearest")
    axis.set_xlim(0, width)
    axis.set_ylim(height, 0)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title("QD current BEV · map + paths + boxes", fontsize=11)
    axis.grid(False)

    def xy_to_pixels(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xy = np.asarray(xy, dtype=np.float64)
        columns = (15.0 - xy[:, 1]) / 0.15
        rows = (30.0 - xy[:, 0]) / 0.15
        return columns, rows

    scores = np.asarray(prediction["scores"], dtype=np.float64)
    keep = scores >= score_threshold
    boxes = np.asarray(prediction["boxes"], dtype=np.float64)[keep]
    labels = np.asarray(prediction["labels"], dtype=np.int64)[keep]
    if len(boxes):
        corners = geometry.box_corners(boxes)[:, :4, :2]
        for footprint, label in zip(corners, labels, strict=True):
            # The map image is ego-forward up and ego-left left.
            columns, rows = xy_to_pixels(footprint)
            axis.plot(
                np.append(columns, columns[0]),
                np.append(rows, rows[0]),
                color=np.asarray(
                    DET_BOX_COLORS[DET_CLASS_NAMES[int(np.clip(label, 0, 6))]],
                    dtype=np.float64,
                )
                / 255.0,
                linewidth=1.0,
            )

    base_columns, base_rows = xy_to_pixels(base_trajectory)
    candidate_columns, candidate_rows = xy_to_pixels(candidate_trajectory)
    axis.plot(
        base_columns,
        base_rows,
        color="tab:blue",
        linewidth=2.8,
        label="RL base",
        zorder=6,
    )
    axis.plot(
        candidate_columns,
        candidate_rows,
        color="tab:orange",
        linewidth=2.8,
        label=candidate_label,
        zorder=7,
    )

    # Simplified 4.6 m x 1.8 m ego footprint, with a small nose chevron.
    ego_outline = np.asarray(
        [
            [-1.5, -0.9],
            [1.9, -0.9],
            [2.8, -0.6],
            [2.8, 0.6],
            [1.9, 0.9],
            [-1.5, 0.9],
        ],
        dtype=np.float64,
    )
    ego_columns, ego_rows = xy_to_pixels(ego_outline)
    axis.fill(
        ego_columns,
        ego_rows,
        facecolor="white",
        edgecolor="black",
        linewidth=1.3,
        zorder=8,
    )
    windshield_columns, windshield_rows = xy_to_pixels(
        np.asarray([[1.15, -0.72], [1.15, 0.72]], dtype=np.float64)
    )
    axis.plot(
        windshield_columns,
        windshield_rows,
        color="black",
        linewidth=1.0,
        zorder=9,
    )
    nose_columns, nose_rows = xy_to_pixels(
        np.asarray([[2.8, -0.6], [3.15, 0.0], [2.8, 0.6]], dtype=np.float64)
    )
    axis.fill(
        nose_columns,
        nose_rows,
        facecolor="black",
        edgecolor="black",
        linewidth=0.8,
        zorder=9,
    )
    axis.legend(loc="lower right", fontsize=6.5, framealpha=0.9)

    return {
        "score_threshold": float(score_threshold),
        "boxes": float(len(boxes)),
        "max_box_score": float(scores.max()) if len(scores) else 0.0,
        "occupancy_nonempty_fraction": float(
            (np.asarray(prediction["occ"]) != 9).mean()
        ),
        "map_nonbackground_fraction": float((map_labels != 0).mean()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--candidate-index", type=int, default=None)
    parser.add_argument(
        "--comparison",
        choices=("best-of-n", "first"),
        default="best-of-n",
        help="candidate chosen for the right panel",
    )
    parser.add_argument(
        "--perception-dir",
        type=Path,
        default=None,
        help="directory of cached QD Perception .npz files, keyed by scene token",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=240)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str((output.parent / ".mplconfig").resolve()))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base_records = _load_predictions(args.baseline)
    candidate_records = _load_predictions(args.candidates)
    if args.token not in base_records or args.token not in candidate_records:
        raise SystemExit(f"token not present in both prediction files: {args.token}")
    base_record = base_records[args.token]
    candidate_record = candidate_records[args.token]
    scene_record = _find_scene_record(args.scenes, args.token)

    base_rfs, candidate_rfs = _rfs_for_base_and_candidates(base_record, candidate_record)
    if args.comparison == "first":
        index = 0
    else:
        index = (
            int(args.candidate_index)
            if args.candidate_index is not None
            else int(np.argmax(candidate_rfs))
        )
    if not 0 <= index < len(candidate_record["trajectories"]):
        raise SystemExit("--candidate-index is out of range")

    base = _trajectory_points(base_record, 0)
    candidates = np.asarray(candidate_record["trajectories"], dtype=np.float64)[..., :2]
    selected = candidates[index]
    ground_truth = _resample_xy(
        candidate_record["future_trajectory"],
        len(selected),
    )
    nav_command = scene_record.get("meta_info", {}).get("intent")
    if nav_command is None:
        command_index = int(scene_record.get("trajectory", {}).get("nav_command", 0))
        nav_command = ("GO_STRAIGHT", "TURN_LEFT", "TURN_RIGHT")[command_index]
    preferences = [
        _resample_xy(_pad_preference(path), len(selected))
        for path in candidate_record.get("preference_trajectories", [])
    ]
    preference_scores = candidate_record.get("preference_scores", [])
    base_official = resample_to_4hz(base, 20, official_grid=True)
    selected_official = resample_to_4hz(selected, 20, official_grid=True)
    ground_truth_official = resample_to_4hz(
        ground_truth,
        20,
        official_grid=True,
    )
    preferences_official = [
        _pad_preference(path)
        for path in candidate_record.get("preference_trajectories", [])
    ]

    camera_paths = _camera_paths(args.image_root, candidate_record, args.token)
    by_label = dict(camera_paths)
    camera_titles = ("Front left", "Front", "Front right")

    # Keep the trajectory landscape-oriented, but use a compact canvas so its
    # physical width is roughly half of the earlier 18-inch layout.
    figure = plt.figure(figsize=(14.5, 8.6))
    outer = figure.add_gridspec(
        2,
        1,
        height_ratios=(0.92, 1.08),
        hspace=0.13,
    )
    camera_grid = outer[0].subgridspec(1, 3, wspace=0.045)
    camera_axes = [figure.add_subplot(camera_grid[0, index]) for index in range(3)]
    bottom_grid = outer[1].subgridspec(
        1,
        2,
        width_ratios=(1.0, 1.0),
        wspace=0.055,
    )
    bev_axis = (
        figure.add_subplot(bottom_grid[0, 0])
        if args.perception_dir is not None
        else None
    )
    trajectory_axis = figure.add_subplot(
        bottom_grid[0, 1] if bev_axis is not None else bottom_grid[0, :]
    )

    for axis, title in zip(camera_axes, camera_titles, strict=True):
        _draw_camera_axis(axis, title, by_label.get(title))

    for preference, score in zip(preferences, preference_scores, strict=True):
        trajectory_axis.plot(
            preference[:, 0],
            preference[:, 1],
            linestyle=":",
            linewidth=2.4 if score == max(preference_scores) else 1.2,
            alpha=0.95 if score == max(preference_scores) else 0.55,
            color="tab:purple" if score == max(preference_scores) else "tab:brown",
            label=f"Human rated {score:g}",
        )
    trajectory_axis.scatter([0.0], [0.0], marker=">", s=80, color="black", zorder=8)

    trajectory_axis.plot(
        ground_truth[:, 0],
        ground_truth[:, 1],
        color="tab:green",
        linestyle="--",
        linewidth=2.4,
        label="Recorded future (GT)",
    )

    for candidate_index, path in enumerate(candidates):
        trajectory_axis.plot(
            path[:, 0],
            path[:, 1],
            color="tab:gray",
            linewidth=0.8,
            alpha=0.18,
            label="20 anchor candidates" if candidate_index == 0 else None,
        )

    trajectory_axis.plot(
        base[:, 0],
        base[:, 1],
        linewidth=2.8,
        color="tab:blue",
        label=f"RL base · RFS {base_rfs:.4f}",
    )

    candidate_label = (
        "Oracle / best-of-N"
        if args.comparison == "best-of-n"
        else "First candidate"
    )
    trajectory_axis.plot(
        selected[:, 0],
        selected[:, 1],
        linewidth=2.8,
        color="tab:orange",
        label=f"{candidate_label} #{index} · RFS {candidate_rfs[index]:.4f}",
    )
    trajectory_axis.set_title("Trajectory comparison · shared ego frame", fontsize=11)
    trajectory_axis.set_xlabel("x forward [m]")
    trajectory_axis.set_ylabel("y lateral [m]")
    trajectory_axis.set_aspect("equal", adjustable="datalim")
    trajectory_axis.grid(alpha=0.25)
    trajectory_axis.legend(loc="best", fontsize=7.5, ncol=2)

    perception_stats = None
    if bev_axis is not None:
        perception_path = args.perception_dir / f"{args.token}.npz"
        if not perception_path.exists():
            raise SystemExit(f"QD Perception cache missing: {perception_path}")
        with np.load(perception_path) as cached:
            perception = {key: cached[key] for key in cached.files}
        perception_stats = _render_qd_bev(
            bev_axis,
            perception,
            score_threshold=0.25,
            base_trajectory=base,
            candidate_trajectory=selected,
            candidate_label=candidate_label,
        )

    plotted_paths = [base, selected, ground_truth, *preferences, *candidates]
    plotted_xy = np.concatenate(plotted_paths, axis=0)
    x_min, y_min = plotted_xy.min(axis=0)
    x_max, y_max = plotted_xy.max(axis=0)
    x_pad = max((x_max - x_min) * 0.04, 1.0)
    y_pad = max((y_max - y_min) * 0.04, 1.0)
    trajectory_axis.set_xlim(x_min - x_pad, x_max + x_pad)
    trajectory_axis.set_ylim(y_min - y_pad, y_max + y_pad)

    figure.suptitle(
        (
            "Base vs Best-of-N"
            if args.comparison == "best-of-n"
            else "Base vs First Candidate"
        )
        + f"  |  NAV: {nav_command}  |  "
        f"RFS {base_rfs:.3f} -> {candidate_rfs[index]:.3f} "
        f"({candidate_rfs[index] - base_rfs:+.3f})",
        fontsize=15,
        fontweight="bold",
        y=0.975,
    )
    figure.text(
        0.5,
        0.928,
        "QD trajectory input: "
        "FRONT + FRONT_LEFT + FRONT_RIGHT × 4 history frames (12 images)",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#444444",
    )
    figure.subplots_adjust(left=0.04, right=0.985, top=0.865, bottom=0.06)
    figure.savefig(output.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)

    manifest = {
        "token": args.token,
        "nav_command": nav_command,
        "comparison": args.comparison,
        "candidate_index": index,
        "base_rfs": base_rfs,
        "candidate_rfs": float(candidate_rfs[index]),
        "rfs_gain": float(candidate_rfs[index] - base_rfs),
        "qd_current_bev": (
            None
            if perception_stats is None
            else {
                **perception_stats,
                "source": "QD Perception prediction on the current frame",
                "cache": str((args.perception_dir / f"{args.token}.npz").resolve()),
            }
        ),
        "ground_truth_ade": {
            "base": float(
                np.linalg.norm(base_official - ground_truth_official, axis=-1).mean()
            ),
            "candidate": float(
                np.linalg.norm(
                    selected_official - ground_truth_official,
                    axis=-1,
                ).mean()
            ),
        },
        "preference_scores": preference_scores,
        "preference_ade": {
            "base": [
                float(np.linalg.norm(base_official - preference, axis=-1).mean())
                for preference in preferences_official
            ],
            "candidate": [
                float(np.linalg.norm(selected_official - preference, axis=-1).mean())
                for preference in preferences_official
            ],
        },
        "oracle_warning": "candidate_index uses validation labels and is analysis-only",
        "camera_coverage": {
            "displayed_cameras": [label for label, _ in camera_paths],
            "trajectory_input_cameras": list(camera_titles),
            "history_frames_per_camera": int(
                scene_record.get("meta_info", {}).get("num_history_frames", 4)
            ),
            "trajectory_input_image_count": 3
            * int(scene_record.get("meta_info", {}).get("num_history_frames", 4)),
        },
        "baseline": str(args.baseline.resolve()),
        "candidates": str(args.candidates.resolve()),
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print(f"wrote {output.with_suffix('.png')}")


if __name__ == "__main__":
    main()
