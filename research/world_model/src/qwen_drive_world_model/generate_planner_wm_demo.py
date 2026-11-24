"""Generate one planner + WM + Reward Model qualitative comparison figure."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .build_selector_features import (
    _downsample_map,
    _load_independent,
    _occupancy_from_qd,
)
from .perception_gt import GT_MAP_PALETTE


HORIZONS = (0.5, 1.0, 2.0, 3.0, 5.0)
POINT_INDEX = (1, 3, 7, 11, 19)
GT_KEYS = ("map_0p5s", "map_1s", "map_2s", "map_3s", "map_5s")
GT_OCC_KEYS = ("occ_0p5s", "occ_1s", "occ_2s", "occ_3s", "occ_5s")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token", required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--selection-key", default="current")
    parser.add_argument("--bev-cache", type=Path, required=True)
    parser.add_argument("--pair-cache", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--h0p5-run", type=Path, required=True)
    parser.add_argument("--h1-run", type=Path, required=True)
    parser.add_argument("--h2-run", type=Path, required=True)
    parser.add_argument("--h3-run", type=Path, required=True)
    parser.add_argument("--h5-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _map_rgb(labels):
    palette = np.asarray(GT_MAP_PALETTE, dtype=np.uint8)
    return palette[np.asarray(labels, dtype=np.int64)].transpose(1, 0, 2)[::-1, ::-1]


def _rgba(values, color, alpha):
    values = np.asarray(values, dtype=np.float32)
    result = np.empty((*values.shape, 4), dtype=np.float32)
    result[..., :3] = np.asarray(color, dtype=np.float32) / 255.0
    result[..., 3] = np.clip(values, 0, 1) * alpha
    return result.transpose(1, 0, 2)[::-1, ::-1]


def _pixels(points):
    points = np.asarray(points)
    return np.stack([(15.0 - points[:, 1]) / 0.15, (30.0 - points[:, 0]) / 0.15], 1)


def _target_frame(path, pose):
    dx, dy, yaw = pose
    cosine, sine = np.cos(yaw), np.sin(yaw)
    x, y = path[:, 0] - dx, path[:, 1] - dy
    return np.column_stack(
        [cosine * x + sine * y, -sine * x + cosine * y]
    )


def _draw_path(axis, path, color, label=None, width=2.2, alpha=1.0):
    pixels = _pixels(path)
    axis.plot(
        pixels[:, 0],
        pixels[:, 1],
        color=color,
        linewidth=width + 1.8,
        alpha=alpha * 0.9,
        solid_capstyle="round",
        zorder=12,
    )
    axis.plot(
        pixels[:, 0],
        pixels[:, 1],
        color=color,
        linewidth=width,
        alpha=alpha,
        label=label,
        solid_capstyle="round",
        zorder=13,
    )


def _draw_ego(axis):
    from matplotlib.patches import Polygon

    corners = np.asarray(
        [[-1.4, -0.9], [1.8, -0.9], [2.7, -0.6], [2.7, 0.6], [1.8, 0.9], [-1.4, 0.9]]
    )
    axis.add_patch(
        Polygon(
            _pixels(corners),
            closed=True,
            facecolor="white",
            edgecolor="#202020",
            linewidth=1.0,
            zorder=15,
        )
    )


def _render_bev(axis, map_labels, occupancy, title):
    axis.imshow(_map_rgb(map_labels), interpolation="nearest")
    axis.imshow(
        _rgba(occupancy, (242, 77, 34), 0.42),
        interpolation="nearest",
    )
    _draw_ego(axis)
    axis.set_xlim(0, 400)
    axis.set_ylim(200, 0)
    axis.set_aspect("equal")
    axis.set_title(title, fontsize=10, pad=5)
    axis.axis("off")


def main():
    args = parse_args()
    os.environ.setdefault("MPLCONFIGDIR", str((args.output.parent / ".mplconfig").resolve()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    candidates = np.load(args.candidates)
    predictions = np.load(args.predictions)
    scene_index = int(np.flatnonzero(candidates["tokens"] == args.token)[0])
    prediction_index = (
        int(np.flatnonzero(predictions["tokens"] == args.token)[0])
        if "tokens" in predictions.files
        else scene_index
    )
    base_index = 0
    selected_index = int(predictions[args.selection_key][prediction_index])
    oracle_index = int(candidates["candidate_rfs"][scene_index].argmax())
    paths = candidates["trajectories"][scene_index]
    rfs = candidates["candidate_rfs"][scene_index]
    pair_rows = {
        row["token"]: row
        for row in (json.loads(line) for line in args.pairs.open())
    }
    pair = pair_rows[args.token]

    with np.load(args.bev_cache / f"{args.token}.npz") as payload:
        current_map = payload["map_t"]
        current_occ, current_height = _occupancy_from_qd(payload["occ_t"])
        current_bev = np.concatenate(
            [_downsample_map(current_map), current_occ[None], current_height]
        )
    with np.load(args.pair_cache / f"{args.token}.npz") as payload:
        actual_maps = [payload[key] for key in GT_KEYS]
        actual_occs = [_occupancy_from_qd(payload[key])[0] for key in GT_OCC_KEYS]

    device = torch.device(args.device)
    models = [
        _load_independent(path, device)
        for path in (args.h0p5_run, args.h1_run, args.h2_run, args.h3_run, args.h5_run)
    ]
    ego = torch.zeros((1, 8), dtype=torch.float32, device=device)
    ego[0, 0] = float(candidates["initial_speed"][scene_index])
    predicted_maps = []
    predicted_occs = []
    with torch.inference_mode():
        for model in models:
            output = model(
                torch.from_numpy(current_bev)[None].to(device),
                torch.from_numpy(paths[selected_index][None].astype(np.float32)).to(device),
                ego,
            )
            predicted_maps.append(
                F.interpolate(
                    output["map_logits"].argmax(1)[:, None].float(),
                    size=(200, 400),
                    mode="nearest",
                )[0, 0].cpu().numpy().astype(np.int64)
            )
            predicted_occs.append(
                output["occupancy_logits"][0].sigmoid().cpu().numpy()
            )

    figure = plt.figure(figsize=(24, 18), facecolor="#f4f5f7")
    grid = figure.add_gridspec(
        4,
        10,
        height_ratios=(1.15, 1.55, 1.45, 1.45),
        hspace=0.24,
        wspace=0.08,
        left=0.025,
        right=0.98,
        top=0.92,
        bottom=0.06,
    )
    camera_dir = args.frames / args.token / "current" / "images"
    for column, name in enumerate(("FRONT_LEFT", "FRONT", "FRONT_RIGHT")):
        axis = figure.add_subplot(grid[0, column * 3 : column * 3 + 3] if column < 2 else grid[0, 6:10])
        axis.imshow(Image.open(camera_dir / f"{name}.jpg"))
        axis.set_title(name.replace("_", " "), fontsize=12, pad=6)
        axis.axis("off")

    axis_current = figure.add_subplot(grid[1, 0:5])
    _render_bev(axis_current, current_map, current_occ, "Current QD BEV + candidate trajectories")
    for index, path in enumerate(paths):
        if index not in (base_index, selected_index, oracle_index):
            _draw_path(axis_current, path, "#7f8c8d", alpha=0.20, width=0.75)
    _draw_path(axis_current, paths[base_index], "#2e7d32", "RL base", 2.5)
    _draw_path(axis_current, paths[selected_index], "#e65100", "RM selected", 3.0)
    _draw_path(axis_current, paths[oracle_index], "#0277bd", "Oracle", 2.5)

    axis_text = figure.add_subplot(grid[1, 5:10])
    axis_text.axis("off")
    axis_text.text(
        0.02,
        0.98,
        f"Scene: {args.token}\n"
        f"RL base #{base_index}: RFS {rfs[base_index]:.3f}\n"
        f"RM selected #{selected_index}: RFS {rfs[selected_index]:.3f}\n"
        f"Oracle #{oracle_index}: RFS {rfs[oracle_index]:.3f}\n\n"
        "Orange occupancy = model probability\n"
        "Green/blue/orange paths are shown in the\n"
        "ego frame of each displayed BEV.",
        va="top",
        ha="left",
        fontsize=15,
        family="monospace",
        linespacing=1.45,
    )

    logged_poses = np.asarray([pair["relative_poses"][str(h)] for h in HORIZONS])
    selected_poses = paths[selected_index, list(POINT_INDEX)]
    for horizon_index, horizon in enumerate(HORIZONS):
        axis = figure.add_subplot(grid[2, horizon_index * 2 : horizon_index * 2 + 2])
        _render_bev(
            axis,
            predicted_maps[horizon_index],
            predicted_occs[horizon_index],
            f"{horizon:g}s WM prediction",
        )
        selected_pose = selected_poses[horizon_index]
        for index, color in (
            (base_index, "#2e7d32"),
            (selected_index, "#e65100"),
            (oracle_index, "#0277bd"),
        ):
            _draw_path(axis, _target_frame(paths[index], selected_pose), color, width=1.8)

        axis = figure.add_subplot(grid[3, horizon_index * 2 : horizon_index * 2 + 2])
        _render_bev(
            axis,
            actual_maps[horizon_index],
            actual_occs[horizon_index],
            f"{horizon:g}s future teacher BEV / GT",
        )
        for index, color in (
            (base_index, "#2e7d32"),
            (selected_index, "#e65100"),
            (oracle_index, "#0277bd"),
        ):
            _draw_path(axis, _target_frame(paths[index], logged_poses[horizon_index]), color, width=1.8)

    handles = [
        Line2D([0], [0], color="#2e7d32", lw=3, label="RL base"),
        Line2D([0], [0], color="#e65100", lw=3, label="Current RM selected"),
        Line2D([0], [0], color="#0277bd", lw=3, label="Oracle"),
        Patch(facecolor=(242 / 255, 77 / 255, 34 / 255, 0.42), label="Occupancy"),
    ]
    figure.legend(handles=handles, loc="upper center", ncol=4, fontsize=13, frameon=False, bbox_to_anchor=(0.5, 0.955))
    figure.suptitle(
        "Qwen-Drive Planner + Action-Conditioned WM: Candidate Evaluation",
        fontsize=20,
        fontweight="bold",
        y=0.985,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight", facecolor=figure.get_facecolor())
    metadata = {
        "token": args.token,
        "base_index": base_index,
        "selected_index": selected_index,
        "oracle_index": oracle_index,
        "rfs": {
            "base": float(rfs[base_index]),
            "selected": float(rfs[selected_index]),
            "oracle": float(rfs[oracle_index]),
        },
        "note": "Future GT is camera-based QD teacher BEV, not LiDAR ground truth.",
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
