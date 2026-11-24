"""Generate qualitative trajectory figures and quantitative metric charts."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

__all__ = ["main", "parse_args"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    qualitative = commands.add_parser("qualitative", help="candidate/selection figure")
    qualitative.add_argument("--analysis", required=True)
    qualitative.add_argument("--baseline", default=None)
    qualitative.add_argument("--scenes", required=True)
    qualitative.add_argument("--image-root", required=True)
    qualitative.add_argument("--output-dir", required=True)
    qualitative.add_argument("--token", default=None)
    qualitative.add_argument("--baseline-index", type=int, default=0)
    qualitative.add_argument("--highlight-index", type=int, default=None)
    qualitative.add_argument("--highlight-label", default="Highlighted candidate")
    qualitative.add_argument("--score-field", default="candidate_scores")
    qualitative.add_argument("--selection", choices=("auto", "none", "oracle"), default="auto")
    qualitative.add_argument("--selection-label", default="Selected candidate")
    qualitative.add_argument(
        "--layout",
        choices=("auto", "s1", "s1_detail", "full"),
        default="auto",
    )
    qualitative.add_argument("--max-speed", type=float, default=30.0)
    qualitative.add_argument("--max-oracle-ade", type=float, default=4.0)
    qualitative.add_argument("--dpi", type=int, default=220)

    metrics = commands.add_parser("metrics", help="bar charts from evaluation summaries")
    metrics.add_argument("--run", action="append", required=True, help="NAME=summary.json")
    metrics.add_argument("--output", required=True)
    metrics.add_argument("--dpi", type=int, default=220)
    return parser.parse_args(argv)


def _load_predictions(path: str | Path) -> dict[str, dict]:
    rows = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            rows[str(record["token"])] = record
    return rows


def _resample_xy(values, count: int = 50) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError(f"expected [T, >=2] trajectory, got {array.shape}")
    if len(array) == count:
        return array[:, :2]
    source = np.linspace(0.0, 1.0, len(array))
    target = np.linspace(0.0, 1.0, count)
    return np.stack(
        [np.interp(target, source, array[:, channel]) for channel in range(2)],
        axis=1,
    )


def _candidate_errors(record: dict) -> tuple[np.ndarray, np.ndarray | None]:
    trajectories = np.asarray(record["trajectories"], dtype=np.float64)
    if record.get("future_trajectory") is not None:
        ground_truth = _resample_xy(record["future_trajectory"], len(trajectories[0]))
        gt_errors = np.linalg.norm(
            trajectories[..., :2] - ground_truth[None, :, :],
            axis=-1,
        ).mean(axis=1)
    else:
        gt_errors = np.full(len(trajectories), np.nan)
    preferences = record.get("preference_trajectories") or []
    scores = record.get("preference_scores") or []
    if preferences and scores:
        best = _resample_xy(preferences[int(np.argmax(scores))], len(trajectories[0]))
        pref_errors = np.linalg.norm(
            trajectories[..., :2] - best[None, :, :],
            axis=-1,
        ).mean(axis=1)
    else:
        pref_errors = None
    return gt_errors, pref_errors


def _max_speed(trajectories: np.ndarray) -> float:
    velocity = np.diff(trajectories[..., :2], axis=1) / 0.1
    return float(np.linalg.norm(velocity, axis=-1).max())


def _selection_index(
    record: dict,
    *,
    selection: str,
    score_field: str,
) -> int | None:
    if selection == "none":
        return None
    gt_errors, _ = _candidate_errors(record)
    if selection == "oracle":
        return int(np.nanargmin(gt_errors))
    selected = record.get("selected_index")
    if selected is not None and 0 <= int(selected) < len(record["trajectories"]):
        return int(selected)
    scores = record.get(score_field)
    if scores is not None and len(scores) == len(record["trajectories"]):
        return int(np.argmax(scores))
    return None


def _scene_choice(
    analysis: dict[str, dict],
    baseline: dict[str, dict],
    *,
    selection: str,
    score_field: str,
    max_speed: float,
    max_oracle_ade: float,
) -> tuple[str, dict]:
    common = sorted(set(analysis) & set(baseline))
    candidates = []
    for token in common:
        record = analysis[token]
        base_record = baseline[token]
        gt_errors, _ = _candidate_errors(record)
        if np.all(np.isnan(gt_errors)) or not base_record.get("trajectories"):
            continue
        baseline_gt, _ = _candidate_errors(base_record)
        baseline_error = float(baseline_gt[0])
        oracle_error = float(np.nanmin(gt_errors))
        peak_speed = _max_speed(np.asarray(record["trajectories"]))
        if peak_speed > max_speed or oracle_error > max_oracle_ade:
            continue
        selected = _selection_index(
            record,
            selection=selection,
            score_field=score_field,
        )
        selected_error = None if selected is None else float(gt_errors[selected])
        selected_success = (
            selected_error is not None
            and selected_error < baseline_error
            and selected_error <= oracle_error + 0.35
        )
        gain = baseline_error - oracle_error
        candidates.append(
            (
                int(selected_success),
                gain,
                baseline_error,
                -peak_speed,
                token,
                {
                    "baseline_ade": baseline_error,
                    "oracle_ade": oracle_error,
                    "selected_index": selected,
                    "selected_ade": selected_error,
                    "max_speed": peak_speed,
                },
            )
        )
    if not candidates:
        raise SystemExit("no common scene satisfies the plotting constraints")
    candidates.sort(reverse=True)
    _, _, _, _, token, metrics = candidates[0]
    return token, metrics


def _find_scene_record(scenes_path: str | Path, token: str) -> dict:
    for line in Path(scenes_path).read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if str(record.get("meta_info", {}).get("token")) == token:
            return record
    raise ValueError(f"scene token not found: {token}")


def _camera_paths(
    image_root: str | Path,
    record: dict,
    token: str,
) -> list[tuple[str, Path]]:
    scene_token = str(record.get("scene_token") or token.rsplit("-", 1)[0])
    frame = token.rsplit("-", 1)[-1]
    paths = []
    for label, camera in (
        ("Front left", "FRONT_LEFT"),
        ("Front", "FRONT"),
        ("Front right", "FRONT_RIGHT"),
    ):
        path = Path(image_root) / "val" / scene_token / camera / f"{frame}.jpg"
        if path.exists():
            paths.append((label, path))
    return paths


def _draw_camera_axis(axis, title: str, path: Path | None) -> None:
    import matplotlib.pyplot as plt

    if path is None:
        axis.text(0.5, 0.5, f"{title} camera unavailable", ha="center", va="center")
    else:
        axis.imshow(plt.imread(path))
    axis.set_title(f"{title} camera · current frame")
    axis.axis("off")


def _draw_camera_montage(axis, paths: list[tuple[str, Path]]) -> None:
    import matplotlib.pyplot as plt

    axis.axis("off")
    if not paths:
        axis.text(0.5, 0.5, "camera views unavailable", ha="center", va="center")
        axis.set_title("Input cameras")
        return
    count = len(paths)
    for index, (label, path) in enumerate(paths):
        inset = axis.inset_axes([index / count, 0.0, 1.0 / count, 1.0])
        inset.imshow(plt.imread(path))
        inset.set_title(label, fontsize=9)
        inset.axis("off")
    axis.set_title("Input cameras · current frame")


def _plot_trajectory_axis(
    axis,
    trajectories: np.ndarray,
    ground_truth: np.ndarray,
    history: np.ndarray | None,
    *,
    title: str,
    alpha_candidates: float = 0.25,
    selected_index: int | None = None,
    top_indices: list[int] | None = None,
    selected_label: str = "Selected",
) -> None:
    for index, trajectory in enumerate(trajectories):
        axis.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            linewidth=0.8,
            alpha=alpha_candidates,
            color="tab:gray",
        )
    if top_indices:
        for rank, index in enumerate(top_indices):
            axis.plot(
                trajectories[index, :, 0],
                trajectories[index, :, 1],
                linewidth=1.4,
                alpha=0.85,
                label=f"rank {rank + 1}",
            )
    if selected_index is not None:
        axis.plot(
            trajectories[selected_index, :, 0],
            trajectories[selected_index, :, 1],
            color="tab:orange",
            linewidth=2.8,
            label=selected_label,
        )
    axis.plot(
        ground_truth[:, 0],
        ground_truth[:, 1],
        color="tab:green",
        linestyle="--",
        linewidth=2.4,
        label="GT",
    )
    if history is not None and len(history):
        axis.plot(
            history[:, 0],
            history[:, 1],
            color="black",
            linewidth=1.5,
            alpha=0.65,
            label="history",
        )
    axis.scatter([0.0], [0.0], marker=">", s=70, color="black", zorder=6)
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(alpha=0.2)
    axis.set_xlabel("x forward [m]")
    axis.set_ylabel("y lateral [m]")
    axis.set_title(title)


def _plot_endpoints(
    axis,
    trajectories: np.ndarray,
    ground_truth: np.ndarray,
) -> None:
    endpoints = trajectories[:, -1, :2]
    colors = np.arange(len(endpoints), dtype=np.float64)
    points = axis.scatter(
        endpoints[:, 0],
        endpoints[:, 1],
        c=colors,
        cmap="viridis",
        s=52,
        edgecolor="white",
        linewidth=0.6,
        label="candidate endpoints",
    )
    for index, endpoint in enumerate(endpoints):
        axis.annotate(
            str(index),
            endpoint,
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=6,
        )
    axis.scatter(
        ground_truth[-1, 0],
        ground_truth[-1, 1],
        marker="*",
        s=180,
        color="tab:green",
        edgecolor="black",
        linewidth=0.5,
        label="GT endpoint",
        zorder=5,
    )
    axis.scatter([0.0], [0.0], marker=">", s=70, color="black", label="ego")
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(alpha=0.25)
    axis.set_xlabel("x endpoint [m]")
    axis.set_ylabel("y endpoint [m]")
    axis.set_title(f"{len(endpoints)} candidate endpoints · no merging")
    axis.legend(handles=[points], loc="best", fontsize=7)


def _qualitative(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str((output / ".mplconfig").resolve()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    analysis = _load_predictions(args.analysis)
    baseline = _load_predictions(args.baseline) if args.baseline else {}
    if args.token:
        token = args.token
        if token not in analysis:
            raise SystemExit(f"token missing from analysis file: {token}")
        chosen_metrics = {}
    elif baseline:
        token, chosen_metrics = _scene_choice(
            analysis,
            baseline,
            selection=args.selection,
            score_field=args.score_field,
            max_speed=args.max_speed,
            max_oracle_ade=args.max_oracle_ade,
        )
    else:
        token = next(iter(analysis))
        chosen_metrics = {}

    analysis_record = analysis[token]
    scene_record = _find_scene_record(args.scenes, token)
    trajectories = np.asarray(analysis_record["trajectories"], dtype=np.float64)
    ground_truth = _resample_xy(
        analysis_record["future_trajectory"],
        len(trajectories[0]),
    )
    history = np.asarray(
        scene_record.get("trajectory", {}).get("hist_traj", []),
        dtype=np.float64,
    )
    selected = _selection_index(
        analysis_record,
        selection=args.selection,
        score_field=args.score_field,
    )
    layout = args.layout
    if layout == "auto":
        layout = "full" if selected is not None else "s1"
    if layout == "full" and selected is None:
        raise SystemExit("full layout requires candidate scores or selected_index")

    figure_path = output / f"qualitative_{token}.png"
    figure_pdf = output / f"qualitative_{token}.pdf"
    detail_axis = None
    camera_axes = None
    if layout == "s1_detail":
        figure = plt.figure(figsize=(18, 10))
        grid = figure.add_gridspec(2, 3, height_ratios=(1.0, 1.08))
        camera_axes = [figure.add_subplot(grid[0, index]) for index in range(3)]
        baseline_axis = figure.add_subplot(grid[1, 0])
        candidate_axis = figure.add_subplot(grid[1, 1])
        detail_axis = figure.add_subplot(grid[1, 2])
        selection_axis = None
    elif layout == "s1":
        figure, axes = plt.subplots(1, 3, figsize=(16, 5.2))
        camera_axis, baseline_axis, candidate_axis = axes
        selection_axis = None
    else:
        figure, axes = plt.subplots(2, 2, figsize=(12, 10))
        camera_axis, baseline_axis = axes[0]
        candidate_axis = axes[1, 0]
        selection_axis = axes[1, 1]

    camera_paths = _camera_paths(args.image_root, analysis_record, token)
    if camera_axes is not None:
        by_label = dict(camera_paths)
        for axis, title in zip(
            camera_axes,
            ("Front left", "Front", "Front right"),
            strict=True,
        ):
            _draw_camera_axis(axis, title, by_label.get(title))
    else:
        _draw_camera_montage(camera_axis, camera_paths)

    if baseline:
        base_record = baseline[token]
        base_trajectories = np.asarray(base_record["trajectories"], dtype=np.float64)
        if not 0 <= args.baseline_index < len(base_trajectories):
            raise SystemExit("--baseline-index is out of range")
        baseline_xy = _resample_xy(
            base_trajectories[args.baseline_index],
            len(ground_truth),
        )
        baseline_error = float(
            np.linalg.norm(baseline_xy - ground_truth, axis=1).mean()
        )
        baseline_axis.plot(
            ground_truth[:, 0],
            ground_truth[:, 1],
            color="tab:green",
            linestyle="--",
            linewidth=2.4,
            label="GT",
        )
        baseline_axis.plot(
            baseline_xy[:, 0],
            baseline_xy[:, 1],
            color="tab:blue",
            linewidth=2.4,
            label="Qwen-Drive",
        )
        baseline_axis.scatter([0.0], [0.0], marker=">", s=70, color="black")
        baseline_axis.set_aspect("equal", adjustable="datalim")
        baseline_axis.grid(alpha=0.2)
        baseline_axis.set_xlabel("x forward [m]")
        baseline_axis.set_ylabel("y lateral [m]")
        baseline_axis.set_title(f"Qwen-Drive baseline · ADE {baseline_error:.3f} m")
        baseline_axis.legend(loc="best")
    else:
        _plot_trajectory_axis(
            baseline_axis,
            np.empty((0, len(ground_truth), 2)),
            ground_truth,
            history,
            title="Ground truth",
        )

    gt_errors, pref_errors = _candidate_errors(analysis_record)
    oracle_index = int(np.nanargmin(gt_errors))
    _plot_trajectory_axis(
        candidate_axis,
        trajectories,
        ground_truth,
        history,
        title=(
            f"{len(trajectories)} candidates · "
            f"Oracle ADE {gt_errors[oracle_index]:.3f} m"
        ),
        selected_index=args.highlight_index,
        selected_label=args.highlight_label,
    )
    candidate_axis.plot(
        trajectories[oracle_index, :, 0],
        trajectories[oracle_index, :, 1],
        color="tab:purple",
        linestyle=":",
        linewidth=2.0,
        label="GT-nearest (oracle)",
    )
    candidate_axis.legend(loc="best")

    if detail_axis is not None:
        _plot_endpoints(detail_axis, trajectories, ground_truth)

    if selection_axis is not None:
        scores = analysis_record.get(args.score_field)
        top_indices = None
        if scores is not None and len(scores) == len(trajectories):
            top_indices = np.argsort(-np.asarray(scores))[:5].tolist()
        _plot_trajectory_axis(
            selection_axis,
            trajectories,
            ground_truth,
            history,
            title=f"{args.selection_label} · ADE {gt_errors[selected]:.3f} m",
            selected_index=selected,
            top_indices=top_indices,
            selected_label=f"{args.selection_label} #{selected}",
        )
        selection_axis.legend(loc="best")
        if scores is not None and len(scores) == len(trajectories):
            order = np.argsort(-np.asarray(scores))[:5]
            lines = [
                f"#{int(index)}  score {float(scores[index]):.3f}  ADE {gt_errors[index]:.3f}"
                for index in order
            ]
            selection_axis.text(
                0.02,
                0.02,
                "\n".join(lines),
                transform=selection_axis.transAxes,
                va="bottom",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
            )

    intent = scene_record.get("meta_info", {}).get("intent")
    if intent is None:
        intent = scene_record.get("trajectory", {}).get("driving_command", "")
    figure.suptitle(f"{token}  |  intent={intent}", fontsize=12)
    figure.tight_layout()
    figure.savefig(figure_path, dpi=args.dpi, bbox_inches="tight")
    figure.savefig(figure_pdf, bbox_inches="tight")
    plt.close(figure)

    manifest = {
        "token": token,
        "layout": layout,
        "selected_index": selected,
        "oracle_index": oracle_index,
        "gt_errors": gt_errors.tolist(),
        "preference_errors": None if pref_errors is None else pref_errors.tolist(),
        "scene_metrics": chosen_metrics,
        "analysis": str(Path(args.analysis).resolve()),
        "baseline": None if not args.baseline else str(Path(args.baseline).resolve()),
    }
    (output / f"qualitative_{token}.json").write_text(json.dumps(manifest, indent=2))
    print(f"selected token: {token}")
    print(chosen_metrics)
    print(f"wrote {figure_path}")


def _parse_runs(items: list[str]) -> list[tuple[str, dict]]:
    runs = []
    single_input = len(items) == 1
    for item in items:
        name, separator, path = item.partition("=")
        if not separator or not name or not path:
            raise ValueError(f"invalid --run {item!r}; expected NAME=PATH")
        summary = json.loads(Path(path).read_text())
        for run_name, values in summary.get("runs", {}).items():
            runs.append(
                (
                    run_name
                    if single_input
                    else f"{name}/{run_name}" if len(summary["runs"]) > 1 else name,
                    values,
                )
            )
    return runs


def _metrics(args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str((output.parent / ".mplconfig").resolve()),
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = _parse_runs(args.run)
    names = [name for name, _ in runs]
    panels = (
        ("RFS", "RFS ↑"),
        ("ADE", "ADE [m] ↓"),
        ("DDv2_Div", "DDv2 Div [%] ↑"),
        ("RFS gap", "RFS gap ↓"),
    )
    figure, axes = plt.subplots(1, 4, figsize=(18, 4.8))
    for axis, (key, title) in zip(axes, panels):
        values = [float(run.get(key, math.nan)) for _, run in runs]
        axis.bar(names, values, color="tab:blue")
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.25)
        finite = [value for value in values if math.isfinite(value)]
        upper = max(finite, default=1.0)
        axis.set_ylim(0, upper * 1.2 if upper > 0 else 1.0)
        for x, value in enumerate(values):
            if math.isfinite(value):
                suffix = "%" if key == "DDv2_Div" else ""
                axis.text(x, value, f"{value:.1f}{suffix}", ha="center", va="bottom", fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=args.dpi, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {output}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "qualitative":
        _qualitative(args)
    else:
        _metrics(args)


if __name__ == "__main__":
    main()
