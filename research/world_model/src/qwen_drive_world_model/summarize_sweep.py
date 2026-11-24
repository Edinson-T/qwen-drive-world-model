"""Summarize world-model horizon runs into one table and plot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="LABEL=metrics.json or LABEL=run-directory",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> tuple[dict, dict]:
    if path.is_dir():
        path = path / "metrics.json"
    metrics = json.loads(path.read_text())
    horizon = float(metrics["args"]["target_horizon"])
    best = min(metrics["history"], key=lambda row: row["val_loss"])
    return metrics, {"horizon": horizon, "best": best, "final": metrics["history"][-1]}


def main() -> None:
    args = parse_args()
    rows = []
    for item in args.run:
        label, separator, value = item.partition("=")
        if not separator:
            raise SystemExit(f"invalid --run {item!r}; expected LABEL=PATH")
        _, summary = _load(Path(value))
        summary["label"] = label
        summary["improvement_pct"] = (
            100.0
            * (summary["best"]["copy_smooth_l1"] - summary["best"]["val_smooth_l1"])
            / summary["best"]["copy_smooth_l1"]
        )
        rows.append(summary)
    rows.sort(key=lambda row: row["horizon"])

    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str((args.output / ".mplconfig").resolve()))
    (args.output / "summary.json").write_text(
        json.dumps({"runs": rows}, indent=2) + "\n"
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    horizons = [row["horizon"] for row in rows]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    axes[0].plot(
        horizons,
        [row["best"]["val_smooth_l1"] for row in rows],
        marker="o",
        linewidth=2,
        label="WM best validation",
    )
    axes[0].plot(
        horizons,
        [row["best"]["copy_smooth_l1"] for row in rows],
        marker="s",
        linestyle="--",
        linewidth=2,
        label="copy-current",
    )
    axes[0].set_ylabel("normalized SmoothL1")
    axes[0].set_title("Future latent error")

    axes[1].plot(
        horizons,
        [row["best"]["val_cosine"] for row in rows],
        marker="o",
        linewidth=2,
        label="WM best validation",
    )
    axes[1].plot(
        horizons,
        [row["best"]["copy_cosine"] for row in rows],
        marker="s",
        linestyle="--",
        linewidth=2,
        label="copy-current",
    )
    axes[1].set_ylabel("cosine similarity")
    axes[1].set_title("Future latent similarity")

    axes[2].plot(
        horizons,
        [row["best"]["action_pairwise_distance"] for row in rows],
        marker="o",
        linewidth=2,
    )
    axes[2].set_ylabel("mean left/straight/right distance")
    axes[2].set_title("Action sensitivity")

    for axis in axes:
        axis.set_xlabel("prediction horizon [s]")
        axis.set_xticks(horizons)
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    figure.savefig(args.output / "horizon_comparison.png", dpi=180, bbox_inches="tight")

    print("horizon  wm_smooth  copy_smooth  improvement  wm_cos  copy_cos  action_dist")
    for row in rows:
        best = row["best"]
        print(
            f"{row['horizon']:>7g}  {best['val_smooth_l1']:>9.5f}  "
            f"{best['copy_smooth_l1']:>11.5f}  {row['improvement_pct']:>10.2f}%  "
            f"{best['val_cosine']:>6.4f}  {best['copy_cosine']:>8.4f}  "
            f"{best['action_pairwise_distance']:>11.6f}"
        )


if __name__ == "__main__":
    main()
