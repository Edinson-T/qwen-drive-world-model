"""Plot optimizer-step training loss components from train_metrics.jsonl."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

__all__ = ["main", "parse_args"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--smooth", type=float, default=0.9)
    return parser.parse_args(argv)


def _ema(values: np.ndarray, weight: float) -> np.ndarray:
    if len(values) == 0:
        return values
    out = np.empty_like(values, dtype=np.float64)
    out[0] = values[0]
    for index in range(1, len(values)):
        out[index] = weight * out[index - 1] + (1 - weight) * values[index]
    return out


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    metrics_path = Path(args.metrics)
    output = Path(args.output_dir) if args.output_dir else metrics_path.parent / "plots"
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str((output / ".mplconfig").resolve()))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [
        json.loads(line)
        for line in metrics_path.read_text().splitlines()
        if line.strip()
    ]
    if not rows:
        raise SystemExit("metrics file is empty")
    steps = np.asarray([row["step"] for row in rows], dtype=np.float64)

    def series(key: str) -> np.ndarray:
        return np.asarray([row.get(key, np.nan) for row in rows], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for key, label in (
        ("loss", "total"),
        ("trajectory_loss", "positive trajectory"),
        ("classification_loss", "classification + ranking"),
        ("ranking_loss", "candidate ranking"),
        ("mode_offset_scale", "mode offset scale"),
    ):
        values = series(key)
        if np.all(np.isnan(values)):
            continue
        axes[0].plot(steps, _ema(values, args.smooth), label=label)
    axes[0].set_xlabel("optimizer step")
    axes[0].set_ylabel("value")
    axes[0].set_yscale("symlog", linthresh=1e-4)
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[0].set_title("Training loss components (EMA)")

    axes[1].plot(steps, series("learning_rate"))
    axes[1].set_xlabel("optimizer step")
    axes[1].set_ylabel("learning rate")
    axes[1].grid(alpha=0.25)
    axes[1].set_title("Learning rate")
    fig.tight_layout()
    fig.savefig(output / "training_losses.png", dpi=180)
    plt.close(fig)
    print(f"wrote {output / 'training_losses.png'}")


if __name__ == "__main__":
    main()

