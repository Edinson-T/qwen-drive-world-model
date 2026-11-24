"""Draw the current anchor-conditioned BaseResidual planner architecture."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


COLORS = {
    "frozen": ("#dbeafe", "#2563eb"),
    "trainable": ("#ffedd5", "#ea580c"),
    "tensor": ("#dcfce7", "#16a34a"),
    "process": ("#f3f4f6", "#6b7280"),
    "output": ("#fef9c3", "#ca8a04"),
}


def box(
    axis,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    body: str = "",
    *,
    kind: str = "process",
    fontsize: float = 9,
) -> None:
    face, edge = COLORS[kind]
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.5,
        edgecolor=edge,
        facecolor=face,
    )
    axis.add_patch(patch)
    axis.text(
        x + width / 2,
        y + height * 0.66,
        title,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color="#111827",
    )
    if body:
        axis.text(
            x + width / 2,
            y + height * 0.28,
            body,
            ha="center",
            va="center",
            fontsize=fontsize - 1,
            color="#374151",
            linespacing=1.25,
        )


def arrow(axis, start, end, *, color: str = "#4b5563", width: float = 1.5) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=width,
            color=color,
            shrinkA=3,
            shrinkB=3,
        )
    )


def build_figure(output: Path) -> None:
    figure, axis = plt.subplots(figsize=(20, 11.8))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    figure.suptitle(
        "Qwen-Drive Anchor BaseResidual Planner (current checkpoint)",
        fontsize=17,
        fontweight="bold",
        y=0.985,
    )

    # Main pipeline: scene input and frozen VLM prefill.
    box(
        axis,
        0.015,
        0.79,
        0.13,
        0.11,
        "Driving inputs",
        "3 front views x 4 frames\nhistory + navigation + ego",
        kind="tensor",
    )
    box(
        axis,
        0.18,
        0.79,
        0.14,
        0.11,
        "Frozen VLM + reasoning",
        "one prefill per scene\noptional reasoning sentence",
        kind="frozen",
    )
    box(
        axis,
        0.355,
        0.79,
        0.14,
        0.11,
        "Scene KV cache",
        "reused by all candidates\n8 cache groups",
        kind="tensor",
    )
    arrow(axis, (0.145, 0.845), (0.18, 0.845))
    arrow(axis, (0.32, 0.845), (0.355, 0.845))

    # Anchor and candidate branch.
    box(
        axis,
        0.015,
        0.58,
        0.13,
        0.11,
        "SKM anchor bank",
        "K=20 anchors\nnormalized 50 x 3",
        kind="frozen",
    )
    box(
        axis,
        0.18,
        0.58,
        0.14,
        0.11,
        "Mode setup",
        "center: a_k - mean(a)\nsmooth noise: std=0.02",
        kind="process",
    )
    box(
        axis,
        0.355,
        0.58,
        0.14,
        0.11,
        "Candidate input",
        "anchor + noise\nhistory + nav + ego + anchor ID",
        kind="tensor",
    )
    box(
        axis,
        0.53,
        0.58,
        0.14,
        0.11,
        "Joint-attention Expert",
        "32 frozen layers + LoRA\nscene KV + waypoint tokens",
        kind="frozen",
    )
    arrow(axis, (0.145, 0.635), (0.18, 0.635))
    arrow(axis, (0.32, 0.635), (0.355, 0.635))
    arrow(axis, (0.495, 0.635), (0.53, 0.635))
    arrow(axis, (0.425, 0.79), (0.60, 0.69), color="#2563eb")
    arrow(axis, (0.08, 0.58), (0.08, 0.48), color="#4b5563")
    axis.text(0.086, 0.53, "mean anchor", fontsize=7.5, va="center", color="#374151")

    # Base branch.
    box(
        axis,
        0.015,
        0.38,
        0.13,
        0.10,
        "Scene base query",
        "anchor center (mean)\nt = 0.9",
        kind="tensor",
    )
    box(
        axis,
        0.18,
        0.38,
        0.14,
        0.10,
        "Frozen Planner + LoRA",
        "predict_endpoint at t=0.9",
        kind="frozen",
    )
    box(
        axis,
        0.355,
        0.38,
        0.14,
        0.10,
        "Shared base B",
        "50 x 3",
        kind="tensor",
    )
    box(
        axis,
        0.53,
        0.38,
        0.14,
        0.10,
        "Candidate hidden H_k",
        "20 x 50 x 1024",
        kind="tensor",
    )
    arrow(axis, (0.145, 0.43), (0.18, 0.43))
    arrow(axis, (0.32, 0.43), (0.355, 0.43))
    arrow(axis, (0.60, 0.58), (0.60, 0.48))
    arrow(axis, (0.425, 0.79), (0.25, 0.48), color="#2563eb")

    # Candidate heads.
    box(
        axis,
        0.36,
        0.205,
        0.15,
        0.09,
        "Score head",
        "pool -> LN -> 1024 -> 512 -> 1",
        kind="trainable",
    )
    box(
        axis,
        0.53,
        0.205,
        0.14,
        0.09,
        "Residual head",
        "Linear 1024 -> 3\nbounded tanh",
        kind="trainable",
    )
    arrow(axis, (0.565, 0.38), (0.44, 0.295))
    arrow(axis, (0.64, 0.38), (0.60, 0.295))

    # Fusion.
    box(
        axis,
        0.17,
        0.065,
        0.33,
        0.10,
        "Candidate fusion",
        "candidate_k = B + lambda * centered(a_k) + delta_k",
        kind="process",
        fontsize=10,
    )
    box(
        axis,
        0.015,
        0.065,
        0.13,
        0.10,
        "mode scale lambda",
        "trainable scalar\nrange [0.04, 0.08]",
        kind="trainable",
    )
    arrow(axis, (0.145, 0.115), (0.17, 0.115))
    arrow(axis, (0.425, 0.38), (0.36, 0.165))
    arrow(axis, (0.60, 0.205), (0.45, 0.165))

    # Output and future WM.
    box(
        axis,
        0.53,
        0.065,
        0.14,
        0.10,
        "20 candidate trajectories",
        "20 x 50 x 3\n+ 20 scores",
        kind="output",
    )
    box(
        axis,
        0.695,
        0.53,
        0.285,
        0.11,
        "Next: future BEV WM + Reward Model",
        "future BEV per candidate -> reward -> chosen top-1 path",
        kind="output",
    )
    box(
        axis,
        0.695,
        0.38,
        0.285,
        0.11,
        "Current selector (control)",
        "argmax(candidate score)\nnot the final paper method",
        kind="trainable",
    )
    arrow(axis, (0.50, 0.115), (0.53, 0.115))

    # Parameter and layer details.
    box(
        axis,
        0.695,
        0.66,
        0.285,
        0.25,
        "Inside each of 32 Planner layers",
        "hidden = 1024 | heads = 16 | KV heads = 4 | head_dim = 256\n"
        "RMSNorm + time/nav/ego/anchor AdaIN condition\n"
        "qkv_proj 1024 -> 10240 -> joint attention over [scene KV | waypoints]\n"
        "output gate -> o_proj 4096 -> 1024\n"
        "SwiGLU FFN: 1024 -> gate/up 7168 -> down 3584 -> 1024",
        kind="frozen",
        fontsize=9,
    )
    box(
        axis,
        0.695,
        0.075,
        0.285,
        0.275,
        "Trainable parameters in current adapter",
        "LoRA rank 16, alpha 32: 14,942,208\n"
        "  qkv 5,767,168 | o 2,621,440\n"
        "  gate_up 4,194,304 | down 2,359,296\n"
        "AnchorHeads: 1,596,417\n"
        "  embedding 20x1024 | projection 1024x1024\n"
        "  score head 1024 -> 512 -> 1\n"
        "Residual: Linear 1024 -> 3, 3,076 parameters\n"
        "  XY cap 12 m | heading cap 0.35 rad\n"
        "Total trainable: 16,541,701",
        kind="trainable",
        fontsize=8.5,
    )

    # Compact legend row.
    legend_items = [
        ("frozen", "frozen", 0.015),
        ("trainable", "trainable", 0.165),
        ("tensor", "tensor", 0.315),
        ("process", "operation", 0.445),
        ("output", "output", 0.575),
    ]
    for kind, label, x in legend_items:
        face, edge = COLORS[kind]
        axis.add_patch(
            FancyBboxPatch(
                (x, 0.935),
                0.035,
                0.025,
                boxstyle="round,pad=0.004,rounding_size=0.006",
                facecolor=face,
                edgecolor=edge,
                linewidth=1.0,
            )
        )
        axis.text(x + 0.042, 0.947, label, va="center", fontsize=8.5)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=240, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/visualizations/anchored_flow/architecture/"
            "planner_base_residual_architecture.png"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_figure(args.output)
    print(f"wrote {args.output.with_suffix('.png')}")
    print(f"wrote {args.output.with_suffix('.pdf')}")
    print(f"wrote {args.output.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
