"""Visual diagnostics for the BEV world-model feasibility pilot."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _pca_rgb(latent: np.ndarray, basis: np.ndarray | None = None):
    """Map a channel-first BEV latent to three orthogonal PCA components."""
    values = np.asarray(latent, dtype=np.float32)
    channels, height, width = values.shape
    pixels = values.reshape(channels, -1).T
    mean = pixels.mean(axis=0, keepdims=True)
    centered = pixels - mean
    if basis is None:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        basis = vt[:3].T
    rgb = centered @ basis
    rgb = rgb.reshape(height, width, -1)
    if rgb.shape[-1] < 3:
        rgb = np.pad(rgb, ((0, 0), (0, 0), (0, 3 - rgb.shape[-1])))
    result = np.zeros_like(rgb, dtype=np.float32)
    for channel in range(3):
        plane = rgb[..., channel]
        low, high = np.percentile(plane, (1.0, 99.0))
        result[..., channel] = np.clip(
            (plane - low) / max(high - low, 1e-6),
            0.0,
            1.0,
        )
    return result, mean, basis


def _shared_pca(*latents: np.ndarray):
    channels, height, width = np.asarray(latents[0]).shape
    pixels = np.concatenate(
        [np.asarray(value, dtype=np.float32).reshape(channels, -1).T for value in latents],
        axis=0,
    )
    mean = pixels.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(pixels - mean, full_matrices=False)
    return mean, vt[:3].T, (height, width)


def _render_shared(ax, latent, mean, basis, shape, title):
    channels = latent.shape[0]
    pixels = np.asarray(latent, dtype=np.float32).reshape(channels, -1).T - mean
    rgb = (pixels @ basis).reshape(*shape, -1)
    if rgb.shape[-1] < 3:
        rgb = np.pad(rgb, ((0, 0), (0, 0), (0, 3 - rgb.shape[-1])))
    rgb = np.stack(
        [
            np.clip(
                (rgb[..., channel] - np.percentile(rgb[..., channel], 1))
                / max(
                    np.percentile(rgb[..., channel], 99)
                    - np.percentile(rgb[..., channel], 1),
                    1e-6,
                ),
                0,
                1,
            )
            for channel in range(3)
        ],
        axis=-1,
    )
    ax.imshow(rgb, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def plot_quad(
    current: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    output: Path,
    *,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    current = np.asarray(current, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    prediction = np.asarray(prediction, dtype=np.float32)
    mean, basis, shape = _shared_pca(current, target, prediction)
    error = np.linalg.norm(prediction - target, axis=0)
    figure, axes = plt.subplots(1, 4, figsize=(14, 4.0), constrained_layout=True)
    _render_shared(axes[0], current, mean, basis, shape, "Current BEV latent")
    _render_shared(axes[1], target, mean, basis, shape, "GT future BEV latent")
    _render_shared(axes[2], prediction, mean, basis, shape, "WM future BEV latent")
    image = axes[3].imshow(error, cmap="magma", interpolation="nearest")
    axes[3].set_title("Prediction error", fontsize=9)
    axes[3].axis("off")
    figure.colorbar(image, ax=axes[3], fraction=0.046, pad=0.02)
    figure.suptitle(title, fontsize=11)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_copy_quad(
    current: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    output: Path,
    *,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    current = np.asarray(current, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    prediction = np.asarray(prediction, dtype=np.float32)
    mean, basis, shape = _shared_pca(current, target, prediction)
    figure, axes = plt.subplots(1, 4, figsize=(14, 4.0), constrained_layout=True)
    _render_shared(axes[0], current, mean, basis, shape, "Current")
    _render_shared(axes[1], current, mean, basis, shape, "Copy-current")
    _render_shared(axes[2], prediction, mean, basis, shape, "WM prediction")
    _render_shared(axes[3], target, mean, basis, shape, "GT future")
    figure.suptitle(title, fontsize=11)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_action_sensitivity(
    current: np.ndarray,
    predictions: dict[str, np.ndarray],
    output: Path,
    *,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(predictions)
    values = [np.asarray(predictions[name], dtype=np.float32) for name in names]
    mean, basis, shape = _shared_pca(*values)
    base = values[0]
    figure, axes = plt.subplots(2, len(names), figsize=(4.2 * len(names), 7.0))
    if len(names) == 1:
        axes = np.asarray(axes).reshape(2, 1)
    for index, (name, value) in enumerate(zip(names, values, strict=True)):
        _render_shared(axes[0, index], value, mean, basis, shape, f"{name}: predicted future")
        difference = np.linalg.norm(value - base, axis=0)
        image = axes[1, index].imshow(difference, cmap="magma", interpolation="nearest")
        axes[1, index].set_title(f"|{name} - {names[0]}|")
        axes[1, index].axis("off")
        figure.colorbar(image, ax=axes[1, index], fraction=0.046, pad=0.02)
    figure.suptitle(title, fontsize=11)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_training_curve(history: list[dict[str, float]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not history:
        return
    steps = [row["step"] for row in history]
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    for axis, keys in zip(
        axes,
        (
            ("train_smooth_l1", "val_smooth_l1", "copy_smooth_l1"),
            ("train_cosine", "val_cosine", "copy_cosine"),
            ("action_pairwise_distance",),
        ),
        strict=True,
    ):
        for key in keys:
            values = [row.get(key, np.nan) for row in history]
            if np.isfinite(values).any():
                axis.plot(steps, values, linewidth=1.8, label=key)
        axis.set_xlabel("step")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    axes[0].set_title("Latent loss")
    axes[1].set_title("Cosine similarity")
    axes[2].set_title("Action sensitivity")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
