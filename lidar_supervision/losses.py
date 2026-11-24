"""Masked depth losses and metrics."""

from __future__ import annotations

import torch

__all__ = ["masked_log_l1", "abs_rel", "delta1", "normalize_depth"]


def normalize_depth(depth: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6):
    """Scale each sample by its own mean valid depth (scale-invariant loss)."""
    denom = (depth * mask).sum(dim=(1, 2, 3), keepdim=True) / mask.sum(dim=(1, 2, 3), keepdim=True).clamp_min(eps)
    return depth / denom.clamp_min(eps)


def masked_log_l1(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean absolute log-depth error over valid pixels, scale-normalised per sample."""
    pred_n = normalize_depth(pred, mask)
    gt_n = normalize_depth(gt, mask)
    diff = (torch.log(pred_n.clamp_min(1e-3)) - torch.log(gt_n.clamp_min(1e-3))).abs()
    return (diff * mask).sum() / mask.sum().clamp_min(1.0)


@torch.no_grad()
def abs_rel(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> float:
    """Absolute relative error, after per-sample scale normalisation."""
    pred_n = normalize_depth(pred, mask)
    gt_n = normalize_depth(gt, mask)
    err = ((pred_n - gt_n).abs() / gt_n.clamp_min(1e-6)) * mask
    return float(err.sum() / mask.sum().clamp_min(1.0))


@torch.no_grad()
def delta1(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, threshold: float = 1.25) -> float:
    """Fraction of valid pixels within ``threshold`` of the ground truth."""
    pred_n = normalize_depth(pred, mask)
    gt_n = normalize_depth(gt, mask)
    ratio = torch.maximum(pred_n / gt_n.clamp_min(1e-6), gt_n / pred_n.clamp_min(1e-6))
    hit = ((ratio < threshold).float() * mask)
    return float(hit.sum() / mask.sum().clamp_min(1.0))
