"""Latent-to-BEV-map decoder used only for world-model visualization."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.GroupNorm(min(32, out_channels), out_channels),
        nn.GELU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.GroupNorm(min(32, out_channels), out_channels),
        nn.GELU(),
    )


class LatentMapDecoder(nn.Module):
    """Decode a compact WM BEV latent into 200x400 semantic map logits."""

    def __init__(
        self,
        in_channels: int = 128,
        num_classes: int = 6,
        hidden: int = 128,
        output_hw: tuple[int, int] = (200, 400),
    ) -> None:
        super().__init__()
        self.output_hw = output_hw
        self.stem = _block(in_channels, hidden)
        self.mid = _block(hidden, hidden)
        self.head = nn.Sequential(
            nn.Conv2d(hidden, hidden // 2, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, hidden // 2), hidden // 2),
            nn.GELU(),
            nn.Conv2d(hidden // 2, num_classes, 1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        x = latent.float()
        x = self.stem(x)
        x = F.interpolate(x, size=(100, 200), mode="bilinear", align_corners=False)
        x = self.mid(x) + x
        x = F.interpolate(x, size=self.output_hw, mode="bilinear", align_corners=False)
        return self.head(x)
