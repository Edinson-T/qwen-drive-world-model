"""A tiny depth head used only to validate the training pipeline."""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["DepthHead"]


class _Block(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(8, channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + identity)


class DepthHead(nn.Module):
    """Predicts metric depth at 1/4 input resolution.

    GroupNorm (not BatchNorm) so batch size 1/2 is stable. This is a stand-in
    for the real head that will read Qwen-Drive's ViT tap ``F^v``; the data,
    target and loss code is what carries over.
    """

    def __init__(self, width: int = 32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, width * 2),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(_Block(width * 2), _Block(width * 2), _Block(width * 2))
        self.head = nn.Sequential(
            nn.Conv2d(width * 2, width, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, 1, 1),
        )
        self.min_depth = 0.1

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """``[B,3,H,W]`` in ``[0,1]`` -> depth ``[B,1,H/4,W/4]``."""
        return torch.nn.functional.softplus(self.head(self.body(self.stem(images)))) + self.min_depth
