"""Compact action-conditioned BEV world model.

The model predicts a future BEV latent in the future ego frame used by the
perception teacher. The complete candidate trajectory is supplied as the action
condition, including its relative future pose. This is the explicit
"future ego-frame prediction" variant from the feasibility protocol.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class WorldModelConfig:
    input_channels: int = 128
    base_channels: int | None = None
    output_channels: int | None = None
    hidden_size: int = 256
    token_hw: tuple[int, int] = (8, 8)
    target_hw: tuple[int, int] = (16, 32)
    num_layers: int = 2
    num_heads: int = 8
    feedforward_size: int = 512
    dropout: float = 0.0
    trajectory_points: int = 8
    ego_status_dim: int = 8
    progressive_decoder: bool = False


def resample_trajectory(trajectory: torch.Tensor, num_points: int) -> torch.Tensor:
    """Resample ``[B, T, C]`` trajectories to ``[B, num_points, C]``."""
    if trajectory.ndim != 3:
        raise ValueError(f"trajectory must be [B,T,C], got {tuple(trajectory.shape)}")
    if trajectory.shape[1] == num_points:
        return trajectory
    if trajectory.shape[1] == 0:
        raise ValueError("trajectory has no points")
    if trajectory.shape[1] == 1:
        return trajectory.expand(-1, num_points, -1)
    values = trajectory.transpose(1, 2)
    values = F.interpolate(
        values,
        size=num_points,
        mode="linear",
        align_corners=True,
    )
    return values.transpose(1, 2)


class TrajectoryEncoder(nn.Module):
    def __init__(self, config: WorldModelConfig) -> None:
        super().__init__()
        self.config = config
        self.trajectory_encoder = nn.Sequential(
            nn.Linear(config.trajectory_points * 3, config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        self.ego_encoder = nn.Sequential(
            nn.Linear(config.ego_status_dim, config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        self.fuse = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

    def forward(self, trajectory: torch.Tensor, ego_status: torch.Tensor | None) -> torch.Tensor:
        trajectory = resample_trajectory(trajectory, self.config.trajectory_points)
        action = self.trajectory_encoder(trajectory.flatten(1))
        if ego_status is None:
            ego_status = trajectory.new_zeros(trajectory.shape[0], self.config.ego_status_dim)
        else:
            ego_status = ego_status.to(dtype=trajectory.dtype)
            if ego_status.shape[-1] < self.config.ego_status_dim:
                ego_status = F.pad(ego_status, (0, self.config.ego_status_dim - ego_status.shape[-1]))
            elif ego_status.shape[-1] > self.config.ego_status_dim:
                ego_status = ego_status[..., : self.config.ego_status_dim]
        return self.fuse(torch.cat([action, self.ego_encoder(ego_status)], dim=-1))


class BEVBridge(nn.Module):
    """Pool a dense QD BEV map into a compact token grid."""

    def __init__(self, config: WorldModelConfig) -> None:
        super().__init__()
        self.config = config
        self.base_channels = config.base_channels or config.input_channels
        self.height_channels = config.input_channels - self.base_channels
        if self.height_channels < 0:
            raise ValueError("base_channels cannot exceed input_channels")
        self.pooled_projection = nn.Linear(self.base_channels, config.hidden_size)
        self.height_projection = (
            nn.Linear(self.height_channels, config.hidden_size)
            if self.height_channels
            else None
        )
        if self.height_projection is not None:
            nn.init.zeros_(self.height_projection.weight)
            nn.init.zeros_(self.height_projection.bias)
        self.bridge = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        token_count = config.token_hw[0] * config.token_hw[1]
        self.position = nn.Parameter(torch.zeros(1, token_count + 1, config.hidden_size))

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(latent.float(), self.config.token_hw)
        base = pooled[:, : self.base_channels].flatten(2).transpose(1, 2)
        tokens = self.pooled_projection(base)
        if self.height_projection is not None:
            height = pooled[:, self.base_channels :].flatten(2).transpose(1, 2)
            tokens = tokens + self.height_projection(height)
        return self.bridge(tokens)


class BEVWorldModel(nn.Module):
    """One deterministic action-conditioned latent transition."""

    def __init__(self, config: WorldModelConfig) -> None:
        super().__init__()
        self.config = config
        self.output_channels = config.output_channels or config.input_channels
        self.trajectory_encoder = TrajectoryEncoder(config)
        self.bridge = BEVBridge(config)
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.feedforward_size,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transition = nn.TransformerEncoder(layer, num_layers=config.num_layers)
        if config.progressive_decoder:
            self.decoder = nn.Sequential(
                nn.Conv2d(config.hidden_size, config.hidden_size, 3, padding=1),
                nn.GELU(),
                nn.Upsample(size=(16, 32), mode="bilinear", align_corners=False),
                nn.Conv2d(config.hidden_size, config.hidden_size // 2, 3, padding=1),
                nn.GELU(),
                nn.Upsample(size=config.target_hw, mode="bilinear", align_corners=False),
                nn.Conv2d(config.hidden_size // 2, config.hidden_size // 2, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(config.hidden_size // 2, self.output_channels, 1),
            )
        else:
            self.decoder = nn.Sequential(
                nn.Conv2d(config.hidden_size, config.hidden_size, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(config.hidden_size, self.output_channels, 1),
            )
        self.delta_scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        latent: torch.Tensor,
        trajectory: torch.Tensor,
        ego_status: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if latent.ndim != 4:
            raise ValueError(f"latent must be [B,C,H,W], got {tuple(latent.shape)}")
        if latent.shape[1] != self.config.input_channels:
            raise ValueError(
                f"expected {self.config.input_channels} latent channels, "
                f"got {latent.shape[1]}"
            )
        base = F.adaptive_avg_pool2d(
            latent[:, : self.output_channels].float(),
            self.config.target_hw,
        )
        visual_tokens = self.bridge(latent)
        action_token = self.trajectory_encoder(trajectory, ego_status).unsqueeze(1)
        scene = torch.cat([action_token, visual_tokens], dim=1) + self.bridge.position
        future = self.transition(scene)[:, 1:]
        token_h, token_w = self.config.token_hw
        delta = future.transpose(1, 2).reshape(
            latent.shape[0], self.config.hidden_size, token_h, token_w
        )
        delta = self.decoder(delta)
        if delta.shape[-2:] != self.config.target_hw:
            delta = F.interpolate(
                delta,
                size=self.config.target_hw,
                mode="bilinear",
                align_corners=False,
            )
        return base + self.delta_scale * delta

    @staticmethod
    def copy_current(latent: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
        return F.adaptive_avg_pool2d(latent.float(), target_hw)


def normalize_latent(latent: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Per-sample standardization used identically for prediction and target."""
    mean = latent.mean(dim=(1, 2, 3), keepdim=True)
    std = latent.std(dim=(1, 2, 3), keepdim=True, unbiased=False).clamp_min(eps)
    return (latent - mean) / std


def latent_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    cosine_weight: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pred_n = normalize_latent(prediction)
    target_n = normalize_latent(target)
    smooth_l1 = F.smooth_l1_loss(pred_n, target_n)
    cosine = F.cosine_similarity(
        pred_n.flatten(1),
        target_n.flatten(1),
        dim=1,
    ).mean()
    total = smooth_l1 + cosine_weight * (1.0 - cosine)
    return total, {
        "smooth_l1": smooth_l1.detach(),
        "cosine": cosine.detach(),
        "raw_l1": (prediction - target).abs().mean().detach(),
    }
