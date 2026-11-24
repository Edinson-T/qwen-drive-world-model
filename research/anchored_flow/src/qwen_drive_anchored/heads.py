"""Anchor-conditioned heads for the base-residual planner."""

from __future__ import annotations

import torch
from torch import nn

__all__ = [
    "AnchorHeads",
    "SceneBaseResidualHead",
    "attach_anchor_heads",
    "attach_scene_base_residual_head",
    "bound_residual",
    "predict_scene_base_residual_conditioned",
]


def _one_hot(index: torch.Tensor, classes: int, dtype: torch.dtype) -> torch.Tensor:
    valid = (index >= 0) & (index < classes)
    encoded = torch.nn.functional.one_hot(
        index.clamp(0, classes - 1), classes
    ).to(dtype)
    return encoded * valid.to(dtype).unsqueeze(-1)


def bound_residual(
    raw_delta: torch.Tensor,
    scale: torch.Tensor,
    *,
    max_xy_residual_m: float,
    max_heading_residual_rad: float,
) -> torch.Tensor:
    """Bound XY displacement in metres and heading change in radians."""
    if raw_delta.shape[-1] != 3:
        raise ValueError(f"expected residuals with 3 channels, got {raw_delta.shape}")
    if max_xy_residual_m <= 0 or max_heading_residual_rad <= 0:
        raise ValueError("residual bounds must be positive")

    scale = scale.to(device=raw_delta.device, dtype=torch.float32)
    xy_metres = raw_delta[..., :2].float() * scale[:2]
    xy_magnitude = torch.linalg.vector_norm(xy_metres, dim=-1, keepdim=True)
    bounded_xy_metres = xy_metres * (
        max_xy_residual_m
        * torch.tanh(xy_magnitude / max_xy_residual_m)
        / xy_magnitude.clamp_min(1e-8)
    )
    heading_radians = raw_delta[..., 2:].float() * scale[2]
    bounded_heading_radians = max_heading_residual_rad * torch.tanh(
        heading_radians / max_heading_residual_rad
    )
    return torch.cat(
        [
            bounded_xy_metres / scale[:2],
            bounded_heading_radians / scale[2],
        ],
        dim=-1,
    )


class AnchorHeads(nn.Module):
    """Anchor identity conditioning and candidate confidence scoring."""

    def __init__(self, num_anchors: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_anchors, hidden_size)
        nn.init.normal_(self.embedding.weight, std=0.02)
        self.projection = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.normal_(self.projection.weight, std=0.01)
        self.score_norm = nn.LayerNorm(hidden_size)
        self.score_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def anchor_condition(
        self, anchor_ids: torch.Tensor, dtype: torch.dtype
    ) -> torch.Tensor:
        return self.projection(self.embedding(anchor_ids)).to(dtype)

    def score(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pooled = hidden_states.float().mean(dim=1)
        return self.score_mlp(self.score_norm(pooled)).squeeze(-1)


class _BoundedResidual(nn.Module):
    """Zero-initialized correction kept under the existing checkpoint path."""

    def __init__(
        self,
        hidden_size: int,
        *,
        max_xy_residual_m: float,
        max_heading_residual_rad: float,
    ) -> None:
        super().__init__()
        self.max_xy_residual_m = float(max_xy_residual_m)
        self.max_heading_residual_rad = float(max_heading_residual_rad)
        self.projection = nn.Linear(hidden_size, 3)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        base: torch.Tensor,
        mode_offset: torch.Tensor,
        scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw_delta = self.projection(hidden_states.float())
        delta = bound_residual(
            raw_delta,
            scale,
            max_xy_residual_m=self.max_xy_residual_m,
            max_heading_residual_rad=self.max_heading_residual_rad,
        )
        return base.float() + mode_offset.float() + delta, delta


class SceneBaseResidualHead(nn.Module):
    """Preserve a scene-conditioned base while anchors add candidate modes."""

    def __init__(
        self,
        hidden_size: int,
        *,
        max_xy_residual_m: float = 12.0,
        max_heading_residual_rad: float = 0.35,
        mode_offset_scale: float = 0.045,
        min_mode_offset_scale: float = 0.04,
        max_mode_offset_scale: float = 0.08,
    ) -> None:
        super().__init__()
        if not 0 <= min_mode_offset_scale <= mode_offset_scale:
            raise ValueError(
                "min_mode_offset_scale must be non-negative and no greater "
                "than mode_offset_scale"
            )
        if not mode_offset_scale < max_mode_offset_scale:
            raise ValueError(
                "mode_offset_scale must be below max_mode_offset_scale"
            )
        self.residual = _BoundedResidual(
            hidden_size,
            max_xy_residual_m=max_xy_residual_m,
            max_heading_residual_rad=max_heading_residual_rad,
        )
        self.min_mode_offset_scale = float(min_mode_offset_scale)
        self.max_mode_offset_scale = float(max_mode_offset_scale)
        fraction = (
            mode_offset_scale - self.min_mode_offset_scale
        ) / (self.max_mode_offset_scale - self.min_mode_offset_scale)
        self.raw_mode_offset_scale = nn.Parameter(
            torch.tensor(
                torch.logit(torch.tensor(fraction)).item(),
                dtype=torch.float32,
            )
        )

    @property
    def mode_offset_scale(self) -> torch.Tensor:
        span = self.max_mode_offset_scale - self.min_mode_offset_scale
        return self.min_mode_offset_scale + span * torch.sigmoid(
            self.raw_mode_offset_scale
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        base: torch.Tensor,
        centered_anchors: torch.Tensor,
        scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mode_offset = self.mode_offset_scale * centered_anchors
        return self.residual(hidden_states, base, mode_offset, scale)


def attach_anchor_heads(model, num_anchors: int) -> AnchorHeads:
    expert = model.planning_expert
    heads = AnchorHeads(num_anchors, expert.config.hidden_size).to(
        expert.out_proj.weight.device
    )
    model.anchor_heads = heads
    return heads


def attach_scene_base_residual_head(
    model,
    *,
    max_xy_residual_m: float = 12.0,
    max_heading_residual_rad: float = 0.35,
    mode_offset_scale: float = 0.045,
    min_mode_offset_scale: float = 0.04,
    max_mode_offset_scale: float = 0.08,
) -> SceneBaseResidualHead:
    expert = model.planning_expert
    head = SceneBaseResidualHead(
        expert.config.hidden_size,
        max_xy_residual_m=max_xy_residual_m,
        max_heading_residual_rad=max_heading_residual_rad,
        mode_offset_scale=mode_offset_scale,
        min_mode_offset_scale=min_mode_offset_scale,
        max_mode_offset_scale=max_mode_offset_scale,
    ).to(expert.out_proj.weight.device)
    model.scene_base_residual_head = head
    return head


def _predict_anchor_hidden(
    expert,
    heads: AnchorHeads,
    waypoints: torch.Tensor,
    flow_time: torch.Tensor,
    history_queries: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    scene_cache: list[tuple[torch.Tensor, torch.Tensor]],
    position_anchor: torch.Tensor,
    anchor_ids: torch.Tensor,
    nav_command: torch.Tensor,
    ego_status: torch.Tensor,
) -> torch.Tensor:
    dtype = expert.dtype
    waypoints = waypoints.to(dtype)
    batch, length, _ = waypoints.shape
    pose_query, velocity_query, acceleration_query = history_queries
    time_condition = expert.time_mlp(expert.time_embed(flow_time).to(dtype))
    waypoint_index = torch.arange(length, device=waypoints.device)
    hidden_states = expert.query_fusion(
        torch.cat(
            [
                expert.trajectory_proj(waypoints),
                expert.fourier_encoder(waypoints),
                time_condition.unsqueeze(1).expand(-1, length, -1),
                pose_query.unsqueeze(1).expand(-1, length, -1),
                expert.waypoint_embed(waypoint_index)
                .unsqueeze(0)
                .expand(batch, -1, -1)
                .to(dtype),
                velocity_query.unsqueeze(1).expand(-1, length, -1),
                acceleration_query.unsqueeze(1).expand(-1, length, -1),
            ],
            dim=-1,
        )
    )
    nav_onehot = _one_hot(nav_command, expert.config.nav_command_classes, dtype)
    condition = (
        time_condition
        + expert.nav_mlp(nav_onehot)
        + expert.ego_mlp(ego_status.to(dtype))
        + heads.anchor_condition(anchor_ids, dtype)
    )
    positions = expert._waypoint_positions(position_anchor, length)
    cos, sin = expert.rotary_emb(positions, dtype)
    layers_per_kv = expert.config.layers_per_kv
    for index, layer in enumerate(expert.layers):
        scene_key, scene_value = scene_cache[index // layers_per_kv]
        hidden_states = layer(
            hidden_states,
            scene_key.expand(batch, -1, -1, -1),
            scene_value.expand(batch, -1, -1, -1),
            cos,
            sin,
            condition,
        )
    return expert.final_layernorm(hidden_states)


def predict_scene_base_residual_conditioned(
    expert,
    heads: AnchorHeads,
    residual_head: SceneBaseResidualHead,
    waypoints: torch.Tensor,
    flow_time: torch.Tensor,
    history_queries: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    scene_cache: list[tuple[torch.Tensor, torch.Tensor]],
    position_anchor: torch.Tensor,
    base_prediction: torch.Tensor,
    centered_anchors: torch.Tensor,
    anchor_ids: torch.Tensor,
    nav_command: torch.Tensor,
    ego_status: torch.Tensor,
    scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Predict a shared scene base plus bounded anchor-specific offsets."""
    hidden_states = _predict_anchor_hidden(
        expert,
        heads,
        waypoints,
        flow_time,
        history_queries,
        scene_cache,
        position_anchor,
        anchor_ids,
        nav_command,
        ego_status,
    )
    predictions, deltas = residual_head(
        hidden_states,
        base_prediction,
        centered_anchors,
        scale,
    )
    return predictions, heads.score(hidden_states), deltas
