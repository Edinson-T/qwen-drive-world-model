"""Shared encoder/decoder world model for five future horizons."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .gt_model import ObjectHeadConfig, ObjectQueryHead
from .model import BEVWorldModel, WorldModelConfig


class SharedHorizonWorldModel(nn.Module):
    """Encode the current state once and decode all horizons with conditioning."""

    def __init__(
        self,
        config: WorldModelConfig,
        object_config: ObjectHeadConfig,
        num_horizons: int = 5,
    ) -> None:
        super().__init__()
        self.config = config
        self.object_config = object_config
        self.num_horizons = num_horizons
        self.output_channels = config.output_channels or config.input_channels
        self.backbone = BEVWorldModel(config)
        self.object_head = ObjectQueryHead(
            self.output_channels,
            config.hidden_size,
            object_config,
        )
        self.horizon_embedding = nn.Embedding(num_horizons, config.hidden_size)
        self.pose_projection = nn.Sequential(
            nn.Linear(3, config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        self.condition_mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size * 2),
        )
        nn.init.zeros_(self.horizon_embedding.weight)
        last = self.condition_mlp[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(
        self,
        current_bev: torch.Tensor,
        trajectory: torch.Tensor,
        ego_status: torch.Tensor | None,
        relative_poses: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if relative_poses.ndim != 3 or relative_poses.shape[1] != self.num_horizons:
            raise ValueError(
                f"relative_poses must be [B,{self.num_horizons},3], "
                f"got {tuple(relative_poses.shape)}"
            )
        visual_tokens = self.backbone.bridge(current_bev)
        action_token = self.backbone.trajectory_encoder(
            trajectory,
            ego_status,
        ).unsqueeze(1)
        scene = torch.cat([action_token, visual_tokens], dim=1)
        scene = scene + self.backbone.bridge.position
        shared_tokens = self.backbone.transition(scene)[:, 1:]
        batch = current_bev.shape[0]
        base = F.adaptive_avg_pool2d(
            current_bev[:, : self.output_channels].float(),
            self.config.target_hw,
        )

        rasters = []
        boxes = []
        box_logits = []
        token_h, token_w = self.config.token_hw
        for horizon in range(self.num_horizons):
            condition = self.horizon_embedding.weight[horizon].unsqueeze(0).expand(
                batch, -1
            )
            condition = condition + self.pose_projection(relative_poses[:, horizon])
            gamma, beta = self.condition_mlp(condition).chunk(2, dim=-1)
            conditioned = shared_tokens * (1.0 + gamma[:, None]) + beta[:, None]
            delta = conditioned.transpose(1, 2).reshape(
                batch,
                self.config.hidden_size,
                token_h,
                token_w,
            )
            delta = self.backbone.decoder(delta)
            delta = F.interpolate(
                delta,
                size=self.config.target_hw,
                mode="bilinear",
                align_corners=False,
            )
            raster = base + self.backbone.delta_scale * delta
            object_output = self.object_head(raster)
            rasters.append(raster)
            boxes.append(object_output["boxes"])
            box_logits.append(object_output["logits"])
        raster = torch.stack(rasters, dim=1)
        return {
            "raster": raster,
            "map_logits": raster[:, :, :8],
            "occupancy_logits": raster[:, :, 8],
            "boxes": torch.stack(boxes, dim=1),
            "box_logits": torch.stack(box_logits, dim=1),
        }

