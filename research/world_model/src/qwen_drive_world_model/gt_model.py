"""GT BEV world model with raster and future-object query heads."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import nms

from .model import BEVWorldModel, WorldModelConfig


@dataclass(frozen=True)
class ObjectHeadConfig:
    num_queries: int = 48
    num_classes: int = 4
    token_hw: tuple[int, int] = (8, 8)
    num_layers: int = 2
    num_heads: int = 8
    feedforward_size: int = 512
    x_range: tuple[float, float] = (-20.0, 60.0)
    y_range: tuple[float, float] = (-30.0, 30.0)
    z_range: tuple[float, float] = (-1.0, 5.5)
    max_dimensions: tuple[float, float, float] = (16.0, 5.0, 5.0)


class ObjectQueryHead(nn.Module):
    """DETR-style future object queries over the predicted GT BEV raster."""

    def __init__(self, input_channels: int, hidden_size: int, config: ObjectHeadConfig):
        super().__init__()
        self.config = config
        self.memory_projection = nn.Linear(
            input_channels, hidden_size
        )
        self.query_embedding = nn.Embedding(config.num_queries, hidden_size)
        self.reference_projection = nn.Linear(hidden_size, 2)
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.feedforward_size,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=config.num_layers)
        self.class_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, config.num_classes + 1),
        )
        self.box_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 8),
        )

    def forward(self, raster: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled = F.adaptive_avg_pool2d(raster, self.config.token_hw)
        memory = self.memory_projection(pooled.flatten(2).transpose(1, 2))
        query = self.query_embedding.weight.unsqueeze(0).expand(raster.shape[0], -1, -1)
        hidden = self.decoder(query, memory)
        logits = self.class_head(hidden)
        raw = self.box_head(hidden)

        xy = raw[..., :2].sigmoid()
        x = xy[..., 0] * (self.config.x_range[1] - self.config.x_range[0]) + self.config.x_range[0]
        y = xy[..., 1] * (self.config.y_range[1] - self.config.y_range[0]) + self.config.y_range[0]
        z = raw[..., 2].sigmoid() * (
            self.config.z_range[1] - self.config.z_range[0]
        ) + self.config.z_range[0]
        dimensions = raw[..., 3:6].sigmoid() * raster.new_tensor(
            self.config.max_dimensions
        )
        yaw_vector = F.normalize(raw[..., 6:8], dim=-1, eps=1e-5)
        yaw = torch.atan2(yaw_vector[..., 1], yaw_vector[..., 0])
        boxes = torch.stack(
            [
                x,
                y,
                z,
                dimensions[..., 0],
                dimensions[..., 1],
                dimensions[..., 2],
                yaw,
            ],
            dim=-1,
        )
        return {
            "boxes": boxes,
            "logits": logits,
            "raw_queries": torch.cat([raw, F.normalize(raw[..., 6:8], dim=-1)], dim=-1),
        }


class GTBEVWorldModel(nn.Module):
    """Predict future map, occupancy and object boxes from a current GT BEV."""

    def __init__(self, config: WorldModelConfig, object_config: ObjectHeadConfig):
        super().__init__()
        self.config = config
        self.object_config = object_config
        self.backbone = BEVWorldModel(config)
        self.object_head = ObjectQueryHead(
            config.output_channels or config.input_channels,
            config.hidden_size,
            object_config,
        )

    def forward(
        self,
        current_bev: torch.Tensor,
        trajectory: torch.Tensor,
        ego_status: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        raster = self.backbone(current_bev, trajectory, ego_status)
        objects = self.object_head(raster)
        return {
            "raster": raster,
            "map_logits": raster[:, :8],
            "occupancy_logits": raster[:, 8],
            "boxes": objects["boxes"],
            "box_logits": objects["logits"],
        }


def _object_cost(
    predicted_boxes: torch.Tensor,
    predicted_logits: torch.Tensor,
    target_boxes: torch.Tensor,
    target_labels: torch.Tensor,
) -> torch.Tensor:
    probability = predicted_logits.softmax(-1)
    class_cost = -probability[:, :, target_labels]
    center_cost = torch.cdist(predicted_boxes[:, :, :2], target_boxes[:, :, :2])
    dims_cost = (
        predicted_boxes[:, :, None, 3:6] - target_boxes[:, None, :, 3:6]
    ).abs().sum(-1)
    predicted_yaw = torch.stack(
        [predicted_boxes[..., 6].sin(), predicted_boxes[..., 6].cos()], -1
    )
    target_yaw = torch.stack(
        [target_boxes[..., 6].sin(), target_boxes[..., 6].cos()], -1
    )
    yaw_cost = torch.cdist(predicted_yaw, target_yaw)
    return 2.0 * class_cost + center_cost / 5.0 + dims_cost / 5.0 + yaw_cost


def object_query_loss(
    predicted_boxes: torch.Tensor,
    predicted_logits: torch.Tensor,
    target_boxes: torch.Tensor,
    target_labels: torch.Tensor,
    box_counts: torch.Tensor,
    *,
    class_weights: torch.Tensor | None = None,
    focal_gamma: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Hungarian set loss for variable-length future object sets."""

    batch, queries, _ = predicted_boxes.shape
    no_object = predicted_logits.shape[-1] - 1
    assignments = []
    for batch_index in range(batch):
        count = int(box_counts[batch_index].item())
        if count == 0:
            assignments.append(
                (
                    np.zeros(0, dtype=np.int64),
                    np.zeros(0, dtype=np.int64),
                )
            )
            continue
        cost = _object_cost(
            predicted_boxes[batch_index : batch_index + 1],
            predicted_logits[batch_index : batch_index + 1],
            target_boxes[batch_index, :count].unsqueeze(0),
            target_labels[batch_index, :count],
        )[0].detach().cpu().numpy()
        query_indices, target_indices = linear_sum_assignment(cost)
        assignments.append((query_indices, target_indices))

    class_targets = torch.full(
        (batch, queries),
        no_object,
        dtype=torch.long,
        device=predicted_logits.device,
    )
    for batch_index, (query_indices, target_indices) in enumerate(assignments):
        if len(query_indices):
            class_targets[batch_index, torch.as_tensor(query_indices, device=class_targets.device)] = (
                target_labels[batch_index, torch.as_tensor(target_indices, device=target_labels.device)]
            )
    if class_weights is None:
        class_weights = predicted_logits.new_tensor([1.0, 3.0, 3.0, 3.0, 0.25])
    log_probability = F.log_softmax(predicted_logits, dim=-1)
    probability = log_probability.exp()
    target_probability = probability.gather(-1, class_targets.unsqueeze(-1)).squeeze(-1)
    sample_loss = F.nll_loss(
        log_probability.flatten(0, 1),
        class_targets.flatten(),
        reduction="none",
    )
    if focal_gamma > 0:
        sample_loss = (1.0 - target_probability.reshape(-1)).pow(focal_gamma) * sample_loss
    sample_weights = class_weights[class_targets.reshape(-1)]
    classification = (sample_loss * sample_weights).sum() / sample_weights.sum().clamp_min(1e-6)

    matched_boxes = []
    matched_targets = []
    for batch_index, (query_indices, target_indices) in enumerate(assignments):
        if not len(query_indices):
            continue
        query_tensor = torch.as_tensor(query_indices, device=predicted_boxes.device)
        target_tensor = torch.as_tensor(target_indices, device=predicted_boxes.device)
        matched_boxes.append(predicted_boxes[batch_index, query_tensor])
        matched_targets.append(target_boxes[batch_index, target_tensor])
    if matched_boxes:
        predicted = torch.cat(matched_boxes, dim=0)
        target = torch.cat(matched_targets, dim=0)
        center_l1 = (predicted[:, :2] - target[:, :2]).abs().mean()
        size_l1 = (predicted[:, 3:6] - target[:, 3:6]).abs().mean()
        yaw_l1 = torch.atan2(
            torch.sin(predicted[:, 6] - target[:, 6]),
            torch.cos(predicted[:, 6] - target[:, 6]),
        ).abs().mean()
        regression = center_l1 + size_l1 + yaw_l1
    else:
        center_l1 = predicted_boxes.sum() * 0.0
        size_l1 = center_l1
        yaw_l1 = center_l1
        regression = center_l1
    total = classification + 2.0 * regression
    return total, {
        "object_classification": classification.detach(),
        "object_center_l1": center_l1.detach(),
        "object_size_l1": size_l1.detach(),
        "object_yaw_l1": yaw_l1.detach(),
        "object_regression": regression.detach(),
    }


def bev_gt_loss(
    output: dict[str, torch.Tensor],
    target_map: torch.Tensor,
    target_occupancy: torch.Tensor,
    map_class_weights: torch.Tensor,
    *,
    occupancy_weight: float = 0.5,
    occupancy_positive_weight: float = 10.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    map_cross_entropy = F.cross_entropy(
        output["map_logits"],
        target_map,
        weight=map_class_weights,
    )
    map_probability = output["map_logits"].softmax(1)
    target_one_hot = F.one_hot(
        target_map,
        num_classes=map_probability.shape[1],
    ).permute(0, 3, 1, 2).to(map_probability.dtype)
    intersection = (map_probability * target_one_hot).sum((0, 2, 3))
    denominator = map_probability.sum((0, 2, 3)) + target_one_hot.sum((0, 2, 3))
    dice = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
    present = target_one_hot.sum((0, 2, 3)) > 0
    map_dice = dice[present].mean() if present.any() else dice.sum() * 0.0

    occupancy_cross_entropy = F.binary_cross_entropy_with_logits(
        output["occupancy_logits"],
        target_occupancy,
        pos_weight=target_occupancy.new_tensor(occupancy_positive_weight),
    )
    occupancy_probability = output["occupancy_logits"].sigmoid()
    occ_intersection = (occupancy_probability * target_occupancy).sum()
    occ_denominator = occupancy_probability.sum() + target_occupancy.sum()
    occupancy_dice = 1.0 - (2.0 * occ_intersection + 1.0) / (occ_denominator + 1.0)

    map_loss = map_cross_entropy + 0.5 * map_dice
    occupancy_loss = occupancy_cross_entropy + 0.5 * occupancy_dice
    total = map_loss + occupancy_weight * occupancy_loss
    return total, {
        "map_loss": map_loss,
        "map_cross_entropy": map_cross_entropy.detach(),
        "map_dice": map_dice.detach(),
        "occupancy_loss": occupancy_loss,
        "occupancy_cross_entropy": occupancy_cross_entropy.detach(),
        "occupancy_dice": occupancy_dice.detach(),
    }


@torch.no_grad()
def segmentation_metrics(
    map_logits: torch.Tensor,
    target_map: torch.Tensor,
    occupancy_logits: torch.Tensor,
    target_occupancy: torch.Tensor,
    num_classes: int = 8,
) -> dict[str, float]:
    prediction = map_logits.argmax(1)
    confusion = torch.bincount(
        target_map.flatten() * num_classes + prediction.flatten(),
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)
    diagonal = torch.diag(confusion).float()
    union = confusion.sum(0) + confusion.sum(1) - diagonal
    iou = torch.where(union > 0, diagonal / union.clamp_min(1), torch.zeros_like(diagonal))
    present = confusion.sum(1) > 0
    occupancy_prediction = occupancy_logits > 0
    occupancy_target = target_occupancy > 0.5
    occ_union = (occupancy_prediction | occupancy_target).sum()
    occ_iou = (
        (occupancy_prediction & occupancy_target).sum().float() / occ_union.float()
        if occ_union
        else torch.tensor(1.0)
    )
    return {
        "map_miou": float(iou[present].mean()) if present.any() else 0.0,
        "map_pixel_accuracy": float(diagonal.sum() / confusion.sum().clamp_min(1)),
        "occupancy_iou": float(occ_iou),
    }


def class_aware_nms_indices(
    boxes: torch.Tensor,
    logits: torch.Tensor,
    *,
    score_threshold: float = 0.99,
    iou_threshold: float = 0.5,
) -> list[torch.Tensor]:
    """Apply class-specific NMS to decoded oriented boxes using AABB overlap."""

    scores, classes = logits[..., :-1].softmax(-1).max(-1)
    centers = boxes[..., :2]
    half_x = (
        boxes[..., 3].abs() * boxes[..., 6].cos().abs()
        + boxes[..., 4].abs() * boxes[..., 6].sin().abs()
    ) / 2.0
    half_y = (
        boxes[..., 3].abs() * boxes[..., 6].sin().abs()
        + boxes[..., 4].abs() * boxes[..., 6].cos().abs()
    ) / 2.0
    aabb = torch.stack(
        [
            centers[..., 0] - half_x,
            centers[..., 1] - half_y,
            centers[..., 0] + half_x,
            centers[..., 1] + half_y,
        ],
        dim=-1,
    )
    result = []
    for batch_index in range(boxes.shape[0]):
        selected = []
        for class_index in range(logits.shape[-1] - 1):
            indices = torch.nonzero(
                (classes[batch_index] == class_index)
                & (scores[batch_index] >= score_threshold),
                as_tuple=False,
            ).flatten()
            if len(indices):
                kept = nms(
                    aabb[batch_index, indices],
                    scores[batch_index, indices],
                    iou_threshold,
                )
                selected.extend(indices[kept].tolist())
        selected = sorted(
            selected,
            key=lambda index: float(scores[batch_index, index]),
            reverse=True,
        )
        result.append(torch.tensor(selected, dtype=torch.long, device=boxes.device))
    return result

