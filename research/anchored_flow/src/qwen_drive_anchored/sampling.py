"""Candidate sampling for the scene-base residual planner."""

from __future__ import annotations

import torch

from .anchors import AnchorBank, select_anchor_ids
from .data import speed_bin_indices
from .heads import predict_scene_base_residual_conditioned

__all__ = ["low_frequency_noise", "sample_anchor2_control", "sample_base_residual"]


def low_frequency_noise(
    shape: tuple[int, int],
    *,
    std: float,
    modes: int,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate smooth cosine-subspace noise over the trajectory horizon."""
    length = shape[0]
    if modes <= 0:
        return std * torch.randn(
            shape, generator=generator, device=device, dtype=dtype
        )
    modes = min(modes, length)
    time = (torch.arange(length, device=device, dtype=dtype) + 0.5) / length
    frequency = torch.arange(modes, device=device, dtype=dtype)
    basis = torch.cos(torch.pi * time[:, None] * frequency[None])
    basis[:, 0] = 1.0
    basis = basis / basis.norm(dim=0, keepdim=True)
    coefficients = torch.randn(
        (modes, shape[1]), generator=generator, device=device, dtype=dtype
    )
    return std * basis @ coefficients


def _tile(tensor: torch.Tensor, count: int) -> torch.Tensor:
    return tensor.repeat_interleave(count, dim=0)


@torch.no_grad()
def sample_anchor2_control(
    expert,
    *,
    anchor_bank: AnchorBank,
    scene_cache: list[tuple[torch.Tensor, torch.Tensor]],
    position_anchor: torch.Tensor,
    history: torch.Tensor,
    history_velocity: torch.Tensor,
    history_acceleration: torch.Tensor,
    nav_command: torch.Tensor,
    ego_status: torch.Tensor,
    num_samples: int,
    noise_std: float,
    seed: int,
    initial_speed: float | None = None,
    flow_time: float = 0.8,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two-step anchor-initialized endpoint control used only for evaluation."""
    device = device or history.device
    if num_samples < anchor_bank.num_anchors and initial_speed is not None:
        anchor_ids = select_anchor_ids(
            anchor_bank,
            num_samples,
            nav_command=int(nav_command[0]),
            speed_bin=int(speed_bin_indices(np.asarray([initial_speed]))[0]),
        )
    else:
        anchor_ids = anchor_bank.candidate_ids(num_samples)
    anchors = torch.from_numpy(anchor_bank.normalized[anchor_ids]).to(device)
    noise = []
    for offset in range(num_samples):
        generator = torch.Generator(device=device).manual_seed(seed + offset)
        noise.append(
            low_frequency_noise(
                anchors.shape[1:],
                std=1.0,
                modes=getattr(expert, "noise_modes", 6),
                generator=generator,
                device=device,
            )
        )
    waypoints = anchors.float() + noise_std * torch.stack(noise)
    history_queries = expert.encode_history(
        _tile(history, num_samples),
        _tile(nav_command, num_samples),
        _tile(history_velocity, num_samples),
        _tile(history_acceleration, num_samples),
    )
    for current_time, next_time in ((flow_time, 0.9), (0.9, 1.0)):
        endpoint = expert.predict_endpoint(
            waypoints,
            torch.full(
                (num_samples,),
                current_time,
                device=device,
                dtype=torch.float32,
            ),
            history_queries,
            scene_cache,
            position_anchor.expand(-1, num_samples),
            _tile(nav_command, num_samples),
            _tile(ego_status, num_samples),
        )
        dt = next_time - current_time
        remaining = max(1.0 - current_time, 0.1)
        waypoints = waypoints + (endpoint.float() - waypoints) / remaining * dt
    return waypoints, torch.from_numpy(anchor_ids)


@torch.no_grad()
def sample_base_residual(
    expert,
    heads,
    residual_head,
    *,
    anchor_bank: AnchorBank,
    scale: torch.Tensor,
    scene_cache: list[tuple[torch.Tensor, torch.Tensor]],
    position_anchor: torch.Tensor,
    history: torch.Tensor,
    history_velocity: torch.Tensor,
    history_acceleration: torch.Tensor,
    nav_command: torch.Tensor,
    ego_status: torch.Tensor,
    num_samples: int,
    noise_std: float,
    seed: int,
    initial_speed: float | None = None,
    flow_time: float = 0.9,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate scene-base candidates with bounded anchor-specific offsets."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")
    device = device or history.device
    if num_samples < anchor_bank.num_anchors and initial_speed is not None:
        anchor_ids = select_anchor_ids(
            anchor_bank,
            num_samples,
            nav_command=int(nav_command[0]),
            speed_bin=int(speed_bin_indices(np.asarray([initial_speed]))[0]),
        )
    else:
        anchor_ids = anchor_bank.candidate_ids(num_samples)
    anchors = torch.from_numpy(anchor_bank.normalized[anchor_ids]).to(device)
    anchor_center = torch.from_numpy(
        anchor_bank.normalized.mean(axis=0)
    ).to(device)
    centered_anchors = anchors.float() - anchor_center.float().unsqueeze(0)
    noise = []
    for offset in range(num_samples):
        generator = torch.Generator(device=device).manual_seed(seed + offset)
        noise.append(
            low_frequency_noise(
                anchors.shape[1:],
                std=1.0,
                modes=getattr(expert, "noise_modes", 6),
                generator=generator,
                device=device,
            )
        )
    waypoints = anchors.float() + noise_std * torch.stack(noise)
    ids_tensor = torch.from_numpy(anchor_ids).to(device)
    base_history_queries = expert.encode_history(
        history,
        nav_command,
        history_velocity,
        history_acceleration,
    )
    base_prediction = expert.predict_endpoint(
        anchor_center.float().unsqueeze(0),
        torch.full((1,), flow_time, device=device, dtype=torch.float32),
        base_history_queries,
        scene_cache,
        position_anchor,
        nav_command,
        ego_status,
    )
    history_queries = expert.encode_history(
        _tile(history, num_samples),
        _tile(nav_command, num_samples),
        _tile(history_velocity, num_samples),
        _tile(history_acceleration, num_samples),
    )
    normalized, scores, _ = predict_scene_base_residual_conditioned(
        expert,
        heads,
        residual_head,
        waypoints,
        torch.full(
            (num_samples,),
            flow_time,
            device=device,
            dtype=torch.float32,
        ),
        history_queries,
        scene_cache,
        position_anchor.expand(-1, num_samples),
        base_prediction,
        centered_anchors,
        ids_tensor,
        _tile(nav_command, num_samples),
        _tile(ego_status, num_samples),
        scale,
    )
    return normalized, torch.from_numpy(anchor_ids), scores.detach()
