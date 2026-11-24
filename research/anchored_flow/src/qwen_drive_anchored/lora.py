"""Minimal dependency-free LoRA adapters for the Qwen-Drive planning expert."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F

__all__ = [
    "LoRALinear",
    "adapter_state_dict",
    "inject_lora",
    "load_trainable_adapter",
    "load_lora_adapter",
    "save_trainable_adapter",
    "save_lora_adapter",
    "trainable_state_dict",
]


class LoRALinear(nn.Module):
    """Frozen linear layer plus a trainable low-rank residual."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        self.base = base
        self.rank = rank
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        device = base.weight.device
        self.lora_a = nn.Parameter(
            torch.empty(rank, base.in_features, dtype=torch.float32, device=device)
        )
        self.lora_b = nn.Parameter(
            torch.zeros(base.out_features, rank, dtype=torch.float32, device=device)
        )
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        low_rank = F.linear(self.dropout(x).float(), self.lora_a)
        residual = F.linear(low_rank, self.lora_b)
        return y + residual.to(dtype=y.dtype) * self.scale


def inject_lora(
    module: nn.Module,
    target_names: tuple[str, ...],
    *,
    rank: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.0,
) -> list[LoRALinear]:
    """Replace matching immediate ``Linear`` children with ``LoRALinear``."""
    wrappers: list[LoRALinear] = []
    for parent in module.modules():
        for name, child in list(parent.named_children()):
            if name in target_names and isinstance(child, nn.Linear):
                wrapper = LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout)
                setattr(parent, name, wrapper)
                wrappers.append(wrapper)
    if not wrappers:
        raise ValueError(f"no Linear modules matched targets {target_names}")
    return wrappers


def adapter_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    state = {}
    for name, parameter in module.named_parameters():
        if isinstance(parameter, nn.Parameter) and name.endswith((".lora_a", ".lora_b")):
            state[name] = parameter.detach().cpu()
    if not state:
        raise ValueError("module contains no LoRA parameters")
    return state


def save_lora_adapter(module: nn.Module, path: str | Path, metadata: dict[str, str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(adapter_state_dict(module), str(path), metadata=metadata)


def trainable_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    """Save every trainable tensor attached to a frozen base model."""
    state = {
        name: parameter.detach().cpu()
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }
    if not state:
        raise ValueError("module contains no trainable parameters")
    return state


def save_trainable_adapter(
    module: nn.Module,
    path: str | Path,
    metadata: dict[str, str],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(trainable_state_dict(module), str(path), metadata=metadata)


def load_trainable_adapter(module: nn.Module, path: str | Path) -> None:
    weights = load_file(str(path))
    parameters = {
        name: parameter
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }
    missing = sorted(set(parameters) - set(weights))
    unexpected = sorted(set(weights) - set(parameters))
    if missing or unexpected:
        raise ValueError(
            f"adapter mismatch: missing={missing[:8]} unexpected={unexpected[:8]}"
        )
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(weights[name].to(device=parameter.device, dtype=parameter.dtype))


def load_lora_adapter(module: nn.Module, path: str | Path) -> None:
    weights = load_file(str(path))
    parameters = {
        name: parameter
        for name, parameter in module.named_parameters()
        if name.endswith((".lora_a", ".lora_b"))
    }
    missing = sorted(set(parameters) - set(weights))
    unexpected = sorted(set(weights) - set(parameters))
    if missing or unexpected:
        raise ValueError(
            f"adapter mismatch: missing={missing[:8]} unexpected={unexpected[:8]}"
        )
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(weights[name].to(device=parameter.device, dtype=parameter.dtype))

