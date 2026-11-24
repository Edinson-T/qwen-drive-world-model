"""Model loading and source/checkpoint-safe adapter attachment."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from qwen_drive import QwenDriveForPlanning

from .lora import inject_lora

__all__ = [
    "LORA_TARGETS",
    "attach_lora",
    "file_sha256",
    "freeze_module",
    "load_released_planner",
]

LORA_TARGETS = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj")


def file_sha256(path: str | Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def load_released_planner(
    model_dir: str | Path,
    planner_dir: str | Path,
    *,
    device: str | torch.device,
    dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "sdpa",
) -> QwenDriveForPlanning:
    """Load the released VLM and planner without writing to either directory."""
    model = QwenDriveForPlanning.from_pretrained(
        str(model_dir),
        planner=str(planner_dir),
        dtype=dtype,
        attn_implementation=attn_implementation,
    )
    return model.to(device).eval()


def freeze_module(module: torch.nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def attach_lora(
    model: QwenDriveForPlanning,
    *,
    rank: int,
    alpha: float,
    dropout: float,
) -> list[torch.nn.Module]:
    """Attach LoRA to every planning-expert transformer block."""
    freeze_module(model)
    wrappers = inject_lora(
        model.planning_expert,
        LORA_TARGETS,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
    )
    for wrapper in wrappers:
        wrapper.lora_a.requires_grad_(True)
        wrapper.lora_b.requires_grad_(True)
    return wrappers

