"""Dataset helpers for anchored-flow planner fine-tuning."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from qwen_drive.benchmarks import BenchmarkSample, read_scene_file
from qwen_drive.trajectory import normalize_trajectory

from .anchors import AnchorBank, normalize_futures

__all__ = [
    "AnchorBalanceSampler",
    "PlannerDataset",
    "StratifiedAnchorSampler",
    "load_training_samples",
]


def speed_bin_indices(
    speeds: np.ndarray,
    edges: tuple[float, ...] = (0.0, 2.0, 5.0, 10.0, float("inf")),
) -> np.ndarray:
    edges_array = np.asarray(edges, dtype=np.float64)
    if len(edges_array) < 2 or np.any(np.diff(edges_array) <= 0):
        raise ValueError("speed edges must be strictly increasing")
    return np.digitize(
        np.asarray(speeds, dtype=np.float64),
        edges_array[1:-1],
        right=False,
    ).astype(np.int64)


def load_training_samples(
    scenes: str | Path,
    image_root: str | Path,
    *,
    num_history_points: int,
    limit: int | None = None,
) -> list[BenchmarkSample]:
    samples = list(
        read_scene_file(
            scenes,
            image_root=image_root,
            num_history_points=num_history_points,
            limit=limit,
        )
    )
    usable = [
        sample
        for sample in samples
        if sample.future_trajectory is not None
        and len(sample.future_trajectory) > 1
    ]
    if not usable:
        raise ValueError("scene file contains no usable future trajectories")
    return usable


class PlannerDataset(Dataset):
    def __init__(
        self,
        samples: list[BenchmarkSample],
        processor,
        anchor_bank: AnchorBank,
        trajectory_scale: tuple[float, float, float],
        *,
        with_reasoning: bool = False,
    ) -> None:
        self.samples = samples
        self.processor = processor
        self.anchor_bank = anchor_bank
        self.scale = np.asarray(trajectory_scale, dtype=np.float32)
        self.with_reasoning = with_reasoning
        self.anchor_ids = self._anchor_ids()
        self.nav_commands = np.asarray(
            [sample.scene.nav_command for sample in self.samples],
            dtype=np.int64,
        )
        self.initial_speeds = np.asarray(
            [sample.initial_speed for sample in self.samples],
            dtype=np.float32,
        )

    def _anchor_ids(self) -> np.ndarray:
        futures = np.stack([sample.future_trajectory for sample in self.samples])
        normalized = normalize_futures(futures, self.scale)
        return np.asarray(
            [self.anchor_bank.nearest_xy(trajectory) for trajectory in normalized],
            dtype=np.int64,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        inputs = self.processor(
            sample.scene,
            with_reasoning=self.with_reasoning,
            device="cpu",
        )
        return {
            "inputs": inputs,
            "future": torch.from_numpy(np.asarray(sample.future_trajectory, dtype=np.float32)),
            "anchor_id": int(self.anchor_ids[index]),
            "token": sample.token,
        }


class AnchorBalanceSampler(torch.utils.data.Sampler[int]):
    """Oversample scenes assigned to rare anchors."""

    def __init__(
        self,
        anchor_ids: np.ndarray,
        *,
        num_samples: int | None = None,
        replacement: bool = True,
        seed: int = 2027,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.anchor_ids = np.asarray(anchor_ids, dtype=np.int64)
        counts = np.bincount(self.anchor_ids)
        weights = 1.0 / counts[self.anchor_ids]
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = len(self.anchor_ids) if num_samples is None else num_samples
        self.replacement = replacement
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return (self.num_samples + self.world_size - 1 - self.rank) // self.world_size

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights,
            self.num_samples,
            replacement=self.replacement,
            generator=generator,
        ).tolist()
        yield from indices[self.rank :: self.world_size]


class StratifiedAnchorSampler(torch.utils.data.Sampler[int]):
    """Balance navigation command, speed bin, and GT-nearest anchor."""

    def __init__(
        self,
        anchor_ids: np.ndarray,
        nav_commands: np.ndarray,
        initial_speeds: np.ndarray,
        *,
        speed_edges: tuple[float, ...] = (0.0, 2.0, 5.0, 10.0, float("inf")),
        num_samples: int | None = None,
        seed: int = 2027,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.anchor_ids = np.asarray(anchor_ids, dtype=np.int64)
        self.nav_commands = np.asarray(nav_commands, dtype=np.int64)
        self.speed_bins = speed_bin_indices(initial_speeds, speed_edges)
        if not (
            len(self.anchor_ids)
            == len(self.nav_commands)
            == len(self.speed_bins)
        ):
            raise ValueError("sampler labels must have equal length")
        keys = list(
            zip(
                self.nav_commands.tolist(),
                self.speed_bins.tolist(),
                self.anchor_ids.tolist(),
            )
        )
        counts: dict[tuple[int, int, int], int] = {}
        for key in keys:
            counts[key] = counts.get(key, 0) + 1
        weights = np.asarray(
            [1.0 / np.sqrt(counts[key]) for key in keys],
            dtype=np.float64,
        )
        weights /= weights.sum()
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = len(self.anchor_ids) if num_samples is None else num_samples
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return (self.num_samples + self.world_size - 1 - self.rank) // self.world_size

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights,
            self.num_samples,
            replacement=True,
            generator=generator,
        ).tolist()
        yield from indices[self.rank :: self.world_size]

