"""Multi-horizon Motion dataset for a shared world model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import _load_array_shards, horizon_key


class MotionMultiHorizonDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        include_highres: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.manifest = json.loads((self.data_dir / "dataset.json").read_text())
        self.horizons = tuple(float(value) for value in self.manifest["horizons"])
        self.keys = [horizon_key(value) for value in self.horizons]
        required = {
            "z_t",
            "height_t",
            "split",
            "scenario_ids",
            "tokens",
            "t_frames",
            "nav_command",
            "ego_velocity",
            "ego_acceleration",
            "driving_command",
            "trajectory_5s",
            "relative_poses",
        }
        for key in self.keys:
            required.update(
                {
                    f"z_{key}",
                    f"boxes_{key}",
                    f"box_labels_{key}",
                    f"box_ids_{key}",
                    f"box_counts_{key}",
                }
            )
        if include_highres:
            required.update(
                {
                    "map_t",
                    "occ_t",
                    "height_t_highres",
                    "boxes_t",
                    "box_labels_t",
                    "box_counts_t",
                }
            )
            for key in self.keys:
                required.update({f"map_{key}", f"occ_{key}"})
        self.arrays = _load_array_shards(self.data_dir, required)
        self.indices = np.flatnonzero(self.arrays["split"] == split)
        if not len(self.indices):
            raise ValueError(f"Motion split {split!r} is empty")
        self.include_highres = include_highres

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict:
        source = int(self.indices[index])
        ego_status = np.concatenate(
            [
                self.arrays["ego_velocity"][source],
                self.arrays["ego_acceleration"][source],
                self.arrays["driving_command"][source],
            ]
        )
        futures = [self.arrays[f"z_{key}"][source] for key in self.keys]
        target_maps = torch.stack(
            [torch.from_numpy(value[:8]).argmax(0) for value in futures]
        ).long()
        target_occupancy = torch.stack(
            [torch.from_numpy(value[8]).clamp(0, 1) for value in futures]
        ).float()
        box_targets = torch.stack(
            [
                torch.from_numpy(self.arrays[f"boxes_{key}"][source]).float()
                for key in self.keys
            ]
        )
        box_labels = torch.stack(
            [
                torch.from_numpy(self.arrays[f"box_labels_{key}"][source]).long()
                for key in self.keys
            ]
        )
        box_ids = torch.stack(
            [
                torch.from_numpy(self.arrays[f"box_ids_{key}"][source]).long()
                for key in self.keys
            ]
        )
        box_counts = torch.tensor(
            [self.arrays[f"box_counts_{key}"][source] for key in self.keys],
            dtype=torch.long,
        )
        sample = {
            "z_t": torch.from_numpy(self.arrays["z_t"][source]).float(),
            "height_t": torch.from_numpy(self.arrays["height_t"][source]).float(),
            "target_maps": target_maps,
            "target_occupancy": target_occupancy,
            "box_targets": box_targets,
            "box_labels": box_labels,
            "box_ids": box_ids,
            "box_counts": box_counts,
            "trajectory_5s": torch.from_numpy(
                self.arrays["trajectory_5s"][source]
            ).float(),
            "ego_status": torch.from_numpy(ego_status).float(),
            "relative_poses": torch.from_numpy(
                self.arrays["relative_poses"][source]
            ).float(),
            "scenario_id": str(self.arrays["scenario_ids"][source]),
            "token": str(self.arrays["tokens"][source]),
            "t_frame": int(self.arrays["t_frames"][source]),
        }
        if self.include_highres:
            sample.update(
                {
                    "map_t": torch.from_numpy(self.arrays["map_t"][source]).long(),
                    "occ_t": torch.from_numpy(self.arrays["occ_t"][source]).float(),
                    "height_t_highres": torch.from_numpy(
                        self.arrays["height_t_highres"][source]
                    ).float(),
                    "boxes_t": torch.from_numpy(
                        self.arrays["boxes_t"][source]
                    ).float(),
                    "box_labels_t": torch.from_numpy(
                        self.arrays["box_labels_t"][source]
                    ).long(),
                    "box_count_t": int(self.arrays["box_counts_t"][source]),
                    "map_future": torch.stack(
                        [
                            torch.from_numpy(self.arrays[f"map_{key}"][source]).long()
                            for key in self.keys
                        ]
                    ),
                    "occupancy_future": torch.stack(
                        [
                            torch.from_numpy(self.arrays[f"occ_{key}"][source]).float()
                            for key in self.keys
                        ]
                    ),
                }
            )
        return sample

