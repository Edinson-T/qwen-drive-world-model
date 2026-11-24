"""Dataset loader for Waymo Motion GT world-model windows."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import _load_array_shards, horizon_key


class MotionWorldDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        horizon: float = 1.0,
        split: str | None = "train",
        include_highres: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.manifest = json.loads((self.data_dir / "dataset.json").read_text())
        if float(horizon) not in {float(value) for value in self.manifest["horizons"]}:
            raise ValueError(f"unsupported Motion horizon {horizon}")
        self.horizon = float(horizon)
        key = horizon_key(self.horizon)
        self.horizon_index = list(map(float, self.manifest["horizons"])).index(self.horizon)
        required = {
            "z_t",
            "height_t",
            f"z_{key}",
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
            f"boxes_{key}",
            f"box_labels_{key}",
            f"box_ids_{key}",
            f"box_counts_{key}",
        }
        if include_highres:
            required.update(
                {
                    "map_t",
                    "occ_t",
                    "height_t_highres",
                    "boxes_t",
                    "box_labels_t",
                    "box_counts_t",
                    f"map_{key}",
                    f"occ_{key}",
                }
            )
        self.arrays = _load_array_shards(self.data_dir, required)
        self.include_highres = include_highres
        self.indices = (
            np.arange(len(self.arrays["z_t"]), dtype=np.int64)
            if split is None
            else np.flatnonzero(self.arrays["split"] == split)
        )
        if not len(self.indices):
            raise ValueError(f"Motion split {split!r} is empty")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        source = int(self.indices[index])
        key = horizon_key(self.horizon)
        ego_status = np.concatenate(
            [
                self.arrays["ego_velocity"][source],
                self.arrays["ego_acceleration"][source],
                self.arrays["driving_command"][source],
            ]
        )
        target = torch.from_numpy(self.arrays[f"z_{key}"][source]).float()
        sample = {
            "z_t": torch.from_numpy(self.arrays["z_t"][source]).float(),
            "z_future": target,
            "height_t": torch.from_numpy(self.arrays["height_t"][source]).float(),
            "target_map": target[:8].argmax(0).long(),
            "target_occupancy": target[8].clamp(0, 1),
            "box_targets": torch.from_numpy(self.arrays[f"boxes_{key}"][source]).float(),
            "box_labels": torch.from_numpy(
                self.arrays[f"box_labels_{key}"][source]
            ).long(),
            "box_ids": torch.from_numpy(
                self.arrays[f"box_ids_{key}"][source]
            ).long(),
            "box_count": int(self.arrays[f"box_counts_{key}"][source]),
            "trajectory_5s": torch.from_numpy(
                self.arrays["trajectory_5s"][source]
            ).float(),
            "ego_status": torch.from_numpy(ego_status).float(),
            "relative_pose": torch.from_numpy(
                self.arrays["relative_poses"][source, self.horizon_index]
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
                    "map_future": torch.from_numpy(
                        self.arrays[f"map_{key}"][source]
                    ).long(),
                    "occupancy_future": torch.from_numpy(
                        self.arrays[f"occ_{key}"][source]
                    ).float(),
                }
            )
        return sample

