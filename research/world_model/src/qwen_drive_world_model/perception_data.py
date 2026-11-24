"""Dataset loader for Waymo Perception GT world-model windows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .data import _load_array_shards, horizon_key


class PerceptionWorldDataset(Dataset):
    """Current GT BEV + action -> future GT BEV map, occupancy and 3D boxes."""

    def __init__(
        self,
        data_dir: str | Path,
        horizon: float = 1.0,
        split: str | None = "train",
        include_highres: bool = False,
        use_highres_targets: bool = False,
        allow_missing_boxes: bool = False,
        target_hw: tuple[int, int] | None = None,
        input_key: str = "z_t",
    ) -> None:
        self.data_dir = Path(data_dir)
        import json

        self.manifest = json.loads((self.data_dir / "dataset.json").read_text())
        if float(horizon) not in {float(value) for value in self.manifest["horizons"]}:
            raise ValueError(f"unsupported horizon {horizon}")
        self.horizon = float(horizon)
        key = horizon_key(self.horizon)
        required = {
            input_key,
            f"z_{key}",
            "split",
            "scene_tokens",
            "tokens",
            "pair_ids",
            "t_frames",
            "nav_command",
            "ego_velocity",
            "ego_acceleration",
            "driving_command",
            "trajectory_5s",
            "relative_poses",
        }
        if not allow_missing_boxes:
            required.update(
                {
                    f"boxes_{key}",
                    f"box_labels_{key}",
                    f"box_counts_{key}",
                }
            )
        if use_highres_targets:
            include_highres = True
        if include_highres:
            required.update(
                {
                    "map_t",
                    "occ_t",
                    f"map_{key}",
                    f"occ_{key}",
                }
            )
        self.arrays = _load_array_shards(self.data_dir, required)
        self.include_highres = include_highres
        self.use_highres_targets = use_highres_targets
        self.target_hw = target_hw
        self.input_key = input_key
        self.indices = (
            np.arange(len(self.arrays["z_t"]), dtype=np.int64)
            if split is None
            else np.flatnonzero(self.arrays["split"] == split)
        )
        if not len(self.indices):
            raise ValueError(f"dataset split {split!r} is empty")

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
        z_future = torch.from_numpy(self.arrays[f"z_{key}"][source]).float()
        if (
            f"boxes_{key}" in self.arrays
            and f"box_labels_{key}" in self.arrays
            and f"box_counts_{key}" in self.arrays
        ):
            box_targets = torch.from_numpy(
                self.arrays[f"boxes_{key}"][source]
            ).float()
            box_labels = torch.from_numpy(
                self.arrays[f"box_labels_{key}"][source]
            ).long()
            box_count = int(self.arrays[f"box_counts_{key}"][source])
        else:
            box_targets = torch.zeros((0, 7), dtype=torch.float32)
            box_labels = torch.zeros((0,), dtype=torch.long)
            box_count = 0
        sample = {
            "z_t": torch.from_numpy(self.arrays[self.input_key][source]).float(),
            "z_future": z_future,
            "target_map": z_future[:8].argmax(0).long(),
            "target_occupancy": z_future[8].clamp(0, 1),
            "box_targets": box_targets,
            "box_labels": box_labels,
            "box_count": box_count,
            "trajectory_5s": torch.from_numpy(
                self.arrays["trajectory_5s"][source]
            ).float(),
            "ego_status": torch.from_numpy(ego_status).float(),
            "relative_pose": torch.from_numpy(
                self.arrays["relative_poses"][source, int(
                    np.flatnonzero(
                        np.asarray(self.manifest["horizons"], float) == self.horizon
                    )[0]
                )]
            ).float(),
            "scene_token": str(self.arrays["scene_tokens"][source]),
            "token": str(self.arrays["tokens"][source]),
            "pair_id": str(self.arrays["pair_ids"][source]),
            "t_frame": int(self.arrays["t_frames"][source]),
        }
        if self.include_highres:
            sample.update(
                {
                    "map_t": torch.from_numpy(self.arrays["map_t"][source]).long(),
                    "occ_t": torch.from_numpy(self.arrays["occ_t"][source]).float(),
                    "map_future": torch.from_numpy(
                        self.arrays[f"map_{key}"][source]
                    ).long(),
                    "occupancy_future": torch.from_numpy(
                        self.arrays[f"occ_{key}"][source]
                    ).float(),
                }
            )
            if self.use_highres_targets:
                target_hw = self.target_hw or tuple(
                    self.arrays[self.input_key].shape[-2:]
                )
                future_map = torch.from_numpy(
                    self.arrays[f"map_{key}"][source]
                )[None, None].float()
                occupancy_classes = np.asarray(
                    self.arrays[f"occ_{key}"][source]
                )
                future_occ = torch.from_numpy(
                    ((occupancy_classes >= 0) & (occupancy_classes <= 6)).max(-1)
                ).float()[None, None]
                sample["target_map"] = F.interpolate(
                    future_map,
                    size=target_hw,
                    mode="nearest",
                )[0, 0].long()
                sample["target_occupancy"] = F.interpolate(
                    future_occ,
                    size=target_hw,
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
        return sample

