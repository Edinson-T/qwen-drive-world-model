"""Dataset loading for cached compact BEV world-model pairs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def horizon_key(seconds: float) -> str:
    return f"{float(seconds):g}s".replace(".", "p")


def trajectory_key(seconds: float) -> str:
    return f"traj_{horizon_key(seconds)}"


def target_latent_key(seconds: float) -> str:
    return f"z_{horizon_key(seconds)}"


def target_feature_key(seconds: float) -> str:
    return f"f_{horizon_key(seconds)}"


def _load_array_shards(directory: Path, keys: set[str]) -> dict[str, np.ndarray]:
    manifest_path = directory / "dataset.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing dataset manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    parts = {key: [] for key in keys}
    for shard in manifest["shards"]:
        with np.load(shard["path"], allow_pickle=False) as payload:
            missing = [key for key in keys if key not in payload.files]
            if missing:
                raise KeyError(f"{shard['path']} is missing arrays {missing}")
            for key in keys:
                parts[key].append(payload[key])
    return {key: np.concatenate(values, axis=0) for key, values in parts.items()}


class WorldWindowDataset(Dataset):
    """One current/future latent pair with a full-horizon action trajectory."""

    def __init__(
        self,
        data_dir: str | Path,
        horizon: float = 0.5,
        split: str | None = "train",
        include_maps: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        manifest = json.loads((self.data_dir / "dataset.json").read_text())
        available_horizons = {float(value) for value in manifest["horizons"]}
        if float(horizon) not in available_horizons:
            raise ValueError(
                f"horizon {horizon} not in dataset horizons {sorted(available_horizons)}"
            )
        self.horizon = float(horizon)
        tkey = horizon_key(self.horizon)
        self.target_key = f"z_{tkey}"
        trajectory_key_name = f"traj_{tkey}"
        required = {
            "z_t",
            self.target_key,
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
            trajectory_key_name,
            f"traj_valid_{tkey}",
            "relative_poses",
        }
        if include_maps:
            required.update({"map_t", f"map_{tkey}"})
        self.arrays = _load_array_shards(self.data_dir, required)
        self.trajectory_key = trajectory_key_name
        self.include_maps = include_maps
        if split is None:
            self.indices = np.arange(len(self.arrays["z_t"]), dtype=np.int64)
        else:
            self.indices = np.flatnonzero(self.arrays["split"] == split)
        if not len(self.indices):
            raise ValueError(f"dataset split {split!r} is empty")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        source_index = int(self.indices[index])
        ego_status = np.concatenate(
            [
                self.arrays["ego_velocity"][source_index],
                self.arrays["ego_acceleration"][source_index],
                self.arrays["driving_command"][source_index],
            ]
        )
        sample = {
            "z_t": torch.from_numpy(self.arrays["z_t"][source_index]).float(),
            "z_future": torch.from_numpy(
                self.arrays[self.target_key][source_index]
            ).float(),
            "trajectory": torch.from_numpy(
                self.arrays[self.trajectory_key][source_index]
            ).float(),
            "trajectory_5s": torch.from_numpy(
                self.arrays["trajectory_5s"][source_index]
            ).float(),
            "ego_status": torch.from_numpy(ego_status).float(),
            "nav_command": int(self.arrays["nav_command"][source_index]),
            "relative_pose": torch.from_numpy(
                self.arrays["relative_poses"][source_index, :, :]
            ).float(),
            "scene_token": str(self.arrays["scene_tokens"][source_index]),
            "token": str(self.arrays["tokens"][source_index]),
            "pair_id": str(self.arrays["pair_ids"][source_index]),
            "t_frame": int(self.arrays["t_frames"][source_index]),
        }
        if self.include_maps:
            sample["map_t"] = self.arrays["map_t"][source_index].astype(np.int64)
            sample["map_future"] = self.arrays[f"map_{horizon_key(self.horizon)}"][
                source_index
            ].astype(np.int64)
        return sample


def overfit_view(dataset: WorldWindowDataset, max_scenes: int, max_samples: int) -> list[int]:
    """Select a deterministic tiny scene-grouped subset."""
    scenes = []
    selected = []
    for dataset_index, source_index in enumerate(dataset.indices):
        scene = str(dataset.arrays["scene_tokens"][source_index])
        if scene not in scenes:
            if len(scenes) >= max_scenes:
                continue
            scenes.append(scene)
        if scene in scenes:
            selected.append(dataset_index)
        if len(selected) >= max_samples:
            break
    return selected
