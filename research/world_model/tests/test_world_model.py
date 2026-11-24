from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from qwen_drive_world_model.data import WorldWindowDataset
from qwen_drive_world_model.map_decoder import LatentMapDecoder
from qwen_drive_world_model.model import (
    BEVWorldModel,
    WorldModelConfig,
    latent_loss,
    resample_trajectory,
)
from qwen_drive_perception.modeling_perception import _compress_feature_map
from qwen_drive_world_model.train import EarlyStopping


class ModelTests(unittest.TestCase):
    def test_resample_preserves_endpoints(self):
        trajectory = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 2.0, 0.1], [3.0, 4.0, 0.2]]])
        result = resample_trajectory(trajectory, 8)
        self.assertEqual(result.shape, (1, 8, 3))
        torch.testing.assert_close(result[:, 0], trajectory[:, 0])
        torch.testing.assert_close(result[:, -1], trajectory[:, -1])

    def test_forward_matches_target_shape(self):
        config = WorldModelConfig(
            input_channels=8,
            hidden_size=32,
            token_hw=(2, 2),
            target_hw=(4, 6),
            num_layers=1,
            num_heads=4,
            feedforward_size=64,
        )
        model = BEVWorldModel(config)
        z = torch.randn(2, 8, 4, 6)
        trajectory = torch.randn(2, 21, 3)
        ego = torch.randn(2, 8)
        output = model(z, trajectory, ego)
        self.assertEqual(output.shape, (2, 8, 4, 6))
        self.assertTrue(torch.isfinite(output).all())

    def test_identical_latents_have_zero_loss(self):
        target = torch.randn(2, 8, 4, 6)
        loss, parts = latent_loss(target, target)
        self.assertAlmostEqual(float(loss), 0.0, places=6)
        self.assertAlmostEqual(float(parts["cosine"]), 1.0, places=5)

    def test_early_stopping_tracks_patience(self):
        stopper = EarlyStopping(patience=2, min_delta=0.01)
        self.assertTrue(stopper.update(1.0, 10))
        self.assertFalse(stopper.update(1.005, 20))
        self.assertFalse(stopper.update(1.006, 30))
        self.assertTrue(stopper.should_stop)
        self.assertEqual(stopper.best_step, 10)

    def test_map_decoder_shape(self):
        decoder = LatentMapDecoder(in_channels=8, hidden=32, output_hw=(20, 40))
        self.assertEqual(tuple(decoder(torch.randn(2, 8, 4, 6)).shape), (2, 6, 20, 40))


class DatasetTests(unittest.TestCase):
    def test_dataset_keys_and_scene_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arrays = {
                "z_t": np.random.randn(2, 8, 4, 6).astype(np.float16),
                "z_0p5s": np.random.randn(2, 8, 4, 6).astype(np.float16),
                "z_1s": np.random.randn(2, 8, 4, 6).astype(np.float16),
                "z_2s": np.random.randn(2, 8, 4, 6).astype(np.float16),
                "z_3s": np.random.randn(2, 8, 4, 6).astype(np.float16),
                "z_5s": np.random.randn(2, 8, 4, 6).astype(np.float16),
                "split": np.asarray(["train", "val"]),
                "scene_tokens": np.asarray(["scene-a", "scene-b"]),
                "tokens": np.asarray(["scene-a-10", "scene-b-10"]),
                "pair_ids": np.asarray(["scene-a-10", "scene-b-10"]),
                "t_frames": np.asarray([10, 10], dtype=np.int32),
                "nav_command": np.asarray([0, 1], dtype=np.int64),
                "ego_velocity": np.zeros((2, 2), dtype=np.float32),
                "ego_acceleration": np.zeros((2, 2), dtype=np.float32),
                "driving_command": np.asarray([[0, 1, 0, 0], [1, 0, 0, 0]], dtype=np.float32),
                "trajectory_5s": np.random.randn(2, 21, 3).astype(np.float32),
                "traj_0p5s": np.random.randn(2, 3, 3).astype(np.float32),
                "traj_valid_0p5s": np.ones((2, 3), dtype=bool),
                "traj_1s": np.random.randn(2, 5, 3).astype(np.float32),
                "traj_valid_1s": np.ones((2, 5), dtype=bool),
                "traj_2s": np.random.randn(2, 9, 3).astype(np.float32),
                "traj_valid_2s": np.ones((2, 9), dtype=bool),
                "traj_3s": np.random.randn(2, 13, 3).astype(np.float32),
                "traj_valid_3s": np.ones((2, 13), dtype=bool),
                "traj_5s": np.random.randn(2, 21, 3).astype(np.float32),
                "traj_valid_5s": np.ones((2, 21), dtype=bool),
                "relative_poses": np.random.randn(2, 3, 3).astype(np.float32),
                "map_t": np.zeros((2, 20, 40), dtype=np.int8),
                "map_0p5s": np.zeros((2, 20, 40), dtype=np.int8),
            }
            shard = root / "shard.npz"
            np.savez_compressed(shard, **arrays)
            manifest = {
                "horizons": [0.5, 1.0, 2.0, 3.0, 5.0],
                "shards": [{"path": str(shard)}],
            }
            (root / "dataset.json").write_text(json.dumps(manifest))
            dataset = WorldWindowDataset(root, horizon=0.5, split="train")
            self.assertEqual(len(dataset), 1)
            sample = dataset[0]
            self.assertEqual(sample["scene_token"], "scene-a")
            self.assertEqual(tuple(sample["z_future"].shape), (8, 4, 6))
            dataset_5s = WorldWindowDataset(root, horizon=5.0, split="val")
            self.assertEqual(len(dataset_5s), 1)
            self.assertEqual(tuple(dataset_5s[0]["trajectory"].shape), (21, 3))
            self.assertEqual(tuple(WorldWindowDataset(root, horizon=1.0, split="train")[0]["trajectory"].shape), (5, 3))
            self.assertEqual(tuple(WorldWindowDataset(root, horizon=2.0, split="train")[0]["trajectory"].shape), (9, 3))
            with_maps = WorldWindowDataset(root, horizon=0.5, split="train", include_maps=True)
            self.assertEqual(tuple(with_maps[0]["map_future"].shape), (20, 40))


class PerceptionLatentTests(unittest.TestCase):
    def test_compresses_tokens_to_configured_latent(self):
        tokens = torch.randn(1, 8 * 8, 256)
        result = _compress_feature_map(
            tokens,
            bev_h=8,
            bev_w=8,
            output_hw=(4, 6),
            output_channels=128,
        )
        self.assertEqual(tuple(result.shape), (1, 128, 4, 6))

    def test_compresses_sequence_first_tokens(self):
        tokens = torch.randn(8 * 8, 1, 256)
        result = _compress_feature_map(
            tokens,
            bev_h=8,
            bev_w=8,
            output_hw=(4, 6),
            output_channels=128,
        )
        self.assertEqual(tuple(result.shape), (1, 128, 4, 6))


if __name__ == "__main__":
    unittest.main()
