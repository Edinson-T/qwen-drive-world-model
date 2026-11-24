from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from qwen_drive_anchored.anchors import (
    AnchorBank,
    fit_anchor_bank,
    fit_stratified_medoid_bank,
    select_anchor_ids,
)
from qwen_drive_anchored.data import StratifiedAnchorSampler, speed_bin_indices
from qwen_drive_anchored.heads import (
    AnchorHeads,
    SceneBaseResidualHead,
    bound_residual,
)
from qwen_drive_anchored.lora import (
    inject_lora,
    load_lora_adapter,
    save_lora_adapter,
)
from qwen_drive_anchored.paper_figures import (
    _resample_xy,
    _selection_index,
)
from qwen_drive_anchored.prepare_waymo_perception_frames import (
    optical_camera_to_vehicle,
)
from qwen_drive_anchored.sampling import (
    low_frequency_noise,
    sample_base_residual,
)


class AnchorTests(unittest.TestCase):
    def test_kmeans_separates_two_modes_and_allocates_candidates(self) -> None:
        rng = np.random.default_rng(0)
        left = np.zeros((30, 5, 3), dtype=np.float32)
        right = np.ones((30, 5, 3), dtype=np.float32)
        left += rng.normal(scale=0.01, size=left.shape)
        right += rng.normal(scale=0.01, size=right.shape)
        bank = fit_anchor_bank(
            np.concatenate([left, right]),
            np.ones(3, dtype=np.float32),
            num_anchors=2,
            seed=3,
        )
        self.assertEqual(bank.counts.tolist(), [30, 30])
        self.assertEqual(bank.candidate_ids(2).tolist(), [0, 1])
        self.assertEqual(bank.candidate_ids(5).tolist(), [0, 1, 0, 1, 0])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "anchors.npz"
            bank.save(path)
            loaded = AnchorBank.load(path)
            np.testing.assert_allclose(loaded.normalized, bank.normalized)
            np.testing.assert_allclose(loaded.physical, bank.physical)

    def test_stratified_medoids_use_real_trajectories_and_cover_groups(self) -> None:
        rng = np.random.default_rng(4)
        trajectories = rng.normal(size=(24, 5, 3)).astype(np.float32)
        nav = np.repeat(np.arange(3), 8)
        speeds = np.repeat([0.5, 3.0, 7.0, 12.0], 6)
        bank = fit_stratified_medoid_bank(
            trajectories,
            nav,
            speeds,
            np.ones(3, dtype=np.float32),
            num_anchors=12,
        )
        expected_groups = set(
            zip(
                nav.tolist(),
                speed_bin_indices(speeds).tolist(),
            )
        )
        actual_groups = set(
            zip(
                bank.nav_commands.tolist(),
                bank.speed_bins.tolist(),
            )
        )
        self.assertTrue(expected_groups <= actual_groups)
        physical = trajectories.reshape(len(trajectories), -1)
        anchors = bank.physical.reshape(len(bank.physical), -1)
        for anchor in anchors:
            self.assertTrue(np.any(np.all(physical == anchor, axis=1)))
        self.assertEqual(len(bank.counts), bank.num_anchors)

    def test_stratified_sampler_balances_groups_deterministically(self) -> None:
        sampler = StratifiedAnchorSampler(
            anchor_ids=np.asarray([0] * 20 + [1]),
            nav_commands=np.asarray([0] * 20 + [1]),
            initial_speeds=np.asarray([8.0] * 20 + [1.0]),
            num_samples=50,
            seed=7,
        )
        first = list(sampler)
        second = list(sampler)
        self.assertEqual(first, second)
        self.assertTrue(any(first.count(index) > 0 for index in range(20)))
        self.assertGreater(first.count(20), 0)

    def test_scene_anchor_selection_keeps_required_anchor(self) -> None:
        bank = AnchorBank(
            normalized=np.zeros((5, 3, 3), dtype=np.float32),
            physical=np.zeros((5, 3, 3), dtype=np.float32),
            counts=np.ones(5, dtype=np.int64),
            scale=np.ones(3, dtype=np.float32),
            nav_commands=np.asarray([0, 0, 1, 1, 2]),
            speed_bins=np.asarray([0, 1, 0, 1, 0]),
            speed_edges=np.asarray([0, 2, 5, 10, np.inf]),
        )
        selected = select_anchor_ids(
            bank,
            3,
            nav_command=0,
            speed_bin=1,
            required=4,
        )
        self.assertEqual(int(selected[0]), 4)
        self.assertIn(1, selected.tolist())
        self.assertEqual(len(selected), 3)


class SamplingTests(unittest.TestCase):
    def test_low_frequency_noise_is_smoother_than_iid_noise(self) -> None:
        generator = torch.Generator().manual_seed(7)
        smooth = low_frequency_noise(
            (50, 3),
            std=1.0,
            modes=6,
            generator=generator,
            device=torch.device("cpu"),
        )
        iid = torch.randn(50, 3, generator=generator)
        self.assertLess(
            float(torch.diff(smooth, dim=0).square().mean()),
            float(torch.diff(iid, dim=0).square().mean()) * 0.1,
        )

    def test_base_sampler_uses_one_shared_scene_base(self) -> None:
        class BaseExpert:
            num_history_points = 16

            @staticmethod
            def encode_history(history, nav, velocity, acceleration):
                return history, velocity, acceleration

            @staticmethod
            def predict_endpoint(
                waypoints,
                flow_time,
                history_queries,
                scene_cache,
                position_anchor,
                nav_command,
                ego_status,
            ):
                return waypoints.float()

        anchors = AnchorBank(
            normalized=np.arange(60, dtype=np.float32).reshape(3, 5, 4)[..., :3],
            physical=np.zeros((3, 5, 3), dtype=np.float32),
            counts=np.ones(3, dtype=np.int64),
            scale=np.ones(3, dtype=np.float32),
        )
        observed = {}

        def predict(
            expert,
            heads,
            residual_head,
            waypoints,
            flow_time,
            history_queries,
            scene_cache,
            position_anchor,
            base_prediction,
            centered_anchors,
            anchor_ids,
            nav_command,
            ego_status,
            scale,
        ):
            observed["base_shape"] = base_prediction.shape
            observed["centered"] = centered_anchors
            return (
                base_prediction.expand(3, -1, -1),
                base_prediction.new_zeros(3),
                base_prediction.new_zeros(3, 5, 3),
            )

        with patch(
            "qwen_drive_anchored.sampling.predict_scene_base_residual_conditioned",
            side_effect=predict,
        ):
            normalized, ids, scores = sample_base_residual(
                BaseExpert(),
                object(),
                object(),
                anchor_bank=anchors,
                scale=torch.ones(3),
                scene_cache=[],
                position_anchor=torch.zeros(1, 1),
                history=torch.zeros(1, 15, 3),
                history_velocity=torch.zeros(1, 16, 2),
                history_acceleration=torch.zeros(1, 16, 2),
                nav_command=torch.zeros(1, dtype=torch.long),
                ego_status=torch.zeros(1, 8),
                num_samples=3,
                noise_std=0.0,
                seed=4,
                flow_time=0.9,
            )
        self.assertEqual(normalized.shape, (3, 5, 3))
        self.assertEqual(ids.tolist(), [0, 1, 2])
        self.assertEqual(scores.shape, (3,))
        self.assertEqual(observed["base_shape"], (1, 5, 3))
        torch.testing.assert_close(
            observed["centered"].mean(dim=0),
            torch.zeros(5, 3),
        )


class HeadTests(unittest.TestCase):
    def test_anchor_heads_start_with_distinct_small_conditions(self) -> None:
        heads = AnchorHeads(num_anchors=20, hidden_size=16)
        condition = heads.anchor_condition(torch.arange(4), torch.float32)
        self.assertGreater(int(torch.count_nonzero(condition)), 0)
        self.assertFalse(torch.equal(condition[0], condition[1]))
        self.assertEqual(heads.score(torch.randn(4, 5, 16)).shape, (4,))

    def test_residual_bounds_are_physical(self) -> None:
        scale = torch.tensor([165.0, 25.0, 1.5703125])
        bounded = bound_residual(
            torch.tensor([[[1000.0, 1000.0, 1000.0]]]),
            scale,
            max_xy_residual_m=12.0,
            max_heading_residual_rad=0.35,
        )
        self.assertLessEqual(
            float(torch.linalg.vector_norm(bounded[..., :2] * scale[:2], dim=-1)),
            12.0 + 1e-5,
        )
        self.assertLessEqual(
            float((bounded[..., 2] * scale[2]).abs()),
            0.35 + 1e-5,
        )

    def test_scene_base_residual_starts_from_base_plus_modes(self) -> None:
        head = SceneBaseResidualHead(
            16,
            mode_offset_scale=0.1,
            min_mode_offset_scale=0.05,
            max_mode_offset_scale=0.25,
        )
        hidden = torch.randn(4, 5, 16)
        base = torch.randn(1, 5, 3).expand(4, -1, -1)
        centered = torch.randn(4, 5, 3)
        scale = torch.tensor([165.0, 25.0, 1.5703125])
        prediction, delta = head(hidden, base, centered, scale)
        torch.testing.assert_close(prediction, base + 0.1 * centered)
        self.assertLessEqual(torch.count_nonzero(delta), 0)

    def test_scene_mode_scale_stays_in_range(self) -> None:
        head = SceneBaseResidualHead(
            16,
            mode_offset_scale=0.1,
            min_mode_offset_scale=0.05,
            max_mode_offset_scale=0.25,
        )
        with torch.no_grad():
            head.raw_mode_offset_scale.fill_(100.0)
        self.assertLessEqual(float(head.mode_offset_scale.detach()), 0.25)
        with torch.no_grad():
            head.raw_mode_offset_scale.fill_(-100.0)
        self.assertGreaterEqual(float(head.mode_offset_scale.detach()), 0.05)

    def test_lora_adapter_roundtrips(self) -> None:
        model = nn.Sequential(nn.Linear(8, 8), nn.SiLU(), nn.Linear(8, 3))
        x = torch.randn(4, 8)
        before = model(x).detach()
        wrappers = inject_lora(model, ("0", "2"), rank=2, alpha=4.0)
        torch.testing.assert_close(model(x), before)
        with torch.no_grad():
            wrappers[0].lora_b.normal_()
        changed = model(x)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter.safetensors"
            save_lora_adapter(model, path, {"format": "test"})
            with torch.no_grad():
                for wrapper in wrappers:
                    wrapper.lora_b.zero_()
            load_lora_adapter(model, path)
        torch.testing.assert_close(model(x), changed)


class FigureTests(unittest.TestCase):
    def test_trajectory_resampling_keeps_endpoints(self) -> None:
        trajectory = np.asarray([[0.0, 0.0], [2.0, 1.0], [4.0, 2.0]])
        sampled = _resample_xy(trajectory, 5)
        np.testing.assert_allclose(sampled[0], trajectory[0])
        np.testing.assert_allclose(sampled[-1], trajectory[-1])

    def test_selection_prefers_saved_index(self) -> None:
        record = {
            "trajectories": [[[0.0, 0.0], [1.0, 0.0]], [[0.0, 0.0], [2.0, 0.0]]],
            "future_trajectory": [[0.0, 0.0], [2.0, 0.0]],
            "candidate_scores": [2.0, 1.0],
            "selected_index": 1,
        }
        self.assertEqual(
            _selection_index(record, selection="auto", score_field="candidate_scores"),
            1,
        )


class CalibrationTests(unittest.TestCase):
    def test_waymo_camera_axes_convert_to_opencv_optical(self) -> None:
        rotation, translation = optical_camera_to_vehicle(np.eye(4, dtype=np.float32))
        # Optical +Z (camera forward) must point along ego +X (vehicle forward).
        np.testing.assert_allclose(rotation[:, 2], [1.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(translation, [0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
