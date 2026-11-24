from __future__ import annotations

import unittest

import numpy as np

from qwen_drive_world_model.evaluate_selector_cv import (
    _policy_selection,
    _scene_folds,
)
from qwen_drive_world_model.train_pairwise_linear import fit_pairwise_linear
from qwen_drive_world_model.train_selector import _mode_features


class SelectorTests(unittest.TestCase):
    def test_scene_folds_keep_repeated_scene_together(self) -> None:
        scenes = np.asarray(["a", "b", "a", "c", "d", "c"])
        folds = _scene_folds(scenes, 2, seed=0)
        self.assertEqual(int(folds[0]), int(folds[2]))
        self.assertEqual(int(folds[3]), int(folds[5]))

    def test_max_risk_feature_modes_select_horizon_maxima(self) -> None:
        data = {
            "base": np.zeros((2, 3, 23), dtype=np.float32),
            "current": np.zeros((2, 3, 40), dtype=np.float32),
            "independent": np.arange(
                2 * 3 * 15, dtype=np.float32
            ).reshape(2, 3, 15),
            "shared": np.arange(
                2 * 3 * 15, dtype=np.float32
            ).reshape(2, 3, 15)
            + 1,
        }
        independent = _mode_features(data, "independent_max")
        shared = _mode_features(data, "shared_max")
        maxima = (1, 4, 7, 10, 13)
        np.testing.assert_array_equal(
            independent[..., -5:],
            data["independent"][..., maxima],
        )
        np.testing.assert_array_equal(
            shared[..., -5:],
            data["shared"][..., maxima],
        )

    def test_pairwise_fit_returns_finite_weights(self) -> None:
        rng = np.random.default_rng(0)
        values = rng.normal(size=(4, 5, 7))
        targets = np.arange(5, dtype=np.float64)[None].repeat(4, axis=0)
        checkpoint = fit_pairwise_linear(values, targets, alpha=0.1)
        self.assertEqual(checkpoint["weights"].shape, (7,))
        self.assertTrue(np.isfinite(checkpoint["weights"]).all())
        self.assertGreater(checkpoint["pairs"], 0)

    def test_baseline_policy_requires_a_margin_to_switch(self) -> None:
        scores = np.asarray([[1.0, 1.4], [1.0, 2.5]])
        np.testing.assert_array_equal(
            _policy_selection(scores, baseline_index=0, margin=0.5),
            np.asarray([0, 1]),
        )


if __name__ == "__main__":
    unittest.main()
