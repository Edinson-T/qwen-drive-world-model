from __future__ import annotations

import unittest

import torch

from qwen_drive_world_model.train_reward_model import RewardModel, _loss
from qwen_drive_world_model.train_geometry_reward_model import GeometryRewardModel, _delta_loss


class RewardModelTests(unittest.TestCase):
    def test_future_geometry_receives_ranking_gradients(self):
        torch.manual_seed(11)
        model = GeometryRewardModel('predicted_geometry', hidden=16).eval()
        geometry = torch.randn(2, 3, 30, requires_grad=True)
        scores = model(torch.zeros(2,3,20,3), torch.tensor([[-1,0,1],[-1,0,1]]),
                       torch.ones(2), torch.zeros(2,32,4,8), geometry)
        _delta_loss(scores, torch.tensor([[8.,9.,7.],[8.,7.,9.]])).backward()
        self.assertTrue(torch.isfinite(geometry.grad).all())
        self.assertGreater(float(geometry.grad.abs().sum()), 0.)

    def test_current_baseline_ignores_future_geometry(self):
        model = GeometryRewardModel('current', hidden=16).eval()
        args = (torch.zeros(2,3,20,3), torch.tensor([[-1,0,1],[-1,0,1]]),
                torch.ones(2), torch.zeros(2,32,4,8))
        torch.testing.assert_close(model(*args, torch.zeros(2,3,30)),
                                   model(*args, torch.randn(2,3,30)*100))

    def test_all_reward_model_inputs_produce_candidate_scores(self) -> None:
        batch, candidates = 2, 3
        trajectories = torch.zeros(batch, candidates, 20, 3)
        anchor_ids = torch.tensor([[-1, 0, 1], [-1, 2, 3]])
        speed = torch.full((batch,), 5.0)
        current = torch.zeros(batch, 32, 4, 8)
        future = torch.zeros(batch, candidates, 5, 32, 4, 8)
        poses = torch.zeros(batch, candidates, 5, 3)
        targets = torch.tensor(
            [
                [8.4, 8.5, 8.3],
                [8.4, 8.2, 8.6],
            ]
        )
        for mode in ("trajectory", "current", "future"):
            model = RewardModel(mode, hidden=16)
            scores = model(
                trajectories,
                anchor_ids,
                speed,
                current,
                future,
                poses,
            )
            self.assertEqual(scores.shape, (batch, candidates))
            self.assertTrue(torch.isfinite(_loss(scores, targets)))


if __name__ == "__main__":
    unittest.main()
