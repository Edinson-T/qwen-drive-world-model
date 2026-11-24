import unittest

import numpy as np
import torch

from qwen_drive_world_model.latent_motion_model import LatentConfig, LatentPredictor
from qwen_drive_world_model.train_motion_task import (
    ProximityHead, proximity_labels, summarize_task, task_loss,
    copy_head_training_state,
)


class MotionTaskTest(unittest.TestCase):
    def test_warmup_copy_does_not_couple_optimizer_controls(self):
        source, target = torch.nn.Linear(2, 1), torch.nn.Linear(2, 1)
        source_opt = torch.optim.AdamW(source.parameters())
        target_opt = torch.optim.AdamW(target.parameters())
        source(torch.ones(1, 2)).sum().backward()
        source_opt.step()
        copy_head_training_state(source, source_opt, target, target_opt)
        source_state = source_opt.state[next(source.parameters())]
        target_state = target_opt.state[next(target.parameters())]
        before = {k:v.clone() for k,v in source_state.items()}
        for key in before:
            self.assertTrue(torch.equal(source_state[key], target_state[key]))
            self.assertNotEqual(source_state[key].data_ptr(), target_state[key].data_ptr())
        target(torch.ones(1, 2)*3).sum().backward()
        target_opt.step()
        for key in before:
            self.assertTrue(torch.equal(source_state[key], before[key]))
        self.assertFalse(torch.equal(target_state['exp_avg'], before['exp_avg']))

    def test_ego_overlap_and_empty_annotation_crop(self):
        occupancy = np.zeros((1, 3, 4, 50, 100), dtype=np.uint8)
        occupancy[0, 0, 0, 25, 50] = 1
        occupancy[0, 1, 0, 25, 60] = 1
        labels = proximity_labels(occupancy)
        self.assertEqual(labels[0, 0, 0], 1)
        self.assertEqual(labels[0, 1, 0], 0)
        self.assertGreater(labels[0, 1, 1], labels[0, 2, 1])
        self.assertAlmostEqual(float(labels[0, 2, 1]), np.exp(-8/3), places=6)

    def test_selection_regret_groups_three_candidates(self):
        labels = np.array([[[0, .8], [0, .4], [0, .2]],
                           [[0, .1], [0, .3], [0, .7]]], dtype=np.float32)
        exact = summarize_task(labels, labels, ['scene1', 'scene2'])
        self.assertEqual(exact['endpoint_selection_regret'], 0)
        wrong = summarize_task(labels[:, ::-1], labels, ['scene1', 'scene2'])
        self.assertAlmostEqual(wrong['endpoint_selection_regret'], .6, places=6)

    def test_task_reaches_predictor_without_target_input(self):
        torch.manual_seed(12)
        config = LatentConfig(channels=16, token_h=4, token_w=8,
                              ego_warp_residual=True, attention_h=4, attention_w=8,
                              motion_context=True)
        predictor = LatentPredictor(config)
        head = ProximityHead(16)
        history = torch.randn(2, 4, 16, 4, 8)
        action, status = torch.zeros(2, 21, 3), torch.zeros(2, 4)
        logits = head(history, predictor(history, action, status), action, status)
        target = torch.tensor([[1., 1.], [0., .2]])
        gradients = torch.autograd.grad(task_loss(logits, target, torch.tensor(2.)),
                                        tuple(predictor.parameters()), allow_unused=True)
        self.assertGreater(sum(float(g.abs().sum()) for g in gradients if g is not None), 0)


if __name__ == '__main__':
    unittest.main()
