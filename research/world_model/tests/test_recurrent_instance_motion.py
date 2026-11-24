import unittest

import torch

from qwen_drive_world_model.instance_motion_model import history_features
from qwen_drive_world_model.recurrent_instance_motion import (
    RecurrentInstanceMotionPredictor, physical_displacements, rollout_boxes,
    multihorizon_loss, OUTPUT_STEPS,
)


class RecurrentInstanceMotionTest(unittest.TestCase):
    def inputs(self):
        history = torch.tensor([[[-4.5, 0., 4., 2., 0.], [-3., 0., 4., 2., 0.],
                                 [-1.5, 0., 4., 2., 0.], [0., 0., 4., 2., 0.]]])
        return history_features(history, torch.ones(1, 4), torch.zeros(1, dtype=torch.long))

    def test_physical_experts_match_v8_at_one_second(self):
        torch.manual_seed(17)
        history = torch.randn(12, 4, 5)
        history[:, :, 2:4] = history[:, :, 2:4].abs() + 1
        f, b, w, _ = history_features(history, torch.ones(12, 4), torch.zeros(12, dtype=torch.long))
        torch.testing.assert_close(physical_displacements(f, b, w, 1.), b)

    def test_constant_velocity_and_prefix_consistency(self):
        f, b, w, c = self.inputs()
        model = RecurrentInstanceMotionPredictor(hidden=16)
        full = model(f, b, w)
        prefix = model(f, b, w, steps=2)
        for a, z in zip(full, prefix):
            torch.testing.assert_close(a[:, :2], z)
        boxes = rollout_boxes(c, full)
        torch.testing.assert_close(boxes[0, :, 0], torch.arange(1, 11) * 1.5)
        torch.testing.assert_close(boxes[0, :, 1], torch.zeros(10))

    def test_later_predictions_backpropagate_through_earlier_transition(self):
        f, b, w, _ = self.inputs()
        model = RecurrentInstanceMotionPredictor(hidden=16)
        values = []
        def hook(module, args, value):
            value.retain_grad()
            values.append(value)
        handle = model.head.register_forward_hook(hook)
        prediction = model(f, b, w)
        prediction[0][:, -1].sum().backward()
        handle.remove()
        self.assertGreater(float(values[0].grad[:, 5:7].abs().sum()), 0)

    def test_multihorizon_loss_supervises_requested_steps(self):
        f, b, w, c = self.inputs()
        model = RecurrentInstanceMotionPredictor(hidden=16)
        pred = model(f, b, w)
        target = rollout_boxes(c, pred).detach()
        loss, per_horizon = multihorizon_loss(pred, c, target, torch.ones(1, 10))
        self.assertEqual(len(per_horizon), len(OUTPUT_STEPS))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))


if __name__ == '__main__':
    unittest.main()
