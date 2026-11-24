from __future__ import annotations

import unittest

import numpy as np
import torch

from qwen_drive_world_model.gt_model import (
    GTBEVWorldModel,
    ObjectHeadConfig,
    bev_gt_loss,
    object_query_loss,
)
from qwen_drive_world_model.model import WorldModelConfig
from qwen_drive_world_model.model import BEVWorldModel
from qwen_drive_world_model.perception_gt import (
    MapFeatureData,
    downsample_gt_bev,
    rasterize_boxes,
    rasterize_map_features,
)


class PerceptionGtRasterTests(unittest.TestCase):
    def test_map_box_raster_and_compression(self):
        feature = MapFeatureData(
            kind=5,
            subtype=1,
            points=np.asarray(
                [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
                dtype=np.float64,
            ),
        )
        semantic = rasterize_map_features([feature], np.eye(4))
        boxes = np.asarray(
            [[10.0, 0.0, 1.0, 4.0, 2.0, 1.5, 0.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        occupancy = rasterize_boxes(boxes)
        compact = downsample_gt_bev(semantic, occupancy, (50, 100))
        self.assertEqual(semantic.shape, (200, 400))
        self.assertEqual(int((semantic == 3).sum()) > 0, True)
        self.assertEqual(int(occupancy.sum()) > 0, True)
        self.assertEqual(compact.shape, (9, 50, 100))
        self.assertEqual(compact.dtype, np.float16)


class GtWorldModelTests(unittest.TestCase):
    def test_geometry_input_is_zero_initialized_and_output_channels_decouple(self):
        base_config = WorldModelConfig(
            input_channels=9,
            base_channels=9,
            output_channels=9,
            hidden_size=32,
            token_hw=(2, 2),
            target_hw=(10, 20),
            num_layers=1,
            num_heads=4,
            feedforward_size=64,
        )
        old_model = BEVWorldModel(base_config)
        new_config = WorldModelConfig(
            input_channels=13,
            base_channels=9,
            output_channels=9,
            hidden_size=32,
            token_hw=(2, 2),
            target_hw=(10, 20),
            num_layers=1,
            num_heads=4,
            feedforward_size=64,
        )
        new_model = BEVWorldModel(new_config)
        missing, unexpected = new_model.load_state_dict(
            old_model.state_dict(),
            strict=False,
        )
        self.assertEqual(
            set(missing),
            {
                "bridge.height_projection.weight",
                "bridge.height_projection.bias",
            },
        )
        self.assertEqual(unexpected, [])
        base = torch.rand(2, 9, 10, 20)
        with_height = torch.cat([base, torch.rand(2, 4, 10, 20)], dim=1)
        trajectory = torch.randn(2, 21, 3)
        old_output = old_model(base, trajectory)
        new_output = new_model(with_height, trajectory)
        self.assertEqual(tuple(new_output.shape), (2, 9, 10, 20))
        torch.testing.assert_close(old_output, new_output)

    def test_raster_and_object_outputs(self):
        config = WorldModelConfig(
            input_channels=9,
            hidden_size=32,
            token_hw=(2, 2),
            target_hw=(10, 20),
            num_layers=1,
            num_heads=4,
            feedforward_size=64,
        )
        object_config = ObjectHeadConfig(
            num_queries=8,
            token_hw=(2, 2),
            num_layers=1,
            num_heads=4,
            feedforward_size=64,
        )
        model = GTBEVWorldModel(config, object_config)
        current = torch.rand(2, 9, 10, 20)
        trajectory = torch.randn(2, 21, 3)
        ego = torch.randn(2, 8)
        output = model(current, trajectory, ego)
        self.assertEqual(tuple(output["raster"].shape), (2, 9, 10, 20))
        self.assertEqual(tuple(output["boxes"].shape), (2, 8, 7))
        self.assertEqual(tuple(output["box_logits"].shape), (2, 8, 5))
        self.assertTrue(torch.isfinite(output["raster"]).all())

        target_map = torch.randint(0, 8, (2, 10, 20))
        target_occ = torch.randint(0, 2, (2, 10, 20)).float()
        weights = torch.ones(8)
        bev_loss, _ = bev_gt_loss(output, target_map, target_occ, weights)
        box_targets = torch.rand(2, 4, 9)
        box_targets[..., 3:6] += 1.0
        box_labels = torch.tensor([[0, 1, 2, 3], [0, 1, -1, -1]])
        box_loss, _ = object_query_loss(
            output["boxes"],
            output["box_logits"],
            box_targets,
            box_labels,
            torch.tensor([4, 2]),
        )
        self.assertTrue(torch.isfinite(bev_loss))
        self.assertTrue(torch.isfinite(box_loss))


if __name__ == "__main__":
    unittest.main()

