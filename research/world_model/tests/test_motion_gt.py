from __future__ import annotations

import unittest

import numpy as np
import torch

from qwen_drive_world_model.build_motion_gt_dataset import _split_for_scenario
from qwen_drive_world_model.gt_model import ObjectHeadConfig, class_aware_nms_indices
from qwen_drive_world_model.motion_gt import (
    MotionMapFeature,
    compress_gt_rasters,
    global_to_ego,
    rasterize_boxes,
    rasterize_map_features,
)
from qwen_drive_world_model.model import WorldModelConfig
from qwen_drive_world_model.shared_horizon_model import SharedHorizonWorldModel


class MotionGtTests(unittest.TestCase):
    def test_class_aware_nms_suppresses_duplicate_box(self):
        boxes = torch.tensor(
            [
                [
                    [10.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0],
                    [10.1, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0],
                ]
            ]
        )
        logits = torch.tensor(
            [
                [
                    [5.0, 0.0, 0.0, 0.0, -5.0],
                    [4.0, 0.0, 0.0, 0.0, -5.0],
                ]
            ]
        )
        keep = class_aware_nms_indices(
            boxes,
            logits,
            score_threshold=0.9,
            iou_threshold=0.5,
        )[0]
        self.assertEqual(keep.tolist(), [0])

    def test_shared_horizon_outputs(self):
        config = WorldModelConfig(
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
        object_config = ObjectHeadConfig(
            num_queries=8,
            token_hw=(2, 2),
            num_layers=1,
            num_heads=4,
            feedforward_size=64,
        )
        model = SharedHorizonWorldModel(config, object_config, num_horizons=5)
        output = model(
            torch.rand(2, 13, 10, 20),
            torch.randn(2, 51, 3),
            torch.randn(2, 8),
            torch.randn(2, 5, 3),
        )
        self.assertEqual(tuple(output["raster"].shape), (2, 5, 9, 10, 20))
        self.assertEqual(tuple(output["boxes"].shape), (2, 5, 8, 7))

    def test_scenario_split_is_whole_scene(self):
        splits = [_split_for_scenario(index, 10) for index in range(25)]
        self.assertEqual(splits[0], "val")
        self.assertEqual(splits[10], "val")
        self.assertEqual(splits[20], "val")
        self.assertEqual(splits[1], "train")

    def test_qd_map_bounds_and_orientation(self):
        feature = MotionMapFeature(
            kind=5,
            subtype=1,
            points=np.asarray([[-5.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
        )
        semantic = rasterize_map_features([feature], np.asarray([0.0, 0.0, 0.0]))
        self.assertEqual(semantic.shape, (200, 400))
        self.assertGreater(int((semantic == 3).sum()), 0)
        self.assertGreater(int(semantic[:, 200:].sum()), 0)
        self.assertEqual(int(semantic[:, :150].sum()), 0)

    def test_height_bands_and_compression(self):
        boxes = np.asarray(
            [[10.0, 0.0, 1.0, 4.0, 2.0, 2.0, 0.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        occupancy, height = rasterize_boxes(boxes, np.asarray([0]))
        self.assertGreater(int(occupancy.sum()), 0)
        self.assertEqual(height[0].any(), True)
        self.assertEqual(height[1].any(), True)
        self.assertEqual(height[2].any(), True)
        self.assertEqual(height[3].any(), False)
        wm_input, target = compress_gt_rasters(occupancy * 0, occupancy, height)
        self.assertEqual(wm_input.shape, (13, 50, 100))
        self.assertEqual(target.shape, (9, 50, 100))

    def test_global_to_ego_rotation(self):
        points = np.asarray([[1.0, 1.0, 0.0]])
        local = global_to_ego(points, np.asarray([1.0, 0.0, np.pi / 2]))
        np.testing.assert_allclose(local[0, :2], [1.0, 0.0], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
