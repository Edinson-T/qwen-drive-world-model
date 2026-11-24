import unittest
import numpy as np

from qwen_drive_world_model.export_instance_reward_histories import qd_to_motion_boxes


class InstanceRewardAdapterTest(unittest.TestCase):
    def test_order_dimensions_transform_classes_and_ignored_velocity(self):
        boxes = np.array([[1., 0., 0., 2., 4., 1.5, 0., 900., -900.],
                          [0., 0., 0., 1., 1., 1.5, 0., 0., 0.],
                          [2., 0., 0., 1., 2., 1.5, 0., 0., 0.]])
        scores = np.array([.9, .9, .1])
        labels = np.array([0, 4, 2])
        pose = [10., 20., np.pi/2]
        result, category = qd_to_motion_boxes(boxes, scores, labels, pose)
        np.testing.assert_allclose(result[:, :2], [[10., 21.], [10., 20.]], atol=1e-5)
        np.testing.assert_allclose(result[:, 2:4], [[4., 2.], [1., 1.]])
        np.testing.assert_array_equal(category, [0, 1])
        changed = boxes.copy()
        changed[:, 7:9] *= -100
        np.testing.assert_array_equal(qd_to_motion_boxes(changed, scores, labels, pose)[0], result)

    def test_empty_detection_frame(self):
        boxes, classes = qd_to_motion_boxes(np.empty((0,9)), np.empty(0), np.empty(0), [0,0,0])
        self.assertEqual(boxes.shape, (0,5))
        self.assertEqual(classes.shape, (0,))


if __name__ == '__main__':
    unittest.main()
