import unittest
import numpy as np

from qwen_drive_world_model.motion_instance_diagnostics import object_mask, summarize_instances


class MotionInstanceDiagnosticsTest(unittest.TestCase):
    def record(self):
        box=np.array([.3,.3,0.,3.,1.8,1.,0.,0.,0.])
        return dict(row=7,track_id=12,category=0,speed=6.,current=box,future=box,fully_inside=True)

    def test_perfect_raster_has_perfect_iou_and_zero_raster_error(self):
        record=self.record();p=np.zeros((4,50,100));p[0]=object_mask(record['future'])
        value=summarize_instances([record],{7:p})['groups']['speed/fast']
        self.assertEqual(value['instance_iou_with_misses'],1.)
        self.assertEqual(value['target_pixel_recall'],1.)
        self.assertEqual(value['miss_rate'],0.)
        self.assertEqual(value['capped_raster_centroid_error_m'],0.)

    def test_missing_prediction_is_not_dropped_from_error_summary(self):
        value=summarize_instances([self.record()],{7:np.zeros((4,50,100))})['groups']['speed/fast']
        self.assertEqual(value['instance_iou_with_misses'],0.)
        self.assertEqual(value['miss_rate'],1.)
        self.assertEqual(value['capped_raster_centroid_error_m'],6.)
        self.assertIsNone(value['matched_full_inside_center_error_m'])


if __name__=='__main__':unittest.main()
