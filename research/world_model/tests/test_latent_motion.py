import unittest

import numpy as np
import torch

from qwen_drive_world_model.latent_motion_data import (
    GRID, occupancy, causal_status, candidate_actions, split_for_scene,
)
from qwen_drive_world_model.latent_motion_model import (
    LatentConfig, StateAutoencoder, LatentPredictor, warp_current, semantic_loss,
)
from qwen_drive_world_model.protos import womd_pb2


def scene():
    value = womd_pb2.Scenario(scenario_id='test', sdc_track_index=0)
    value.timestamps_seconds.extend(np.arange(91)/10)
    for track_id, x in [(1, 0.), (2, 10.)]:
        track = value.tracks.add(id=track_id, object_type=1)
        for i in range(91):
            track.states.add(center_x=x+i, center_y=0, center_z=0,
                             length=4, width=2, height=2, heading=0,
                             velocity_x=10, velocity_y=0, valid=True)
    return value


class MotionProtocolTests(unittest.TestCase):
    def test_grid_and_small_vehicle(self):
        np.testing.assert_allclose(GRID.pixels([[0, 0], [10, 0]]), [[50, 25], [66.6666667, 25]], atol=1e-5)
        s = scene()
        occ = occupancy(s, 0, np.zeros(3))
        self.assertGreater(occ.sum(), 0)
        self.assertFalse(occ[:, 23:27, 48:52].any())  # SDC excluded.
        y, x = np.nonzero(occ[0])
        self.assertLess(abs(((x+.5)*.6-30).mean()-10), .7)
        self.assertLess(abs(((y+.5)*.6-15).mean()), .7)

    def test_constant_velocity_matches_actual(self):
        s = scene()
        np.testing.assert_array_equal(occupancy(s, 0, np.zeros(3), extrapolate=1), occupancy(s, 10, np.zeros(3)))

    def test_actual_timestamps_and_no_future_status(self):
        s = scene()
        actions = candidate_actions(s.tracks[0], 15)
        np.testing.assert_allclose(actions[0, :, 0], np.arange(21)*2.5)
        np.testing.assert_allclose(actions[:, 0], 0)
        self.assertNotEqual(actions[1, 4, 1], actions[2, 4, 1])
        before = causal_status(s.tracks[0], 15)
        s.tracks[0].states[16].velocity_x = 1000
        np.testing.assert_array_equal(causal_status(s.tracks[0], 15), before)

    def test_pose_transform_and_validity(self):
        image = torch.zeros(1, 1, 50, 100)
        image[0, 0, 25, 66] = 1
        warped, valid = warp_current(image, torch.zeros(1, 3))
        torch.testing.assert_close(warped, image)
        self.assertTrue(valid.all())
        moved, mask = warp_current(image, torch.tensor([[6., 0., 0.]]))
        self.assertEqual(float(moved[0, 0, 25, 56]), 1)
        self.assertFalse(mask.all())
        # +90deg future ego yaw maps a marker at current +x to future -y.
        square = torch.zeros(1,1,50,100);square[:,:,24:26,65:68]=1
        rotated,_=warp_current(square,torch.tensor([[0.,0.,np.pi/2]]))
        row,col=torch.nonzero(rotated[0,0],as_tuple=True)
        self.assertLess(float(row.float().mean()), 15)
        self.assertLess(abs(float(col.float().mean())-49.5), 2)

    def test_frozen_teacher_allows_predictor_gradient(self):
        config=LatentConfig(channels=32,layers=1)
        ae=StateAutoencoder(config).eval().requires_grad_(False)
        predictor=LatentPredictor(config)
        h=torch.randn(2,4,32,8,16)
        actions=torch.zeros(2,21,3)
        output=ae.decoder(predictor(h,actions,torch.zeros(2,4)))
        loss=semantic_loss(output,torch.zeros(2,50,100,dtype=torch.long),torch.zeros(2,4,50,100),torch.ones(8))
        loss.backward()
        self.assertTrue(all(p.grad is None for p in ae.parameters()))
        self.assertGreater(float(predictor.output[-1].weight.grad.abs().sum()),0)
        actions[:,0,0]=1
        with self.assertRaises(ValueError): predictor(h,actions,torch.zeros(2,4))

    def test_split_is_stable(self):
        self.assertEqual(split_for_scene('same'), split_for_scene('same'))

    def test_geometry_residual_initialization_and_gradient(self):
        config=LatentConfig(channels=32,layers=1,token_h=16,token_w=32,
                            ego_warp_residual=True,detail_encoder=True)
        model=LatentPredictor(config)
        history=torch.randn(2,4,32,16,32)
        action=torch.zeros(2,21,3);action[:,4,0]=3
        prediction=model(history,action,torch.zeros(2,4))
        expected,_=warp_current(history[:,-1],action[:,4],mode='bilinear')
        torch.testing.assert_close(prediction,expected)
        self.assertEqual(model.position.shape[1],129)  # Attention stays small at higher feature resolution.
        prediction.square().mean().backward()
        self.assertGreater(float(model.local_residual[-1].weight.grad.abs().sum()),0)
        ae=StateAutoencoder(config)
        value=torch.randn(2,12,50,100)
        self.assertEqual(ae.encoder(value).shape,(2,32,16,32))
        self.assertEqual(ae(value).shape,(2,12,50,100))

    def test_motion_upgrade_preserves_pretrained_prediction(self):
        old=LatentPredictor(LatentConfig(channels=32,layers=1,ego_warp_residual=True))
        new=LatentPredictor(LatentConfig(channels=32,layers=1,ego_warp_residual=True,motion_context=True))
        missing,unexpected=new.load_state_dict(old.state_dict(),strict=False)
        self.assertTrue(missing)
        self.assertTrue(all(key.startswith('motion_context.') for key in missing))
        self.assertFalse(unexpected)
        h=torch.randn(2,4,32,8,16);a=torch.zeros(2,21,3);s=torch.zeros(2,4)
        torch.testing.assert_close(old(h,a,s),new(h,a,s))
        new(h,a,s).square().mean().backward()
        self.assertGreater(float(new.motion_context[-1].weight.grad.abs().sum()),0)

    def test_velocity_target_frame_and_training_only_head(self):
        from qwen_drive_world_model.latent_motion_model import velocity_auxiliary_loss
        velocity=torch.zeros(1,2,50,100);velocity[:,0]=10
        mask=torch.ones(1,50,100);pose=torch.tensor([[0.,0.,np.pi/2]])
        correct=torch.zeros(1,2,8,16);correct[:,1]=-10
        self.assertLess(float(velocity_auxiliary_loss(correct,velocity,mask,pose)),1e-8)
        wrong=torch.zeros(1,2,8,16,requires_grad=True)
        loss=velocity_auxiliary_loss(wrong,velocity,mask,pose);loss.backward()
        self.assertGreater(float(wrong.grad.abs().sum()),0)
        old=LatentPredictor(LatentConfig(channels=32,layers=1,ego_warp_residual=True,motion_context=True))
        new=LatentPredictor(LatentConfig(channels=32,layers=1,ego_warp_residual=True,motion_context=True,motion_aux=True))
        missing,unexpected=new.load_state_dict(old.state_dict(),strict=False)
        self.assertEqual(set(missing),{'velocity_head.weight','velocity_head.bias'});self.assertFalse(unexpected)
        h=torch.randn(1,4,32,8,16);a=torch.zeros(1,21,3);s=torch.zeros(1,4)
        torch.testing.assert_close(old(h,a,s),new(h,a,s))


if __name__=='__main__':
    unittest.main()
