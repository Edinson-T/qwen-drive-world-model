import unittest
import torch

from qwen_drive_world_model.motion_transport import forward_splat, class_velocity_loss, TransportWorldModel
from qwen_drive_world_model.latent_motion_model import LatentConfig, StateAutoencoder


class MotionTransportTest(unittest.TestCase):
    def test_zero_residual_modes_have_identical_initial_outputs(self):
        torch.manual_seed(8)
        config=LatentConfig(channels=16,token_h=4,token_w=8,attention_h=4,attention_w=8,
                            ego_warp_residual=True,motion_context=True,motion_aux=True)
        source=TransportWorldModel(config,'transport_only').eval()
        residual=TransportWorldModel(config,'transport_residual').eval()
        residual.load_state_dict(source.state_dict())
        ae=StateAutoencoder(config).eval().requires_grad_(False)
        h=torch.randn(1,4,16,4,8);a=torch.zeros(1,21,3);s=torch.zeros(1,4)
        occ=torch.rand(1,4,4,50,100);maps=torch.randn(1,8,50,100).softmax(1)
        with torch.no_grad():
            z1,l1=source(h,a,s,occ,maps,ae)
            z2,l2=residual(h,a,s,occ,maps,ae)
        self.assertTrue(torch.equal(l1,l2))
        self.assertTrue(torch.equal(z1,z2))
        self.assertTrue(torch.equal(l1[:,8:].sigmoid(),source.semantic_state[:,8:]))

    def test_identity(self):
        torch.manual_seed(7)
        source=torch.rand(1,2,50,100)
        predicted=forward_splat(source,torch.zeros(1,2,2,50,100),torch.zeros(1,3))
        self.assertTrue(torch.allclose(source,predicted,atol=1e-5))

    def test_positive_world_velocity_moves_forward(self):
        source=torch.zeros(1,1,50,100);source[0,0,25,50]=1
        velocity=torch.zeros(1,1,2,50,100);velocity[:,:,0]=.6
        predicted=forward_splat(source,velocity,torch.zeros(1,3))
        self.assertAlmostEqual(float(predicted[0,0,25,51]),1.,places=4)

    def test_compose_before_crop_retains_fast_object(self):
        source=torch.zeros(1,1,50,100);source[0,0,25,90]=1
        velocity=torch.zeros(1,1,2,50,100);velocity[:,:,0]=12.
        predicted=forward_splat(source,velocity,torch.tensor([[12.,0.,0.]]))
        self.assertTrue(torch.allclose(source,predicted,atol=1e-5))

    def test_candidate_yaw_rotates_into_future_axes(self):
        source=torch.zeros(1,1,50,100);source[0,0,25,51]=1
        predicted=forward_splat(source,torch.zeros(1,1,2,50,100),torch.tensor([[0.,0.,torch.pi/2]]))
        self.assertAlmostEqual(float(predicted[0,0,23,50]),1.,places=4)

    def test_velocity_receives_semantic_transport_gradient(self):
        source=torch.zeros(1,1,50,100);source[0,0,25,50]=1
        velocity=torch.zeros(1,1,2,50,100,requires_grad=True)
        offset=velocity+torch.tensor([.2,.1])[None,None,:,None,None]
        predicted=forward_splat(source,offset,torch.zeros(1,3))
        gradient=torch.autograd.grad(predicted[0,0,25,51],velocity)[0]
        self.assertGreater(float(gradient[0,0,0,25,50]),0.)

    def test_velocity_supervision_ignores_unlabeled_pixels(self):
        prediction=torch.randn(1,4,2,5,10,requires_grad=True)
        loss=class_velocity_loss(prediction,torch.zeros(1,2,5,10),torch.zeros(1,5,10),torch.ones(1,4,5,10))
        gradient=torch.autograd.grad(loss,prediction)[0]
        self.assertEqual(float(gradient.abs().sum()),0.)


if __name__=='__main__':unittest.main()
