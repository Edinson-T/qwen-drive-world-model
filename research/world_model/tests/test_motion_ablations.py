import unittest

import torch

from qwen_drive_world_model.ablate_motion_training import latent_term, semantic_parts
from qwen_drive_world_model.latent_motion_model import LatentConfig, StateAutoencoder, semantic_loss


class MotionAblationTest(unittest.TestCase):
    def test_semantic_parts_reproduce_original_objective(self):
        logits=torch.randn(2,12,5,10,requires_grad=True)
        maps=torch.randint(0,8,(2,5,10))
        occ=torch.randint(0,2,(2,4,5,10)).float()
        weights=torch.ones(8)
        parts=semantic_parts(logits,maps,occ,weights)
        combined=.5*parts['map']+parts['bce']+.5*parts['dice']
        self.assertTrue(torch.allclose(combined,semantic_loss(logits,maps,occ,weights)))

    def test_disabling_latent_has_exactly_zero_gradient(self):
        pred=torch.randn(2,8,4,4,requires_grad=True)
        loss=latent_term(pred,torch.randn_like(pred),'none')
        gradient=torch.autograd.grad(loss,pred)[0]
        self.assertEqual(float(gradient.abs().sum()),0.)

    def test_raw_objectives_detect_scale_hidden_by_layernorm(self):
        target=torch.randn(2,8,4,4)
        self.assertLess(float(latent_term(target*2,target,'normalized')),1e-7)
        self.assertGreater(float(latent_term(target*2,target,'raw_smooth')),.01)
        self.assertGreater(float(latent_term(target*2,target,'raw_mse')),.01)

    def test_decoder_thaw_preserves_frozen_input_encoder(self):
        ae=StateAutoencoder(LatentConfig(channels=16,token_h=4,token_w=8)).requires_grad_(False)
        ae.decoder.requires_grad_(True)
        source=torch.randn(1,12,50,100)
        latent=ae.encoder(source)
        ae.decoder(latent).square().mean().backward()
        self.assertTrue(all(p.grad is None for p in ae.encoder.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in ae.decoder.parameters()))


if __name__=='__main__':unittest.main()
