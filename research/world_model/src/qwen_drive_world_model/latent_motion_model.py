"""Motion encoder, frozen latent teacher, and action-conditioned BEV predictor."""
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class LatentConfig:
    channels: int = 128
    token_h: int = 8
    token_w: int = 16
    history: int = 4
    layers: int = 2
    heads: int = 4
    map_classes: int = 8
    object_classes: int = 4
    detail_encoder: bool = False
    ego_warp_residual: bool = False
    attention_h: int = 8
    attention_w: int = 16
    motion_context: bool = False
    motion_aux: bool = False


def pack_state(map_labels, occupancy):
    onehot = F.one_hot(map_labels.long(), 8).movedim(-1, -3).float()
    return torch.cat((onehot, occupancy.float()), -3)


class StateAutoencoder(nn.Module):
    """Explicit states -> learned latents; separate semantic decoder, no raw-channel residual."""
    def __init__(self, config=LatentConfig()):
        super().__init__()
        self.config = config
        c = config.channels
        self.encoder = nn.Sequential(
            nn.Conv2d(12, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, c, 3, stride=2, padding=1), nn.GroupNorm(8, c), nn.GELU(),
            nn.Conv2d(c, c, 3, stride=1 if config.detail_encoder else 2, padding=1), nn.GroupNorm(8, c), nn.GELU(),
            nn.AdaptiveAvgPool2d((config.token_h, config.token_w)),
            nn.Conv2d(c, c, 1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1), nn.GELU(),
            nn.Upsample(size=(25, 50), mode='bilinear', align_corners=False),
            nn.Conv2d(c, 64, 3, padding=1), nn.GELU(),
            nn.Upsample(size=(50, 100), mode='bilinear', align_corners=False),
            nn.Conv2d(64, 64, 3, padding=1), nn.GELU(), nn.Conv2d(64, 12, 1),
        )

    def forward(self, value):
        return self.decoder(self.encoder(value))


class LatentPredictor(nn.Module):
    """128 spatial tokens + action. Four aligned observations fuse before attention."""
    def __init__(self, config=LatentConfig()):
        super().__init__()
        self.config = config
        c = config.channels
        self.history_fuser = nn.Sequential(nn.Conv2d(config.history*(c+1), c, 1), nn.GELU(), nn.Conv2d(c, c, 1))
        self.action_encoder = nn.Sequential(nn.Linear(21*3 + 4, c), nn.GELU(), nn.Linear(c, c))
        position_h = config.attention_h if config.ego_warp_residual else config.token_h
        position_w = config.attention_w if config.ego_warp_residual else config.token_w
        self.position = nn.Parameter(torch.randn(1, position_h*position_w + 1, c) * .01)
        layer = nn.TransformerEncoderLayer(c, config.heads, c*4, dropout=.0, activation='gelu', batch_first=True, norm_first=True)
        self.transition = nn.TransformerEncoder(layer, config.layers, enable_nested_tensor=False)
        self.output = nn.Sequential(nn.LayerNorm(c), nn.Linear(c, c))
        if config.ego_warp_residual:
            self.local_residual = nn.Sequential(nn.Conv2d(2*c,c,3,padding=1),nn.GELU(),nn.Conv2d(c,c,3,padding=1))
            # Starts at a physically aligned feature baseline, learns only corrections.
            nn.init.zeros_(self.local_residual[-1].weight)
            nn.init.zeros_(self.local_residual[-1].bias)
            if config.motion_context:
                self.motion_context=nn.Sequential(
                    nn.Conv2d(2*c,c,3,padding=1),nn.GELU(),
                    nn.Conv2d(c,c,3,padding=2,dilation=2),nn.GELU(),
                    nn.Conv2d(c,c,3,padding=4,dilation=4),nn.GELU(),
                    nn.Conv2d(c,c,1))
                nn.init.zeros_(self.motion_context[-1].weight)
                nn.init.zeros_(self.motion_context[-1].bias)
                if config.motion_aux:
                    self.velocity_head=nn.Conv2d(c,2,1)
                    nn.init.zeros_(self.velocity_head.weight);nn.init.zeros_(self.velocity_head.bias)

    def forward(self, history_latents, actions, status, history_mask=None):
        b, t, c, h, w = history_latents.shape
        if actions.shape[1:] != (21, 3):
            raise ValueError('actions must include t=0 and 20 future waypoints at 4Hz')
        if not torch.allclose(actions[:, 0], torch.zeros_like(actions[:, 0]), atol=1e-4):
            raise ValueError('first action point must be the current ego origin')
        if history_mask is None:
            history_mask = history_latents.new_ones(b, t)
        mask = history_mask[:, :, None, None, None].expand(-1, -1, 1, h, w)
        base = None
        if self.config.ego_warp_residual:
            aligned, valid = warp_current(history_latents.reshape(b,t*c,h,w), actions[:,4], mode='bilinear')
            history_latents=aligned.reshape(b,t,c,h,w)
            mask=mask*valid[:,None,None].to(mask.dtype)
            base=history_latents[:,-1]
        scene = self.history_fuser(torch.cat((history_latents * mask, mask), 2).reshape(b, -1, h, w))
        trajectory = actions / actions.new_tensor([30., 15., 3.14159265])
        ego = status / status.new_tensor([15., 15., 5., 5.])
        action = self.action_encoder(torch.cat((trajectory.flatten(1), ego), -1))[:, None]
        if self.config.ego_warp_residual:
            context=F.adaptive_avg_pool2d(scene,(self.config.attention_h,self.config.attention_w))
        else:
            context=scene
        tokens = torch.cat((action, context.flatten(2).transpose(1, 2)), 1)
        future = self.output(self.transition(tokens + self.position)[:, 1:])
        if self.config.ego_warp_residual:
            global_delta=future.transpose(1,2).reshape(b,c,self.config.attention_h,self.config.attention_w)
            global_delta=F.interpolate(global_delta,size=(h,w),mode='bilinear',align_corners=False)
            combined=torch.cat((scene,global_delta),1)
            delta=self.local_residual(combined)
            if self.config.motion_context:
                motion=self.motion_context[:-1](combined)
                delta=delta+self.motion_context[-1](motion)
                if self.config.motion_aux:
                    self.aux_velocity=self.velocity_head(motion)*15.
            return base+delta
        return future.transpose(1, 2).reshape(b, c, h, w)


def semantic_loss(logits, maps, occupancy, weights):
    map_loss = F.cross_entropy(logits[:, :8], maps.long(), weight=weights)
    occ_logits = logits[:, 8:]
    bce = F.binary_cross_entropy_with_logits(occ_logits, occupancy.float(), pos_weight=logits.new_tensor(4.))
    probability = occ_logits.sigmoid()
    intersection = (probability * occupancy).sum((0, 2, 3))
    denominator = probability.sum((0, 2, 3)) + occupancy.sum((0, 2, 3))
    dice = (1 - (2*intersection+1)/(denominator+1)).mean()
    return .5*map_loss + bce + .5*dice


def latent_distance(prediction, target):
    # Both tensors share a frozen, reconstruction-trained encoder; no moving/collapsing target.
    p = F.layer_norm(prediction.movedim(1, -1), (prediction.shape[1],))
    t = F.layer_norm(target.movedim(1, -1), (target.shape[1],))
    return F.smooth_l1_loss(p, t)


def warp_current(value, pose, mode='nearest'):
    """Sample current-ego raster at candidate-future-ego coordinates, +x columns/+y rows."""
    b, _, h, w = value.shape
    x = (torch.arange(w, device=value.device, dtype=value.dtype)+.5)/w*60-30
    y = (torch.arange(h, device=value.device, dtype=value.dtype)+.5)/h*30-15
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    c, s = pose[:, 2].cos()[:, None, None], pose[:, 2].sin()[:, None, None]
    px = c*xx-s*yy+pose[:, 0, None, None]
    py = s*xx+c*yy+pose[:, 1, None, None]
    grid = torch.stack((px/30, py/15), -1)
    # Pixel-centre arithmetic can differ by a few ulps even for identity transforms.
    valid = (px.abs() <= 30-30/w+1e-5) & (py.abs() <= 15-15/h+1e-5)
    warped = F.grid_sample(value, grid, mode=mode, padding_mode='zeros', align_corners=False)
    return warped, valid


def velocity_auxiliary_loss(prediction,velocity,mask,pose):
    """Rotate/warp training-only current velocity labels into the candidate frame."""
    values,_=warp_current(torch.cat((velocity.float()*mask[:,None],mask[:,None].float()),1),pose)
    numerator=F.adaptive_avg_pool2d(values[:,:2],prediction.shape[-2:])
    weight=F.adaptive_avg_pool2d(values[:,2:3],prediction.shape[-2:])
    target=numerator/weight.clamp_min(1e-6)
    c,s=pose[:,2,None,None].cos(),pose[:,2,None,None].sin()
    vx,vy=target[:,0],target[:,1]
    target=torch.stack((c*vx+s*vy,-s*vx+c*vy),1)
    error=F.smooth_l1_loss(prediction/15.,target/15.,reduction='none').mean(1,keepdim=True)
    return (error*weight).sum()/weight.sum().clamp_min(1.)
