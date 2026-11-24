"""Explicit current-frame forward transport, composed with future ego pose."""
import torch
from torch import nn
import torch.nn.functional as F

from .latent_motion_model import LatentPredictor, pack_state, warp_current


def forward_splat(occupancy, velocity, pose, seconds=1.):
    """B,K,H,W probabilities; B,K,2,H,W current-ego world velocities in m/s.

    Move source cells first in physical coordinates, then transform to future
    ego coordinates BEFORE rasterization. No intermediate crop or backward-flow
    interpretation. Bilinear scatter supports gradients to source and velocity.
    """
    b,k,h,w=occupancy.shape
    if velocity.shape!=(b,k,2,h,w):raise ValueError('source velocity shape mismatch')
    if pose.shape!=(b,3):raise ValueError('future ego pose shape mismatch')
    x=(torch.arange(w,device=occupancy.device,dtype=occupancy.dtype)+.5)*60/w-30
    y=(torch.arange(h,device=occupancy.device,dtype=occupancy.dtype)+.5)*30/h-15
    yy,xx=torch.meshgrid(y,x,indexing='ij')
    px=xx[None,None]+velocity[:,:,0]*seconds-pose[:,None,0,None,None]
    py=yy[None,None]+velocity[:,:,1]*seconds-pose[:,None,1,None,None]
    c=pose[:,None,2,None,None].cos();s=pose[:,None,2,None,None].sin()
    dx=(c*px+s*py+30)*w/60-.5
    dy=(-s*px+c*py+15)*h/30-.5
    x0,y0=dx.floor(),dy.floor();fx,fy=dx-x0,dy-y0
    output=occupancy.new_zeros((b*k,h*w))
    values=occupancy.reshape(b*k,-1)
    for ox,oy,weight in ((0,0,(1-fx)*(1-fy)),(1,0,fx*(1-fy)),
                         (0,1,(1-fx)*fy),(1,1,fx*fy)):
        ix,iy=x0.long()+ox,y0.long()+oy
        valid=(ix>=0)&(ix<w)&(iy>=0)&(iy<h)
        indices=(iy.clamp(0,h-1)*w+ix.clamp(0,w-1)).reshape(b*k,-1)
        contribution=values*(weight*valid).reshape(b*k,-1)
        output=output.scatter_add(1,indices,contribution)
    return output.reshape(b,k,h,w).clamp(0,1)


def class_velocity_loss(prediction, velocity, mask, current_occ):
    """Only training supervision; one valid velocity target per current class pixel."""
    valid=mask[:,None].float()*current_occ.float()
    target=velocity[:,None].float().expand_as(prediction)
    error=F.smooth_l1_loss(prediction/15.,target/15.,reduction='none').mean(2)
    return (error*valid).sum()/valid.sum().clamp_min(1.)


class TransportWorldModel(nn.Module):
    """Same parameters across flow-only, flow+residual and zero-flow+residual.

    Operates on decoded historical features, not raw GT occupancy. The complete
    propagated semantic state is re-encoded with the original frozen encoder
    for latent supervision. Native BEV is supervised directly; D(E(BEV)) is a
    separate compatibility probe, avoiding a second reconstruction bottleneck.
    """
    MODES=('transport_only','transport_residual','static_residual')

    def __init__(self, config, mode):
        super().__init__()
        if mode not in self.MODES:raise ValueError(mode)
        self.mode=mode
        self.backbone=LatentPredictor(config)
        self.reduce=nn.Conv2d(config.channels,32,1)
        self.refine=nn.Sequential(nn.Conv2d(32+16+2,32,3,padding=1),nn.GroupNorm(8,32),nn.GELU(),
                                  nn.Conv2d(32,32,3,padding=2,dilation=2),nn.GELU(),
                                  nn.Conv2d(32,32,3,padding=4,dilation=4),nn.GELU())
        self.velocity_refinement=nn.Conv2d(32,8,1)
        nn.init.zeros_(self.velocity_refinement.weight);nn.init.zeros_(self.velocity_refinement.bias)
        self.action=nn.Sequential(nn.Linear(67,32),nn.GELU(),nn.Linear(32,32))
        self.residual=nn.Sequential(nn.Conv2d(32+8+4+2,32,3,padding=1),nn.GELU(),nn.Conv2d(32,12,1))
        nn.init.zeros_(self.residual[-1].weight);nn.init.zeros_(self.residual[-1].bias)

    def forward(self, history, actions, status, history_occ_prob, current_map_prob, ae, history_mask=None):
        b=history.shape[0]
        # Scene flow follows logged other actors and is independent of ego candidate.
        zero_action=torch.zeros_like(actions)
        current_future=self.backbone(history,zero_action,status,history_mask)
        feature=F.interpolate(self.reduce(current_future),size=(50,100),mode='bilinear',align_corners=False)
        initial_velocity=F.interpolate(self.backbone.aux_velocity,size=(50,100),mode='bilinear',align_corners=False)
        observed=history_occ_prob
        if history_mask is not None:observed=observed*history_mask[:,:,None,None,None]
        hidden=self.refine(torch.cat((feature,observed.flatten(1,2),initial_velocity),1))
        correction=self.velocity_refinement(hidden).reshape(b,4,2,50,100)
        self.velocity=30*torch.tanh((initial_velocity[:,None]+correction)/30)
        used_velocity=self.velocity if self.mode!='static_residual' else self.velocity*0.
        base=forward_splat(history_occ_prob[:,-1],used_velocity,actions[:,4])
        maps,valid=warp_current(current_map_prob,actions[:,4],mode='bilinear')
        # Unknown outside current mapped crop is encoded as background, not free space.
        maps=maps.clone();maps[:,0]+=~valid
        maps=maps/maps.sum(1,keepdim=True).clamp_min(1e-6)
        self.transported=base
        prior_map_logits=maps.clamp_min(.001).log()
        prior_occ_logits=torch.logit(base.clamp(.01,.99))
        residual=base.new_zeros(b,12,50,100)
        if self.mode!='transport_only':
            spatial,_=warp_current(hidden,actions[:,4],mode='bilinear')
            a=actions/actions.new_tensor([30.,15.,3.14159265])
            ego=status/status.new_tensor([15.,15.,5.,5.])
            action_feature=self.action(torch.cat((a.flatten(1),ego),1))[:,:,None,None]
            yy,xx=torch.meshgrid(torch.linspace(-1,1,50,device=history.device),
                                 torch.linspace(-1,1,100,device=history.device),indexing='ij')
            xy=torch.stack((xx,yy),0)[None].expand(b,-1,-1,-1)
            residual=self.residual(torch.cat((spatial+action_feature,maps,base,xy),1))
        map_logits=prior_map_logits+residual[:,:8]
        occ_logits=prior_occ_logits+residual[:,8:]
        maps=map_logits.softmax(1)
        occupancy=occ_logits.sigmoid()
        self.semantic_state=torch.cat((maps,occupancy),1)
        prediction=ae.encoder(self.semantic_state)
        return prediction,torch.cat((map_logits,occ_logits),1)
