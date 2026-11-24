"""Instance-token forecasting from causal boxes; no velocity/ID input fields."""
import torch
from torch import nn
import torch.nn.functional as F


def history_features(history,mask,category):
    current=history[:,-1]
    c,s=current[:,4].cos(),current[:,4].sin()
    delta=history[:,:,:2]-current[:,None,:2]
    body=torch.stack((c[:,None]*delta[:,:,0]+s[:,None]*delta[:,:,1],
                      -s[:,None]*delta[:,:,0]+c[:,None]*delta[:,:,1]),-1)
    full_angle=history[:,:,4]-current[:,None,4]
    angle=.5*torch.atan2((2*full_angle).sin(),(2*full_angle).cos())
    relative_size=(history[:,:,2:4].clamp_min(.05)/current[:,None,2:4].clamp_min(.05)).log().clamp(-2,2)
    m=mask.float()
    per_frame=torch.cat((body/10,full_angle.sin()[:,:,None],full_angle.cos()[:,:,None],relative_size,m[:,:,None]),-1)*m[:,:,None]
    times=history.new_tensor([-1.5,-1.,-.5,0.])[None]
    q=.5*times.square()
    tt=(m*times.square()).sum(1);tq=(m*times*q).sum(1);qq=(m*q.square()).sum(1)+.05
    rhs_t=(m[:,:,None]*times[:,:,None]*body).sum(1)
    rhs_q=(m[:,:,None]*q[:,:,None]*body).sum(1)
    determinant=(tt*qq-tq.square()).clamp_min(1e-6)
    velocity=(qq[:,None]*rhs_t-tq[:,None]*rhs_q)/determinant[:,None]
    acceleration=(tt[:,None]*rhs_q-tq[:,None]*rhs_t)/determinant[:,None]
    velocity=velocity.clamp(-50,50);acceleration=acceleration.clamp(-12,12)
    recent=torch.zeros_like(velocity)
    for i in range(3):recent=torch.where(m[:,i,None].bool(),body[:,i]/times[:,i,None],recent)
    recent=recent.clamp(-50,50)
    omega=((m*times*angle).sum(1)/tt.clamp_min(1e-6)).clamp(-1.2,1.2)
    first=torch.sinc(omega/torch.pi)
    second=.5*omega*torch.sinc(omega/(2*torch.pi)).square()
    rotated=first[:,None]*velocity+second[:,None]*torch.stack((-velocity[:,1],velocity[:,0]),1)
    direction=velocity/velocity.norm(dim=1,keepdim=True).clamp_min(.1)
    longitudinal=(acceleration*direction).sum(1,keepdim=True)*direction
    lagged=[]
    for i in (1,0):lagged.append(torch.where(m[:,i,None].bool(),body[:,i]/times[:,i,None],recent).clamp(-50,50))
    bases=torch.stack((recent,*lagged,velocity+.5*acceleration,rotated+.5*longitudinal),1)
    features=torch.cat((per_frame.flatten(1),(current[:,2:4]/current.new_tensor([4.,2.])).clamp_min(.05).log(),
                        (current[:,:2]/100).clamp(-5,5),F.one_hot(category.long(),4).float(),
                        recent/15,velocity/15,acceleration/5,omega[:,None]),1)
    return features,bases,omega,current


class InstanceMotionPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.network=nn.Sequential(nn.Linear(43,128),nn.GELU(),nn.Linear(128,128),nn.GELU(),nn.Linear(128,11))
        nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)
        with torch.no_grad():self.network[-1].bias[0]=2.;self.network[-1].bias[10]=4.6

    def forward(self,features,bases,omega):
        value=self.network(features)
        mix=value[:,:5].softmax(1)
        displacement=(mix[:,:,None]*bases).sum(1)+value[:,5:7]
        yaw=(mix[:,3]+mix[:,4])*omega+.8*value[:,7].tanh()
        log_size=.25*value[:,8:10].tanh()
        return displacement,yaw,log_size,value[:,10]


def decoded_boxes(current,displacement,yaw,log_size):
    c,s=current[:,4].cos(),current[:,4].sin()
    xy=current[:,:2]+torch.stack((c*displacement[:,0]-s*displacement[:,1],
                                s*displacement[:,0]+c*displacement[:,1]),1)
    return torch.cat((xy,current[:,2:4]*log_size.exp(),(current[:,4]+yaw)[:,None]),1)


def forecast_loss(prediction,current,target,valid):
    displacement,yaw,log_size,existence=prediction
    delta=target[:,:2]-current[:,:2]
    c,s=current[:,4].cos(),current[:,4].sin()
    desired=torch.stack((c*delta[:,0]+s*delta[:,1],-s*delta[:,0]+c*delta[:,1]),1)
    scale=current[:,2:4].clamp_min(1.)
    center=F.smooth_l1_loss(displacement/scale,desired/scale,reduction='none').mean(1)
    desired_yaw=target[:,4]-current[:,4]
    orientation=(1-(2*(yaw-desired_yaw)).cos())*.25
    size=F.smooth_l1_loss(log_size,(target[:,2:4].clamp_min(.05)/current[:,2:4]).log(),reduction='none').mean(1)*.25
    weights=current[:,2:4].prod(1).sqrt().clamp(1,8)*valid.float()
    geometry=((center+orientation+size)*weights).sum()/weights.sum().clamp_min(1.)
    presence=F.binary_cross_entropy_with_logits(existence,valid.float())
    return geometry+.1*presence,dict(center=(center*weights).sum()/weights.sum().clamp_min(1.),
                                   orientation=(orientation*weights).sum()/weights.sum().clamp_min(1.),presence=presence)
