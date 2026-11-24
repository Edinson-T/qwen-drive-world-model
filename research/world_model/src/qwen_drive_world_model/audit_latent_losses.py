"""Read-only loss-gradient and decoder-scale diagnostics; no model updates."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .latent_motion_model import (LatentConfig, LatentPredictor, StateAutoencoder,
                                  latent_distance, velocity_auxiliary_loss)
from .train_latent_motion import Counts, atomic_json


def main():
    root = Path('outputs/world_model')
    out = root/'loss_audit_20261119'
    if out.exists():
        raise FileExistsError(out)
    out.mkdir()
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    path = root/'motion_latent_v4/velocity_aux/best.pt'
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    cfg = LatentConfig(**ckpt['config'])
    wm = LatentPredictor(cfg).to(device).eval()
    wm.load_state_dict(ckpt['predictor'])
    ae = StateAutoencoder(cfg).to(device).eval().requires_grad_(False)
    ae.load_state_dict(ckpt['autoencoder'])
    cache = torch.load(ckpt['protocol']['frozen_latent_cache'], mmap=True,
                       map_location='cpu', weights_only=False)
    with np.load(root/'motion_latent_v3/dataset2048/windows.npz') as data:
        a = {key:data[key] for key in ('scene','anchor','actions','status','target_map','target_occ','split')}
    with np.load(root/'motion_latent_v2/dataset/windows.npz') as ref:
        keys = {(str(ref['scene'][i]),int(ref['anchor'][i])) for i in np.flatnonzero(ref['split']=='val')}
    val = np.array([i for i in range(len(a['scene'])) if (str(a['scene'][i]),int(a['anchor'][i])) in keys])
    # Spread samples across the fixed development windows; no new holdout consumed.
    rows = val[np.linspace(0,len(val)-1,64,dtype=int)]
    choices = np.arange(64)%3
    with np.load(root/'motion_latent_v4/velocity_targets.npz') as v:
        velocity, mask = v['velocity'][rows], v['mask'][rows]
    counts = np.bincount(a['target_map'][a['split']=='train'].ravel(),minlength=8).astype(float)
    weights = 1/np.sqrt(counts/counts.sum()+1e-3)
    weights = torch.tensor(np.clip(weights/weights.mean(),.25,4),device=device,dtype=torch.float32)
    parameters = tuple(wm.parameters())
    metrics = []
    scale_counts = {str(scale):Counts() for scale in (.5,1.,2.)}
    scale_latent = {str(scale):[] for scale in (.5,1.,2.)}
    for start in range(0,64,8):
        r,c = rows[start:start+8],choices[start:start+8]
        h = cache['history'][r].to(device).float()
        target = cache['target'][r,c].to(device).float()
        action = torch.tensor(a['actions'][r,c],device=device)
        status = torch.tensor(a['status'][r],device=device)
        maps = torch.tensor(a['target_map'][r,c],device=device).long()
        occ = torch.tensor(a['target_occ'][r,c],device=device).float()
        prediction = wm(h,action,status)
        logits = ae.decoder(prediction)
        probability = logits[:,8:].sigmoid()
        intersection = (probability*occ).sum((0,2,3))
        denominator = probability.sum((0,2,3))+occ.sum((0,2,3))
        losses = dict(latent=latent_distance(prediction,target),
                      map=.5*F.cross_entropy(logits[:,:8],maps,weight=weights),
                      occupancy=(F.binary_cross_entropy_with_logits(logits[:,8:],occ,pos_weight=logits.new_tensor(4.))
                                 +.5*(1-(2*intersection+1)/(denominator+1)).mean()),
                      velocity=.5*velocity_auxiliary_loss(wm.aux_velocity,
                               torch.tensor(velocity[start:start+8],device=device),
                               torch.tensor(mask[start:start+8],device=device),action[:,4]))
        gradients = {}
        for name,loss in losses.items():
            grad = torch.autograd.grad(loss, parameters,retain_graph=True,allow_unused=True)
            gradients[name] = torch.cat([(torch.zeros_like(p) if g is None else g).flatten()
                                         for p,g in zip(parameters,grad)])
        cosine = lambda x,y:float(F.cosine_similarity(gradients[x],gradients[y],dim=0))
        metrics.append(dict(weighted_losses={n:float(l.detach()) for n,l in losses.items()},
                            wm_gradient_norm={n:float(g.norm()) for n,g in gradients.items()},
                            cosine_with_occupancy={n:cosine(n,'occupancy') for n in ('latent','map','velocity')},
                            prediction_channel_std=float(prediction.std(1,unbiased=False).mean()),
                            target_channel_std=float(target.std(1,unbiased=False).mean())))
        with torch.no_grad():
            for scale in (.5,1.,2.):
                reconstruction = ae.decoder(target*scale)
                scale_counts[str(scale)].update(reconstruction[:,8:].sigmoid(),occ)
                scale_latent[str(scale)].append(float(latent_distance(target*scale,target)))
        print('diagnosed',start+8,flush=True)
    summary = {}
    for field in ('weighted_losses','wm_gradient_norm','cosine_with_occupancy'):
        summary[field] = {name:dict(median=float(np.median([b[field][name] for b in metrics])),
                                   minimum=float(min(b[field][name] for b in metrics)),
                                   maximum=float(max(b[field][name] for b in metrics)))
                          for name in metrics[0][field]}
    result = dict(checkpoint=str(path),checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  samples=64,batch_size=8,split='existing fixed development windows, evenly spaced; no history dropout',
                  note='Local weighted gradients at one checkpoint; no causal loss ablation or model update. Teacher scaling is an artificial sensitivity probe, not a deployed prediction.',
                  gradient_summary=summary,batches=metrics,
                  target_scaling_probe={s:dict(normalized_latent_distance=float(np.mean(scale_latent[s])),
                                               reconstruction=count.result()) for s,count in scale_counts.items()})
    atomic_json(out/'metrics.json',result)
    (out/'source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(summary=summary,scaling=result['target_scaling_probe']),indent=2),flush=True)


if __name__=='__main__':
    main()
