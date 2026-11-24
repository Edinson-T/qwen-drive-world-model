"""Evaluate a fixed Motion latent checkpoint on a separate official validation subset."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .latent_motion_model import (LatentConfig, StateAutoencoder, LatentPredictor,
                                  pack_state, warp_current, latent_distance)
from .train_latent_motion import Counts, Windows, device_batch, state_batch, atomic_json


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--train-data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'metrics.json').exists():
        raise FileExistsError('refusing to overwrite an evaluation')
    torch.set_num_threads(4)
    device=torch.device(args.device)
    checkpoint=torch.load(args.checkpoint,map_location=device,weights_only=False)
    config=LatentConfig(**checkpoint['config'])
    ae=StateAutoencoder(config).to(device).eval().requires_grad_(False)
    wm=LatentPredictor(config).to(device).eval()
    ae.load_state_dict(checkpoint['autoencoder']);wm.load_state_dict(checkpoint['predictor'])
    manifest=json.loads((args.data/'dataset.json').read_text())
    with np.load(args.data/'windows.npz') as payload:
        a={k:payload[k] for k in payload.files}
    with np.load(args.train_data/'windows.npz') as train:
        if set(a['scene']) & set(train['scene']):
            raise ValueError('evaluation scenarios overlap training/development scenarios')
    train_manifest=json.loads((args.train_data/'dataset.json').read_text())
    for key in ('grid','history_times','action_times','horizon','target_frame','object_classes'):
        if manifest[key] != train_manifest[key]:
            raise ValueError(f'data protocol mismatch: {key}')
    names=('world_model','copy','ego_warp','constant_velocity','cv_reconstruction','teacher_reconstruction','shuffled_action','current_only')
    counts={k:Counts() for k in names};scene_counts={};examples=[];latent_total=0;n=0
    loader=DataLoader(Windows(a,np.arange(len(a['scene']))),batch_size=16,shuffle=False)
    for raw in loader:
        b=device_batch(raw,device);size=len(b['row'])
        h=ae.encoder(state_batch(b,history=True)).reshape(size,4,config.channels,config.token_h,config.token_w)
        target=ae.encoder(state_batch(b))
        prediction=wm(h,b['action'],b['status']);logits=ae.decoder(prediction)
        teacher=ae.decoder(target)
        warped,_=warp_current(b['history_occ'][:,-1].float(),b['action'][:,4])
        current_maps=pack_state(b['current_map'],b['history_occ'][:,-1])[:,:8]
        warped_maps,_=warp_current(current_maps,b['action'][:,4])
        cv_decoded=ae(pack_state(warped_maps.argmax(1),b['cv']))
        mask=h.new_zeros(size,4);mask[:,-1]=1
        probabilities=dict(world_model=logits[:,8:].sigmoid(),copy=b['history_occ'][:,-1],ego_warp=warped,
                           constant_velocity=b['cv'],teacher_reconstruction=teacher[:,8:].sigmoid(),
                           cv_reconstruction=cv_decoded[:,8:].sigmoid(),
                           shuffled_action=ae.decoder(wm(h,b['alternate'],b['status']))[:,8:].sigmoid(),
                           current_only=ae.decoder(wm(h,b['action'],b['status'],mask))[:,8:].sigmoid())
        for name,probability in probabilities.items():
            maps=logits[:,:8] if name=='world_model' else teacher[:,:8] if name=='teacher_reconstruction' else None
            counts[name].update(probability,b['occ'],maps,b['maps'])
        latent_total+=float(latent_distance(prediction,target))*size;n+=size
        for j,index in enumerate(raw['row'].tolist()):
            scene=str(a['scene'][index])
            group=scene_counts.setdefault(scene,{k:Counts() for k in ('world_model','ego_warp','constant_velocity')})
            for name,counter in group.items():
                counter.update(probabilities[name][j:j+1],b['occ'][j:j+1])
            if raw['candidate'][j]==0 and len(examples)<6:
                examples.append(dict(target=b['occ'][j].any(0).cpu().numpy(),
                                     current=b['history_occ'][j,-1].any(0).cpu().numpy(),
                                     prediction=probabilities['world_model'][j].amax(0).cpu().numpy(),
                                     warp=warped[j].any(0).cpu().numpy(),cv=b['cv'][j].any(0).cpu().numpy(),
                                     teacher=probabilities['teacher_reconstruction'][j].amax(0).cpu().numpy()))
    # Resample whole scenarios, keeping all windows and candidates together.
    rng=np.random.default_rng(42);keys=sorted(scene_counts)
    tensors={name:np.array([[scene_counts[key][name].tp,scene_counts[key][name].fp,scene_counts[key][name].fn]
                            for key in keys]) for name in ('world_model','ego_warp','constant_velocity')}
    differences=[]
    for _ in range(2000):
        sample=rng.integers(0,len(keys),len(keys));iou={}
        for name,values in tensors.items():
            total=values[sample].sum(0);iou[name]=total[0]/max(total.sum(),1)
        differences.append(iou['world_model']-iou['ego_warp'])
    result=dict(checkpoint=str(args.checkpoint.resolve()),source=manifest,scenarios=len(keys),windows=len(a['scene']),
                candidate_targets=n,latent_loss=latent_total/n,metrics={k:v.result() for k,v in counts.items()},
                wm_minus_ego_warp_scene_bootstrap_95ci=np.quantile(differences,[.025,.975]).tolist(),
                notes=['Official validation subset; fixed checkpoint, no tuning on this evaluation.',
                       'Other agents follow logs; candidate labels change reference frame, not interactive responses.',
                       'constant_velocity uses privileged GT object velocities.',
                       'cv_reconstruction passes CV occupancy and warped current map through the same frozen autoencoder.',
                       'GT CV also uses globally annotated current tracks, including tracks outside the WM cropped input.',
                       'shuffled_action/current_only are inference ablations, not independently trained models.',
                       'Motion teacher latents are not yet aligned to QD feature space.'])
    atomic_json(args.output/'metrics.json',result)
    atomic_json(args.output/'per_scene.json',{key:{name:value.result() for name,value in group.items()} for key,group in scene_counts.items()})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(examples),6,figsize=(18,2.5*len(examples)),squeeze=False)
    for row,example in enumerate(examples):
        for col,name in enumerate(('current','target','prediction','warp','cv','teacher')):
            axes[row,col].imshow(example[name],origin='lower',extent=(-30,30,-15,15),vmin=0,vmax=1,cmap='magma')
            axes[row,col].set_title(name);axes[row,col].set_xlabel('x forward (m)')
    fig.tight_layout();fig.savefig(args.output/'comparison.png',dpi=140);plt.close(fig)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
