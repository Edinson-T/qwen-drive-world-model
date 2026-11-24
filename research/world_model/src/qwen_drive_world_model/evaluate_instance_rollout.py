"""Evaluate preselected shared rollouts on a disjoint official validation subset."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .recurrent_instance_motion import RecurrentInstanceMotionPredictor
from .train_instance_rollout import RolloutTrial
from .train_latent_motion import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('instances','targets','parent','training-instances','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--device',default='cuda:0')
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    selections=[]
    checkpoints=[]
    # Complete all development selections and bind hashes before generating test predictions.
    for run in args.runs:
        status=json.loads((run/'status.json').read_text())
        if status['state'] != 'completed':
            raise ValueError('finish training and development selection first')
        ckpt=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
        checkpoints.append(ckpt)
        selections.append(dict(checkpoint=str(run/'best.pt'),step=ckpt['step'],seed=ckpt['seed'],
                               development_metrics=ckpt['metrics'],
                               sha256=hashlib.sha256((run/'best.pt').read_bytes()).hexdigest()))
    with np.load(args.instances/'instances.npz') as test, np.load(args.training_instances/'instances.npz') as train:
        if set(test['scene']) & set(train['scene']):
            raise ValueError('test overlaps the training/development master')
    args.output.mkdir(parents=True)
    atomic_json(args.output/'selection_before_test.json',dict(models=selections,scene_disjointness_verified=True))
    trial=RolloutTrial(SimpleNamespace(instances=args.instances,targets=args.targets,parent=args.parent,
                                      val_reference=None,output=args.output,device=args.device,evaluation_only=True))
    cv=np.repeat(trial.a['history'][:,-1,None],10,axis=1)
    cv[:,:,:2]+=trial.a['gt_velocity'][:,None]*np.arange(.5,5.01,.5)[None,:,None]
    baseline=trial.evaluate(cv,np.ones(cv.shape[:2],bool))
    oracle=trial.evaluate(trial.y['target'],trial.y['valid'].astype(bool))
    learned=[]
    for ckpt in checkpoints:
        model=RecurrentInstanceMotionPredictor(ckpt['hidden']).to(trial.device)
        model.load_state_dict(ckpt['model'])
        metrics=trial.evaluate(*trial.predict(model))
        learned.append(dict(seed=ckpt['seed'],selected_step=ckpt['step'],**metrics))
    paired={}
    rng=np.random.default_rng(20261119)
    for horizon in baseline['horizons']:
        scenes=sorted(baseline['horizons'][horizon]['per_scene_counts'])
        base=np.asarray([baseline['horizons'][horizon]['per_scene_counts'][s] for s in scenes])
        models=np.asarray([[m['horizons'][horizon]['per_scene_counts'][s] for s in scenes] for m in learned])
        idx=rng.integers(0,len(scenes),(10000,len(scenes)))
        sums=models[:,idx].sum(2)
        base_sums=base[idx].sum(1)
        deltas=(sums[:,:,0]/np.maximum(sums.sum(-1),1)).mean(0)-base_sums[:,0]/np.maximum(base_sums.sum(-1),1)
        mean=float(np.mean([m['horizons'][horizon]['occupancy_iou'] for m in learned]))
        paired[horizon]=dict(mean_learned_iou=mean,delta_vs_gt_cv=mean-baseline['horizons'][horizon]['occupancy_iou'],
                              scenario_bootstrap_ci95=np.quantile(deltas,[.025,.975]).tolist())
    report=dict(gt_velocity_cv=baseline,known_current_actor_future_oracle=oracle,learned=learned,paired=paired,
                scene_disjointness_verified=True,scenes=len(set(trial.a['scene'])),windows=len(trial.val),
                interval_note='Paired scene bootstrap conditional on these trained seeds, not training-population uncertainty',
                scope='GT historical boxes, all current source actors; learned models use no numeric IDs or GT velocities. Five-horizon single shared free-running model, not RFS.')
    atomic_json(args.output/'metrics.json',report)
    print(json.dumps(dict(scenes=report['scenes'],windows=report['windows'],paired=paired)),flush=True)


if __name__=='__main__':
    main()
