"""Evaluate dev-selected instance models once on untouched official scenes."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .confirm_motion_ablations import paired_iou_bootstrap
from .instance_motion_model import InstanceMotionPredictor
from .train_instance_motion import Trial
from .train_latent_motion import Counts,atomic_json


def cached_cv(trial):
    counts=Counts();per={}
    for row in trial.val:
        p=trial.grid['cv_occ'][row];t=trial.grid['target_occ'][row]
        counts.update(torch.tensor(p),torch.tensor(t))
        pu,tu=p.any(1),t.any(1)
        value=np.array([(pu&tu).sum(),(pu&~tu).sum(),(~pu&tu).sum()])
        per.setdefault(str(trial.a['scene'][row]),np.zeros(3,dtype=np.int64))[:]+=value
    return dict(**counts.result(),per_scene_counts={s:v.tolist() for s,v in per.items()})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--instances',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('preserve official evaluation')
    selection=json.loads((args.run/'selection_before_official_test.json').read_text())
    protocol=json.loads((args.run/'protocol.json').read_text())
    settings=protocol['args']
    training_scene=set(np.load(Path(settings['data'])/'windows.npz')['scene'])
    test_scene=set(np.load(args.data/'windows.npz')['scene'])
    if training_scene&test_scene:raise ValueError('official scenes overlap training/development data')
    args.output.mkdir(parents=True)
    settings['instances']=args.instances;settings['data']=args.data;settings['output']=args.output
    settings['val_reference']=Path(settings['val_reference']);settings['evaluation_only']=True
    torch.set_num_threads(4)
    trial=Trial(argparse.Namespace(**settings));reference=cached_cv(trial)
    baselines={}
    for kind in ('gt_velocity','history_cv','history_ca','history_ctrv'):
        boxes,keep=trial.predict(kind);baselines[kind]=trial.evaluate(boxes,keep)
        print('official baseline',kind,baselines[kind]['occupancy_iou'],flush=True)
    if abs(baselines['gt_velocity']['occupancy_iou']-reference['occupancy_iou'])>1e-4:
        raise ValueError('current actor population/renderer differs from cached GT-CV benchmark')
    models=[]
    for seed,entry in selection['models'].items():
        checkpoint=torch.load(entry['checkpoint'],map_location='cpu',weights_only=False)
        if checkpoint['step']!=entry['step']:raise ValueError('selection changed')
        model=InstanceMotionPredictor().to(trial.device);model.load_state_dict(checkpoint['model'])
        boxes,keep=trial.predict('learned',model)
        metrics=trial.evaluate(boxes,keep)
        metrics['seed']=int(seed);metrics['selected_development_step']=checkpoint['step']
        models.append(metrics);atomic_json(args.output/f'seed{seed}.json',metrics)
        np.savez_compressed(args.output/f'predicted_boxes_seed{seed}.npz',boxes=boxes,keep=keep,offsets=trial.a['offsets'])
        print('official learned',seed,metrics['occupancy_iou'],'GT-CV',reference['occupancy_iou'],flush=True)
    summary=dict(scenes=len(test_scene),windows=len(trial.val),gt_velocity_cv=reference,
                 baseline_reproduction=baselines,models=models,
                 mean_learned_iou=float(np.mean([m['occupancy_iou'] for m in models])),
                 paired_vs_gt_cv=paired_iou_bootstrap(models,[reference for _ in models]),
                 selection=selection,scene_disjointness_verified=True,
                 scope='Future 1s annotated-box BEV, identical targets/grid/three ego candidates. Precise instance histories, 16 past detection frames for geometric association; no GT velocities or numeric IDs in predictor inputs. Analytic token decoder, not original V4 latent decoder.')
    atomic_json(args.output/'metrics.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k in ('mean_learned_iou','paired_vs_gt_cv')}),flush=True)


if __name__=='__main__':main()
