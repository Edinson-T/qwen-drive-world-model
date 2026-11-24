"""Preselect ablations on development, then confirm on reserved unused scenes."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .ablate_motion_training import Experiment, VARIANTS
from .latent_motion_model import LatentPredictor, StateAutoencoder
from .train_latent_motion import atomic_json


def paired_iou_bootstrap(candidate, baseline, draws=4000):
    scenes=sorted(candidate[0]['per_scene_counts'])
    arrays=[]
    for group in (candidate,baseline):
        arrays.append(np.asarray([[r['per_scene_counts'][s] for s in scenes] for r in group],dtype=np.float64))
    indices=np.random.default_rng(8675309).integers(0,len(scenes),(draws,len(scenes)))
    scores=[]
    for counts in arrays:
        sampled=counts[:,indices,:].sum(2)
        score=sampled[...,0]/sampled.sum(-1).clip(1)
        scores.append(score.mean(0))
    delta=scores[0]-scores[1]
    pooled=lambda counts:float((counts.sum(1)[:,0]/counts.sum((1,2)).clip(1)).mean())
    return dict(mean_seed_iou_delta=pooled(arrays[0])-pooled(arrays[1]),
                scenario_bootstrap_ci95=np.quantile(delta,[.025,.975]).tolist(),
                scenario_bootstrap_ci99=np.quantile(delta,[.005,.995]).tolist(),
                note='Scenario sampling paired across seeds; 2 training seeds, not a population CI over training randomness.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',type=Path,required=True)
    args=p.parse_args()
    if json.loads((args.suite/'status.json').read_text())['state']!='completed':
        raise RuntimeError('wait for all matched experiments')
    protocol=json.loads((args.suite/'protocol.json').read_text())
    seeds=protocol['args']['seeds']
    summary={}
    statuses={}
    for name in protocol['args']['variants']:
        group=[]
        for seed in seeds:
            status=json.loads((args.suite/f'{name}_seed{seed}'/'status.json').read_text())
            if status['state']!='completed':raise RuntimeError('incomplete '+name)
            group.append(status)
        statuses[name]=group
        summary[name]=dict(mean_best_iou=float(np.mean([s['best_iou'] for s in group])),
                           mean_last_iou=float(np.mean([s['last']['occupancy_iou'] for s in group])),
                           best_steps=[s['best_step'] for s in group],
                           last_ious=[s['last']['occupancy_iou'] for s in group],
                           last_changed_inside=float(np.mean([s['last']['changed_inside_recall'] for s in group])),
                           last_ghost_retention=float(np.mean([s['last']['ghost_retention'] for s in group])),
                           last_map_miou=float(np.mean([s['last']['map_miou'] for s in group])),
                           last_old_decoder_iou=float(np.mean([s['last']['original_decoder_probe']['occupancy_iou'] for s in group])),
                           last_target_reconstruction_iou=float(np.mean([s['last']['target_reconstruction']['occupancy_iou'] for s in group])))
    atomic_json(args.suite/'development_summary.json',summary)
    best=lambda names:max(names,key=lambda name:summary[name]['mean_last_iou'])
    alternatives=[n for n in summary if n!='control']
    chosen=['control',best(alternatives),best(['no_latent','raw_smooth','raw_mse']),
            best(['decoder','decoder_recon','encoder_decoder'])]
    if 'encoder_decoder' in chosen:chosen.append('decoder_recon')
    chosen=list(dict.fromkeys(chosen))
    selection=dict(variants=chosen,seeds=seeds,checkpoint='latest.pt',
                   rule='Preselect strongest overall, latent, and decoder-family by mean final-step development IoU; include control. If online encoder selected, include decoder_recon comparator.',
                   interpretation='Fixed budget final-step comparison, not best-checkpoint fishing; unmodified V4 also included.')
    atomic_json(args.suite/'confirmation_selection.json',selection)
    print('development summary',json.dumps(summary),flush=True)
    print('preselected',chosen,flush=True)
    output=args.suite/'confirmation'
    if output.exists():raise FileExistsError('preserve confirmation results')
    output.mkdir()
    settings=protocol['args']
    for key in ('data','checkpoint','velocity','val_reference'):settings[key]=Path(settings[key])
    settings['output']=output
    torch.set_num_threads(4)
    experiment=Experiment(argparse.Namespace(**settings))
    ids=np.load(args.suite/'confirmation_rows.npy')
    reservation=json.loads((args.suite/'confirmation_reservation.json').read_text())
    if len(set(experiment.a['scene'][ids]))!=reservation['scenes']:
        raise ValueError('confirmation reservation mismatch')
    if set(experiment.a['scene'][ids]) & set(experiment.a['scene'][experiment.train]):
        raise ValueError('training leakage')
    if set(experiment.a['scene'][ids]) & set(experiment.a['scene'][experiment.val]):
        raise ValueError('development leakage')
    results={}
    wm=LatentPredictor(experiment.cfg).to(experiment.device)
    ae=StateAutoencoder(experiment.cfg).to(experiment.device)
    wm.load_state_dict(experiment.ckpt['predictor']);ae.load_state_dict(experiment.ckpt['autoencoder'])
    original=experiment.evaluate(wm,ae,ids)
    results['original_v4']=[original for _ in seeds]
    for name in chosen:
        results[name]=[]
        for seed in seeds:
            path=args.suite/f'{name}_seed{seed}'/'latest.pt'
            checkpoint=torch.load(path,map_location='cpu',weights_only=False)
            wm.load_state_dict(checkpoint['predictor']);ae.load_state_dict(checkpoint['autoencoder'])
            result=experiment.evaluate(wm,ae,ids,checkpoint['protocol']['settings']['encoder'])
            results[name].append(result)
            atomic_json(output/f'{name}_seed{seed}.json',result)
            print(name,seed,result['occupancy_iou'],flush=True)
    comparisons={name:dict(versus_control=paired_iou_bootstrap(group,results['control']),
                           versus_original_v4=paired_iou_bootstrap(group,results['original_v4']))
                 for name,group in results.items() if name not in ('original_v4','control')}
    if 'encoder_decoder' in results:
        comparisons['encoder_decoder']['versus_decoder_recon']=paired_iou_bootstrap(results['encoder_decoder'],results['decoder_recon'])
    means={name:{key:float(np.mean([s[key] for s in group])) for key in
                       ('occupancy_iou','precision','recall','map_miou','changed_inside_recall','changed_outside_recall','ghost_retention','positive_brier','negative_brier')}
           for name,group in results.items()}
    atomic_json(output/'metrics.json',dict(reservation=reservation,selection=selection,mean_metrics=means,
                                          comparisons=comparisons,per_seed=results))
    print('confirmation means',json.dumps(means),flush=True)


if __name__=='__main__':main()
