"""Confirm predefined transport models and compare matched-object errors."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .confirm_motion_ablations import paired_iou_bootstrap
from .motion_transport import TransportWorldModel
from .train_motion_transport import TransportExperiment
from .train_latent_motion import atomic_json


def ranking_bootstrap(candidate,baseline):
    scenes=sorted(candidate[0]['per_scene_regret'])
    index=np.random.default_rng(1984).integers(0,len(scenes),(4000,len(scenes)))
    scores=[];means=[]
    for group in (candidate,baseline):
        values=np.asarray([[v['per_scene_regret'][s] for s in scenes] for v in group],dtype=float)
        sampled=values[:,index,:].sum(2)
        scores.append((sampled[...,0]/sampled[...,1]).mean(0))
        aggregate=values.sum(1);means.append(float((aggregate[:,0]/aggregate[:,1]).mean()))
    delta=scores[0]-scores[1]
    return dict(mean_seed_regret_delta=means[0]-means[1],scenario_bootstrap_ci95=np.quantile(delta,[.025,.975]).tolist(),
                scenario_bootstrap_ci99=np.quantile(delta,[.005,.995]).tolist(),lower_is_better=True,
                note='Conditional on two training seeds; three-candidate 1s geometry risk proxy, not RFS.')


def paired_object_centers(candidate,baseline,oracle):
    a=candidate['per_object'];b=baseline['per_object'];g=oracle['per_object'];groups={}
    for key in a.keys()&b.keys()&g.keys():
        if any(value[key]['center_error'] is None for value in (a,b,g)):continue
        values=[a[key]['center_error'],b[key]['center_error'],g[key]['center_error']]
        for group in ('all','speed/'+a[key]['speed_bin'],'class/'+a[key]['category']):
            groups.setdefault(group,[]).append(values)
    return {key:dict(common_objects=len(value),candidate_epe_m=float(np.asarray(value)[:,0].mean()),
                     v4_epe_m=float(np.asarray(value)[:,1].mean()),gt_velocity_cv_epe_m=float(np.asarray(value)[:,2].mean()))
            for key,value in groups.items()}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--suite',type=Path,required=True)
    args=p.parse_args();root=args.suite
    if json.loads((root/'status.json').read_text())['state']!='completed':raise RuntimeError('wait for training')
    protocol=json.loads((root/'protocol.json').read_text());settings=protocol['args']
    out=root/'confirmation'
    if out.exists():raise FileExistsError('preserve confirmation')
    out.mkdir()
    for key in ('data','checkpoint','velocity','val_reference'):settings[key]=Path(settings[key])
    settings['output']=out
    torch.set_num_threads(4)
    experiment=TransportExperiment(argparse.Namespace(**settings))
    ids=np.load(root/'confirmation_rows.npy');reserved=json.loads((root/'confirmation_reservation.json').read_text())
    scenes=set(experiment.a['scene'][ids])
    if scenes!=set(reserved['scene_ids']):raise ValueError('reservation mismatch')
    if scenes&set(experiment.a['scene'][experiment.data.train]):raise ValueError('training leakage')
    if scenes&set(experiment.a['scene'][experiment.data.val]):raise ValueError('development leakage')
    diagnostics=np.asarray(sorted(experiment.instance_rows))
    baseline_objects={name:experiment.evaluate(baseline=name,ids=diagnostics,instances=True)['object_diagnostics']
                      for name in ('v4','gt_velocity_cv')}
    atomic_json(out/'development_object_baselines.json',baseline_objects)
    seeds=settings['seeds'];modes=settings['modes']
    results={}
    for name in ('v4','decoded_ego_warp','gt_velocity_cv'):
        value=experiment.evaluate(baseline=name,ids=ids)
        results[name]=[value for _ in seeds]
        print('confirmation',name,value['occupancy_iou'],value['endpoint_ranking_regret'],flush=True)
    results['v4_matched_continuation']=[]
    for seed in seeds:
        path=Path('outputs/world_model/ablation_v6_20261119')/f'control_seed{seed}'/'latest.pt'
        old=torch.load(path,map_location='cpu',weights_only=False)
        for field in ('dataset_sha256','checkpoint_sha256','teacher_sha256'):
            if old['protocol'][field]!=experiment.data.protocol[field]:raise ValueError('continuation mismatch '+field)
        experiment.original.load_state_dict(old['predictor'])
        results['v4_matched_continuation'].append(experiment.evaluate(baseline='v4',ids=ids))
    object_pairs={};development={}
    for mode in modes:
        results[mode]=[];object_pairs[mode]=[];development[mode]=[]
        for seed in seeds:
            directory=root/f'{mode}_seed{seed}'
            checkpoint=torch.load(directory/'latest.pt',map_location='cpu',weights_only=False)
            if checkpoint['step']!=settings['steps']:raise ValueError('step mismatch')
            if any(not torch.equal(v,checkpoint['autoencoder'][k]) for k,v in experiment.data.ckpt['autoencoder'].items()):
                raise ValueError('frozen teacher changed')
            model=TransportWorldModel(experiment.data.cfg,mode).to(experiment.device)
            model.load_state_dict(checkpoint['model'])
            value=experiment.evaluate(model,ids=ids)
            results[mode].append(value);atomic_json(out/f'{mode}_seed{seed}.json',value)
            objects=experiment.evaluate(model,ids=diagnostics,instances=True)['object_diagnostics']
            atomic_json(out/f'{mode}_seed{seed}_development_objects.json',objects)
            object_pairs[mode].append(paired_object_centers(objects,baseline_objects['v4'],baseline_objects['gt_velocity_cv']))
            status=json.loads((directory/'status.json').read_text());development[mode].append(status['last'])
            print('confirmation',mode,seed,value['occupancy_iou'],value['endpoint_ranking_regret'],flush=True)
    comparisons={}
    for mode in modes:
        comparisons[mode]={}
        for baseline in ('v4','v4_matched_continuation','static_residual'):
            if mode==baseline:continue
            comparisons[mode][baseline]=dict(iou=paired_iou_bootstrap(results[mode],results[baseline]),
                                             ranking=ranking_bootstrap(results[mode],results[baseline]))
    means={name:{key:float(np.mean([v[key] for v in group])) for key in
                    ('occupancy_iou','map_miou','precision','recall','endpoint_ranking_regret','changed_inside_recall','ghost_retention')}
           for name,group in results.items()}
    for mode in modes:
        means[mode]['decoded_latent_iou']=float(np.mean([v['decoded_latent_probe']['occupancy_iou'] for v in results[mode]]))
    atomic_json(out/'metrics.json',dict(reservation=reserved,mean_metrics=means,per_seed=results,
                 comparisons=comparisons,development_final=development,paired_development_object_centers=object_pairs,
                 note='Native BEV and latent-decoded occupancy reported separately. Same-parameter transport+residual versus zero-flow+residual is primary structural contrast. Object-center comparisons use exactly common matched GT IDs, alongside miss/recall metrics.'))
    print('means',json.dumps(means),flush=True)


if __name__=='__main__':main()
