"""Resume causal extraction, then run paired RM comparisons without mixing IDs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--reference',type=Path,default=Path('outputs/selector_fullval/logged_geometry200'))
    p.add_argument('--candidates',type=Path,default=Path('outputs/selector_fullval/candidates_with_rl_base.npz'))
    p.add_argument('--current-cache',type=Path,default=Path('outputs/selector_fullval/qd_current_bev'))
    args=p.parse_args()
    root=args.root
    env=os.environ.copy()
    status_path=root/'pipeline_status.json'
    def status(stage,state,**extra):
        status_path.write_text(json.dumps(dict(pid=os.getpid(),stage=stage,state=state,**extra),indent=2)+'\n')
    def run(stage,module,arguments):
        status(stage,'running')
        with (root/f'{stage}.log').open('a') as log:
            result=subprocess.run([sys.executable,'-u','-m','qwen_drive_world_model.'+module,*map(str,arguments)],
                                  env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            status(stage,'failed',returncode=result.returncode,log=str(root/f'{stage}.log'))
            raise RuntimeError(f'{stage} failed, see log')
    with np.load(args.candidates) as a, np.load(args.reference/'candidates.npz') as b:
        lookup={str(t):i for i,t in enumerate(a['tokens'])}
        required=[lookup[str(t)] for t in b['tokens']]
        for key in ('scene_tokens','trajectories','candidate_rfs','anchor_ids'):
            np.testing.assert_array_equal(a[key][required],b[key])
    # Existing four-scene smoke caches remain usable. Limit covers every exact reference token.
    run('extract_histories','export_instance_reward_histories',[
        '--pairs',root/'paired/pairs.jsonl','--output',root/'qd_histories','--limit',max(required)+1])
    raw=root/'predicted_geometry_all'
    if not (raw/'protocol.json').exists():
        run('predict_geometry','build_instance_reward_geometry',[
            '--candidates',args.candidates,'--histories',root/'qd_histories/histories',
            '--checkpoint',args.checkpoint,'--output',raw])
    aligned=root/'predicted_geometry_reference200'
    aligned.mkdir(exist_ok=True)
    with np.load(raw/'candidates.npz') as a,np.load(raw/'geometry.npz') as g,np.load(args.reference/'candidates.npz') as b:
        lookup={str(t):i for i,t in enumerate(a['tokens'])}
        indices=[lookup[str(t)] for t in b['tokens']]
        for key in ('scene_tokens','trajectories','candidate_rfs','anchor_ids'):
            np.testing.assert_array_equal(a[key][indices],b[key])
        np.savez_compressed(aligned/'candidates.npz',**{k:a[k][indices] for k in a.files})
        np.savez_compressed(aligned/'geometry.npz',geometry=g['geometry'][indices])
    (aligned/'provenance.json').write_text(json.dumps(dict(
        checkpoint=str(args.checkpoint),reference=str(args.reference),
        prediction='QD historical detections + shared recurrent Motion-pretrained forecast',
        reference_semantics='logged future QD occupancy predictions, NOT LiDAR GT',
        pairing='exact tokens and identical candidate trajectories/labels; original row ordering restored'),indent=2)+'\n')
    rm=root/'reward_model'
    if not (rm/'metrics.json').exists():
        run('train_reward','train_geometry_reward_model',[
            '--data',args.reference,'--predicted-data',aligned,'--bev-cache',args.current_cache,
            '--output',rm,'--modes','current','predicted_geometry','logged_geometry','--epochs',25,
            '--folds',5,'--device','cuda:0'])
    with np.load(aligned/'candidates.npz') as a,np.load(rm/'predictions.npz') as p:
        labels=a['candidate_rfs']
        scenes=sorted(set(a['scene_tokens'].tolist()))
        masks=[a['scene_tokens']==s for s in scenes]
        indices=np.random.default_rng(20261119).integers(0,len(scenes),(10000,len(scenes)))
        selected={name:labels[np.arange(len(labels)),p[name]] for name in ('current','predicted_geometry','logged_geometry')}
        comparisons={}
        for name,base in (('predicted_geometry','current'),('predicted_geometry','rl_base'),('logged_geometry','current')):
            reference=labels[:,0] if base=='rl_base' else selected[base]
            delta=selected[name]-reference
            sums=np.asarray([delta[m].sum() for m in masks])
            counts=np.asarray([m.sum() for m in masks])
            samples=sums[indices].sum(1)/counts[indices].sum(1)
            comparisons[f'{name}_minus_{base}']=dict(mean=float(delta.mean()),
                scenario_bootstrap_ci95=np.quantile(samples,[.025,.975]).tolist(),
                wins=int((delta>0).sum()),losses=int((delta<0).sum()),ties=int((delta==0).sum()))
    (rm/'paired_comparisons.json').write_text(json.dumps(comparisons,indent=2)+'\n')
    status('finished','completed',comparisons=comparisons)


if __name__=='__main__':
    main()
