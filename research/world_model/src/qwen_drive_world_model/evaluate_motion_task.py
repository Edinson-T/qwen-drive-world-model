"""Evaluate preselected Motion endpoint-proximity controls on held-out scenes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .latent_motion_model import LatentConfig, StateAutoencoder, LatentPredictor, pack_state, warp_current
from .train_latent_motion import Counts, atomic_json
from .train_motion_task import ProximityHead, proximity_labels, summarize_task


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--exclude-data', type=Path, nargs='+', required=True)
    p.add_argument('--joint-run', type=Path, required=True)
    p.add_argument('--semantic-run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve existing evaluations')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    with np.load(args.data/'windows.npz', allow_pickle=False) as data:
        a = {k:data[k] for k in data.files}
    seen = set(a['scene'])
    for other in args.exclude_data:
        with np.load(other/'windows.npz', allow_pickle=False) as prior:
            if seen & set(prior['scene']):
                raise ValueError('held-out scenes overlap '+str(other))
    paths = {name:args.joint_run/(name+'_best.pt')
             for name in ('history_static', 'frozen_future', 'joint_future')}
    paths['semantic_future'] = args.semantic_run/'joint_future_best.pt'
    for path in (args.joint_run, args.semantic_run):
        if json.loads((path/'status.json').read_text())['state']!='completed':
            raise ValueError('select checkpoints only after training completes')
    checkpoints = {name:torch.load(path, map_location='cpu', weights_only=False) for name,path in paths.items()}
    first = checkpoints['joint_future']
    cfg = LatentConfig(**first['config'])
    ae = StateAutoencoder(cfg).to(device).eval()
    ae.load_state_dict(first['autoencoder'])
    history = torch.empty(len(a['scene']), 4, cfg.channels, cfg.token_h, cfg.token_w, dtype=torch.float16)
    args.output.mkdir(parents=True)
    with torch.inference_mode():
        for start in range(0, len(history), 16):
            end = min(start+16, len(history))
            maps = torch.as_tensor(a['history_map'][start:end], device=device)[:,None].expand(-1,4,-1,-1)
            occ = torch.as_tensor(a['history_occ'][start:end], device=device)
            latent = ae.encoder(pack_state(maps, occ).flatten(0,1))
            history[start:end] = latent.reshape(end-start,4,cfg.channels,cfg.token_h,cfg.token_w).half().cpu()
        target = proximity_labels(a['target_occ'])
        results = {}
        arrays = {}
        for name, checkpoint in checkpoints.items():
            if checkpoint['config']!=first['config']:
                raise ValueError('architecture mismatch')
            if any(not torch.equal(v, checkpoint['autoencoder'][k]) for k,v in first['autoencoder'].items()):
                raise ValueError('encoder mismatch')
            wm = LatentPredictor(cfg).to(device).eval()
            wm.load_state_dict(checkpoint['predictor'])
            head = ProximityHead(cfg.channels).to(device).eval()
            head.load_state_dict(checkpoint['head'])
            predictions = []
            counts = Counts()
            for start in range(0, len(history)*3, 16):
                ids = np.arange(start, min(start+16, len(history)*3))
                rows, choices = ids//3, ids%3
                h = history[rows].to(device).float()
                action = torch.as_tensor(a['actions'][rows, choices], device=device)
                status = torch.as_tensor(a['status'][rows], device=device)
                future = (warp_current(h[:,-1], action[:,4], mode='bilinear')[0]
                          if name=='history_static' else wm(h, action, status))
                predictions.append(head(h, future, action, status).sigmoid().cpu().numpy())
                if name!='history_static':
                    truth = torch.as_tensor(a['target_occ'][rows, choices], device=device)
                    counts.update(ae.decoder(future)[:,8:].sigmoid(), truth)
            arrays[name] = np.concatenate(predictions).reshape(-1,3,2)
            results[name] = dict(checkpoint=str(paths[name]), step=checkpoint['step'],
                                 checkpoint_sha256=hashlib.sha256(paths[name].read_bytes()).hexdigest(),
                                 task=summarize_task(arrays[name], target, a['scene']),
                                 occupancy=counts.result() if name!='history_static' else None)
            print(name, results[name]['task']['proximity_mae'], flush=True)
        pairs = {}
        for baseline in ('history_static', 'frozen_future', 'semantic_future'):
            mae = lambda name: np.abs(arrays[name][:,:,1]-target[:,:,1]).mean(1)
            delta = mae('joint_future')-mae(baseline)
            scene_delta = [float(delta[a['scene']==s].mean()) for s in sorted(seen)]
            boot = np.random.default_rng(123).choice(scene_delta,(2000,len(scene_delta))).mean(1)
            pairs['joint_minus_'+baseline] = dict(proximity_mae_delta=float(np.mean(scene_delta)),
                                                 scene_bootstrap_ci95=np.quantile(boot,[.025,.975]).tolist())
        result = dict(scenes=len(seen), windows=len(history), results=results, paired=pairs,
                      selection='minimum development proximity MAE separately per control; no test-based selection',
                      task='nominal ego 1s endpoint proximity only; not RFS, full-route collision or simulation',
                      dataset_sha256=hashlib.sha256((args.data/'windows.npz').read_bytes()).hexdigest())
        atomic_json(args.output/'metrics.json', result)
        np.savez_compressed(args.output/'predictions.npz', target=target, scene=a['scene'], **arrays)
        print(json.dumps(result['paired']), flush=True)


if __name__ == '__main__':
    main()
