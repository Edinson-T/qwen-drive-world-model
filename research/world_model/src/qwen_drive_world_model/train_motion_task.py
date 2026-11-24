"""Small controlled task-gradient experiment in the existing 1s Motion domain.

The task is endpoint proximity to a fixed 4.8m x 2m ego footprint, NOT
trajectory safety, simulated reward, PDMS, or RFS. Labels are training-only.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .latent_motion_model import (LatentConfig, LatentPredictor, StateAutoencoder,
                                  latent_distance, semantic_loss, warp_current)
from .train_latent_motion import Counts, atomic_json


def proximity_labels(occupancy):
    """[endpoint box overlap, exp(-clearance/3m)] from labeled box pixels.

    Fixed nominal ego dimensions; raster centers approximate geometry. Empty
    local crops have clearance capped at 8m and do not certify free space.
    """
    h, w = occupancy.shape[-2:]
    x = (np.arange(w) + .5) * 60 / w - 30
    y = (np.arange(h) + .5) * 30 / h - 15
    # Subtract half a raster cell for conservative footprint intersection.
    dx = np.maximum(np.abs(x) - 2.4 - 30/w, 0)
    dy = np.maximum(np.abs(y) - 1.0 - 15/h, 0)
    distance = np.hypot(dy[:, None], dx[None, :])
    present = occupancy.any(-3)
    clearance = np.where(present, distance, 8.).min((-2, -1)).clip(0, 8)
    return np.stack((clearance <= 1e-6, np.exp(-clearance/3)), -1).astype(np.float32)


class ProximityHead(nn.Module):
    """Same capacity and observed history for all three controls."""
    def __init__(self, channels=128):
        super().__init__()
        self.visual = nn.Sequential(nn.Conv2d(channels*5, 64, 1), nn.GELU(),
                                    nn.Conv2d(64, 32, 3, padding=1), nn.GELU(),
                                    nn.AdaptiveAvgPool2d((4, 8)), nn.Flatten())
        self.head = nn.Sequential(nn.Linear(32*4*8+67, 128), nn.GELU(), nn.Linear(128, 2))

    def forward(self, history, future, action, status):
        feature = self.visual(torch.cat((history.flatten(1, 2), future), 1))
        a = action / action.new_tensor([30., 15., np.pi])
        s = status / status.new_tensor([15., 15., 5., 5.])
        return self.head(torch.cat((feature, a.flatten(1), s), -1))


def task_loss(logits, labels, positive_weight):
    overlap = F.binary_cross_entropy_with_logits(logits[:, 0], labels[:, 0],
                                                 pos_weight=positive_weight)
    proximity = F.mse_loss(logits[:, 1].sigmoid(), labels[:, 1])
    return .25*overlap + proximity


def copy_head_training_state(source_head, source_optimizer, target_head, target_optimizer):
    """Start matched controls with equal values and independent Adam storage.

    Optimizer.load_state_dict can retain tensor storage on the same device.
    A shallow state_dict transfer couples the controls' moments and step count.
    """
    target_head.load_state_dict(source_head.state_dict())
    target_optimizer.load_state_dict(copy.deepcopy(source_optimizer.state_dict()))


def summarize_task(prediction, labels, scenes):
    p, y = prediction.reshape(-1, 3, 2), labels.reshape(-1, 3, 2)
    # Endpoint proximity alone. Lower risk is preferred. No full-route claims.
    chosen = p[:, :, 1].argmin(1)
    chosen_risk = y[np.arange(len(y)), chosen, 1]
    oracle = y[:, :, 1].min(1)
    informative = np.ptp(y[:, :, 1], axis=1) > .05
    unique = sorted(set(scenes))
    per_scene = [float((chosen_risk-oracle)[np.asarray(scenes)==s].mean()) for s in unique]
    return dict(proximity_mae=float(np.abs(p[:, :, 1]-y[:, :, 1]).mean()),
                overlap_brier=float(((p[:, :, 0]-y[:, :, 0])**2).mean()),
                endpoint_selection_regret=float((chosen_risk-oracle).mean()),
                informative_windows=int(informative.sum()),
                informative_regret=float((chosen_risk-oracle)[informative].mean()) if informative.any() else None,
                mean_candidate_endpoint_risk=float(y[:, :, 1].mean()),
                logged_candidate_endpoint_risk=float(y[:, 0, 1].mean()),
                oracle_endpoint_risk=float(oracle.mean()),
                per_scene_regret=dict(zip(unique, per_scene)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--val-reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=1600)
    parser.add_argument('--warmup', type=int, default=400)
    parser.add_argument('--eval-every', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--wm-lr', type=float, default=1e-5)
    parser.add_argument('--distill-weight', type=float, default=0.)
    parser.add_argument('--stop-task-gradient', action='store_true',
                        help='Matched semantic-only WM update control; head still trains.')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('use a fresh run directory')
    if not 0 < args.warmup < args.steps:
        raise ValueError('require 0 < warmup < steps')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no silent CPU fallback')
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device('cuda:0')
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = LatentConfig(**ckpt['config'])
    frozen = LatentPredictor(cfg).to(device).eval()
    frozen.load_state_dict(ckpt['predictor'])
    frozen.requires_grad_(False)
    joint = copy.deepcopy(frozen).requires_grad_(True)
    ae = StateAutoencoder(cfg).to(device).eval()
    ae.load_state_dict(ckpt['autoencoder'])
    ae.requires_grad_(False)
    with np.load(args.data/'windows.npz', allow_pickle=False) as payload:
        a = {k: payload[k] for k in ('scene', 'anchor', 'split', 'actions', 'status', 'target_occ', 'target_map')}
    expected_hash = ckpt['protocol']['dataset_sha256']
    actual_hash = hashlib.sha256((args.data/'windows.npz').read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError('checkpoint and dataset differ')
    cache = torch.load(args.cache, map_location='cpu', mmap=True, weights_only=False)
    # Compare teacher parameters rather than trusting paths or cache shape alone.
    teacher_path = Path(ckpt['protocol']['frozen_latent_cache']).parent/'autoencoder_best.pt'
    teacher = torch.load(teacher_path, map_location='cpu', weights_only=False)
    if cache['teacher_sha256'] != hashlib.sha256(teacher_path.read_bytes()).hexdigest():
        raise ValueError('cache teacher hash mismatch')
    if any(not torch.equal(v, teacher['model'][k]) for k, v in ckpt['autoencoder'].items()):
        raise ValueError('checkpoint uses a different teacher')
    train = np.flatnonzero(a['split']=='train')
    with np.load(args.val_reference/'windows.npz', allow_pickle=False) as ref:
        selected = np.flatnonzero(ref['split']=='val')
        lookup = {(str(s), int(t)): i for i, (s,t) in enumerate(zip(a['scene'], a['anchor']))}
        val = np.array([lookup[(str(ref['scene'][i]), int(ref['anchor'][i]))] for i in selected])
        for key in ('actions', 'target_occ', 'target_map', 'status'):
            if not np.array_equal(a[key][val], ref[key][selected]):
                raise ValueError('validation data changed: '+key)
    if set(a['scene'][train]) & set(a['scene'][val]):
        raise ValueError('scene leakage')
    labels = proximity_labels(a['target_occ'])
    prevalence = float(labels[train, :, 0].mean())
    positive_weight = torch.tensor(min(20., (1-prevalence)/max(prevalence, .001)), device=device)
    names = ('history_static', 'frozen_future', 'joint_future')
    first_head = ProximityHead(cfg.channels).to(device)
    heads = {name: copy.deepcopy(first_head) for name in names}
    optimizers = {name: torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=1e-3)
                  for name, head in heads.items()}
    wm_optimizer = torch.optim.AdamW(joint.parameters(), lr=args.wm_lr, weight_decay=1e-3)
    counts = np.bincount(a['target_map'][train].ravel(), minlength=8).astype(float)
    weights = 1/np.sqrt(counts/counts.sum()+1e-3)
    weights = torch.tensor(np.clip(weights/weights.mean(), .25, 4), dtype=torch.float32, device=device)
    source = args.output/'source'
    source.mkdir()
    for name in ('train_motion_task.py', 'latent_motion_model.py'):
        (source/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    protocol = dict(args={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                    dataset_sha256=actual_hash, checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                    task='1s endpoint box overlap and proximity, nominal 4.8x2m ego; NOT RFS or full-trajectory safety',
                    controls=list(names), train_windows=len(train), train_scenes=len(set(a['scene'][train])),
                    validation_windows=len(val), validation_scenes=len(set(a['scene'][val])),
                    overlap_prevalence=prevalence, warmup='identical frozen/joint head updates before WM unfreeze',
                    optimizer_state_isolation='deepcopy at warmup; fixes pre-2026-11-19 shared Adam moments',
                    other_agents='logged Motion future; no reactive simulation',
                    limitations='three logged-path perturbations; narrow 1s headroom; development set reused; joint also receives semantic updates, so comparison does not isolate task-gradient causality',
                    config=asdict(cfg))
    atomic_json(args.output/'protocol.json', protocol)

    def batch(rows, choices):
        tensor = lambda value: torch.as_tensor(value, device=device)
        return (cache['history'][rows].to(device).float(),
                tensor(a['actions'][rows, choices]), tensor(a['status'][rows]),
                tensor(labels[rows, choices]))

    @torch.no_grad()
    def evaluate(step):
        predictions = {name: [] for name in names}
        occupancy = {'frozen_future': Counts(), 'joint_future': Counts()}
        for head in heads.values():
            head.eval()
        joint.eval()
        for start in range(0, len(val)*3, args.batch_size):
            ids = np.arange(start, min(start+args.batch_size, len(val)*3))
            rows, choices = val[ids//3], ids%3
            h, action, status, target = batch(rows, choices)
            base, _ = warp_current(h[:, -1], action[:, 4], mode='bilinear')
            futures = dict(history_static=base, frozen_future=frozen(h, action, status),
                           joint_future=joint(h, action, status))
            for name in names:
                predictions[name].append(heads[name](h, futures[name], action, status).sigmoid().cpu().numpy())
            truth = torch.as_tensor(a['target_occ'][rows, choices], device=device)
            for name in occupancy:
                occupancy[name].update(ae.decoder(futures[name])[:, 8:].sigmoid(), truth)
        result = dict(step=step, controls={name: summarize_task(np.concatenate(predictions[name]),
                     labels[val], a['scene'][val]) for name in names},
                     occupancy={name: c.result() for name,c in occupancy.items()})
        # Identical paired validation scenes, descriptive rather than a final test.
        pairs = [result['controls']['joint_future']['per_scene_regret'][s]
                 - result['controls']['history_static']['per_scene_regret'][s]
                 for s in sorted(set(a['scene'][val]))]
        bootstrap = np.random.default_rng(123).choice(pairs, (2000, len(pairs)), replace=True).mean(1)
        result['joint_minus_history_regret_ci95'] = np.quantile(bootstrap, [.025, .975]).tolist()
        with (args.output/'metrics.jsonl').open('a') as f:
            f.write(json.dumps(result)+'\n')
        brief = {n:{k:v for k,v in r.items() if k!='per_scene_regret'} for n,r in result['controls'].items()}
        print(json.dumps(dict(step=step, controls=brief, occupancy_iou={n:r['occupancy_iou'] for n,r in result['occupancy'].items()})), flush=True)
        atomic_json(args.output/'status.json', dict(state='running', step=step, metrics=result))
        return result

    print(json.dumps(protocol), flush=True)
    initial = evaluate(0)
    best = {name: float('inf') for name in names}
    for step in range(1, args.steps+1):
        rows = rng.choice(train, args.batch_size, replace=True)
        choices = rng.integers(0, 3, args.batch_size)
        h, action, status, target = batch(rows, choices)
        with torch.no_grad():
            static, _ = warp_current(h[:, -1], action[:, 4], mode='bilinear')
            fixed = frozen(h, action, status)
        for name in ('history_static', 'frozen_future'):
            heads[name].train()
            optimizers[name].zero_grad(set_to_none=True)
            pred = heads[name](h, static if name=='history_static' else fixed, action, status)
            loss = task_loss(pred, target, positive_weight)
            loss.backward()
            optimizers[name].step()
        if step <= args.warmup:
            heads['joint_future'].load_state_dict(heads['frozen_future'].state_dict())
            if step == args.warmup:
                copy_head_training_state(heads['frozen_future'], optimizers['frozen_future'],
                                         heads['joint_future'], optimizers['joint_future'])
        else:
            joint.train()
            heads['joint_future'].train()
            wm_optimizer.zero_grad(set_to_none=True)
            optimizers['joint_future'].zero_grad(set_to_none=True)
            future = joint(h, action, status)
            risk_feature = future.detach() if args.stop_task_gradient else future
            risk = task_loss(heads['joint_future'](h, risk_feature, action, status), target, positive_weight)
            if step == args.warmup+1 and not args.stop_task_gradient:
                gradients = torch.autograd.grad(risk, tuple(joint.parameters()), retain_graph=True, allow_unused=True)
                norm = sum(float(g.square().sum()) for g in gradients if g is not None)**.5
                if not np.isfinite(norm) or norm <= 0:
                    raise RuntimeError('task loss does not reach WM')
                atomic_json(args.output/'gradient_check.json', dict(task_to_wm_l2=norm, step=step))
            target_latent = cache['target'][rows, choices].to(device).float()
            maps = torch.as_tensor(a['target_map'][rows, choices], device=device)
            occ = torch.as_tensor(a['target_occ'][rows, choices], device=device)
            semantic = semantic_loss(ae.decoder(future), maps, occ, weights)
            loss = (risk + semantic + .2*latent_distance(future, target_latent)
                    + args.distill_weight*F.mse_loss(future, fixed))
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite joint loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(joint.parameters(), 1.)
            wm_optimizer.step()
            optimizers['joint_future'].step()
        if step % args.eval_every == 0 or step == args.steps:
            result = evaluate(step)
            for name in names:
                score = result['controls'][name]['proximity_mae']
                if score < best[name]:
                    best[name] = score
                    torch.save(dict(step=step, predictor=(joint if name=='joint_future' else frozen).state_dict(),
                                    head=heads[name].state_dict(), autoencoder=ae.state_dict(), config=asdict(cfg),
                                    control=name, protocol=protocol, metrics=result), args.output/(name+'_best.pt'))
            torch.save(dict(step=step, predictor=joint.state_dict(), autoencoder=ae.state_dict(),
                            heads={n:v.state_dict() for n,v in heads.items()}, config=asdict(cfg), protocol=protocol,
                            optimizers={n:o.state_dict() for n,o in optimizers.items()}, wm_optimizer=wm_optimizer.state_dict(),
                            rng_state=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state_all()), args.output/'latest.pt')
    atomic_json(args.output/'status.json', dict(state='completed', step=step, metrics=result,
                best_proximity_mae=best, caveat=protocol['limitations']))


if __name__ == '__main__':
    main()
