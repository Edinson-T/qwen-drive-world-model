"""Train ONE free-running object model jointly at .5/1/2/3/5 seconds."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .instance_motion_model import history_features
from .recurrent_instance_motion import (RecurrentInstanceMotionPredictor, rollout_boxes,
                                        multihorizon_loss, OUTPUT_STEPS, OUTPUT_HORIZONS)
from .train_instance_motion import rasterize
from .train_latent_motion import atomic_json


class RolloutTrial:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        with np.load(args.instances / 'instances.npz') as a:
            self.a = {k: a[k] for k in ('history', 'mask', 'category', 'scene', 'anchor', 'offsets', 'split', 'gt_velocity')}
        with np.load(args.targets / 'targets.npz') as a:
            self.y = {k: a[k] for k in a.files}
        meta = json.loads((args.targets / 'dataset.json').read_text())
        if hashlib.sha256((args.instances / 'instances.npz').read_bytes()).hexdigest() != meta['instance_sha256']:
            raise ValueError('history cache identity mismatch')
        for k in ('scene', 'anchor', 'offsets'):
            np.testing.assert_array_equal(self.a[k], self.y[k])
        with np.load(args.parent / 'windows.npz') as a:
            for k in ('scene', 'anchor'):
                np.testing.assert_array_equal(self.a[k], a[k])
            self.actions = a['actions']
        lookup = {(str(s), int(t)): i for i, (s, t) in enumerate(zip(self.a['scene'], self.a['anchor']))}
        evaluation_only = getattr(args, 'evaluation_only', False)
        if evaluation_only:
            self.val = np.arange(len(self.a['scene']))
        else:
            with np.load(args.val_reference / 'windows.npz') as a:
                self.val = np.asarray([lookup[(str(a['scene'][i]), int(a['anchor'][i]))]
                                       for i in np.flatnonzero(a['split'] == 'val')])
        train_scenes = set(self.a['scene'][self.a['split'] == 'train'])
        if not evaluation_only and train_scenes & set(self.a['scene'][self.val]):
            raise ValueError('scene leakage')
        self.train_indices = np.flatnonzero(np.repeat(self.a['split'] == 'train', np.diff(self.a['offsets'])))
        tensor = lambda v: torch.as_tensor(v, device=self.device)
        self.features, self.bases, self.omega, self.current = history_features(
            tensor(self.a['history']).float(), tensor(self.a['mask']), tensor(self.a['category']))
        self.target = tensor(self.y['target']).float()
        self.valid = tensor(self.y['valid'])
        self.eval_indices = np.concatenate([np.arange(*self.a['offsets'][row:row+2]) for row in self.val])
        # Cache full future targets: includes objects NOT present at the current query time.
        self.target_occ = {}
        for row in self.val:
            for h in OUTPUT_STEPS:
                begin, end = self.y['future_offsets'][row*10+h:row*10+h+2]
                poses = self.actions[row, :, (h+1)*2]
                self.target_occ[(row, h)] = np.stack([
                    rasterize(self.y['future_boxes'][begin:end], self.y['future_classes'][begin:end], pose,
                              np.ones(end-begin, bool)).any(0) for pose in poses])
        atomic_json(args.output / 'protocol.json', dict(
            args={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
            train_scenes=len(train_scenes), train_actor_windows=len(self.train_indices),
            development_scenes=len(set(self.a['scene'][self.val])), development_windows=len(self.val),
            output_times=OUTPUT_HORIZONS, transition_seconds=.5, shared_transition=True,
            free_running_training=True, teacher_forcing=False, future_targets_are_inputs=False,
            selection='evaluation only, preselected checkpoint' if evaluation_only else 'mean five-horizon development occupancy IoU; official holdout untouched',
            annotation_validity='not physical survival probability',
            scope='GT historical boxes; no numeric ID/GT velocity inputs; perturbed logged candidates, not RFS'))
        source = args.output / 'source'
        source.mkdir()
        for name in ('recurrent_instance_motion.py', 'train_instance_rollout.py', 'build_instance_rollout_targets.py'):
            (source/name).write_bytes(Path(__file__).with_name(name).read_bytes())

    @torch.no_grad()
    def predict(self, model):
        model.eval()
        boxes = np.zeros((len(self.current), 10, 5), np.float32)
        keep = np.ones((len(self.current), 10), bool)
        for start in range(0, len(self.eval_indices), 8192):
            indices = self.eval_indices[start:start+8192]
            p = model(self.features[indices], self.bases[indices], self.omega[indices])
            boxes[indices] = rollout_boxes(self.current[indices], p).cpu().numpy()
            keep[indices] = (p[3] > 0).cpu().numpy()
        return boxes, keep

    def evaluate(self, boxes, keep):
        report = {}
        for h, seconds in zip(OUTPUT_STEPS, OUTPUT_HORIZONS):
            counts = np.zeros(3, np.int64)
            counts_all = np.zeros(3, np.int64)
            per_scene = {}
            errors = []
            for row in self.val:
                begin, end = self.a['offsets'][row:row+2]
                poses = self.actions[row, :, (h+1)*2]
                target = self.target_occ[(row, h)]
                for forced, counter in ((False, counts), (True, counts_all)):
                    pred = np.stack([rasterize(boxes[begin:end, h], self.a['category'][begin:end], pose,
                                              np.ones(end-begin, bool) if forced else keep[begin:end, h]).any(0)
                                     for pose in poses])
                    added = np.array([(pred & target).sum(), (pred & ~target).sum(), (~pred & target).sum()])
                    counter += added
                    if not forced:
                        sid = str(self.a['scene'][row])
                        per_scene.setdefault(sid, np.zeros(3, np.int64))[:] += added
                valid = self.y['valid'][begin:end, h].astype(bool)
                # Report on all future-valid currently queried actors, no keep/crop filtering.
                errors.extend(np.linalg.norm(boxes[begin:end, h, :2][valid] - self.y['target'][begin:end, h, :2][valid], axis=-1))
            iou = lambda c: float(c[0]/max(c.sum(), 1))
            report[str(seconds)] = dict(occupancy_iou=iou(counts), keep_all_iou=iou(counts_all),
                                        center_epe_m=float(np.mean(errors)) if errors else None,
                                        matched_actor_occurrences=len(errors), counts=counts.tolist(),
                                        per_scene_counts={s:v.tolist() for s,v in per_scene.items()})
        return dict(horizons=report, mean_iou=float(np.mean([v['occupancy_iou'] for v in report.values()])))

    def train(self):
        args = self.args
        torch.manual_seed(args.seed)
        rng = np.random.default_rng(args.seed)
        model = RecurrentInstanceMotionPredictor(args.hidden).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        # Equal source population, full target population, identical candidate/grid per horizon.
        cv = np.repeat(self.a['history'][:, -1, None], 10, axis=1)
        cv[:, :, :2] += self.a['gt_velocity'][:, None] * np.arange(.5, 5.01, .5)[None, :, None]
        baseline = self.evaluate(cv, np.ones(cv.shape[:2], bool))
        atomic_json(args.output / 'baselines.json', dict(gt_velocity_cv=baseline))
        print('baseline', json.dumps(baseline), flush=True)
        best = -1.
        losses = []
        started = time.monotonic()
        for step in range(args.steps+1):
            if step:
                model.train()
                indices = rng.choice(self.train_indices, args.batch_size, replace=True)
                pred = model(self.features[indices], self.bases[indices], self.omega[indices])
                loss, per_horizon = multihorizon_loss(pred, self.current[indices], self.target[indices], self.valid[indices])
                if not torch.isfinite(loss):
                    raise RuntimeError('nonfinite loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
                losses.append(float(loss.detach()))
            if step % args.eval_every == 0 or step == args.steps:
                metrics = self.evaluate(*self.predict(model))
                row = dict(step=step, seconds=time.monotonic()-started,
                           train_loss=float(np.mean(losses)) if losses else None, val=metrics)
                losses = []
                with (args.output/'metrics.jsonl').open('a') as f:
                    f.write(json.dumps(row)+'\n')
                checkpoint = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=step,
                                  hidden=args.hidden, seed=args.seed, horizons=OUTPUT_HORIZONS, metrics=metrics)
                torch.save(checkpoint, args.output/'latest.pt')
                if metrics['mean_iou'] > best:
                    best = metrics['mean_iou']
                    torch.save(checkpoint, args.output/'best.pt')
                atomic_json(args.output/'status.json', dict(state='completed' if step == args.steps else 'running',
                                                           step=step, best_mean_iou=best, latest=row))
                print(json.dumps({**row, 'val': dict(mean_iou=metrics['mean_iou'], horizons={
                    h:{k:v for k,v in m.items() if k != 'per_scene_counts'} for h,m in metrics['horizons'].items()})}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('instances', 'targets', 'parent', 'val-reference', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--steps', type=int, default=4000)
    p.add_argument('--batch-size', type=int, default=2048)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--eval-every', type=int, default=500)
    p.add_argument('--seed', type=int, default=43)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    RolloutTrial(args).train()


if __name__ == '__main__':
    main()
