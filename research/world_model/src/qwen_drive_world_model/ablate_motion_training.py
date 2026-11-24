"""Matched V4 continuation ablations. No QD model or final RFS claims."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .latent_motion_model import (LatentConfig, LatentPredictor, StateAutoencoder,
    latent_distance, pack_state, velocity_auxiliary_loss, warp_current)
from .train_latent_motion import Counts, atomic_json


VARIANTS = {
    'control': {},
    'no_latent': {'latent': 'none'},
    'latent02': {'latent_weight': .2},
    'raw_smooth': {'latent': 'raw_smooth'},
    'raw_mse': {'latent': 'raw_mse'},
    'map02': {'map_weight': .2},
    'no_velocity': {'velocity_weight': 0.},
    'decoder': {'decoder': True},
    'decoder_recon': {'decoder': True, 'reconstruction_weight': 1.},
    'encoder_decoder': {'encoder': True, 'decoder': True, 'reconstruction_weight': 1.},
}


def latent_term(prediction, target, kind):
    if kind == 'normalized':
        return latent_distance(prediction, target)
    if kind == 'raw_smooth':
        return F.smooth_l1_loss(prediction, target)
    if kind == 'raw_mse':
        return F.mse_loss(prediction, target)
    if kind == 'none':
        return prediction.sum()*0.
    raise ValueError(kind)


def semantic_parts(logits, maps, occ, weights):
    map_loss = F.cross_entropy(logits[:,:8], maps.long(), weight=weights)
    bce = F.binary_cross_entropy_with_logits(logits[:,8:], occ.float(), pos_weight=logits.new_tensor(4.))
    p = logits[:,8:].sigmoid()
    intersection = (p*occ).sum((0,2,3))
    denominator = p.sum((0,2,3))+occ.sum((0,2,3))
    dice = (1-(2*intersection+1)/(denominator+1)).mean()
    return dict(map=map_loss, bce=bce, dice=dice)


def aggregate_scene_iou(counts):
    values = np.asarray(list(counts.values()),dtype=np.int64)
    tp, fp, fn = values.sum(0)
    return float(tp/max(tp+fp+fn,1))


class Experiment:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda:0')
        self.ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        self.cfg = LatentConfig(**self.ckpt['config'])
        with np.load(args.data/'windows.npz',allow_pickle=False) as data:
            self.a = {k:data[k] for k in data.files}
        actual_hash = hashlib.sha256((args.data/'windows.npz').read_bytes()).hexdigest()
        if actual_hash != self.ckpt['protocol']['dataset_sha256']:
            raise ValueError('dataset differs from parent')
        self.train = np.flatnonzero(self.a['split']=='train')
        with np.load(args.val_reference/'windows.npz',allow_pickle=False) as ref:
            selected = np.flatnonzero(ref['split']=='val')
            lookup = {(str(s),int(t)):i for i,(s,t) in enumerate(zip(self.a['scene'],self.a['anchor']))}
            self.val = np.array([lookup[(str(ref['scene'][i]),int(ref['anchor'][i]))] for i in selected])
            for k in ('history_map','history_occ','actions','status','target_map','target_occ'):
                if not np.array_equal(self.a[k][self.val],ref[k][selected]):
                    raise ValueError('fixed development data changed: '+k)
        if set(self.a['scene'][self.train]) & set(self.a['scene'][self.val]):
            raise ValueError('scene leakage')
        self.cache = torch.load(self.ckpt['protocol']['frozen_latent_cache'],map_location='cpu',mmap=True,weights_only=False)
        teacher_path = Path(self.ckpt['protocol']['frozen_latent_cache']).parent/'autoencoder_best.pt'
        if self.cache['teacher_sha256']!=hashlib.sha256(teacher_path.read_bytes()).hexdigest():
            raise ValueError('teacher cache hash mismatch')
        teacher = torch.load(teacher_path,map_location='cpu',weights_only=False)
        if any(not torch.equal(v,teacher['model'][k]) for k,v in self.ckpt['autoencoder'].items()):
            raise ValueError('teacher and parent differ')
        meta = json.loads(args.velocity.with_suffix('.json').read_text())
        if meta['dataset_sha256']!=actual_hash:
            raise ValueError('velocity data mismatch')
        with np.load(args.velocity) as vel:
            if not np.array_equal(vel['scene'],self.a['scene']) or not np.array_equal(vel['anchor'],self.a['anchor']):
                raise ValueError('velocity row mismatch')
            self.velocity,self.velocity_mask = vel['velocity'],vel['mask']
        freq = np.bincount(self.a['target_map'][self.train].ravel(),minlength=8).astype(float)
        weights = 1/np.sqrt(freq/freq.sum()+1e-3)
        self.weights = torch.tensor(np.clip(weights/weights.mean(),.25,4),device=self.device,dtype=torch.float32)
        self.original = StateAutoencoder(self.cfg).to(self.device).eval().requires_grad_(False)
        self.original.load_state_dict(self.ckpt['autoencoder'])
        self.protocol = dict(dataset_sha256=actual_hash,checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                             teacher_sha256=self.cache['teacher_sha256'],train_windows=len(self.train),val_windows=len(self.val),
                             args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                             config=asdict(self.cfg),
                             interpretation='Continuation from selected V4, not training from scratch; all optimizers reset.',
                             raw_loss_scaling='Match initial WM latent-only gradient norm on 4 training batches; coefficients then fixed.',
                             target_encoder='Original frozen encoder; targets always original verified cache even when online input encoder trains.',
                             regions='changed/stable are target versus static ego-warp raster regions; not exact moving/static objects',
                             checkpoint_selection='maximum fixed development pooled occupancy IoU, includes initial step0',
                             randomness='same candidate sequence/history masks per seed across variants; CUDA kernels may be nondeterministic')
        atomic_json(args.output/'protocol.json',self.protocol)
        source = args.output/'source'
        source.mkdir()
        for name in ('ablate_motion_training.py','latent_motion_model.py','train_latent_motion.py'):
            (source/name).write_bytes(Path(__file__).with_name(name).read_bytes())

    def tensor(self, value):
        return torch.as_tensor(value,device=self.device)

    def states(self, rows):
        maps = self.tensor(self.a['history_map'][rows])[:,None].expand(-1,4,-1,-1)
        return pack_state(maps,self.tensor(self.a['history_occ'][rows])).flatten(0,1)

    def history(self, ae, rows, online):
        if online:
            return ae.encoder(self.states(rows)).reshape(len(rows),4,self.cfg.channels,self.cfg.token_h,self.cfg.token_w)
        return self.cache['history'][rows].to(self.device).float()

    def latent_scale(self, wm, kind):
        if kind in ('none','normalized'):
            return 1.,[]
        parameters = tuple(wm.parameters())
        rng = np.random.default_rng(1729)
        ratios=[]
        for _ in range(4):
            rows=rng.choice(self.train,self.args.batch_size,replace=True)
            choices=rng.integers(0,3,len(rows))
            h=self.history(self.original,rows,False)
            pred=wm(h,self.tensor(self.a['actions'][rows,choices]),self.tensor(self.a['status'][rows]))
            target=self.cache['target'][rows,choices].to(self.device).float()
            norms=[]
            for mode in ('normalized',kind):
                grad=torch.autograd.grad(latent_term(pred,target,mode),parameters,retain_graph=True,allow_unused=True)
                norms.append(sum(float(g.square().sum()) for g in grad if g is not None)**.5)
            ratios.append(norms[0]/max(norms[1],1e-12))
        return float(np.median(ratios)),ratios

    @torch.no_grad()
    def evaluate(self, wm, ae, ids, online=False, probe=True):
        wm.eval();ae.eval()
        counts, old_decoder, reconstruction = Counts(),Counts(),Counts()
        region_sums = np.zeros(8,dtype=np.int64)
        brier_sums=np.zeros(4,dtype=np.float64)
        scene_counts={}
        losses={k:0. for k in ('normalized_latent','raw_smooth','raw_mse','map','bce','dice','velocity')}
        n=0
        for start in range(0,len(ids)*3,32):
            flat=np.arange(start,min(start+32,len(ids)*3))
            rows,choices=ids[flat//3],flat%3
            h=self.history(ae,rows,online)
            action=self.tensor(self.a['actions'][rows,choices]);status=self.tensor(self.a['status'][rows])
            target=self.cache['target'][rows,choices].to(self.device).float()
            occ=self.tensor(self.a['target_occ'][rows,choices]);maps=self.tensor(self.a['target_map'][rows,choices])
            pred=wm(h,action,status);logits=ae.decoder(pred);p=logits[:,8:].sigmoid()
            counts.update(p,occ,logits[:,:8],maps)
            parts=semantic_parts(logits,maps,occ,self.weights)
            vals=dict(normalized_latent=latent_term(pred,target,'normalized'),raw_smooth=latent_term(pred,target,'raw_smooth'),
                      raw_mse=latent_term(pred,target,'raw_mse'),**parts,
                      velocity=velocity_auxiliary_loss(wm.aux_velocity,self.tensor(self.velocity[rows]),
                                                     self.tensor(self.velocity_mask[rows]),action[:,4]))
            for key,value in vals.items():losses[key]+=float(value)*len(rows)
            n+=len(rows)
            truth=occ.bool().any(1);prediction=p.ge(.5).any(1)
            warped,valid=warp_current(self.tensor(self.a['history_occ'][rows,-1]).float(),action[:,4])
            current=warped.bool().any(1)
            stable=truth&current;changed=truth&~current;ghost=current&~truth
            values=[(prediction&stable).sum(),stable.sum(),(prediction&changed&valid).sum(),(changed&valid).sum(),
                    (prediction&changed&~valid).sum(),(changed&~valid).sum(),(prediction&ghost).sum(),ghost.sum()]
            region_sums+=np.array([int(v) for v in values])
            error=(p-occ.float()).square()
            brier_sums+=np.array([float(error[occ.bool()].sum()),int(occ.bool().sum()),
                                 float(error[~occ.bool()].sum()),int((~occ.bool()).sum())])
            per=torch.stack(((prediction&truth).sum((1,2)),(prediction&~truth).sum((1,2)),(~prediction&truth).sum((1,2))),1).cpu().numpy()
            for scene,value in zip(self.a['scene'][rows],per):
                scene_counts.setdefault(str(scene),np.zeros(3,dtype=np.int64))[:] += value
            if probe:
                old_decoder.update(self.original.decoder(pred)[:,8:].sigmoid(),occ)
                reconstruction.update(ae.decoder(target)[:,8:].sigmoid(),occ)
        result=dict(**counts.result(),losses={k:v/n for k,v in losses.items()},
                    stable_target_recall=float(region_sums[0]/max(region_sums[1],1)),
                    changed_inside_recall=float(region_sums[2]/max(region_sums[3],1)),
                    changed_outside_recall=float(region_sums[4]/max(region_sums[5],1)),
                    ghost_retention=float(region_sums[6]/max(region_sums[7],1)),
                    positive_brier=float(brier_sums[0]/max(brier_sums[1],1)),
                    negative_brier=float(brier_sums[2]/max(brier_sums[3],1)),
                    per_scene_counts={s:v.tolist() for s,v in scene_counts.items()})
        if probe:result.update(original_decoder_probe=old_decoder.result(),target_reconstruction=reconstruction.result())
        return result

    def run(self, name, seed, subset=None):
        settings=dict(latent='normalized',latent_weight=1.,map_weight=.5,velocity_weight=.5,
                      decoder=False,encoder=False,reconstruction_weight=0.,**{})
        settings.update(VARIANTS[name] if name!='overfit32' else {})
        directory=self.args.output/f'{name}_seed{seed}'
        if directory.exists():
            status=directory/'status.json'
            if status.exists() and json.loads(status.read_text())['state']=='completed':
                print('already completed',directory.name,flush=True)
                return
            raise FileExistsError('refusing to overwrite incomplete experiment '+str(directory))
        directory.mkdir()
        wm=LatentPredictor(self.cfg).to(self.device)
        wm.load_state_dict(self.ckpt['predictor'])
        ae=copy.deepcopy(self.original)
        ae.decoder.requires_grad_(settings['decoder'])
        ae.encoder.requires_grad_(settings['encoder'])
        scale,ratios=self.latent_scale(wm,settings['latent'])
        scale*=settings['latent_weight']
        # Reset sampling streams after all construction/calibration for matched runs.
        rng=np.random.default_rng(seed)
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        ids=self.train if subset is None else np.asarray(subset)
        steps=self.args.steps if subset is None else self.args.overfit_steps
        lr=self.args.lr if subset is None else 1e-4
        groups=[dict(params=wm.parameters(),lr=lr)]
        if settings['decoder']:groups.append(dict(params=ae.decoder.parameters(),lr=self.args.ae_lr))
        if settings['encoder']:groups.append(dict(params=ae.encoder.parameters(),lr=self.args.ae_lr))
        optimizer=torch.optim.AdamW(groups,weight_decay=1e-3)
        run_protocol=dict(**self.protocol,variant=name,seed=seed,settings=settings,
                          latent_coefficient=scale,latent_gradient_matching_ratios=ratios,
                          actual_steps=steps,actual_lr=lr,train_windows_this_run=len(ids),
                          fitting_diagnostic=subset is not None)
        atomic_json(directory/'protocol.json',run_protocol)
        initial=self.evaluate(wm,ae,self.val,settings['encoder'])
        best=initial['occupancy_iou'];best_step=0;selected=initial
        started=time.perf_counter();interval={};interval_n=0

        def checkpoint(step,result,filename,include_optimizer=False):
            value=dict(predictor=wm.state_dict(),autoencoder=ae.state_dict(),config=asdict(self.cfg),
                       step=step,metrics=result,protocol=run_protocol)
            if include_optimizer:value.update(optimizer=optimizer.state_dict(),numpy_rng=rng.bit_generator.state,
                                               torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
            temp=directory/(filename+'.tmp');torch.save(value,temp);temp.replace(directory/filename)

        def record(step,result,train_losses=None,fit=None):
            payload=dict(step=step,seconds=time.perf_counter()-started,val=result,
                         train_interval_mean=train_losses,fit_training=fit)
            with (directory/'metrics.jsonl').open('a') as f:f.write(json.dumps(payload)+'\n')
            atomic_json(directory/'status.json',dict(state='running',step=step,best_step=best_step,
                                                    best_iou=best,latest=payload))
            print(json.dumps(dict(run=directory.name,step=step,val_iou=result['occupancy_iou'],
                                  changed_inside=result['changed_inside_recall'],ghost=result['ghost_retention'],
                                  fit_iou=None if fit is None else fit['occupancy_iou'])),flush=True)

        checkpoint(0,initial,'best.pt');record(0,initial)
        for step in range(1,steps+1):
            wm.train();ae.train()
            rows=rng.choice(ids,self.args.batch_size,replace=True)
            choices=rng.integers(0,3,len(rows))
            h=self.history(ae,rows,settings['encoder'])
            action=self.tensor(self.a['actions'][rows,choices]);status=self.tensor(self.a['status'][rows])
            target=self.cache['target'][rows,choices].to(self.device).float()
            maps=self.tensor(self.a['target_map'][rows,choices]);occ=self.tensor(self.a['target_occ'][rows,choices])
            history_mask=(torch.rand(len(rows),4,device=self.device)>.1).float();history_mask[:,-1]=1
            pred=wm(h,action,status,history_mask)
            parts=semantic_parts(ae.decoder(pred),maps,occ,self.weights)
            latent=latent_term(pred,target,settings['latent'])
            velocity=velocity_auxiliary_loss(wm.aux_velocity,self.tensor(self.velocity[rows]),
                                           self.tensor(self.velocity_mask[rows]),action[:,4])
            reconstruction=pred.new_zeros(())
            if settings['reconstruction_weight']:
                rp=semantic_parts(ae.decoder(h[:,-1]),self.tensor(self.a['history_map'][rows]),
                                  self.tensor(self.a['history_occ'][rows,-1]),self.weights)
                reconstruction=.5*rp['map']+rp['bce']+.5*rp['dice']
            loss=(scale*latent+settings['map_weight']*parts['map']+parts['bce']+.5*parts['dice']
                  +settings['velocity_weight']*velocity+settings['reconstruction_weight']*reconstruction)
            if not torch.isfinite(loss):raise RuntimeError('nonfinite loss '+directory.name)
            optimizer.zero_grad(set_to_none=True);loss.backward()
            # Keep WM clipping identical across controls; clip new AE groups separately.
            torch.nn.utils.clip_grad_norm_(wm.parameters(),1.)
            if settings['encoder']:torch.nn.utils.clip_grad_norm_(ae.encoder.parameters(),1.)
            if settings['decoder']:torch.nn.utils.clip_grad_norm_(ae.decoder.parameters(),1.)
            optimizer.step()
            values=dict(total=loss,latent=latent,weighted_latent=scale*latent,**parts,velocity=velocity,reconstruction=reconstruction)
            for k,v in values.items():interval[k]=interval.get(k,0.)+float(v.detach())
            interval_n+=1
            if step%self.args.eval_every==0 or step==steps:
                result=self.evaluate(wm,ae,self.val,settings['encoder'])
                fit=self.evaluate(wm,ae,ids,settings['encoder'],probe=False) if subset is not None else None
                means={k:v/interval_n for k,v in interval.items()};interval={};interval_n=0
                if result['occupancy_iou']>best:
                    best=result['occupancy_iou'];best_step=step;selected=result
                    checkpoint(step,result,'best.pt')
                record(step,result,means,fit)
                checkpoint(step,result,'latest.pt',include_optimizer=True)
                if fit is not None and fit['occupancy_iou']>=.95:break
        atomic_json(directory/'status.json',dict(state='completed',step=step,best_step=best_step,
                    best_iou=best,selected=selected,last=result,
                    fit_training=fit if subset is not None else None,seconds=time.perf_counter()-started))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('outputs/world_model/motion_latent_v3/dataset2048'))
    p.add_argument('--checkpoint',type=Path,default=Path('outputs/world_model/motion_latent_v4/velocity_aux/best.pt'))
    p.add_argument('--velocity',type=Path,default=Path('outputs/world_model/motion_latent_v4/velocity_targets.npz'))
    p.add_argument('--val-reference',type=Path,default=Path('outputs/world_model/motion_latent_v2/dataset'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=2400)
    p.add_argument('--overfit-steps',type=int,default=4000)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--lr',type=float,default=3e-5)
    p.add_argument('--ae-lr',type=float,default=1e-5)
    p.add_argument('--eval-every',type=int,default=800)
    p.add_argument('--seeds',type=int,nargs='+',default=[43,44])
    p.add_argument('--variants',nargs='+',choices=list(VARIANTS),default=list(VARIANTS))
    p.add_argument('--skip-overfit',action='store_true')
    args=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    if args.output.exists():raise FileExistsError('use a fresh suite directory')
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    experiment=Experiment(args)
    atomic_json(args.output/'status.json',dict(state='running',completed=[]))
    completed=[]
    if not args.skip_overfit:
        subset=np.random.default_rng(123).choice(experiment.train,32,replace=False)
        np.save(args.output/'overfit_rows.npy',subset)
        experiment.run('overfit32',args.seeds[0],subset)
        completed.append('overfit32')
    for seed in args.seeds:
        for name in args.variants:
            experiment.run(name,seed)
            completed.append(f'{name}_seed{seed}')
            atomic_json(args.output/'status.json',dict(state='running',completed=completed))
    atomic_json(args.output/'status.json',dict(state='completed',completed=completed))


if __name__=='__main__':main()
