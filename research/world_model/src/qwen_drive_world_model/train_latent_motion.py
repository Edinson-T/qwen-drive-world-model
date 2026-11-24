"""Two-stage Motion latent WM pilot. Does not load or overwrite legacy models."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .latent_motion_model import (LatentConfig, StateAutoencoder, LatentPredictor,
                                  pack_state, semantic_loss, latent_distance, warp_current)


class Windows(Dataset):
    def __init__(self, arrays, indices):
        self.a, self.indices = arrays, np.asarray(indices)

    def __len__(self):
        return len(self.indices) * 3

    def __getitem__(self, i):
        row, candidate = int(self.indices[i//3]), i % 3
        a = self.a
        sample=dict(row=row, candidate=candidate,
                    maps=a['target_map'][row, candidate], occ=a['target_occ'][row, candidate],
                    action=a['actions'][row, candidate], alternate=a['actions'][row, (candidate+1)%3],
                    status=a['status'][row], current_map=a['history_map'][row],
                    history_occ=a['history_occ'][row], cv=a['cv_occ'][row, candidate])
        if 'motion_velocity' in a:
            sample.update(motion_velocity=a['motion_velocity'][row],motion_mask=a['motion_mask'][row])
        return sample


def device_batch(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def state_batch(batch, history=False):
    if history:
        maps = batch['current_map'][:, None].expand(-1, 4, -1, -1)
        return pack_state(maps, batch['history_occ']).flatten(0, 1)
    return pack_state(batch['maps'], batch['occ'])


class Counts:
    def __init__(self):
        self.tp = self.fp = self.fn = 0
        self.class_counts = np.zeros((4, 3), dtype=np.int64)
        self.confusion = np.zeros((8, 8), dtype=np.int64)
        self.positives = self.targets = self.pixels = 0

    def update(self, probability, target, maps=None, labels=None, valid=None):
        prediction = probability >= .5
        truth = target.bool()
        union_prediction, union_truth = prediction.any(1), truth.any(1)
        mask = torch.ones_like(union_truth) if valid is None else valid
        self.tp += int((union_prediction & union_truth & mask).sum())
        self.fp += int((union_prediction & ~union_truth & mask).sum())
        self.fn += int((~union_prediction & union_truth & mask).sum())
        self.positives += int((union_prediction & mask).sum())
        self.targets += int((union_truth & mask).sum())
        self.pixels += int(mask.sum())
        for c in range(4):
            p, t = prediction[:, c], truth[:, c]
            self.class_counts[c] += [int((p&t&mask).sum()), int((p&~t&mask).sum()), int((~p&t&mask).sum())]
        if maps is not None:
            values = labels.long()*8+maps.argmax(1)
            self.confusion += torch.bincount(values[mask].flatten(), minlength=64).reshape(8, 8).cpu().numpy()

    def result(self):
        d = np.diag(self.confusion)
        union = self.confusion.sum(0)+self.confusion.sum(1)-d
        present = union > 0
        return dict(occupancy_iou=self.tp/max(self.tp+self.fp+self.fn, 1),
                    precision=self.tp/max(self.tp+self.fp, 1), recall=self.tp/max(self.tp+self.fn, 1),
                    predicted_fraction=self.positives/max(self.pixels, 1), target_fraction=self.targets/max(self.pixels, 1),
                    class_iou=[float(tp/max(tp+fp+fn, 1)) for tp,fp,fn in self.class_counts],
                    map_miou=float((d[present]/union[present]).mean()) if present.any() else None)


def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def selection_score(metrics, criterion):
    # Lower is better internally. Semantic losses remain logged for every selection policy.
    return -metrics['occupancy_iou'] if criterion=='occupancy_iou' else metrics['loss']


def cycle(loader):
    while True:
        yield from loader


def provenance():
    result = {}
    for name in ('latent_motion_data.py', 'latent_motion_model.py', 'train_latent_motion.py'):
        path = Path(__file__).with_name(name)
        result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--ae-steps', type=int, default=1500)
    p.add_argument('--wm-steps', type=int, default=2000)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--eval-every', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--overfit-windows', type=int, default=0)
    p.add_argument('--channels', type=int, default=128)
    p.add_argument('--token-height',type=int,default=8)
    p.add_argument('--token-width',type=int,default=16)
    p.add_argument('--detail-encoder',action='store_true')
    p.add_argument('--ego-warp-residual',action='store_true')
    p.add_argument('--motion-context',action='store_true')
    p.add_argument('--motion-aux',action='store_true')
    p.add_argument('--upgrade-motion-aux',action='store_true')
    p.add_argument('--motion-targets',type=Path,default=None)
    p.add_argument('--upgrade-motion-context',action='store_true',
                   help='Allow adding only the zero-initialized wider motion branch to a pretrained WM.')
    p.add_argument('--reuse-latent-cache',action='store_true')
    p.add_argument('--init-autoencoder',type=Path,default=None,
                   help='Reuse a frozen reconstruction teacher for a predictor-only architecture experiment.')
    p.add_argument('--finetune-from',type=Path,default=None,
                   help='Transfer WM and frozen teacher to new training data; rebuild latents and reset optimizer.')
    p.add_argument('--val-scenes-from',type=Path,default=None,
                   help='Keep exactly the validation windows from this original windows.npz for data expansion.')
    p.add_argument('--wm-lr',type=float,default=3e-4)
    p.add_argument('--selection-metric',choices=('loss','occupancy_iou'),default='loss')
    p.add_argument('--resume', type=Path, default=None,
                   help='Continue WM weights and optimizer; reuse its frozen autoencoder and latent cache.')
    p.add_argument('--early-stop-patience', type=int, default=0,
                   help='Validation checks without a loss improvement; 0 disables early stopping.')
    p.add_argument('--early-stop-min-delta', type=float, default=1e-4)
    args = p.parse_args()
    config=LatentConfig(channels=args.channels,token_h=args.token_height,token_w=args.token_width,
                        detail_encoder=args.detail_encoder,ego_warp_residual=args.ego_warp_residual,
                        motion_context=args.motion_context,motion_aux=args.motion_aux)
    if args.motion_aux and (not args.motion_context or not args.motion_targets):
        raise ValueError('motion auxiliary training requires motion-context and motion-targets')
    if sum(v is not None for v in (args.resume,args.init_autoencoder,args.finetune_from))>1:
        raise ValueError('choose only one of resume, init-autoencoder, finetune-from')
    if (args.output/'status.json').exists():
        raise FileExistsError('use a fresh output directory to preserve run provenance')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    torch.set_num_threads(4)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; refusing to silently start a long CPU run')
    device = torch.device(args.device)
    manifest = json.loads((args.data/'dataset.json').read_text())
    if manifest['version'] != 'motion_latent_v2.0':
        raise ValueError('incompatible data protocol')
    resumed = None
    transferred = None
    if args.resume:
        resumed = torch.load(args.resume, map_location=device, weights_only=False)
        if resumed['protocol']['dataset'] != manifest:
            raise ValueError('resume dataset protocol differs from checkpoint')
        if asdict(LatentConfig(**resumed['config'])) != asdict(config):
            raise ValueError('resume architecture differs from requested configuration')
        if resumed['protocol']['args']['batch_size'] != args.batch_size:
            raise ValueError('keep batch size fixed for the continuation control experiment')
        if args.wm_steps <= resumed['step']:
            raise ValueError('--wm-steps is the TOTAL step budget and must exceed the checkpoint step')
        if bool(args.overfit_windows) != bool(resumed['protocol']['args'].get('overfit_windows', 0)):
            raise ValueError('cannot mix overfit diagnostics with a generalization run')
    if args.finetune_from:
        transferred=torch.load(args.finetune_from,map_location=device,weights_only=False)
        previous_config=asdict(LatentConfig(**transferred['config']))
        if args.upgrade_motion_context:
            if previous_config['motion_context'] or not config.motion_context or not config.ego_warp_residual:
                raise ValueError('only false->true motion-context upgrade is allowed')
            previous_config['motion_context']=True
        if args.upgrade_motion_aux:
            if previous_config['motion_aux'] or not config.motion_aux:
                raise ValueError('only false->true motion auxiliary upgrade is allowed')
            previous_config['motion_aux']=True
        if previous_config!=asdict(config):
            raise ValueError('finetune architecture mismatch')
        previous=transferred['protocol']['dataset']
        for key in ('version','grid','history_times','action_times','horizon','target_frame','object_classes'):
            if manifest[key]!=previous[key]:
                raise ValueError(f'finetune data semantics mismatch: {key}')
    with np.load(args.data/'windows.npz', allow_pickle=False) as payload:
        a = {k: payload[k] for k in payload.files}
    if args.motion_targets:
        target_meta=json.loads(args.motion_targets.with_suffix('.json').read_text())
        if target_meta['dataset_sha256']!=hashlib.sha256((args.data/'windows.npz').read_bytes()).hexdigest():
            raise ValueError('motion supervision belongs to a different dataset')
        with np.load(args.motion_targets,allow_pickle=False) as supervision:
            if not np.array_equal(supervision['scene'],a['scene']) or not np.array_equal(supervision['anchor'],a['anchor']):
                raise ValueError('motion supervision row alignment mismatch')
            a['motion_velocity']=supervision['velocity'];a['motion_mask']=supervision['mask']
    train = np.flatnonzero(a['split']=='train'); val = np.flatnonzero(a['split']=='val')
    if set(a['scene'][train]) & set(a['scene'][val]):
        raise ValueError('scenario leakage')
    if not len(train) or not len(val):
        raise ValueError('both train and val are required; no fallback to training set')
    if args.val_scenes_from:
        with np.load(args.val_scenes_from,allow_pickle=False) as reference:
            selected=np.flatnonzero(reference['split']=='val')
            scenes=set(reference['scene'][selected])
            if scenes & set(a['scene'][train]):
                raise ValueError('original validation scenes would enter expanded training set')
            lookup={(str(a['scene'][i]),int(a['anchor'][i])):int(i) for i in val}
            keys=[(str(reference['scene'][i]),int(reference['anchor'][i])) for i in selected]
            if not all(key in lookup for key in keys):
                raise ValueError('expanded dataset is missing original validation windows')
            val=np.asarray([lookup[key] for key in keys])
            for key in ('history_map','history_occ','target_map','target_occ','actions','status','cv_occ'):
                if not np.array_equal(a[key][val],reference[key][selected]):
                    raise ValueError(f'fixed validation arrays changed: {key}')
    if args.overfit_windows:
        train = train[:args.overfit_windows]; val = train.copy()
    loaders = {k: DataLoader(Windows(a, ids), batch_size=args.batch_size, shuffle=k=='train', num_workers=0,
                             pin_memory=device.type=='cuda') for k,ids in [('train',train),('val',val)]}
    ae = StateAutoencoder(config).to(device)
    counts = np.bincount(a['target_map'][train].ravel(), minlength=8).astype(float)
    weights = 1/np.sqrt(counts/counts.sum()+1e-3); weights /= weights.mean()
    weights = torch.tensor(np.clip(weights, .25, 4), dtype=torch.float32, device=device)
    history = []
    if resumed and (args.resume.parent/'metrics.json').exists():
        parent_summary=json.loads((args.resume.parent/'metrics.json').read_text())
        history=[row for row in parent_summary['history']
                 if row['stage']=='autoencoder' or row['step']<=resumed['step']]
    started = time.perf_counter()
    metadata = dict(args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                    config=asdict(config), source_hashes=provenance(), dataset=manifest,
                    train_windows=len(train), val_windows=len(val),
                    validation_protocol='overfit diagnostic only' if args.overfit_windows else 'held-out scenario development split',
                    transfer_status='Motion encoder space only; not aligned to QD latent space',
                    input_semantics='historical annotated-box occupancy + current map; no observed free-space claim')
    metadata['dataset_sha256']=hashlib.sha256((args.data/'windows.npz').read_bytes()).hexdigest()
    metadata['steps_per_epoch']=len(loaders['train'])
    if args.motion_targets:
        metadata['motion_supervision']=dict(path=str(args.motion_targets.resolve()),
            sha256=hashlib.sha256(args.motion_targets.read_bytes()).hexdigest(),role='training target only; not passed to predictor')
    if args.val_scenes_from:
        metadata['fixed_validation']=dict(source=str(args.val_scenes_from.resolve()),
                                          sha256=hashlib.sha256(args.val_scenes_from.read_bytes()).hexdigest(),
                                          all_window_arrays_identical=True)
    if transferred:
        metadata['finetune']=dict(checkpoint=str(args.finetune_from.resolve()),parent_step=transferred['step'],
                                 checkpoint_sha256=hashlib.sha256(args.finetune_from.read_bytes()).hexdigest(),
                                 optimizer_restored=False,learning_rate=args.wm_lr,autoencoder_frozen=True,
                                 new_latent_cache=not args.reuse_latent_cache,
                                 upgrade_motion_context=args.upgrade_motion_context)
        parent_metrics=args.finetune_from.parent/'metrics.json'
        if parent_metrics.exists():
            history=[r for r in json.loads(parent_metrics.read_text())['history'] if r['stage']=='autoencoder']
    if args.init_autoencoder:
        parent_config=json.loads((args.init_autoencoder.parent/'config.json').read_text())
        if parent_config['dataset'] != manifest:
            raise ValueError('initial teacher was trained with a different dataset protocol')
        parent_data=Path(parent_config['args']['data'])/'windows.npz'
        if hashlib.sha256(parent_data.read_bytes()).hexdigest()!=metadata['dataset_sha256']:
            raise ValueError('initial teacher dataset bytes differ')
        parent=LatentConfig(**parent_config['config'])
        for key in ('channels','token_h','token_w','detail_encoder','map_classes','object_classes'):
            if getattr(parent,key)!=getattr(config,key):
                raise ValueError(f'initial teacher architecture mismatch: {key}')
        metadata['initial_teacher']=dict(path=str(args.init_autoencoder.resolve()),
                                        sha256=hashlib.sha256(args.init_autoencoder.read_bytes()).hexdigest(),
                                        predictor_initialized_from_scratch=True)
        parent_metrics=args.init_autoencoder.parent/'metrics.json'
        if parent_metrics.exists():
            history=[r for r in json.loads(parent_metrics.read_text())['history'] if r['stage']=='autoencoder']
    if resumed:
        parent_data=Path(resumed['protocol']['args']['data'])/'windows.npz'
        if hashlib.sha256(parent_data.read_bytes()).hexdigest() != metadata['dataset_sha256']:
            raise ValueError('resume dataset arrays differ from parent run')
        metadata['resume']=dict(checkpoint=str(args.resume.resolve()),step=resumed['step'],
                                checkpoint_sha256=hashlib.sha256(args.resume.read_bytes()).hexdigest(),
                                optimizer_restored=True,autoencoder_frozen=True,
                                data_order='new deterministic shuffle stream; old checkpoint has no sampler position',
                                parent_source_hashes=resumed['protocol']['source_hashes'])
    atomic_json(args.output/'config.json', metadata)
    snapshot=args.output/'source'
    snapshot.mkdir(exist_ok=True)
    for name,digest in metadata['source_hashes'].items():
        source=Path(__file__).with_name(name)
        if hashlib.sha256(source.read_bytes()).hexdigest()!=digest:
            raise RuntimeError('source changed during run initialization')
        shutil.copyfile(source,snapshot/name)

    def log(stage, step, values):
        row=dict(stage=stage, step=step, epoch=step/len(loaders['train']), seconds=time.perf_counter()-started, **values)
        history.append(row)
        with (args.output/'metrics.jsonl').open('a') as f: f.write(json.dumps(row, allow_nan=False)+'\n')
        atomic_json(args.output/'status.json', dict(state='running', **row))
        print(json.dumps(row, allow_nan=False), flush=True)

    @torch.no_grad()
    def eval_ae():
        ae.eval(); metrics=Counts(); total=0.; n=0
        for raw in loaders['val']:
            b=device_batch(raw,device); logits=ae(state_batch(b))
            loss=semantic_loss(logits,b['maps'],b['occ'],weights)
            total+=float(loss)*len(b['row']); n+=len(b['row'])
            metrics.update(logits[:,8:].sigmoid(),b['occ'],logits[:,:8],b['maps'])
        return dict(loss=total/n, **metrics.result())

    if resumed or args.init_autoencoder or transferred:
        source_ae=(args.resume.parent/'autoencoder_best.pt' if resumed else
                   args.finetune_from.parent/'autoencoder_best.pt' if transferred else args.init_autoencoder)
        ae_checkpoint=torch.load(source_ae,map_location=device,weights_only=False)
        ae.load_state_dict(resumed['autoencoder'] if resumed else transferred['autoencoder'] if transferred else ae_checkpoint['model'])
        if not all(torch.equal(value,ae_checkpoint['model'][name]) for name,value in ae.state_dict().items()):
            raise ValueError('frozen autoencoder file does not match checkpoint')
        shutil.copyfile(source_ae,args.output/'autoencoder_best.pt')
    else:
        best=math.inf
        optimizer=torch.optim.AdamW(ae.parameters(),lr=3e-4,weight_decay=1e-4)
        stream=cycle(loaders['train'])
        log('autoencoder',0,dict(val=eval_ae()))
        for step in range(1,args.ae_steps+1):
            ae.train();b=device_batch(next(stream),device)
            # Current/history and future states use the same encoder/decoder and training-only samples.
            if step%2:
                value=state_batch(b); maps,occ=b['maps'],b['occ']
            else:
                index=random.randrange(4);maps,occ=b['current_map'],b['history_occ'][:,index]
                value=pack_state(maps,occ)
            logits=ae(value); loss=semantic_loss(logits,maps,occ,weights)
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(ae.parameters(),1.);optimizer.step()
            if step%args.eval_every==0 or step==args.ae_steps:
                result=eval_ae();log('autoencoder',step,dict(train_loss=float(loss.detach()),val=result))
                if result['loss']<best:
                    best=result['loss'];torch.save(dict(model=ae.state_dict(),config=asdict(config),step=step,metrics=result),args.output/'autoencoder_best.pt')
    ae_checkpoint=torch.load(args.output/'autoencoder_best.pt',map_location=device,weights_only=False)
    ae.load_state_dict(ae_checkpoint['model']);ae.eval();ae.requires_grad_(False)
    if not args.overfit_windows and ae_checkpoint['metrics']['occupancy_iou'] < .1:
        atomic_json(args.output/'status.json',dict(state='failed_reconstruction_gate',metrics=ae_checkpoint['metrics']))
        raise RuntimeError('reconstruction occupancy IoU <0.1; inspect representation before training WM')

    # Cache frozen teacher features, not arbitrary averages of semantic channels.
    if resumed or args.init_autoencoder or (transferred and args.reuse_latent_cache):
        if transferred:
            parent_data=Path(transferred['protocol']['args']['data'])/'windows.npz'
            if hashlib.sha256(parent_data.read_bytes()).hexdigest()!=metadata['dataset_sha256']:
                raise ValueError('cannot reuse cache when training arrays change')
        source_cache=(args.resume.parent if resumed else args.finetune_from.parent if transferred else args.init_autoencoder.parent)/'frozen_latents.pt'
        if not source_cache.exists():
            if resumed:
                source_cache=Path(resumed['protocol']['frozen_latent_cache'])
            elif transferred:
                source_cache=Path(transferred['protocol']['frozen_latent_cache'])
            else:
                source_cache=Path(parent_config['frozen_latent_cache'])
        cache=torch.load(source_cache,map_location='cpu',weights_only=False)
        if cache['teacher_sha256']!=hashlib.sha256((args.output/'autoencoder_best.pt').read_bytes()).hexdigest():
            raise ValueError('latent cache belongs to a different frozen teacher')
        latent_history,latent_target=cache['history'],cache['target']
        if latent_history.shape!=(len(a['scene']),4,args.channels,config.token_h,config.token_w):
            raise ValueError('latent cache shape mismatch')
        metadata['frozen_latent_cache']=str(source_cache.resolve())
        print('reused verified frozen teacher and latent cache',flush=True)
    else:
        print('encoding frozen history and target features',flush=True)
        latent_history=torch.empty(len(a['scene']),4,args.channels,config.token_h,config.token_w,dtype=torch.float16)
        latent_target=torch.empty(len(a['scene']),3,args.channels,config.token_h,config.token_w,dtype=torch.float16)
        with torch.inference_mode():
            for start in range(0,len(a['scene']),args.batch_size):
                end=min(start+args.batch_size,len(a['scene']))
                maps=torch.tensor(a['history_map'][start:end],device=device)[:,None].expand(-1,4,-1,-1)
                occ=torch.tensor(a['history_occ'][start:end],device=device)
                value=ae.encoder(pack_state(maps,occ).flatten(0,1))
                latent_history[start:end]=value.reshape(end-start,4,args.channels,config.token_h,config.token_w).half().cpu()
                maps=torch.tensor(a['target_map'][start:end],device=device)
                occ=torch.tensor(a['target_occ'][start:end],device=device)
                value=ae.encoder(pack_state(maps,occ).flatten(0,1))
                latent_target[start:end]=value.reshape(end-start,3,args.channels,config.token_h,config.token_w).half().cpu()
        torch.save(dict(history=latent_history,target=latent_target,teacher_sha256=hashlib.sha256((args.output/'autoencoder_best.pt').read_bytes()).hexdigest()),args.output/'frozen_latents.pt')
        metadata['frozen_latent_cache']=str((args.output/'frozen_latents.pt').resolve())
    atomic_json(args.output/'config.json',metadata)
    predictor=LatentPredictor(config).to(device)
    optimizer=torch.optim.AdamW(predictor.parameters(),lr=args.wm_lr,weight_decay=1e-3)
    if resumed:
        predictor.load_state_dict(resumed['predictor'])
        optimizer.load_state_dict(resumed['optimizer'])
    elif transferred:
        if args.upgrade_motion_context or args.upgrade_motion_aux:
            missing,unexpected=predictor.load_state_dict(transferred['predictor'],strict=False)
            expected={key for key in predictor.state_dict()
                      if (args.upgrade_motion_context and key.startswith('motion_context.'))
                      or (args.upgrade_motion_aux and key.startswith('velocity_head.'))}
            if set(missing)!=expected or unexpected:
                raise ValueError(f'unsafe motion upgrade: {missing}, {unexpected}')
        else:
            predictor.load_state_dict(transferred['predictor'])

    def features(raw,b):
        rows,choices=raw['row'],raw['candidate']
        return latent_history[rows].to(device).float(),latent_target[rows,choices].to(device).float()

    @torch.no_grad()
    def evaluate(diagnostics=False):
        predictor.eval();counts=Counts();valid_counts=Counts();total=latent=0.;n=0;example=None
        comparisons={k:Counts() for k in ('copy','ego_warp','constant_velocity','teacher_reconstruction','shuffled_action','current_only')} if diagnostics else {}
        for raw in loaders['val']:
            b=device_batch(raw,device);h,target=features(raw,b)
            pred=predictor(h,b['action'],b['status']);logits=ae.decoder(pred)
            ld=latent_distance(pred,target);sl=semantic_loss(logits,b['maps'],b['occ'],weights)
            total+=float(ld+sl)*len(h);latent+=float(ld)*len(h);n+=len(h)
            counts.update(logits[:,8:].sigmoid(),b['occ'],logits[:,:8],b['maps'])
            warped,valid=warp_current(b['history_occ'][:,-1].float(),b['action'][:,4])
            valid_counts.update(logits[:,8:].sigmoid(),b['occ'],valid=valid)
            if diagnostics:
                comparisons['copy'].update(b['history_occ'][:,-1],b['occ'])
                comparisons['ego_warp'].update(warped,b['occ'])
                comparisons['constant_velocity'].update(b['cv'],b['occ'])
                truth_logits=ae.decoder(target)
                comparisons['teacher_reconstruction'].update(truth_logits[:,8:].sigmoid(),b['occ'],truth_logits[:,:8],b['maps'])
                shuffled=ae.decoder(predictor(h,b['alternate'],b['status']))
                comparisons['shuffled_action'].update(shuffled[:,8:].sigmoid(),b['occ'])
                mask=h.new_zeros(len(h),4);mask[:,-1]=1
                single=ae.decoder(predictor(h,b['action'],b['status'],mask))
                comparisons['current_only'].update(single[:,8:].sigmoid(),b['occ'])
                if example is None:
                    example=dict(target=b['occ'].any(1).cpu().numpy(),prediction=logits[:,8:].sigmoid().amax(1).cpu().numpy(),
                                 copy=b['history_occ'][:,-1].any(1).cpu().numpy(),warp=warped.any(1).cpu().numpy(),
                                 cv=b['cv'].any(1).cpu().numpy(),teacher=truth_logits[:,8:].sigmoid().amax(1).cpu().numpy())
        result=dict(loss=total/n,latent_loss=latent/n,**counts.result(),valid_input_region=valid_counts.result())
        if diagnostics:
            result['baselines']={k:v.result() for k,v in comparisons.items()}
            result['baseline_note']='constant_velocity uses privileged current GT object velocities; current_only/shuffled_action are inference ablations, not separately trained models'
        return result,example

    initial_step=resumed['step'] if resumed else 0
    best=math.inf;best_step=initial_step;stream=cycle(loaders['train'])
    result,_=evaluate();log('world_model',initial_step,dict(val=result))
    if resumed or transferred:
        best=selection_score(result,args.selection_metric)
        if resumed and abs(result['loss']-resumed['metrics']['loss'])>1e-4:
            raise ValueError('resumed validation loss differs from checkpoint; inspect protocol/cache')
        if transferred and args.val_scenes_from and abs(result['occupancy_iou']-transferred['metrics']['occupancy_iou'])>1e-4:
            raise ValueError('transferred model does not reproduce original fixed validation IoU')
        torch.save(dict(predictor=predictor.state_dict(),autoencoder=ae.state_dict(),config=asdict(config),step=initial_step,
                        metrics=result,protocol=metadata,optimizer=optimizer.state_dict()),args.output/'best.pt')
    stale=0;stop_reason='max_steps';interval_losses=[]
    for step in range(initial_step+1,args.wm_steps+1):
        predictor.train();raw=next(stream);b=device_batch(raw,device);h,target=features(raw,b)
        mask=(torch.rand(len(h),4,device=device)>.1).float();mask[:,-1]=1
        pred=predictor(h,b['action'],b['status'],mask)
        logits=ae.decoder(pred)
        ld=latent_distance(pred,target);sl=semantic_loss(logits,b['maps'],b['occ'],weights);loss=ld+sl
        if args.motion_aux:
            from .latent_motion_model import velocity_auxiliary_loss
            loss=loss+.5*velocity_auxiliary_loss(predictor.aux_velocity,b['motion_velocity'],b['motion_mask'],b['action'][:,4])
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(predictor.parameters(),1.);optimizer.step()
        interval_losses.append(float(loss.detach()))
        if step%args.eval_every==0 or step==args.wm_steps:
            result,_=evaluate();log('world_model',step,dict(train_loss=float(loss.detach()),train_mean_loss=float(np.mean(interval_losses)),train_latent=float(ld.detach()),val=result))
            interval_losses=[]
            checkpoint=dict(predictor=predictor.state_dict(),autoencoder=ae.state_dict(),config=asdict(config),step=step,
                            metrics=result,protocol=metadata,optimizer=optimizer.state_dict())
            torch.save(checkpoint,args.output/'latest.pt')
            score=selection_score(result,args.selection_metric)
            if score<best-args.early_stop_min_delta:
                best=score;best_step=step;torch.save(checkpoint,args.output/'best.pt')
                stale=0
            else:
                stale+=1
            if args.early_stop_patience and stale>=args.early_stop_patience:
                stop_reason='early_stopping'
                break
    checkpoint=torch.load(args.output/'best.pt',map_location=device,weights_only=False)
    predictor.load_state_dict(checkpoint['predictor'])
    result,example=evaluate(diagnostics=True)
    if example:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(4,6,figsize=(18,8),squeeze=False)
        for row in range(min(4,len(example['target']))):
            for col,name in enumerate(('target','prediction','copy','warp','cv','teacher')):
                axes[row,col].imshow(example[name][row],origin='lower',extent=(-30,30,-15,15),vmin=0,vmax=1,cmap='magma')
                axes[row,col].set_title(name);axes[row,col].set_xlabel('x forward (m)')
        fig.tight_layout();fig.savefig(args.output/'validation_comparison.png',dpi=150);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for stage,axis in zip(('autoencoder','world_model'),axes):
            rows=[r for r in history if r['stage']==stage]
            axis.plot([r['step'] for r in rows],[r['val']['loss'] for r in rows],label='val loss')
            training_rows=[r for r in rows if 'train_mean_loss' in r]
            if training_rows:
                axis.plot([r['step'] for r in training_rows],[r['train_mean_loss'] for r in training_rows],label='train interval mean')
            axis.set_xlabel('optimizer step')
            axis.set_title(stage);axis.legend()
        fig.tight_layout();fig.savefig(args.output/'training_curves.png',dpi=150);plt.close(fig)
    verdict='needs_improvement'
    if result['occupancy_iou']>result['baselines']['ego_warp']['occupancy_iou']:
        verdict='beats_ego_warp_on_development_split'
    summary=dict(**metadata,best_step=best_step,last_step=step,stop_reason=stop_reason,
                 total_wm_epochs=step/len(loaders['train']),autoencoder_metrics=ae_checkpoint['metrics'],metrics=result,
                 verdict=verdict,seconds=time.perf_counter()-started,
                 cuda_peak_memory_mb=torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else 0,
                 history=history)
    atomic_json(args.output/'metrics.json',summary)
    atomic_json(args.output/'status.json',dict(state='completed',best_step=best_step,last_step=step,
                 total_wm_epochs=step/len(loaders['train']),stop_reason=stop_reason,verdict=verdict,metrics=result))
    print(json.dumps(dict(best_step=best_step,verdict=verdict,metrics=result),indent=2),flush=True)


if __name__=='__main__':
    main()
