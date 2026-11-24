"""Motion-field transport pilot with matched-capacity zero-flow control."""
import argparse
import copy
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .ablate_motion_training import Experiment, semantic_parts
from .latent_motion_model import LatentPredictor, latent_distance, warp_current
from .motion_transport import TransportWorldModel, class_velocity_loss
from .motion_instance_diagnostics import extract_instances, summarize_instances
from .train_latent_motion import Counts, atomic_json


def risk_weights(device):
    x=(torch.arange(100,device=device)+.5)*.6-30
    y=(torch.arange(50,device=device)+.5)*.6-15
    dx=(x.abs()-2.4-.3).clamp_min(0)
    dy=(y.abs()-1.-.3).clamp_min(0)
    return torch.exp(-torch.hypot(dy[:,None],dx[None,:]).clamp_max(8)/3)


class TransportExperiment:
    def __init__(self,args):
        self.args=args;self.root=args.output
        data_args=copy.copy(args);data_args.output=self.root/'data_protocol';data_args.output.mkdir()
        self.data=Experiment(data_args);self.device=self.data.device;self.a=self.data.a
        self.ae=self.data.original
        self.weights=risk_weights(self.device)
        self.rows=self.data.val
        self.instances,instance_meta=extract_instances(self.a,self.rows,self.root/'diagnostic_instances.npz')
        atomic_json(self.root/'diagnostic_instances.json',instance_meta)
        self.instance_rows={r['row'] for r in self.instances}
        self.original=LatentPredictor(self.data.cfg).to(self.device).eval().requires_grad_(False)
        self.original.load_state_dict(self.data.ckpt['predictor'])
        self.protocol=dict(data=self.data.protocol,args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                source_inputs='Only original frozen history latents and their decoded probabilities, ego status and candidate actions. No raw GT occupancy, IDs or velocities enter predictors.',
                scene_frame='Predict forward current-ego per-class world velocity; compose object movement and candidate future ego transform before bilinear splatting.',
                latent_contract='Native propagated BEV is the semantic prediction; re-encode with original frozen E for latent supervision. D(E(predicted BEV)) is evaluated separately as latent compatibility probe.',
                loss='normalized latent SmoothL1 + 0.5 map CE + occupancy BCE + 0.5 Dice + 0.5 current-frame class-specific velocity SmoothL1',
                velocity_label_difference='Same current GT velocity labels, now at 50x100 per-class current coordinates rather than pooled 16x32 candidate-frame auxiliary grid. Matched new-architecture controls share this definition.',
                ranking='Continuous weighted proximity proxy at 1s, 3 logged-path perturbations; NOT 5s trajectory safety, 20-candidate QD ranking, Waymax, PDMS or RFS.',
                target_teacher_frozen=True,source_decoder_frozen=True,
                candidate_physics='Other actors follow logs; motion field does not depend on ego candidate, residual is candidate-conditioned.',
                selection='Primary compare final fixed step budget; save development-best separately, including step0.')
        atomic_json(self.root/'protocol.json',self.protocol)
        source=self.root/'source';source.mkdir()
        for name in ('train_motion_transport.py','motion_transport.py','motion_instance_diagnostics.py',
                     'latent_motion_model.py','ablate_motion_training.py'):
            (source/name).write_bytes(Path(__file__).with_name(name).read_bytes())
        self.decode_observations()

    @torch.no_grad()
    def decode_observations(self):
        n=len(self.a['scene'])
        self.occ_cache=torch.empty(n,4,4,50,100,dtype=torch.float16)
        self.map_cache=torch.empty(n,8,50,100,dtype=torch.float16)
        for start in range(0,n,32):
            end=min(start+32,n)
            h=self.data.cache['history'][start:end].to(self.device).float()
            decoded=self.ae.decoder(h.flatten(0,1)).reshape(end-start,4,12,50,100)
            self.occ_cache[start:end]=decoded[:,:,8:].sigmoid().half().cpu()
            self.map_cache[start:end]=decoded[:,-1,:8].softmax(1).half().cpu()
            if end%1024==0:print('decoded input observations',end,'/',n,flush=True)
        atomic_json(self.root/'decoded_cache.json',dict(in_memory_only=True,windows=n,
                    teacher_sha256=self.data.protocol['teacher_sha256'],dataset_sha256=self.data.protocol['dataset_sha256'],
                    note='Inference of frozen D on original historical latents only; future features are not used.'))

    def inputs(self,rows,choices):
        h=self.data.cache['history'][rows].to(self.device).float()
        action=self.data.tensor(self.a['actions'][rows,choices]);status=self.data.tensor(self.a['status'][rows])
        source=self.occ_cache[rows].to(self.device).float()
        maps=self.map_cache[rows].to(self.device).float()
        return h,action,status,source,maps

    @torch.no_grad()
    def evaluate(self,model=None,baseline=None,ids=None,instances=False):
        ids=self.rows if ids is None else ids
        if model is not None:model.eval()
        counts=Counts();state_counts=Counts();transport_counts=Counts();decoded_counts=Counts()
        loss_sums={k:0. for k in ('latent','map','bce','dice','velocity')};n=0
        regions=np.zeros(8,dtype=np.int64);per_scene={};predictions={}
        predicted_risk=[];true_risk=[];velocity_error=velocity_weight=0.
        for start in range(0,len(ids)*3,32):
            flat=np.arange(start,min(start+32,len(ids)*3));rows,choices=ids[flat//3],flat%3
            h,action,status,source,map_source=self.inputs(rows,choices)
            maps=self.data.tensor(self.a['target_map'][rows,choices])
            occ=self.data.tensor(self.a['target_occ'][rows,choices])
            target=self.data.cache['target'][rows,choices].to(self.device).float()
            if baseline=='v4':
                latent=self.original(h,action,status);logits=self.ae.decoder(latent);p=logits[:,8:].sigmoid()
            elif baseline in ('decoded_ego_warp','gt_velocity_cv'):
                p=(warp_current(source[:,-1],action[:,4],mode='bilinear')[0] if baseline=='decoded_ego_warp'
                   else self.data.tensor(self.a['cv_occ'][rows,choices]).float())
                logits_map,_=warp_current(map_source,action[:,4],mode='bilinear')
                logits_map=logits_map.clamp_min(1e-4).log()
                logits=torch.cat((logits_map,torch.logit(p.clamp(.001,.999))),1)
                latent=None
            else:
                latent,logits=model(h,action,status,source,map_source,self.ae)
                p=logits[:,8:].sigmoid()
                state_counts.update(model.semantic_state[:,8:],occ)
                transport_counts.update(model.transported,occ)
                decoded_counts.update(self.ae.decoder(latent)[:,8:].sigmoid(),occ)
                parts_velocity=class_velocity_loss(model.velocity,self.data.tensor(self.data.velocity[rows]),
                            self.data.tensor(self.data.velocity_mask[rows]),self.data.tensor(self.a['history_occ'][rows,-1]))
                loss_sums['velocity']+=float(parts_velocity)*len(rows)
                valid=self.data.tensor(self.data.velocity_mask[rows])[:,None].float()*self.data.tensor(self.a['history_occ'][rows,-1]).float()
                delta=(model.velocity-self.data.tensor(self.data.velocity[rows])[:,None]).square().sum(2).sqrt()
                velocity_error+=float((delta*valid).sum());velocity_weight+=float(valid.sum())
            counts.update(p,occ,logits[:,:8],maps)
            if latent is not None:loss_sums['latent']+=float(latent_distance(latent,target))*len(rows)
            parts=semantic_parts(logits,maps,occ,self.data.weights)
            for key,value in parts.items():loss_sums[key]+=float(value)*len(rows)
            n+=len(rows)
            prediction=p.ge(.5).any(1);truth=occ.bool().any(1)
            warped,valid=warp_current(self.data.tensor(self.a['history_occ'][rows,-1]).float(),action[:,4])
            current=warped.bool().any(1);stable=current&truth;changed=truth&~current;ghost=current&~truth
            values=[(prediction&stable).sum(),stable.sum(),(prediction&changed&valid).sum(),(changed&valid).sum(),
                    (prediction&changed&~valid).sum(),(changed&~valid).sum(),(prediction&ghost).sum(),ghost.sum()]
            regions+=np.array([int(v) for v in values])
            per=torch.stack(((prediction&truth).sum((1,2)),(prediction&~truth).sum((1,2)),(~prediction&truth).sum((1,2))),1).cpu().numpy()
            for scene,value in zip(self.a['scene'][rows],per):per_scene.setdefault(str(scene),np.zeros(3,dtype=np.int64))[:]+=value
            floor=float(np.exp(-8/3))
            predicted_risk.append((p.amax(1)*self.weights).flatten(1).amax(1).clamp_min(floor).cpu().numpy())
            true_risk.append((truth*self.weights).flatten(1).amax(1).clamp_min(floor).cpu().numpy())
            if instances:
                for index,(row,choice) in enumerate(zip(rows,choices)):
                    if choice==0 and int(row) in self.instance_rows:predictions[int(row)]=p[index].cpu().numpy()
        risk=np.concatenate(predicted_risk).reshape(-1,3);gt=np.concatenate(true_risk).reshape(-1,3)
        chosen=risk.argmin(1);regret=gt[np.arange(len(ids)),chosen]-gt.min(1)
        scene_regret={str(s):[float(regret[self.a['scene'][ids]==s].sum()),int((self.a['scene'][ids]==s).sum())]
                      for s in sorted(set(self.a['scene'][ids]))}
        result=dict(**counts.result(),losses={k:v/n for k,v in loss_sums.items()},
                    stable_target_recall=float(regions[0]/max(regions[1],1)),
                    changed_inside_recall=float(regions[2]/max(regions[3],1)),
                    changed_outside_recall=float(regions[4]/max(regions[5],1)),
                    ghost_retention=float(regions[6]/max(regions[7],1)),
                    endpoint_ranking_regret=float(regret.mean()),risk_mae=float(np.abs(risk-gt).mean()),
                    original_logged_endpoint_regret=float((gt[:,0]-gt.min(1)).mean()),
                    per_scene_counts={s:v.tolist() for s,v in per_scene.items()},per_scene_regret=scene_regret)
        if model is not None:result.update(raw_semantic_state=state_counts.result(),transport_only_state=transport_counts.result(),
                                          decoded_latent_probe=decoded_counts.result(),
                                          source_velocity_epe_mps=velocity_error/max(velocity_weight,1))
        if instances:result['object_diagnostics']=summarize_instances(self.instances,predictions)
        return result

    def run(self,mode,seed):
        directory=self.root/f'{mode}_seed{seed}';directory.mkdir()
        torch.manual_seed(seed)
        model=TransportWorldModel(self.data.cfg,mode).to(self.device)
        model.backbone.load_state_dict(self.data.ckpt['predictor'])
        optimizer=torch.optim.AdamW(model.parameters(),lr=self.args.lr,weight_decay=1e-3)
        rng=np.random.default_rng(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        metadata=dict(**self.protocol,mode=mode,seed=seed,parameters=sum(p.numel() for p in model.parameters()))
        atomic_json(directory/'protocol.json',metadata)
        best=-1.;best_step=0;started=time.perf_counter();interval={};interval_n=0

        def save(step,metrics,name,include_optimizer=False):
            checkpoint=dict(model=model.state_dict(),config=asdict(self.data.cfg),mode=mode,step=step,
                            protocol=metadata,metrics=metrics,autoencoder=self.ae.state_dict())
            if include_optimizer:checkpoint.update(optimizer=optimizer.state_dict(),numpy_rng=rng.bit_generator.state,
                                                    torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
            tmp=directory/(name+'.tmp');torch.save(checkpoint,tmp);tmp.replace(directory/name)

        def record(step,metrics,losses=None):
            row=dict(step=step,seconds=time.perf_counter()-started,val=metrics,train_interval_mean=losses)
            with (directory/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            atomic_json(directory/'status.json',dict(state='running',step=step,best_iou=best,best_step=best_step,latest=row))
            print(json.dumps(dict(run=directory.name,step=step,iou=metrics['occupancy_iou'],
                                  raw_state_iou=metrics['raw_semantic_state']['occupancy_iou'],
                                  velocity_epe=metrics['source_velocity_epe_mps'],regret=metrics['endpoint_ranking_regret'])),flush=True)

        initial=self.evaluate(model,instances=True);best=initial['occupancy_iou'];selected=initial
        record(0,initial);save(0,initial,'best.pt')
        for step in range(1,self.args.steps+1):
            model.train()
            rows=rng.choice(self.data.train,self.args.batch_size,replace=True)
            choices=rng.integers(0,3,len(rows))
            h,action,status,source,maps_source=self.inputs(rows,choices)
            history_mask=(torch.rand(len(rows),4,device=self.device)>.1).float();history_mask[:,-1]=1
            latent,logits=model(h,action,status,source,maps_source,self.ae,history_mask)
            target=self.data.cache['target'][rows,choices].to(self.device).float()
            occ=self.data.tensor(self.a['target_occ'][rows,choices]);maps=self.data.tensor(self.a['target_map'][rows,choices])
            parts=semantic_parts(logits,maps,occ,self.data.weights)
            ld=latent_distance(latent,target)
            vl=class_velocity_loss(model.velocity,self.data.tensor(self.data.velocity[rows]),
                    self.data.tensor(self.data.velocity_mask[rows]),self.data.tensor(self.a['history_occ'][rows,-1]))
            semantic=.5*parts['map']+parts['bce']+.5*parts['dice']
            if step==1:
                grad=torch.autograd.grad(ld+semantic,model.velocity_refinement.weight,retain_graph=True,allow_unused=True)[0]
                norm=0. if grad is None else float(grad.norm())
                if mode!='static_residual' and (not np.isfinite(norm) or norm<=0):
                    raise RuntimeError('transport does not receive prediction gradient')
                atomic_json(directory/'gradient_check.json',dict(semantic_latent_to_velocity_refinement_l2=norm,
                            expected='positive for transport; zero for capacity-matched zero-flow control'))
            loss=ld+semantic+.5*vl
            if not torch.isfinite(loss):raise RuntimeError('nonfinite transport loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            values=dict(total=loss,latent=ld,**parts,velocity=vl)
            for key,value in values.items():interval[key]=interval.get(key,0.)+float(value.detach())
            interval_n+=1
            if step%self.args.eval_every==0 or step==self.args.steps:
                metrics=self.evaluate(model,instances=step==self.args.steps)
                if metrics['occupancy_iou']>best:
                    best=metrics['occupancy_iou'];best_step=step;selected=metrics;save(step,metrics,'best.pt')
                record(step,metrics,{k:v/interval_n for k,v in interval.items()});interval={};interval_n=0
                save(step,metrics,'latest.pt',include_optimizer=True)
        atomic_json(directory/'status.json',dict(state='completed',step=step,best_iou=best,best_step=best_step,
                    selected=selected,last=metrics,seconds=time.perf_counter()-started))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('outputs/world_model/motion_latent_v3/dataset2048'))
    p.add_argument('--checkpoint',type=Path,default=Path('outputs/world_model/motion_latent_v4/velocity_aux/best.pt'))
    p.add_argument('--velocity',type=Path,default=Path('outputs/world_model/motion_latent_v4/velocity_targets.npz'))
    p.add_argument('--val-reference',type=Path,default=Path('outputs/world_model/motion_latent_v2/dataset'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=2400)
    p.add_argument('--eval-every',type=int,default=800)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--lr',type=float,default=3e-5)
    p.add_argument('--seeds',type=int,nargs='+',default=[43,44])
    p.add_argument('--modes',nargs='+',choices=TransportWorldModel.MODES,default=list(TransportWorldModel.MODES))
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('preserve experiment outputs')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    args.output.mkdir(parents=True);torch.set_num_threads(4)
    experiment=TransportExperiment(args)
    baseline={name:experiment.evaluate(baseline=name,instances=True) for name in ('v4','decoded_ego_warp','gt_velocity_cv')}
    atomic_json(args.output/'baselines.json',baseline)
    print('baseline motion groups',json.dumps({name:value['object_diagnostics']['groups'] for name,value in baseline.items()}),flush=True)
    completed=[];atomic_json(args.output/'status.json',dict(state='running',completed=completed))
    for seed in args.seeds:
        for mode in args.modes:
            experiment.run(mode,seed)
            completed.append(f'{mode}_seed{seed}')
            atomic_json(args.output/'status.json',dict(state='running',completed=completed))
    atomic_json(args.output/'status.json',dict(state='completed',completed=completed))


if __name__=='__main__':main()
