"""Train precise-box motion tokens and compare exact same-grid GT-velocity CV."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw
import torch

from .instance_motion_model import InstanceMotionPredictor,history_features,decoded_boxes,forecast_loss
from .latent_motion_data import GRID
from .motion_gt import _box_footprint
from .train_latent_motion import Counts,atomic_json


def rasterize(boxes,category,pose,keep):
    c,s=np.cos(pose[2]),np.sin(pose[2])
    xy=boxes[:,:2]-pose[:2]
    centers=np.stack((c*xy[:,0]+s*xy[:,1],-s*xy[:,0]+c*xy[:,1]),1)
    images=[Image.new('L',(400,200)) for _ in range(4)];draws=[ImageDraw.Draw(v) for v in images]
    for i in np.flatnonzero(keep):
        x,y=centers[i];length,width=boxes[i,2:4]
        if abs(x)>30+max(length,width) or abs(y)>15+max(length,width):continue
        box=np.array([x,y,0.,length,width,1.,boxes[i,4]-pose[2]])
        points=GRID.pixels(_box_footprint(box))*4
        if np.isfinite(points).all():draws[int(category[i])].polygon([tuple(v) for v in points],fill=1)
    return np.stack([np.asarray(v).reshape(50,4,100,4).max((1,3)) for v in images])


class Trial:
    def __init__(self,args):
        self.args=args;self.device=torch.device('cuda:0');self.out=args.output
        with np.load(args.instances/'instances.npz') as p:self.a={k:p[k] for k in p.files}
        meta=json.loads((args.instances/'dataset.json').read_text())
        if hashlib.sha256((args.data/'windows.npz').read_bytes()).hexdigest()!=meta['parent_dataset_sha256']:
            raise ValueError('source data hash mismatch')
        with np.load(args.data/'windows.npz') as p:self.grid={k:p[k] for k in ('scene','anchor','actions','target_occ','cv_occ')}
        if not np.array_equal(self.a['scene'],self.grid['scene']) or not np.array_equal(self.a['anchor'],self.grid['anchor']):raise ValueError('row mismatch')
        evaluation_only=getattr(args,'evaluation_only',False)
        lookup={(str(s),int(t)):i for i,(s,t) in enumerate(zip(self.a['scene'],self.a['anchor']))}
        if evaluation_only:self.val=np.arange(len(self.a['scene']))
        else:
            with np.load(args.val_reference/'windows.npz') as p:
                selected=np.flatnonzero(p['split']=='val')
                self.val=np.asarray([lookup[(str(p['scene'][i]),int(p['anchor'][i]))] for i in selected])
        self.row_for_actor=np.repeat(np.arange(len(self.a['scene'])),np.diff(self.a['offsets']))
        self.train_indices=np.flatnonzero(self.a['split'][self.row_for_actor]=='train')
        if not evaluation_only and set(self.a['scene'][self.val])&set(self.a['scene'][self.a['split']=='train']):raise ValueError('leakage')
        tensor=lambda x:torch.as_tensor(x,device=self.device)
        history=tensor(self.a['history']).float();mask=tensor(self.a['mask']);category=tensor(self.a['category'])
        self.features,self.bases,self.omega,self.current=history_features(history,mask,category)
        self.target=tensor(self.a['target']).float();self.valid=tensor(self.a['future_valid'])
        self.oracle_features,self.oracle_bases,self.oracle_omega,_=history_features(tensor(self.a['oracle_history']).float(),tensor(self.a['oracle_mask']),category)
        atomic_json(self.out/'protocol.json',dict(args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                    dataset=meta,scope='Instance-token forecast with precise historical boxes and full current actor population, beyond original cropped BEV-latent input.',
                    default_input='Hungarian-associated past boxes; no GT IDs or GT velocities enter forecasting features.',
                    target='Actual +1s future box and existence; IDs only assign training targets.',
                    baseline='Original cached cv_occ and original target_occ, same scenes/windows/candidates/grid/threshold.',
                    decoding='Analytic object-token rasterization, no learned AE compression; separate future token architecture.'))
        source=self.out/'source';source.mkdir()
        for name in ('instance_motion_model.py','train_instance_motion.py','build_instance_motion.py'):(source/name).write_bytes(Path(__file__).with_name(name).read_bytes())

    @torch.no_grad()
    def predict(self,kind,model=None):
        if kind=='learned':
            model.eval();parts=[];presence=[]
            for start in range(0,len(self.current),8192):
                end=min(start+8192,len(self.current))
                prediction=model(self.features[start:end],self.bases[start:end],self.omega[start:end])
                parts.append(decoded_boxes(self.current[start:end],*prediction[:3]).cpu().numpy())
                presence.append(prediction[3].sigmoid().cpu().numpy())
            return np.concatenate(parts),np.concatenate(presence)>.5
        if kind=='gt_current_future_oracle':return self.a['target'],self.a['future_valid'].astype(bool)
        if kind=='gt_velocity':
            boxes=self.a['history'][:,-1].copy();boxes[:,:2]+=self.a['gt_velocity']
            return boxes,np.ones(len(boxes),bool)
        oracle=kind.startswith('oracle_');bases=self.oracle_bases if oracle else self.bases
        omega=self.oracle_omega if oracle else self.omega
        which=0 if kind.endswith('cv') else 3 if kind.endswith('ca') else 4
        displacement=bases[:,which]
        yaw=torch.zeros_like(omega) if which==0 else omega
        boxes=decoded_boxes(self.current,displacement,yaw,torch.zeros(len(self.current),2,device=self.device))
        return boxes.cpu().numpy(),np.ones(len(boxes),bool)

    def evaluate(self,boxes,keep,ids=None):
        ids=self.val if ids is None else ids
        counts=Counts();per_scene={};reference=Counts()
        for row in ids:
            start,end=self.a['offsets'][row:row+2]
            pred=np.stack([rasterize(boxes[start:end],self.a['category'][start:end],pose[4],keep[start:end]) for pose in self.grid['actions'][row]])
            target=self.grid['target_occ'][row]
            counts.update(torch.tensor(pred),torch.tensor(target))
            reference.update(torch.tensor(self.grid['cv_occ'][row]),torch.tensor(target))
            p,t=pred.any(1),target.any(1)
            value=np.array([(p&t).sum(),(p&~t).sum(),(~p&t).sum()])
            per_scene.setdefault(str(self.a['scene'][row]),np.zeros(3,dtype=np.int64))[:]+=value
        return dict(**counts.result(),per_scene_counts={k:v.tolist() for k,v in per_scene.items()},
                    gt_velocity_reference=reference.result())

    def train(self,seed):
        path=self.out/f'seed{seed}';path.mkdir()
        torch.manual_seed(seed);rng=np.random.default_rng(seed)
        model=InstanceMotionPredictor().to(self.device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=self.args.lr,weight_decay=1e-4)
        best=-1.;best_step=0;interval=[];start=time.perf_counter()
        for step in range(self.args.steps+1):
            if step:
                model.train();indices=torch.as_tensor(rng.choice(self.train_indices,self.args.batch_size,replace=True),device=self.device)
                pred=model(self.features[indices],self.bases[indices],self.omega[indices])
                loss,parts=forecast_loss(pred,self.current[indices],self.target[indices],self.valid[indices])
                if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
                optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
                interval.append(float(loss.detach()))
            if step%self.args.eval_every==0 or step==self.args.steps:
                boxes,keep=self.predict('learned',model);metrics=self.evaluate(boxes,keep)
                row=dict(step=step,seconds=time.perf_counter()-start,val=metrics,train_loss=float(np.mean(interval)) if interval else None)
                interval=[]
                with (path/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                checkpoint=dict(model=model.state_dict(),step=step,metrics=metrics,optimizer=optimizer.state_dict(),seed=seed)
                torch.save(checkpoint,path/'latest.pt')
                if metrics['occupancy_iou']>best:
                    best=metrics['occupancy_iou'];best_step=step;torch.save(checkpoint,path/'best.pt')
                atomic_json(path/'status.json',dict(state='running',step=step,best_iou=best,best_step=best_step,latest=row))
                print(json.dumps(dict(seed=seed,step=step,iou=metrics['occupancy_iou'],gt_cv=metrics['gt_velocity_reference']['occupancy_iou'])),flush=True)
        atomic_json(path/'status.json',dict(state='completed',step=step,best_iou=best,best_step=best_step,last=metrics))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--instances',type=Path,default=Path('outputs/world_model/instance_motion_v8_20261119/dataset'))
    p.add_argument('--data',type=Path,default=Path('outputs/world_model/motion_latent_v3/dataset2048'))
    p.add_argument('--val-reference',type=Path,default=Path('outputs/world_model/motion_latent_v2/dataset'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=4000)
    p.add_argument('--batch-size',type=int,default=2048);p.add_argument('--eval-every',type=int,default=500)
    p.add_argument('--lr',type=float,default=3e-4);p.add_argument('--seeds',type=int,nargs='+',default=[43,44])
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('preserve experiment')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    args.output.mkdir(parents=True);torch.set_num_threads(4)
    trial=Trial(args);baselines={}
    for kind in ('gt_velocity','history_cv','history_ca','history_ctrv','oracle_cv','oracle_ca','gt_current_future_oracle'):
        boxes,keep=trial.predict(kind);baselines[kind]=trial.evaluate(boxes,keep)
        print(kind,baselines[kind]['occupancy_iou'],flush=True)
    atomic_json(args.output/'baselines.json',baselines)
    for seed in args.seeds:trial.train(seed)
    atomic_json(args.output/'status.json',dict(state='completed',seeds=args.seeds))


if __name__=='__main__':main()
