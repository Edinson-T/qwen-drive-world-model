"""Precise past-box histories, associated without GT IDs as predictor input."""
import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from .build_motion_gt_dataset import _iter_tfrecords
from .latent_motion_data import HISTORY
from .motion_gt import ego_pose_from_state,global_to_ego,wrap_angle
from .protos import womd_pb2


def detections(scenario,step,pose):
    boxes=[];classes=[];identities=[];velocities=[]
    ego_id=scenario.tracks[scenario.sdc_track_index].id
    c,s=np.cos(pose[2]),np.sin(pose[2])
    for track in scenario.tracks:
        if track.id==ego_id or track.object_type not in (1,2,3,4):continue
        state=track.states[step]
        if not state.valid or state.length<=0 or state.width<=0:continue
        xy=global_to_ego(np.array([[state.center_x,state.center_y,0.]]),pose)[0,:2]
        boxes.append([*xy,state.length,state.width,wrap_angle(state.heading-pose[2])])
        classes.append(track.object_type-1);identities.append(track.id)
        velocities.append([c*state.velocity_x+s*state.velocity_y,-s*state.velocity_x+c*state.velocity_y])
    return (np.asarray(boxes,dtype=np.float32).reshape(-1,5),np.asarray(classes,dtype=np.int64),
            np.asarray(identities,dtype=np.int64),np.asarray(velocities,dtype=np.float32).reshape(-1,2))


def associate(current,classes,current_ids,past):
    n=len(current);history=np.zeros((n,4,5),np.float32);mask=np.zeros((n,4),np.uint8)
    history[:,-1]=current;mask[:,-1]=1
    oracle=np.zeros_like(history);oracle_mask=np.zeros_like(mask)
    oracle[:,-1]=current;oracle_mask[:,-1]=1
    correct=np.zeros(3,np.int64);total=np.zeros(3,np.int64)
    for frame in (2,1,0):
        boxes,labels,identities,_=past[frame]
        lookup={int(identity):j for j,identity in enumerate(identities)}
        for i,identity in enumerate(current_ids):
            if int(identity) in lookup:oracle[i,frame]=boxes[lookup[int(identity)]];oracle_mask[i,frame]=1
        dt=(frame-3)*.5
        estimate=np.zeros((n,2),np.float32)
        seen=np.zeros(n,dtype=bool)
        for previous in range(2,frame,-1):
            usable=mask[:,previous].astype(bool)&~seen
            estimate[usable]=(current[usable,:2]-history[usable,previous,:2])/((3-previous)*.5)
            seen|=usable
        speed=np.linalg.norm(estimate,axis=1)
        estimate*=np.minimum(1.,50/np.maximum(speed,1e-6))[:,None]
        expected=current[:,:2]+estimate*dt
        for category in range(4):
            a=np.flatnonzero(classes==category);b=np.flatnonzero(labels==category)
            if not len(a) or not len(b):continue
            distance=np.linalg.norm(expected[a,None]-boxes[None,b,:2],axis=-1)
            dim=np.abs(np.log(current[a,None,2:4]/boxes[None,b,2:4])).sum(-1)
            full_angle=np.abs(wrap_angle(current[a,None,4]-boxes[None,b,4]))
            shape_angle=np.minimum(full_angle,np.pi-full_angle)
            displacement=np.linalg.norm(current[a,None,:2]-boxes[None,b,:2],axis=-1)
            angle=np.where(displacement<1.,shape_angle,full_angle)
            feasible=(distance<=(35 if category==0 else 15)*abs(dt)+3)&(dim<1.2)&(angle<1.3)
            cost=np.where(feasible,distance+10*dim+5*angle,1e6)
            rows,cols=linear_sum_assignment(cost)
            for i,j in zip(rows,cols):
                if cost[i,j]>=1e6:continue
                source,target=a[i],b[j]
                history[source,frame]=boxes[target];mask[source,frame]=1
                total[frame]+=1;correct[frame]+=int(current_ids[source]==identities[target])
    return history,mask,oracle,oracle_mask,correct,total


def track_dense_boxes(current,classes,frames):
    """Inference tracker: only geometry/classes, no IDs or velocity-label inputs."""
    if len(frames)!=16:raise ValueError('need 16 causal 10Hz detection frames')
    n=len(current);observed=np.zeros((n,16,5),np.float32);valid=np.zeros((n,16),np.uint8)
    observed[:,-1]=current;valid[:,-1]=1
    last=current.copy();last_time=np.zeros(n);velocity=np.zeros((n,2),np.float32);has_velocity=np.zeros(n,bool)
    for frame in range(14,-1,-1):
        boxes,labels=frames[frame][:2];dt=(frame-15)*.1
        elapsed=dt-last_time
        expected=last[:,:2]+velocity*elapsed[:,None]
        for category in range(4):
            a=np.flatnonzero(classes==category);b=np.flatnonzero(labels==category)
            if not len(a) or not len(b):continue
            distance=np.linalg.norm(expected[a,None]-boxes[None,b,:2],axis=-1)
            dimension=np.abs(np.log(last[a,None,2:4]/boxes[None,b,2:4])).sum(-1)
            full=np.abs(wrap_angle(last[a,None,4]-boxes[None,b,4]))
            shape=np.minimum(full,np.pi-full)
            change=np.linalg.norm(last[a,None,:2]-boxes[None,b,:2],axis=-1)
            angle=np.where(change<.5,shape,full)
            feasible=(distance<=(35 if category==0 else 15)*np.abs(elapsed[a,None])+1.5)&(dimension<1.2)&(angle<1.3)
            cost=np.where(feasible,distance+10*dimension+5*angle,1e6)
            rows,cols=linear_sum_assignment(cost)
            for i,j in zip(rows,cols):
                if cost[i,j]>=1e6:continue
                source,target=a[i],b[j]
                measured=(last[source,:2]-boxes[target,:2])/(last_time[source]-dt)
                velocity[source]=.5*(velocity[source]+measured) if has_velocity[source] else measured
                speed=np.linalg.norm(velocity[source])
                if speed>50:velocity[source]*=50/speed
                has_velocity[source]=True;last[source]=boxes[target];last_time[source]=dt
                observed[source,frame]=boxes[target];valid[source,frame]=1
    selected=[0,5,10,15];history=observed[:,selected];mask=valid[:,selected]
    return history,mask


def associate_dense(current,classes,current_ids,frames):
    history,mask=track_dense_boxes(current,classes,frames)
    n=len(current);selected=[0,5,10,15]
    oracle=np.zeros_like(history);oracle_mask=np.zeros_like(mask);oracle[:,-1]=current;oracle_mask[:,-1]=1
    correct=np.zeros(3,np.int64);total=np.zeros(3,np.int64)
    for out,frame in enumerate(selected[:-1]):
        boxes,_,identities,_=frames[frame]
        lookup={int(identity):i for i,identity in enumerate(identities)}
        for i,identity in enumerate(current_ids):
            if int(identity) in lookup:
                oracle[i,out]=boxes[lookup[int(identity)]];oracle_mask[i,out]=1
            if mask[i,out]:
                # Identity is used only after association, to audit correctness.
                distances=np.linalg.norm(boxes[:,:2]-history[i,out,:2],axis=1)
                j=int(distances.argmin());total[out]+=1;correct[out]+=int(identities[j]==identity)
    return history,mask,oracle,oracle_mask,correct,total


def render(payload,anchors,association='four'):
    scenario=womd_pb2.Scenario();scenario.ParseFromString(payload)
    result=[]
    for anchor in anchors:
        pose=ego_pose_from_state(scenario.tracks[scenario.sdc_track_index].states[anchor])
        frames=[detections(scenario,anchor+offset,pose) for offset in HISTORY]
        current,classes,identities,velocities=frames[-1]
        # Ordering is geometric, not the GT numeric identity.
        order=np.lexsort((current[:,1],current[:,0],np.linalg.norm(current[:,:2],axis=1)))
        current,classes,identities,velocities=current[order],classes[order],identities[order],velocities[order]
        if association=='dense':
            full=[detections(scenario,anchor+offset,pose) for offset in range(-15,1)]
            history,mask,oracle,oracle_mask,correct,total=associate_dense(current,classes,identities,full)
        else:history,mask,oracle,oracle_mask,correct,total=associate(current,classes,identities,frames)
        future,_,future_ids,_=detections(scenario,anchor+10,pose)
        lookup={int(identity):i for i,identity in enumerate(future_ids)}
        target=np.zeros_like(current);valid=np.zeros(len(current),np.uint8)
        for i,identity in enumerate(identities):
            if int(identity) in lookup:target[i]=future[lookup[int(identity)]];valid[i]=1
        result.append(dict(scene=scenario.scenario_id,anchor=int(anchor),history=history,mask=mask,
                           oracle_history=oracle,oracle_mask=oracle_mask,category=classes,target=target,
                           future_valid=valid,gt_velocity=velocities,association_correct=correct,association_total=total))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('outputs/world_model/motion_latent_v3/dataset2048'))
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--association',choices=['four','dense'],default='four')
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    with np.load(args.data/'windows.npz') as a:scene,anchor,split=a['scene'],a['anchor'],a['split']
    wanted={str(s):anchor[scene==s].tolist() for s in np.unique(scene)}
    jobs={}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for path in sorted(args.source.glob('*.tfrecord-*')):
            for payload in _iter_tfrecords(path):
                s=womd_pb2.Scenario();s.ParseFromString(payload)
                if s.scenario_id in wanted and s.scenario_id not in jobs:
                    jobs[s.scenario_id]=pool.submit(render,payload,wanted[s.scenario_id],args.association)
                if len(jobs)==len(wanted):break
            if len(jobs)==len(wanted):break
        rows={}
        for i,job in enumerate(jobs.values(),1):
            for row in job.result():rows[(row['scene'],row['anchor'])]=row
            if i%128==0:print('prepared scenes',i,'/',len(wanted),flush=True)
    keys=list(zip(map(str,scene),map(int,anchor)))
    if not all(k in rows for k in keys):raise ValueError('missing requested windows')
    offsets=np.cumsum([0]+[len(rows[k]['category']) for k in keys])
    fields={name:np.concatenate([rows[k][name] for k in keys]) for name in
            ('history','mask','oracle_history','oracle_mask','category','target','future_valid','gt_velocity')}
    correct=sum((rows[k]['association_correct'] for k in keys),np.zeros(3,np.int64))
    total=sum((rows[k]['association_total'] for k in keys),np.zeros(3,np.int64))
    np.savez_compressed(args.output/'instances.npz',scene=scene,anchor=anchor,split=split,offsets=offsets,**fields)
    meta=dict(windows=len(keys),actors=int(offsets[-1]),scenarios=len(wanted),source=str(args.source),
              parent_dataset_sha256=hashlib.sha256((args.data/'windows.npz').read_bytes()).hexdigest(),
              predictor_input='Four precise historical boxes and masks, associated geometrically without IDs. No numeric ID or GT velocity features.',
              oracle_history='GT-associated past history retained for a clearly labeled diagnostic only; default model uses Hungarian history.',
              information_change='Precise object boxes and all currently labeled source actors, beyond original cropped raster latent inputs; this is an instance-token architecture.',
              gt_velocity='Saved only for privileged CV/reference audits, never concatenated to model inputs.',
              association_accuracy=(correct/np.maximum(total,1)).tolist(),association_samples=total.tolist())
    meta['source_code_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    meta['association_mode']=args.association
    meta['additional_observation_note']='Dense mode uses 16 causal 10Hz box detection frames for association, then samples four histories; four mode uses only four 2Hz frames.'
    (args.output/'dataset.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta),flush=True)


if __name__=='__main__':main()
