"""Same-raster-input component association + CV, without GT IDs/velocities."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from .train_latent_motion import Counts, atomic_json


def components(mask):
    labels,count=ndimage.label(mask,structure=np.ones((3,3),dtype=np.uint8))
    result=[]
    for i in range(1,count+1):
        pixels=np.argwhere(labels==i)
        result.append(dict(mask=labels==i,center=pixels.mean(0)*.6,area=len(pixels)))
    return result


def estimate_velocities(history,maximum_speed):
    current=components(history[-1])
    points=[[(0.,obj['center'])] for obj in current]
    # All frames already share current-ego axes. Associations use only rasters.
    for frame,dt in [(2,-.5),(1,-1.),(0,-1.5)]:
        past=components(history[frame])
        if not current or not past:continue
        expected=[]
        for observations in points:
            velocity=fit_velocity(observations)
            expected.append(observations[0][1]+velocity*dt)
        distance=np.linalg.norm(np.asarray(expected)[:,None]-np.asarray([v['center'] for v in past])[None],axis=-1)
        area=np.abs(np.log(np.asarray([v['area'] for v in current])[:,None]/np.asarray([v['area'] for v in past])[None]))
        feasible=(distance<=maximum_speed*abs(dt)+1.2)&(area<np.log(3.))
        cost=np.where(feasible,distance+2*area,1e6)
        rows,cols=linear_sum_assignment(cost)
        for r,c in zip(rows,cols):
            if cost[r,c]<1e6:points[r].append((dt,past[c]['center']))
    values=[]
    for obj,observations in zip(current,points):
        velocity=fit_velocity(observations)
        speed=np.linalg.norm(velocity)
        if speed>maximum_speed:velocity*=maximum_speed/speed
        values.append((obj['mask'],velocity))
    return values


def fit_velocity(observations):
    if len(observations)<2:return np.zeros(2)
    t=np.asarray([v[0] for v in observations]);xy=np.asarray([v[1] for v in observations])
    return ((t-t.mean())[:,None]*(xy-xy.mean(0))).sum(0)/((t-t.mean())**2).sum()


def propagate(mask,velocity_yx,pose):
    c,s=np.cos(pose[2]),np.sin(pose[2])
    matrix=np.array([[c,s],[-s,c]])
    center=(np.asarray(mask.shape)-1)/2
    offset=center-matrix@center+(pose[:2][::-1]-velocity_yx)/.6
    return ndimage.affine_transform(mask.astype(np.uint8),matrix,offset,output_shape=mask.shape,
                                    order=0,mode='constant',cval=0,prefilter=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',type=Path,required=True)
    p.add_argument('--split',choices=['development','confirmation'],required=True)
    args=p.parse_args()
    output=args.suite/('raster_tracking_'+args.split+'.json')
    if output.exists():raise FileExistsError(output)
    protocol=json.loads((args.suite/'protocol.json').read_text())
    with np.load(Path(protocol['args']['data'])/'windows.npz') as a:
        scene,anchor,split=a['scene'],a['anchor'],a['split']
        if args.split=='confirmation':
            if not (args.suite/'confirmation_selection.json').exists():raise RuntimeError('preselect learned candidates first')
            ids=np.load(args.suite/'confirmation_rows.npy')
        else:
            with np.load(Path(protocol['args']['val_reference'])/'windows.npz') as ref:
                keys={(str(ref['scene'][i]),int(ref['anchor'][i])) for i in np.flatnonzero(ref['split']=='val')}
            ids=np.asarray([i for i,(s,t) in enumerate(zip(scene,anchor)) if (str(s),int(t)) in keys])
        history=a['history_occ'][ids];actions=a['actions'][ids];target=a['target_occ'][ids]
    counts=Counts();per_scene={}
    for i,row in enumerate(ids):
        models=[estimate_velocities(history[i,:,k],maximum) for k,maximum in enumerate((30.,5.,15.,20.))]
        predictions=np.zeros_like(target[i])
        for candidate in range(3):
            for category,objects in enumerate(models):
                for mask,velocity in objects:
                    predictions[candidate,category]|=propagate(mask,velocity,actions[i,candidate,4])
        counts.update(torch.tensor(predictions),torch.tensor(target[i]))
        pred=predictions.any(1);truth=target[i].any(1)
        value=np.array([(pred&truth).sum(),(pred&~truth).sum(),(~pred&truth).sum()],dtype=np.int64)
        per_scene.setdefault(str(scene[row]),np.zeros(3,dtype=np.int64))[:]+=value
        if (i+1)%64==0:print('tracked',i+1,'/',len(ids),flush=True)
    result=dict(metrics=counts.result(),scenes=len(per_scene),windows=len(ids),
                per_scene_counts={s:v.tolist() for s,v in per_scene.items()},
                method='4-frame per-class connected components, Hungarian association, linear fit of associated centroids, nearest-neighbor raster propagation',
                inputs='same historical 50x100 object rasters and candidate future ego pose; no raw tracks, GT IDs, GT velocities, or future labels as predictor inputs',
                association=dict(max_speed_by_class=[30.,5.,15.,20.],area_ratio_gate=3.,centroid_cost_area_weight=2.),
                caveat='Fixed heuristic diagnostic, not tuned on confirmation. Touching objects merge; border clipping distorts centroids. Not a production tracker or camera evaluation.')
    atomic_json(output,result)
    print(json.dumps(result['metrics']),flush=True)


if __name__=='__main__':main()
