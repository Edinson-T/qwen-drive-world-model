"""Current object velocity supervision only; never an inference input."""
import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw

from .latent_motion_data import GRID
from .motion_gt import ego_pose_from_state,global_to_ego,wrap_angle,_box_footprint
from .build_motion_gt_dataset import _iter_tfrecords
from .protos import womd_pb2


def render(payload,anchors):
    s=womd_pb2.Scenario();s.ParseFromString(payload);ego=s.tracks[s.sdc_track_index];result=[]
    for anchor in anchors:
        pose=ego_pose_from_state(ego.states[anchor]);c,t=np.cos(pose[2]),np.sin(pose[2])
        fields=np.zeros((2,200,400),np.float32);hits=np.zeros((200,400),np.uint16)
        for track in s.tracks:
            if track.id==ego.id or track.object_type not in (1,2,3,4):continue
            state=track.states[anchor]
            if not state.valid or not track.states[anchor-5].valid or state.length<=0 or state.width<=0:continue
            xyz=global_to_ego(np.array([[state.center_x,state.center_y,state.center_z]]),pose)[0]
            box=np.r_[xyz,state.length,state.width,state.height,wrap_angle(state.heading-pose[2])]
            points=GRID.pixels(_box_footprint(box))*4
            image=Image.new('L',(400,200));ImageDraw.Draw(image).polygon([tuple(p) for p in points],fill=1)
            mask=np.asarray(image).astype(bool)
            velocity=np.array([c*state.velocity_x+t*state.velocity_y,-t*state.velocity_x+c*state.velocity_y])
            fields[:,mask]=velocity[:,None];hits[mask]+=1
        valid=hits==1
        denominator=valid.reshape(50,4,100,4).sum((1,3))
        velocity=(fields*valid).reshape(2,50,4,100,4).sum((2,4))/np.maximum(denominator,1)
        result.append((s.scenario_id,int(anchor),velocity.astype(np.float16),(denominator>0).astype(np.uint8)))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=8);args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((args.data/'dataset.json').read_text())
    with np.load(args.data/'windows.npz') as a:scenes=a['scene'];anchors=a['anchor']
    wanted={str(s):anchors[scenes==s].tolist() for s in np.unique(scenes)};jobs={}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for path in sorted(Path(manifest['source']).glob('*.tfrecord-*')):
            for payload in _iter_tfrecords(path):
                s=womd_pb2.Scenario();s.ParseFromString(payload)
                if s.scenario_id in wanted and s.scenario_id not in jobs:
                    jobs[s.scenario_id]=pool.submit(render,payload,wanted[s.scenario_id])
                if len(jobs)==len(wanted):break
            if len(jobs)==len(wanted):break
        rows={}
        for i,job in enumerate(jobs.values(),1):
            for scene,anchor,velocity,mask in job.result():rows[(scene,anchor)]=(velocity,mask)
            if i%128==0:print(f'velocity labels {i}/{len(wanted)} scenarios',flush=True)
    keys=list(zip(map(str,scenes),map(int,anchors)))
    if not all(k in rows for k in keys):raise ValueError('missing supervision rows')
    np.savez_compressed(args.output,scene=scenes,anchor=anchors,
                        velocity=np.stack([rows[k][0] for k in keys]),mask=np.stack([rows[k][1] for k in keys]))
    args.output.with_suffix('.json').write_text(json.dumps(dict(dataset_sha256=hashlib.sha256((args.data/'windows.npz').read_bytes()).hexdigest(),
        semantics='current annotated object velocity in current ego axes, training auxiliary target only',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),windows=len(keys)),indent=2)+'\n')


if __name__=='__main__':main()
