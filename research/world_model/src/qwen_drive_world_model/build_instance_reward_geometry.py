"""Predict five-horizon geometry from paired causal histories using ONE WM."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .build_logged_future_geometry import _geometry
from .instance_motion_model import history_features
from .recurrent_instance_motion import RecurrentInstanceMotionPredictor, rollout_boxes, OUTPUT_STEPS, OUTPUT_HORIZONS
from .train_instance_motion import rasterize


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidates',type=Path,required=True)
    p.add_argument('--histories',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    args=p.parse_args()
    torch.set_num_threads(4)
    if args.output.exists():
        raise FileExistsError(args.output)
    ckpt=torch.load(args.checkpoint,map_location=args.device,weights_only=False)
    model=RecurrentInstanceMotionPredictor(ckpt['hidden']).to(args.device).eval()
    model.load_state_dict(ckpt['model'])
    with np.load(args.candidates) as a:
        candidates={k:a[k] for k in a.files}
    indices=[i for i,t in enumerate(candidates['tokens']) if (args.histories/f'{t}.npz').exists()]
    if not indices:
        raise ValueError('no paired completed histories')
    geometry=[]
    for i in indices:
        with np.load(args.histories/f"{candidates['tokens'][i]}.npz") as a:
            if str(a['token']) != str(candidates['tokens'][i]) or int(a['candidate_row']) != i:
                raise ValueError('candidate/history row identity mismatch')
            tensor=lambda x:torch.as_tensor(x,device=args.device)
            f,b,w,current=history_features(tensor(a['history']).float(),tensor(a['mask']),tensor(a['category']))
            with torch.inference_mode():
                prediction=model(f,b,w)
                boxes=rollout_boxes(current,prediction).cpu().numpy()
                keep=(prediction[3]>0).cpu().numpy()
            categories=a['category']
        scene=[]
        for trajectory in candidates['trajectories'][i]:
            features=[]
            for step,time in zip(OUTPUT_STEPS,OUTPUT_HORIZONS):
                # Candidate series is 0.25..5s (20 points), unlike Motion's 0..5s.
                pose=trajectory[int(round(time/.25))-1]
                occupancy=rasterize(boxes[:,step],categories,pose,keep[:,step]).any(0)
                features.extend(_geometry(occupancy))
            scene.append(features)
        geometry.append(scene)
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'candidates.npz',**{k:v[indices] for k,v in candidates.items()})
    np.savez_compressed(args.output/'geometry.npz',geometry=np.asarray(geometry,np.float32))
    (args.output/'protocol.json').write_text(json.dumps(dict(
        records=len(indices),requested_records=len(candidates['tokens']),horizons=OUTPUT_HORIZONS,
        single_shared_model=True,checkpoint=str(args.checkpoint),
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        detection_source='QD camera predictions, not Motion GT',
        alignment='causal ego poses estimated from past trajectory metadata',
        labels_used_for_features=False,scope='Dynamic vehicle/pedestrian/bicycle footprint geometry; static map/occ supplied separately'),indent=2)+'\n')
    print(json.dumps(dict(records=len(indices),shape=np.asarray(geometry).shape)),flush=True)


if __name__=='__main__':
    main()
