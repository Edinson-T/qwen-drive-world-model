"""Export causal QD detections for RFS records; no future images/labels are read."""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np

from .build_instance_motion import track_dense_boxes
from .pack_waymo import _write_teacher_frame


def qd_to_motion_boxes(boxes, scores, labels, pose, threshold=.3):
    """Explicit QD [x,y,z,w,l,h,yaw,vx,vy] -> Motion [x,y,l,w,yaw].

    Map vehicle/pedestrian/bicycle only. Other QD categories have no validated
    dynamic-model counterpart and remain the current occupancy branch's job.
    QD-reported velocity columns are deliberately ignored.
    """
    boxes, scores, labels = np.asarray(boxes), np.asarray(scores), np.asarray(labels)
    mapping = {0: 0, 4: 1, 2: 2}
    keep = np.asarray([(int(label) in mapping) for label in labels], dtype=bool) & (scores >= threshold)
    keep &= np.isfinite(boxes[:, :7]).all(1) & (boxes[:, 3] > 0) & (boxes[:, 4] > 0)
    selected = boxes[keep][:, [0, 1, 4, 3, 6]].astype(np.float32).copy()
    classes = np.asarray([mapping[int(v)] for v in labels[keep]], np.int64)
    c, s = np.cos(pose[2]), np.sin(pose[2])
    selected[:, :2] = selected[:, :2] @ np.array([[c, s], [-s, c]]) + np.asarray(pose[:2])
    selected[:, 4] = np.arctan2(np.sin(selected[:, 4]+pose[2]), np.cos(selected[:, 4]+pose[2]))
    return selected, classes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--limit', type=int)
    p.add_argument('--threshold', type=float, default=.3)
    p.add_argument('--vlm', type=Path, default=Path('Qwen-Drive-1.0-4B'))
    p.add_argument('--perception', type=Path, default=Path('Qwen-Drive-1.0-4B/perception'))
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--dtype', default='bfloat16')
    p.add_argument('--attn-implementation', default='sdpa')
    args = p.parse_args()
    entries = [json.loads(line) for line in args.pairs.read_text().splitlines() if line.strip()]
    if args.limit:
        entries = entries[:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.output/'detections'
    cache.mkdir(exist_ok=True)
    histories = args.output/'histories'
    histories.mkdir(exist_ok=True)
    config = dict(pairs=str(args.pairs.resolve()), threshold=args.threshold,
                  coordinate_alignment='estimated causal past ego poses, not measured odometry',
                  cameras='existing QD Perception eight-view packer; not the three-view planner input',
                  classes={'vehicle':0, 'pedestrian':1, 'bicycle':2},
                  velocity_input=False, future_image_input=False, dtype=args.dtype,
                  vlm=str(args.vlm.resolve()), perception=str(args.perception.resolve()))
    config_path = args.output/'protocol.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('existing export configuration differs')
    config_path.write_text(json.dumps(config, indent=2)+'\n')
    from .export_latents import _load_models, _infer_frame
    import e2e_proto
    model = processor = None
    started = time.monotonic()
    for number, entry in enumerate(entries, 1):
        out = histories/f"{entry['token']}.npz"
        if out.exists():
            print(f"{number}/{len(entries)} {entry['token']} cached", flush=True)
            continue
        frames = []
        for k, record in enumerate(entry['records']):
            key = f"{entry['scene']}-{record['frame']}"
            path = Path(entry['current_cache']) if k == 15 else cache/f'{key}.npz'
            if not path.exists():
                if model is None:
                    model, processor, _ = _load_models(args)
                payload = e2e_proto.read_record(Path(record['shard']), record['byte_offset'], record['byte_length'])
                frame = e2e_proto.parse_e2ed_frame(payload)
                if frame.context_name != key:
                    raise ValueError('historical frame identity mismatch')
                scratch = args.output/'scratch_frame'
                _write_teacher_frame(scratch, frame, payload)
                result = _infer_frame(model, processor, scratch, device=args.device,
                                      latent_size=(16,32), latent_channels=128)
                np.savez_compressed(path, boxes_t=result['boxes'], scores_t=result['scores'], labels_t=result['labels'])
                shutil.rmtree(scratch)
            with np.load(path) as data:
                frames.append(qd_to_motion_boxes(data['boxes_t'], data['scores_t'], data['labels_t'],
                                                 entry['estimated_history_ego_poses'][k], args.threshold))
        current, classes = frames[-1]
        order = np.lexsort((current[:,1], current[:,0], np.linalg.norm(current[:,:2], axis=1)))
        current, classes = current[order], classes[order]
        history, mask = track_dense_boxes(current, classes, frames)
        np.savez_compressed(out, history=history, mask=mask, category=classes,
                            token=np.asarray(entry['token']), candidate_row=entry['candidate_row'])
        status = dict(completed=number, total=len(entries), seconds=time.monotonic()-started,
                      current_actors=len(current), state='completed' if number == len(entries) else 'running')
        (args.output/'status.json').write_text(json.dumps(status,indent=2)+'\n')
        print(json.dumps(dict(token=entry['token'], **status)), flush=True)


if __name__ == '__main__':
    main()
