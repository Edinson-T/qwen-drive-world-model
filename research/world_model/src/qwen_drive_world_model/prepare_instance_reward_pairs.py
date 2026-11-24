"""Join RFS candidates to EXACT WOD-E2E historical records, never to Motion IDs."""
import argparse
import gzip
import json
from pathlib import Path
import sys

import numpy as np

from .pack_waymo import _read_record, REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / 'scripts'))
from interpolate_waymo_trajectory import interpolate_history


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidates', type=Path, required=True)
    p.add_argument('--index', type=Path, required=True)
    p.add_argument('--raw-root', type=Path, required=True)
    p.add_argument('--current-cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    with gzip.open(args.index, 'rt') as f:
        index = json.load(f)
    with np.load(args.candidates) as data:
        candidates = {key: data[key] for key in data.files}
    entries = []
    for row, token in enumerate(candidates['tokens']):
        scene, anchor = str(token).rsplit('-', 1)
        anchor = int(anchor)
        if scene != str(candidates['scene_tokens'][row]):
            raise ValueError('candidate scene mismatch')
        available = index['index'].get(scene, {})
        frames = list(range(anchor-15, anchor+1))
        if any(str(frame) not in available for frame in frames):
            raise ValueError(f'{token}: missing historical observation; do not pad with future frames')
        current = args.current_cache / f'{token}.npz'
        if not current.exists():
            raise FileNotFoundError(current)
        records = []
        for frame in frames:
            shard, offset, length = available[str(frame)]
            records.append(dict(frame=frame, time_seconds=(frame-anchor)/10,
                                shard=str((args.raw_root/index['shards'][shard]).resolve()),
                                byte_offset=offset, byte_length=length))
        entries.append(dict(candidate_row=row, token=str(token), scene=scene, anchor=anchor,
                            records=records, current_cache=str(current.resolve())))
    # Sequential shard order reduces mechanical disk seeks. Restore row order for the manifest.
    order = sorted(range(len(entries)), key=lambda i: (entries[i]['records'][-1]['shard'],
                                                       entries[i]['records'][-1]['byte_offset']))
    for count, i in enumerate(order, 1):
        entry = entries[i]
        record, payload = _read_record(entry['scene'], entry['anchor'], index, args.raw_root)
        if record.context_name != entry['token']:
            raise ValueError('raw record identity mismatch')
        past_poses, _, _ = interpolate_history(record.past)
        if past_poses.shape != (16, 3) or not np.isfinite(past_poses).all():
            raise ValueError('invalid causal ego history')
        entry['estimated_history_ego_poses'] = past_poses.tolist()
        entry['camera_count'] = len(record.images)
        # Audit the embedded Frame, rather than defaulting a missing pose to identity.
        import e2e_proto
        frame = next(v for n,w,v in e2e_proto._iter_fields(payload) if n == 1 and w == 2)
        fields = list(e2e_proto._iter_fields(frame))
        entry['raw_has_ego_pose'] = any(n == 3 and w == 2 for n,w,_ in fields)
        entry['raw_lidar_label_messages'] = sum(n == 6 and w == 2 for n,w,_ in fields)
        if count % 32 == 0:
            print(f'paired {count}/{len(entries)}', flush=True)
    with (args.output/'pairs.jsonl').open('w') as f:
        for entry in entries:
            f.write(json.dumps(entry)+'\n')
    np.savez_compressed(args.output/'candidates.npz', **candidates)
    summary = dict(records=len(entries), scene_groups=len(set(candidates['scene_tokens'].tolist())),
                   candidates_per_record=int(candidates['trajectories'].shape[1]),
                   historical_frames=16, history_times=np.arange(-15,1).astype(float).tolist(),
                   raw_pose_records=sum(e['raw_has_ego_pose'] for e in entries),
                   raw_lidar_label_messages=sum(e['raw_lidar_label_messages'] for e in entries),
                   pairing='Exact E2E context token and TFRecord offset; no join to Motion scenario IDs',
                   ego_alignment='Estimated from CURRENT record past trajectory/velocity via existing interpolate_history; not measured 6DOF odometry',
                   future_reference='No LiDAR GT implied. Logged future QD predictions, if used, must be labeled teacher estimates.',
                   status='Raw records and causal estimated poses paired; historical QD detection extraction still required')
    summary['history_times'] = [i/10 for i in range(-15,1)]
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
