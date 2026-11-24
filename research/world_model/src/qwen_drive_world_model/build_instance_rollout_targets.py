"""Add ten future half-second labels to V8 histories without changing inputs."""
import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .build_instance_motion import detections
from .build_motion_gt_dataset import _iter_tfrecords
from .motion_gt import ego_pose_from_state
from .protos import womd_pb2


def render_future(payload, anchors):
    scenario = womd_pb2.Scenario()
    scenario.ParseFromString(payload)
    result = []
    sdc = scenario.tracks[scenario.sdc_track_index]
    for anchor in anchors:
        pose = ego_pose_from_state(sdc.states[anchor])
        current, categories, identities, _ = detections(scenario, anchor, pose)
        order = np.lexsort((current[:, 1], current[:, 0], np.linalg.norm(current[:, :2], axis=1)))
        current, identities = current[order], identities[order]
        target = np.zeros((len(current), 10, 5), np.float32)
        valid = np.zeros((len(current), 10), np.uint8)
        all_boxes, all_classes, future_ego = [], [], []
        for step, offset in enumerate(range(5, 51, 5)):
            boxes, labels, future_ids, _ = detections(scenario, anchor + offset, pose)
            lookup = {int(identity): i for i, identity in enumerate(future_ids)}
            for i, identity in enumerate(identities):
                j = lookup.get(int(identity))
                if j is not None:
                    target[i, step] = boxes[j]
                    valid[i, step] = 1
            all_boxes.append(boxes)
            all_classes.append(labels)
            ego = sdc.states[anchor + offset]
            # Ego reference for evaluation only, never predictor features.
            ego_box = np.array([[ego.center_x, ego.center_y]])
            from .motion_gt import global_to_ego, wrap_angle
            xy = global_to_ego(np.c_[ego_box, np.zeros(1)], pose)[0, :2]
            future_ego.append([*xy, wrap_angle(ego.heading - pose[2])])
        result.append(dict(scene=scenario.scenario_id, anchor=anchor, current=current,
                           target=target, valid=valid, all_boxes=all_boxes,
                           all_classes=all_classes, future_ego=future_ego))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--instances', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with np.load(args.instances / 'instances.npz') as data:
        old = {key: data[key] for key in ('scene', 'anchor', 'offsets', 'history', 'target', 'future_valid')}
    args.output.mkdir(parents=True)
    wanted = {str(s): old['anchor'][old['scene'] == s].tolist() for s in np.unique(old['scene'])}
    jobs, rows = {}, {}
    sources = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for path in sorted(args.source.glob('*.tfrecord-*')):
            with path.open('rb') as handle:
                while True:
                    offset = handle.tell()
                    import struct
                    header = handle.read(8)
                    if not header:
                        break
                    size = struct.unpack('<Q', header)[0]
                    handle.read(4)
                    payload = handle.read(size)
                    handle.read(4)
                    scenario = womd_pb2.Scenario()
                    scenario.ParseFromString(payload)
                    if scenario.scenario_id in wanted and scenario.scenario_id not in jobs:
                        sid = scenario.scenario_id
                        jobs[sid] = pool.submit(render_future, payload, wanted[sid])
                        sources.append(dict(scene=sid, path=str(path.resolve()), offset=offset))
                    if len(jobs) == len(wanted):
                        break
            if len(jobs) == len(wanted):
                break
        for i, job in enumerate(jobs.values(), 1):
            for row in job.result():
                rows[(row['scene'], row['anchor'])] = row
            if i % 32 == 0:
                print(f'rendered {i}/{len(wanted)} scenarios', flush=True)
    ordered = [rows[(str(s), int(t))] for s, t in zip(old['scene'], old['anchor'])]
    for i, row in enumerate(ordered):
        begin, end = old['offsets'][i:i+2]
        np.testing.assert_allclose(row['current'], old['history'][begin:end, -1], atol=1e-5)
        np.testing.assert_allclose(row['target'][:, 1], old['target'][begin:end], atol=1e-5)
        np.testing.assert_array_equal(row['valid'][:, 1], old['future_valid'][begin:end])
    future_boxes = [box for row in ordered for box in row['all_boxes']]
    np.savez_compressed(args.output / 'targets.npz', scene=old['scene'], anchor=old['anchor'],
                        offsets=old['offsets'], target=np.concatenate([r['target'] for r in ordered]),
                        valid=np.concatenate([r['valid'] for r in ordered]),
                        future_ego=np.asarray([r['future_ego'] for r in ordered], np.float32),
                        future_offsets=np.cumsum([0]+[len(v) for v in future_boxes]),
                        future_boxes=np.concatenate(future_boxes),
                        future_classes=np.concatenate([v for r in ordered for v in r['all_classes']]))
    meta = dict(instances=str(args.instances), source=str(args.source), windows=len(ordered),
                instance_sha256=hashlib.sha256((args.instances / 'instances.npz').read_bytes()).hexdigest(),
                label_times=[.5*i for i in range(1, 11)], output_times=[.5, 1, 2, 3, 5],
                one_second_labels_match_v8=True, records=sources,
                label_semantics='Annotation validity, not physical existence/free space; full future population saved for evaluation')
    (args.output / 'dataset.json').write_text(json.dumps(meta, indent=2)+'\n')
    print(json.dumps({k:v for k,v in meta.items() if k != 'records'}), flush=True)


if __name__ == '__main__':
    main()
