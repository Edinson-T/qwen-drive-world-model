"""Versioned Motion data for latent prediction, independent of legacy caches."""
from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .build_motion_gt_dataset import _iter_tfrecords
from .motion_gt import (
    MOTION_OBJECT_CLASS_NAMES, ego_pose_from_state, global_to_ego,
    rasterize_map_features, scenario_map_features, wrap_angle, _box_footprint,
)
from .protos import womd_pb2


@dataclass(frozen=True)
class GridSpec:
    x_min: float = -30.0
    x_max: float = 30.0
    y_min: float = -15.0
    y_max: float = 15.0
    height: int = 50
    width: int = 100
    axes: str = "rows=y, columns=x; both increasing"

    def pixels(self, xy):
        xy = np.asarray(xy, dtype=np.float64)
        return np.stack(((xy[..., 0] - self.x_min) * self.width / (self.x_max - self.x_min),
                         (xy[..., 1] - self.y_min) * self.height / (self.y_max - self.y_min)), -1)


GRID = GridSpec()
HISTORY = (-15, -10, -5, 0)
ACTION_TIMES = np.arange(21, dtype=np.float64) * .25


def split_for_scene(scene_id):
    # Stable across file ordering. All anchors/candidates of a scenario stay together.
    return "val" if int(hashlib.sha256(scene_id.encode()).hexdigest()[:8], 16) % 5 == 0 else "train"


def local_to_global(xy, pose):
    c, s = np.cos(pose[2]), np.sin(pose[2])
    return np.asarray(xy) @ np.array([[c, s], [-s, c]]) + pose[:2]


def candidate_actions(sdc, anchor, count=3):
    states = sdc.states[anchor:anchor + 51]
    pose = ego_pose_from_state(states[0])
    xy = global_to_ego(np.array([[v.center_x, v.center_y, 0] for v in states]), pose)[:, :2]
    yaw = np.unwrap([v.heading for v in states]) - pose[2]
    times = np.arange(51) / 10
    action = np.stack([np.interp(ACTION_TIMES, times, xy[:, 0]),
                       np.interp(ACTION_TIMES, times, xy[:, 1]),
                       np.interp(ACTION_TIMES, times, yaw)], -1).astype(np.float32)
    actions = [action]
    # Controlled lateral perturbations, not reactive simulation or guaranteed drivable paths.
    u = ACTION_TIMES / 5
    smooth = 3*u*u - 2*u*u*u
    tangent_speed = np.maximum(np.gradient(action[:, 0], .25), 2.0)
    for sign in (-1, 1)[:count - 1]:
        value = action.copy()
        lateral = sign * 8 * smooth
        value[:, 1] += lateral
        value[:, 2] = wrap_angle(value[:, 2] + np.arctan2(sign * 8 * (6*u - 6*u*u) / 5, tangent_speed))
        value[0] = 0
        actions.append(value)
    return np.stack(actions)


def occupancy(scenario, step, reference_pose, extrapolate=0.):
    """Class-specific footprint labels. Zeros mean no labeled box, not observed free space."""
    images = [Image.new("L", (GRID.width * 4, GRID.height * 4)) for _ in range(4)]
    draws = [ImageDraw.Draw(im) for im in images]
    ego_id = scenario.tracks[scenario.sdc_track_index].id
    mapping = {1: 0, 2: 1, 3: 2, 4: 3}
    for track in scenario.tracks:
        if track.id == ego_id or track.object_type not in mapping or step >= len(track.states):
            continue
        state = track.states[step]
        if not state.valid or state.length <= 0 or state.width <= 0:
            continue
        center = global_to_ego(np.array([[state.center_x + extrapolate*state.velocity_x,
                                        state.center_y + extrapolate*state.velocity_y, state.center_z]]), reference_pose)[0]
        box = np.array([*center, state.length, state.width, state.height, wrap_angle(state.heading-reference_pose[2])])
        points = GRID.pixels(_box_footprint(box)) * 4
        if not np.isfinite(points).all():
            continue
        draws[mapping[track.object_type]].polygon([tuple(p) for p in points], fill=1)
    # Conservative aggregation from 0.15m, never bilinear point sampling of sparse labels.
    return np.stack([np.asarray(im).reshape(GRID.height, 4, GRID.width, 4).max((1, 3)) for im in images])


def semantic_map(features, pose):
    high = rasterize_map_features(features, pose)
    blocks = np.eye(8, dtype=np.uint8)[high].reshape(50, 4, 100, 4, 8).max((1, 3))
    priority = np.array([.1, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6])
    return (blocks * priority).argmax(-1).astype(np.uint8)


def causal_status(sdc, anchor):
    now, prev = sdc.states[anchor], sdc.states[anchor - 1]
    c, s = np.cos(now.heading), np.sin(now.heading)
    rotation = np.array([[c, s], [-s, c]])
    velocity = rotation @ [now.velocity_x, now.velocity_y]
    acceleration = rotation @ (np.array([now.velocity_x-prev.velocity_x, now.velocity_y-prev.velocity_y]) * 10)
    # No future-derived navigation command or forward-difference acceleration.
    return np.concatenate((velocity, acceleration)).astype(np.float32)


def render_scenario(payload, anchors):
    scenario=womd_pb2.Scenario()
    scenario.ParseFromString(payload)
    sdc=scenario.tracks[scenario.sdc_track_index]
    features=scenario_map_features(scenario)
    rows=[]
    for anchor in anchors:
        pose=ego_pose_from_state(sdc.states[anchor])
        actions=candidate_actions(sdc,anchor)
        future_poses=[np.r_[local_to_global(action[4,:2],pose),wrap_angle(pose[2]+action[4,2])] for action in actions]
        rows.append(dict(scene=scenario.scenario_id,anchor=int(anchor),split=split_for_scene(scenario.scenario_id),
                         history_map=semantic_map(features,pose),
                         history_occ=np.stack([occupancy(scenario,anchor+offset,pose) for offset in HISTORY]),
                         target_map=np.stack([semantic_map(features,p) for p in future_poses]),
                         target_occ=np.stack([occupancy(scenario,anchor+10,p) for p in future_poses]),
                         cv_occ=np.stack([occupancy(scenario,anchor,p,extrapolate=1.) for p in future_poses]),
                         actions=actions,status=causal_status(sdc,anchor)))
    return rows


def build(args):
    if (args.output / "dataset.json").exists():
        raise FileExistsError(f"refusing to replace existing dataset: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    rows, scenes, skipped, rejected = [], set(), set(), 0
    executor=ProcessPoolExecutor(max_workers=args.workers) if args.workers>1 else None
    jobs=[]
    for path in sorted(args.data_dir.glob("*.tfrecord-*")):
        for payload in _iter_tfrecords(path):
            scenario = womd_pb2.Scenario()
            scenario.ParseFromString(payload)
            if not scenario.scenario_id or scenario.scenario_id in scenes or scenario.scenario_id in skipped:
                continue
            if not 0 <= scenario.sdc_track_index < len(scenario.tracks):
                continue
            sdc = scenario.tracks[scenario.sdc_track_index]
            features = scenario_map_features(scenario)
            times = np.asarray(scenario.timestamps_seconds)
            if not features or len(times) < 66 or not np.allclose(np.diff(times), .1, atol=.002):
                rejected += 1
                continue
            possible = [i for i in range(15, len(times)-50)
                        if all(v.valid for v in sdc.states[i-15:i+51])]
            if not possible:
                continue
            if len(skipped)<args.skip_scenarios:
                skipped.add(scenario.scenario_id)
                continue
            anchors = np.unique(np.asarray(possible)[np.linspace(0, len(possible)-1, min(args.anchors, len(possible)), dtype=int)])
            scenes.add(scenario.scenario_id)
            if executor:
                jobs.append(executor.submit(render_scenario,payload,anchors))
            else:
                rows.extend(render_scenario(payload,anchors))
            if len(scenes) % 32 == 0:
                print(f"selected {len(scenes)} scenarios; rendered {len(rows)} windows", flush=True)
            if len(scenes) >= args.scenarios:
                break
        if len(scenes) >= args.scenarios:
            break
    if executor:
        try:
            for index,job in enumerate(jobs,start=1):
                rows.extend(job.result())  # Preserve source order regardless of worker completion order.
                if index%32==0:
                    print(f'rendered {index}/{len(jobs)} scenarios / {len(rows)} windows',flush=True)
        finally:
            executor.shutdown(cancel_futures=True)
    if not rows:
        raise RuntimeError("no eligible Motion windows")
    arrays = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
    np.savez_compressed(args.output / "windows.npz", **arrays)
    manifest = dict(version="motion_latent_v2.0", grid=asdict(GRID), scenarios=len(scenes), windows=len(rows),
                    split_counts={k: int((arrays['split']==k).sum()) for k in ('train', 'val')},
                    source=str(args.data_dir.resolve()), history_times=[-1.5, -1., -.5, 0.],
                    action_times=ACTION_TIMES.tolist(), horizon=1., input_frame="current ego",
                    target_frame="candidate future ego at 1s", object_classes=list(MOTION_OBJECT_CLASS_NAMES),
                    target_semantics="annotated object box footprints; not full occupancy/free-space/visibility GT",
                    excluded_sdc=True, candidates="logged action plus two lateral perturbations; other agents follow logs",
                    status="current vx,vy, backward-difference ax,ay; no navigation", rejected_scenarios=rejected,
                    args={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()})
    (args.output / "dataset.json").write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('/mnt/hdd/waymo_motion_v1_3_1/training'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scenarios', type=int, default=512)
    parser.add_argument('--anchors', type=int, default=4)
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--skip-scenarios',type=int,default=0,
                        help='Skip this many eligible scenarios before selecting a fresh evaluation subset.')
    build(parser.parse_args())
