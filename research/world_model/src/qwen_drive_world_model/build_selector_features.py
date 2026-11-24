"""Cache current/independent/shared WM features for fixed RL+K20 candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from .gt_model import GTBEVWorldModel, ObjectHeadConfig
from .model import WorldModelConfig
from .shared_horizon_model import SharedHorizonWorldModel


QD_MAP_TO_MOTION = np.asarray([0, 1, 2, 3, 4, 7], dtype=np.int64)
MODEL_HW = (50, 100)
HORIZONS = (0.5, 1.0, 2.0, 3.0, 5.0)
HORIZON_INDICES = (1, 3, 7, 11, 19)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--bev-cache", type=Path, required=True)
    parser.add_argument("--shared-run", type=Path, required=True)
    parser.add_argument("--h0p5-run", type=Path, required=True)
    parser.add_argument("--h1-run", type=Path, required=True)
    parser.add_argument("--h2-run", type=Path, required=True)
    parser.add_argument("--h3-run", type=Path, required=True)
    parser.add_argument("--h5-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--occupancy-threshold", type=float, default=0.7)
    parser.add_argument("--corridor-half-width-m", type=float, default=3.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit-scenes", type=int, default=None)
    return parser.parse_args()


def _device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_shared(path: Path, device: torch.device) -> SharedHorizonWorldModel:
    checkpoint = torch.load(path / "best.pt", map_location=device, weights_only=False)
    model = SharedHorizonWorldModel(
        WorldModelConfig(**checkpoint["config"]),
        ObjectHeadConfig(**checkpoint["object_config"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval()


def _load_independent(path: Path, device: torch.device) -> GTBEVWorldModel:
    checkpoint = torch.load(path / "best.pt", map_location=device, weights_only=False)
    model = GTBEVWorldModel(
        WorldModelConfig(**checkpoint["config"]),
        ObjectHeadConfig(**checkpoint["object_config"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval()


def _downsample_map(map_t: np.ndarray) -> np.ndarray:
    motion = QD_MAP_TO_MOTION[np.clip(map_t, 0, 5)]
    out_h, out_w = MODEL_HW
    block_h, block_w = motion.shape[0] // out_h, motion.shape[1] // out_w
    one_hot = np.eye(8, dtype=np.float32)[motion]
    one_hot = one_hot.reshape(out_h, block_h, out_w, block_w, 8).max((1, 3))
    importance = np.asarray([0.5, 1.0, 1.1, 1.1, 1.2, 1.3, 1.2, 1.1], np.float32)
    labels = (one_hot * importance).argmax(-1)
    return np.eye(8, dtype=np.float32)[labels].transpose(2, 0, 1)


def _occupancy_from_qd(occ: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    collision = (occ >= 0) & (occ <= 6)
    x_indices = np.rint(np.linspace(40, 159, 100)).astype(np.int64)
    y_indices = np.rint(np.linspace(70, 129, 50)).astype(np.int64)
    crop = collision[x_indices][:, y_indices, :]
    crop = crop.transpose(1, 0, 2)
    band_slices = (slice(6, 9), slice(9, 11), slice(11, 14), slice(14, 16))
    height = np.stack(
        [crop[..., value].max(-1) for value in band_slices]
    ).astype(np.float32)
    occupancy = crop.max(-1).astype(np.float32)
    return occupancy, height


def _corridor_mask(points: np.ndarray, half_width_m: float) -> np.ndarray:
    columns = (points[:, 0] + 30.0) / 0.6
    rows = (points[:, 1] + 15.0) / 0.6
    canvas = Image.new("L", (100, 50), color=0)
    ImageDraw.Draw(canvas).line(
        list(zip(columns.tolist(), rows.tolist())),
        fill=1,
        width=max(1, int(round(2.0 * half_width_m / 0.6))),
        joint="curve",
    )
    return np.asarray(canvas, dtype=bool)


def _trajectory_in_target_frame(trajectory: np.ndarray, pose: np.ndarray) -> np.ndarray:
    dx, dy, yaw = pose
    cosine, sine = np.cos(yaw), np.sin(yaw)
    x, y = trajectory[:, 0] - dx, trajectory[:, 1] - dy
    return np.column_stack(
        [cosine * x + sine * y, -sine * x + cosine * y, trajectory[:, 2] - yaw]
    )


def _region_stats(values: np.ndarray, mask: np.ndarray, threshold: float) -> list[float]:
    selected = values[mask]
    if not len(selected):
        return [0.0, 0.0, 0.0]
    return [
        float(selected.mean()),
        float(selected.max()),
        float((selected >= threshold).mean()),
    ]


def _trajectory_features(
    trajectory: np.ndarray,
    anchor_id: int,
    initial_speed: float,
) -> list[float]:
    xy = trajectory[:, :2]
    delta = np.diff(np.concatenate([np.zeros((1, 2)), xy], axis=0), axis=0)
    headings = trajectory[:, 2]
    heading_delta = np.diff(np.concatenate([[0.0], headings]))
    return [
        *trajectory[list(HORIZON_INDICES)].reshape(-1).tolist(),
        float(np.linalg.norm(delta, axis=1).sum()),
        float(np.abs(xy[:, 1]).max()),
        float(xy[:, 1].max() - xy[:, 1].min()),
        float(np.abs(heading_delta).sum()),
        float(anchor_id) / 20.0,
        float(initial_speed),
        float(trajectory[-1, 0]),
        float(trajectory[-1, 1]),
    ]


def main() -> None:
    args = parse_args()
    device = _device(args.device)
    cache = np.load(args.candidates)
    shared = _load_shared(args.shared_run, device)
    independent = [
        _load_independent(path, device)
        for path in (args.h0p5_run, args.h1_run, args.h2_run, args.h3_run, args.h5_run)
    ]
    base_rows = []
    current_rows = []
    shared_rows = []
    independent_rows = []
    scene_count = (
        len(cache["tokens"])
        if args.limit_scenes is None
        else min(args.limit_scenes, len(cache["tokens"]))
    )
    for scene_index, token in enumerate(cache["tokens"][:scene_count]):
        token = str(token)
        with np.load(args.bev_cache / f"{token}.npz") as payload:
            map_t = payload["map_t"]
            occ_t = payload["occ_t"]
        map_low = _downsample_map(map_t)
        occupancy_low, height_low = _occupancy_from_qd(occ_t)
        current_bev = np.concatenate(
            [map_low, occupancy_low[None], height_low],
            axis=0,
        )
        scene_trajectories = cache["trajectories"][scene_index].astype(np.float32)
        batch = scene_trajectories.shape[0]
        z = torch.from_numpy(current_bev)[None].repeat(batch, 1, 1, 1).to(device)
        trajectories = torch.from_numpy(scene_trajectories).to(device)
        poses = np.stack(
            [
                scene_trajectories[:, value, :]
                for value in HORIZON_INDICES
            ],
            axis=1,
        )
        poses_tensor = torch.from_numpy(poses).to(device)
        ego = torch.tensor(
            [[cache["initial_speed"][scene_index], 0, 0, 0, 0, 1, 0, 0]],
            dtype=torch.float32,
            device=device,
        ).repeat(batch, 1)
        with torch.no_grad():
            shared_output = shared(z, trajectories, ego, poses_tensor)
            independent_outputs = [
                model(z, trajectories, ego) for model in independent
            ]
        for candidate_index in range(batch):
            trajectory = scene_trajectories[candidate_index]
            base_features = _trajectory_features(
                trajectory,
                int(cache["anchor_ids"][scene_index, candidate_index]),
                float(cache["initial_speed"][scene_index]),
            )
            current_features = []
            for horizon_index, point_index in enumerate(HORIZON_INDICES):
                prefix = trajectory[: point_index + 1]
                mask = _corridor_mask(prefix, args.corridor_half_width_m)
                current_features.extend(
                    _region_stats(occupancy_low, mask, 0.5)
                )
                current_features.extend(
                    float(map_low[class_id][mask].mean())
                    for class_id in (1, 2, 3, 4, 7)
                )
            base_rows.append(base_features)
            current_rows.append(current_features)
            shared_features = []
            independent_features = []
            for horizon_index, pose in enumerate(poses[candidate_index]):
                future_trajectory = _trajectory_in_target_frame(trajectory, pose)
                mask = _corridor_mask(future_trajectory, args.corridor_half_width_m)
                shared_probability = (
                    shared_output["occupancy_logits"][candidate_index, horizon_index]
                    .sigmoid()
                    .cpu()
                    .numpy()
                )
                independent_probability = (
                    independent_outputs[horizon_index]["occupancy_logits"][candidate_index]
                    .sigmoid()
                    .cpu()
                    .numpy()
                )
                shared_features.extend(
                    _region_stats(
                        shared_probability,
                        mask,
                        args.occupancy_threshold,
                    )
                )
                independent_features.extend(
                    _region_stats(
                        independent_probability,
                        mask,
                        args.occupancy_threshold,
                    )
                )
            shared_rows.append(shared_features)
            independent_rows.append(independent_features)
        print(f"{scene_index + 1}/{scene_count} {token}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    candidate_count = int(cache["trajectories"].shape[1])
    np.savez_compressed(
        args.output / "features.npz",
        tokens=cache["tokens"][:scene_count],
        scene_tokens=cache["scene_tokens"][:scene_count],
        split=cache["split"][:scene_count],
        candidate_rfs=cache["candidate_rfs"][:scene_count],
        anchor_ids=cache["anchor_ids"][:scene_count],
        base=np.asarray(base_rows, np.float32).reshape(
            scene_count, candidate_count, -1
        ),
        current=np.asarray(current_rows, np.float32).reshape(
            scene_count, candidate_count, -1
        ),
        independent=np.asarray(independent_rows, np.float32).reshape(
            scene_count, candidate_count, -1
        ),
        shared=np.asarray(shared_rows, np.float32).reshape(
            scene_count, candidate_count, -1
        ),
    )
    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "scenes": scene_count,
                "candidates": candidate_count,
                "feature_dims": {
                    "base": int(np.asarray(base_rows).shape[1]),
                    "current": int(np.asarray(current_rows).shape[1]),
                    "independent": int(np.asarray(independent_rows).shape[1]),
                    "shared": int(np.asarray(shared_rows).shape[1]),
                },
                "occupancy_threshold": args.occupancy_threshold,
                "corridor_half_width_m": args.corridor_half_width_m,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

