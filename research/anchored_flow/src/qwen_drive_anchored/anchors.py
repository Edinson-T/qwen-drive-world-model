"""Motion-anchor construction and deterministic candidate allocation."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "AnchorBank",
    "fit_anchor_bank",
    "fit_stratified_medoid_bank",
    "main",
    "normalize_futures",
    "parse_args",
    "select_anchor_ids",
]


def _wrap_heading(trajectory: np.ndarray) -> np.ndarray:
    out = np.asarray(trajectory, dtype=np.float64).copy()
    out[..., 2] = (out[..., 2] + np.pi) % (2 * np.pi) - np.pi
    return out


def normalize_futures(trajectories: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Normalize ``[N, T, 3]`` trajectories with Qwen-Drive's coordinate scale."""
    trajectories = np.asarray(trajectories, dtype=np.float64)
    if trajectories.ndim != 3 or trajectories.shape[-1] != 3:
        raise ValueError(f"expected [N, T, 3], got {trajectories.shape}")
    return (_wrap_heading(trajectories) / np.asarray(scale, dtype=np.float64)).astype(
        np.float32
    )


def _sq_distances(points: np.ndarray, centres: np.ndarray, chunk: int = 4096) -> np.ndarray:
    distances = np.empty((len(points), len(centres)), dtype=np.float64)
    centre_sq = np.einsum("kd,kd->k", centres, centres)
    for start in range(0, len(points), chunk):
        block = points[start : start + chunk]
        distances[start : start + len(block)] = (
            np.einsum("nd,nd->n", block, block)[:, None]
            - 2.0 * block @ centres.T
            + centre_sq[None]
        )
    return np.maximum(distances, 0.0)


def _kmeans_plus_plus(
    points: np.ndarray, num_clusters: int, rng: np.random.Generator
) -> np.ndarray:
    centres = [points[int(rng.integers(len(points)))]]
    closest = _sq_distances(points, np.stack(centres))[:, 0]
    for _ in range(1, num_clusters):
        total = float(closest.sum())
        if total <= 0:
            index = int(rng.integers(len(points)))
        else:
            index = int(rng.choice(len(points), p=closest / total))
        centres.append(points[index])
        new_distance = _sq_distances(points, points[index : index + 1])[:, 0]
        closest = np.minimum(closest, new_distance)
    return np.stack(centres)


def _kmeans(
    points: np.ndarray,
    num_clusters: int,
    *,
    seed: int,
    max_iter: int,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not 1 <= num_clusters <= len(points):
        raise ValueError(f"num_clusters must be in [1, {len(points)}]")
    rng = np.random.default_rng(seed)
    centres = _kmeans_plus_plus(points, num_clusters, rng)
    labels = np.zeros(len(points), dtype=np.int64)
    for _ in range(max_iter):
        distances = _sq_distances(points, centres)
        new_labels = distances.argmin(axis=1)
        new_centres = centres.copy()
        for cluster in range(num_clusters):
            members = points[new_labels == cluster]
            if len(members):
                new_centres[cluster] = members.mean(axis=0)
            else:
                farthest = int(distances.min(axis=1).argmax())
                new_centres[cluster] = points[farthest]
        shift = float(np.max(np.linalg.norm(new_centres - centres, axis=1)))
        centres = new_centres
        labels = new_labels
        if shift <= tolerance:
            break
    distances = _sq_distances(points, centres)
    labels = distances.argmin(axis=1)
    return centres.astype(np.float32), labels


@dataclass
class AnchorBank:
    """A fixed set of normalized and physical trajectory modes."""

    normalized: np.ndarray
    physical: np.ndarray
    counts: np.ndarray
    scale: np.ndarray
    seed: int = 0
    nav_commands: np.ndarray | None = None
    speed_bins: np.ndarray | None = None
    speed_edges: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.normalized = np.asarray(self.normalized, dtype=np.float32)
        self.physical = np.asarray(self.physical, dtype=np.float32)
        self.counts = np.asarray(self.counts, dtype=np.int64)
        self.scale = np.asarray(self.scale, dtype=np.float32)
        if self.normalized.ndim != 3 or self.normalized.shape[-1] != 3:
            raise ValueError("normalized anchors must have shape [K, T, 3]")
        if self.physical.shape != self.normalized.shape:
            raise ValueError("physical and normalized anchors must have equal shapes")
        if len(self.counts) != len(self.normalized):
            raise ValueError("counts must have one entry per anchor")
        optional = {
            "nav_commands": self.nav_commands,
            "speed_bins": self.speed_bins,
        }
        for name, value in optional.items():
            if value is None:
                continue
            value = np.asarray(value, dtype=np.int64)
            if len(value) != len(self.normalized):
                raise ValueError(f"{name} must contain one value per anchor")
            setattr(self, name, value)
        if self.speed_edges is not None:
            self.speed_edges = np.asarray(self.speed_edges, dtype=np.float32)

    @property
    def num_anchors(self) -> int:
        return len(self.normalized)

    def nearest(self, normalized_trajectory: np.ndarray) -> tuple[int, float]:
        trajectory = np.asarray(normalized_trajectory, dtype=np.float32)
        distances = np.linalg.norm(
            self.normalized.reshape(self.num_anchors, -1)
            - trajectory.reshape(1, -1),
            axis=1,
        )
        index = int(distances.argmin())
        return index, float(distances[index])

    def nearest_xy(self, normalized_trajectory: np.ndarray, reduce: str = "mean") -> int:
        """Match DiffusionDrive's anchor assignment using mean x/y distance."""
        trajectory = np.asarray(normalized_trajectory, dtype=np.float32)
        distances = np.linalg.norm(
            self.normalized[..., :2] - trajectory[None, ..., :2],
            axis=-1,
        )
        if reduce == "mean":
            score = distances.mean(axis=1)
        elif reduce == "sum":
            score = distances.sum(axis=1)
        else:
            raise ValueError(f"unsupported reduce {reduce!r}")
        return int(score.argmin())

    def candidate_ids(self, num_samples: int) -> np.ndarray:
        """Allocate candidates deterministically across anchors.

        If there are fewer candidates than anchors, evenly spaced anchors are
        selected. If there are more, anchors are tiled. This prevents
        ``num_samples`` from degenerating into independent noise seeds.
        """
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if num_samples <= self.num_anchors:
            ids = np.linspace(0, self.num_anchors - 1, num_samples).round().astype(np.int64)
        else:
            repeats = int(np.ceil(num_samples / self.num_anchors))
            ids = np.tile(np.arange(self.num_anchors, dtype=np.int64), repeats)[:num_samples]
        return ids

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            normalized=self.normalized,
            physical=self.physical,
            counts=self.counts,
            scale=self.scale,
            seed=np.asarray(self.seed, dtype=np.int64),
            version=np.asarray("qwen-drive-anchors-v1"),
            **(
                {}
                if self.nav_commands is None
                else {
                    "nav_commands": self.nav_commands,
                    "speed_bins": self.speed_bins,
                    "speed_edges": self.speed_edges,
                }
            ),
        )

    @classmethod
    def load(cls, path: str | Path) -> "AnchorBank":
        with np.load(path, allow_pickle=False) as data:
            return cls(
                normalized=data["normalized"],
                physical=data["physical"],
                counts=data["counts"],
                scale=data["scale"],
                seed=int(data["seed"]),
                nav_commands=data["nav_commands"]
                if "nav_commands" in data
                else None,
                speed_bins=data["speed_bins"] if "speed_bins" in data else None,
                speed_edges=data["speed_edges"] if "speed_edges" in data else None,
            )


def fit_anchor_bank(
    trajectories: np.ndarray,
    scale: np.ndarray,
    num_anchors: int = 20,
    *,
    seed: int = 2027,
    max_iter: int = 100,
    tolerance: float = 1e-5,
) -> AnchorBank:
    """Fit a deterministic K-Means++ bank on normalized future trajectories."""
    physical = np.asarray(trajectories, dtype=np.float32)
    if len(physical) < num_anchors:
        raise ValueError(f"need at least {num_anchors} trajectories, got {len(physical)}")
    normalized = normalize_futures(physical, scale)
    centres, labels = _kmeans(
        normalized.reshape(len(normalized), -1).astype(np.float64),
        num_anchors,
        seed=seed,
        max_iter=max_iter,
        tolerance=tolerance,
    )
    counts = np.bincount(labels, minlength=num_anchors)
    normalized_centres = centres.reshape(num_anchors, *normalized.shape[1:]).astype(np.float32)
    physical_centres = (
        _wrap_heading(normalized_centres.astype(np.float64) * scale.reshape(1, 1, -1))
    ).astype(np.float32)
    return AnchorBank(
        normalized=normalized_centres,
        physical=physical_centres,
        counts=counts,
        scale=np.asarray(scale, dtype=np.float32),
        seed=seed,
    )


def _cluster_features(
    normalized: np.ndarray,
    waypoint_indices: tuple[int, ...] = (9, 19, 29, 49),
) -> np.ndarray:
    """Use sparse waypoint geometry and standardize channels to avoid scale bias."""
    indices = [index for index in waypoint_indices if index < normalized.shape[1]]
    features = normalized[:, indices, :].reshape(len(normalized), -1).astype(np.float64)
    scale = features.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return features / scale


def _farthest_first_indices(points: np.ndarray, count: int) -> np.ndarray:
    """Deterministic coverage-first medoid initialization."""
    mean = points.mean(axis=0)
    selected = [int(np.argmin(np.linalg.norm(points - mean, axis=1)))]
    for _ in range(1, count):
        distances = np.min(
            np.stack(
                [np.linalg.norm(points - points[index], axis=1) for index in selected],
                axis=0,
            ),
            axis=0,
        )
        distances[np.asarray(selected, dtype=np.int64)] = -1.0
        selected.append(int(np.argmax(distances)))
    return np.asarray(selected, dtype=np.int64)


def _fit_stratum_medoids(
    points: np.ndarray,
    count: int,
    *,
    max_iter: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """K-Means assignments with medoid-constrained real-trajectory centers."""
    if not 1 <= count <= len(points):
        raise ValueError(f"need 1 <= medoids <= points, got {count}, {len(points)}")
    medoids = _farthest_first_indices(points, count)
    labels = np.zeros(len(points), dtype=np.int64)
    for _ in range(max_iter):
        distances = np.stack(
            [np.linalg.norm(points - points[index], axis=1) for index in medoids],
            axis=1,
        )
        new_labels = distances.argmin(axis=1)
        new_medoids = medoids.copy()
        for cluster in range(count):
            members = np.flatnonzero(new_labels == cluster)
            if not len(members):
                nearest = np.argmin(distances.min(axis=1))
                new_medoids[cluster] = min(nearest, len(points) - 1)
                continue
            centre = points[members].mean(axis=0)
            local = np.linalg.norm(points[members] - centre, axis=1)
            new_medoids[cluster] = int(members[int(np.argmin(local))])
        shift = int(np.count_nonzero(new_medoids != medoids))
        medoids = new_medoids
        labels = new_labels
        if shift == 0:
            break
    distances = np.stack(
        [np.linalg.norm(points - points[index], axis=1) for index in medoids],
        axis=1,
    )
    return medoids, distances.argmin(axis=1)


def _allocate_medoids(
    stratum_sizes: dict[tuple[int, int], int],
    num_anchors: int,
) -> dict[tuple[int, int], int]:
    """Give each nonempty stratum one slot, then allocate by sqrt(frequency)."""
    if num_anchors < len(stratum_sizes):
        raise ValueError(
            f"{num_anchors} anchors cannot cover {len(stratum_sizes)} strata"
        )
    allocation = {key: 1 for key in stratum_sizes}
    remaining = num_anchors - len(stratum_sizes)
    while remaining:
        candidates = [
            key
            for key, size in stratum_sizes.items()
            if allocation[key] < size
        ]
        if not candidates:
            break
        weights = np.asarray(
            [
                np.sqrt(stratum_sizes[key]) / (allocation[key] + 1)
                for key in candidates
            ],
            dtype=np.float64,
        )
        weights /= weights.sum()
        choice = candidates[int(np.argmax(weights))]
        allocation[choice] += 1
        remaining -= 1
    return allocation


def fit_stratified_medoid_bank(
    trajectories: np.ndarray,
    nav_commands: np.ndarray,
    initial_speeds: np.ndarray,
    scale: np.ndarray,
    num_anchors: int = 20,
    *,
    speed_edges: tuple[float, ...] = (0.0, 2.0, 5.0, 10.0, float("inf")),
    seed: int = 2027,
) -> AnchorBank:
    """Fit command/speed-stratified anchors using real trajectories as medoids."""
    physical = np.asarray(trajectories, dtype=np.float32)
    nav_commands = np.asarray(nav_commands, dtype=np.int64)
    initial_speeds = np.asarray(initial_speeds, dtype=np.float32)
    if len(physical) != len(nav_commands) or len(physical) != len(initial_speeds):
        raise ValueError("trajectories, nav commands, and speeds must have equal length")
    if not 1 <= num_anchors <= len(physical):
        raise ValueError("num_anchors must be between 1 and sample count")
    edges = np.asarray(speed_edges, dtype=np.float32)
    if len(edges) < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("speed_edges must be strictly increasing")
    normalized = normalize_futures(physical, scale)
    features = _cluster_features(normalized)
    speed_bins = np.digitize(initial_speeds, edges[1:-1], right=False).astype(np.int64)
    groups: dict[tuple[int, int], list[int]] = {}
    for index, (nav, speed_bin) in enumerate(zip(nav_commands, speed_bins)):
        groups.setdefault((int(nav), int(speed_bin)), []).append(index)
    allocation = _allocate_medoids(
        {key: len(indices) for key, indices in groups.items()},
        num_anchors,
    )
    medoid_indices: list[int] = []
    counts: list[int] = []
    anchor_nav: list[int] = []
    anchor_speed_bin: list[int] = []
    for key, indices in sorted(groups.items()):
        local_indices = np.asarray(indices, dtype=np.int64)
        local_medoids, local_labels = _fit_stratum_medoids(
            features[local_indices],
            min(allocation[key], len(local_indices)),
        )
        for local_medoid, cluster in zip(local_medoids, range(len(local_medoids))):
            medoid_indices.append(int(local_indices[local_medoid]))
            counts.append(int(np.count_nonzero(local_labels == cluster)))
            anchor_nav.append(int(key[0]))
            anchor_speed_bin.append(int(key[1]))
    selected = np.asarray(medoid_indices, dtype=np.int64)
    return AnchorBank(
        normalized=normalized[selected],
        physical=physical[selected],
        counts=np.asarray(counts, dtype=np.int64),
        scale=np.asarray(scale, dtype=np.float32),
        seed=seed,
        nav_commands=np.asarray(anchor_nav, dtype=np.int64),
        speed_bins=np.asarray(anchor_speed_bin, dtype=np.int64),
        speed_edges=edges,
    )


def select_anchor_ids(
    bank: AnchorBank,
    count: int,
    *,
    nav_command: int,
    speed_bin: int,
    required: int | None = None,
) -> np.ndarray:
    """Select a scene-feasible anchor subset while optionally keeping a GT anchor."""
    if count <= 0:
        raise ValueError("candidate count must be positive")
    if count >= bank.num_anchors:
        return np.arange(bank.num_anchors, dtype=np.int64)
    if bank.nav_commands is None or bank.speed_bins is None:
        return bank.candidate_ids(count)
    selected: list[int] = []
    if required is not None:
        if not 0 <= required < bank.num_anchors:
            raise ValueError(f"required anchor is out of range: {required}")
        selected.append(int(required))
    order = sorted(
        range(bank.num_anchors),
        key=lambda index: (
            int(bank.nav_commands[index]) != nav_command,
            abs(int(bank.speed_bins[index]) - speed_bin),
            index,
        ),
    )
    for index in order:
        if index not in selected:
            selected.append(index)
        if len(selected) == count:
            break
    return np.asarray(selected, dtype=np.int64)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a K-Means anchor bank from a WOD-E2E training scene file."
    )
    parser.add_argument("--scenes", required=True, help="WOD-E2E training scene JSONL")
    parser.add_argument("--output", required=True, help="destination .npz")
    parser.add_argument("--num-anchors", type=int, default=20)
    parser.add_argument(
        "--method",
        choices=("global-kmeans", "stratified-medoids"),
        default="stratified-medoids",
    )
    parser.add_argument(
        "--speed-edges",
        type=float,
        nargs="+",
        default=(0.0, 2.0, 5.0, 10.0, float("inf")),
    )
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--scale",
        type=float,
        nargs=3,
        default=(165.0, 25.0, 1.5703125),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    trajectories = []
    nav_commands = []
    initial_speeds = []
    with open(args.scenes) as handle:
        for line in handle:
            if args.limit is not None and len(trajectories) >= args.limit:
                break
            record = json.loads(line)
            trajectory = record.get("trajectory", {})
            future = trajectory.get("future_traj_10hz")
            if future is None or len(future) < 2:
                continue
            array = np.asarray(future, dtype=np.float32)
            if array.shape[-1] != 3:
                continue
            ego = trajectory.get("ego_status", {}).get("ego_velocity", (0.0, 0.0))
            trajectories.append(array)
            nav_commands.append(int(trajectory.get("nav_command", 0)))
            initial_speeds.append(
                float(
                    trajectory.get("current_state", {}).get(
                        "speed",
                        np.hypot(float(ego[0]), float(ego[1])),
                    )
                )
            )
    if not trajectories:
        raise SystemExit("no future trajectories found")
    lengths = {len(trajectory) for trajectory in trajectories}
    if len(lengths) != 1:
        raise SystemExit(f"inconsistent future lengths: {sorted(lengths)}")
    stacked = np.stack(trajectories)
    scale = np.asarray(args.scale, dtype=np.float32)
    if args.method == "global-kmeans":
        bank = fit_anchor_bank(
            stacked,
            scale,
            num_anchors=args.num_anchors,
            seed=args.seed,
        )
    else:
        bank = fit_stratified_medoid_bank(
            stacked,
            np.asarray(nav_commands, dtype=np.int64),
            np.asarray(initial_speeds, dtype=np.float32),
            scale,
            num_anchors=args.num_anchors,
            speed_edges=tuple(args.speed_edges),
            seed=args.seed,
        )
    bank.save(args.output)
    print(
        f"wrote {bank.num_anchors} anchors to {args.output}; "
        f"method={args.method}; counts={bank.counts.tolist()}"
    )


if __name__ == "__main__":
    main()

