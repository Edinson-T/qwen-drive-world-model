"""Train small baseline-relative Reward Models on QD world-model features."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .evaluate_selector_cv import _policy_selection, _scene_folds


MODES = ("trajectory", "current", "future")
MARGINS = (0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--bev-cache", type=Path, required=True)
    parser.add_argument("--future-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rebuild-features", action="store_true")
    return parser.parse_args()


def _compress_latent(latent: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(latent, dtype=np.float32))
    prefix = tensor.shape[:-3]
    channels = tensor.shape[-3]
    groups = 32
    if channels % groups:
        raise ValueError(f"latent channels must be divisible by {groups}")
    tensor = tensor.reshape(
        *prefix,
        groups,
        channels // groups,
        tensor.shape[-2],
        tensor.shape[-1],
    ).mean(-3)
    tensor = F.adaptive_avg_pool2d(tensor.reshape(-1, groups, 16, 32), (4, 8))
    return tensor.reshape(*prefix, groups, 4, 8).numpy().astype(np.float16)


def _build_feature_store(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    cache_dir = args.output / "feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    current_path = cache_dir / "current.npy"
    future_path = cache_dir / "future.npy"
    candidates = np.load(args.candidates)
    scene_count = len(candidates["tokens"])
    candidate_count = candidates["trajectories"].shape[1]
    if (
        not args.rebuild_features
        and current_path.exists()
        and future_path.exists()
    ):
        current = np.load(current_path, mmap_mode="r")
        future = np.load(future_path, mmap_mode="r")
        return current, future

    current = np.lib.format.open_memmap(
        current_path,
        mode="w+",
        dtype=np.float16,
        shape=(scene_count, 32, 4, 8),
    )
    future = np.lib.format.open_memmap(
        future_path,
        mode="w+",
        dtype=np.float16,
        shape=(scene_count, candidate_count, 5, 32, 4, 8),
    )
    for scene_index, token in enumerate(candidates["tokens"], start=1):
        token = str(token)
        with np.load(args.bev_cache / f"{token}.npz") as payload:
            current[scene_index - 1] = _compress_latent(payload["z_t"])
        with np.load(args.future_cache / f"{token}.npz") as payload:
            future[scene_index - 1] = _compress_latent(payload["future_z"])
        if scene_index % 50 == 0:
            print(f"compressed features {scene_index}/{scene_count}", flush=True)
    current.flush()
    future.flush()
    return current, future


class RewardModel(nn.Module):
    def __init__(self, mode: str, hidden: int) -> None:
        super().__init__()
        self.mode = mode
        self.latent_encoder = nn.Sequential(
            nn.Linear(32 * 4 * 8, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.trajectory_encoder = nn.Sequential(
            nn.Linear(20 * 3 + 1, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.anchor_embedding = nn.Embedding(22, hidden)
        self.pose_encoder = nn.Sequential(
            nn.Linear(3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.horizon_embedding = nn.Parameter(torch.zeros(1, 7, hidden))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=4,
            dim_feedforward=hidden * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        trajectories: torch.Tensor,
        anchor_ids: torch.Tensor,
        speed: torch.Tensor,
        current: torch.Tensor | None,
        future: torch.Tensor | None,
        poses: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, candidates = trajectories.shape[:2]
        trajectory = trajectories.clone()
        trajectory[..., :2] /= 50.0
        trajectory_input = torch.cat(
            [
                trajectory.flatten(2),
                speed[:, None, None].expand(-1, candidates, 1),
            ],
            dim=-1,
        )
        candidate_token = self.trajectory_encoder(trajectory_input)
        candidate_token = candidate_token + self.anchor_embedding(anchor_ids + 1)
        tokens = [candidate_token.unsqueeze(2)]
        if self.mode in ("current", "future") and current is not None:
            current_feature = self.latent_encoder(current.flatten(1))
            tokens.append(current_feature[:, None, None].expand(-1, candidates, 1, -1))
        if self.mode == "future" and future is not None and poses is not None:
            future_feature = self.latent_encoder(
                future.flatten(0, 2).flatten(1)
            ).reshape(batch, candidates, 5, -1)
            normalized_pose = poses.clone()
            normalized_pose[..., 0] /= 50.0
            normalized_pose[..., 1] /= 30.0
            pose_feature = self.pose_encoder(normalized_pose)
            tokens.append(future_feature + pose_feature)
        sequence = torch.cat(tokens, dim=2)
        sequence = sequence + self.horizon_embedding[:, : sequence.shape[2]]
        hidden = self.temporal(sequence.flatten(0, 1))
        hidden = hidden[:, 0].reshape(batch, candidates, -1)
        return self.head(hidden).squeeze(-1)


def _loss(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    delta = targets[:, 1:] - targets[:, :1]
    predicted = scores[:, 1:] - scores[:, :1]
    weight = 1.0 + delta.abs()
    regression = (
        F.smooth_l1_loss(predicted, delta / 2.0, reduction="none") * weight
    ).mean()
    active = delta.abs() > 0.05
    sign = delta.sign()
    ranking = F.softplus(-sign * predicted / 0.2)
    ranking = (ranking * active * delta.abs()).sum() / (
        (active * delta.abs()).sum().clamp_min(1e-6)
    )
    return regression + 0.5 * ranking


def _batched(
    indices: np.ndarray,
    candidates: np.lib.npyio.NpzFile,
    current: np.ndarray,
    future: np.ndarray,
    device: torch.device,
    future_cache: Path,
):
    trajectories = torch.from_numpy(
        candidates["trajectories"][indices].astype(np.float32)
    ).to(device)
    anchor_ids = torch.from_numpy(
        candidates["anchor_ids"][indices].astype(np.int64)
    ).to(device)
    speed = torch.from_numpy(
        candidates["initial_speed"][indices].astype(np.float32)
    ).to(device)
    current_tensor = torch.from_numpy(current[indices].astype(np.float32)).to(device)
    future_tensor = torch.from_numpy(future[indices].astype(np.float32)).to(device)
    relative_poses = []
    for index in indices:
        token = str(candidates["tokens"][index])
        with np.load(future_cache / f"{token}.npz") as payload:
            relative_poses.append(payload["relative_poses"])
    poses = torch.from_numpy(np.stack(relative_poses)).to(device)
    targets = torch.from_numpy(
        candidates["candidate_rfs"][indices].astype(np.float32)
    ).to(device)
    return trajectories, anchor_ids, speed, current_tensor, future_tensor, poses, targets


def _evaluate(
    model: RewardModel,
    indices: np.ndarray,
    candidates: np.lib.npyio.NpzFile,
    current: np.ndarray,
    future: np.ndarray,
    device: torch.device,
    batch_size: int,
    future_cache: Path,
):
    model.eval()
    scores = []
    targets = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            (
                trajectories,
                anchor_ids,
                speed,
                current_tensor,
                future_tensor,
                poses,
                batch_targets,
            ) = _batched(
                batch_indices,
                candidates,
                current,
                future,
                device,
                future_cache,
            )
            scores.append(
                model(
                    trajectories,
                    anchor_ids,
                    speed,
                    current_tensor,
                    future_tensor,
                    poses,
                )
                .cpu()
                .numpy()
            )
            targets.append(batch_targets.cpu().numpy())
    return np.concatenate(scores), np.concatenate(targets)


def _best_margin(scores: np.ndarray, targets: np.ndarray) -> tuple[float, float]:
    best = (float("-inf"), MARGINS[0])
    for margin in MARGINS:
        selected = _policy_selection(scores, baseline_index=0, margin=margin)
        value = float(targets[np.arange(len(targets)), selected].mean())
        if value > best[0]:
            best = (value, margin)
    return best[1], best[0]


def main() -> None:
    args = parse_args()
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    candidates = np.load(args.candidates)
    current, future = _build_feature_store(args)
    scene_folds = _scene_folds(
        np.asarray(candidates["scene_tokens"]),
        args.folds,
        args.seed,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol": "scene-group nested CV baseline-relative reward model",
        "modes": {},
        "device": str(device),
    }
    all_predictions = {}
    all_scores = {}
    for mode_index, mode in enumerate(args.modes):
        predictions = np.zeros(len(candidates["tokens"]), dtype=np.int64)
        outer_scores = np.zeros(
            (len(candidates["tokens"]), candidates["trajectories"].shape[1]),
            dtype=np.float32,
        )
        fold_rows = []
        for outer_fold in range(args.folds):
            remaining = [fold for fold in range(args.folds) if fold != outer_fold]
            val_fold = remaining[outer_fold % len(remaining)]
            train_indices = np.flatnonzero(
                (scene_folds != outer_fold) & (scene_folds != val_fold)
            )
            val_indices = np.flatnonzero(scene_folds == val_fold)
            test_indices = np.flatnonzero(scene_folds == outer_fold)
            torch.manual_seed(args.seed + mode_index * 100 + outer_fold)
            model = RewardModel(mode, args.hidden).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
            best_value = float("-inf")
            best_margin = MARGINS[0]
            best_state = None
            stale = 0
            rng = np.random.default_rng(args.seed + outer_fold)
            for epoch in range(args.epochs):
                model.train()
                order = rng.permutation(train_indices)
                losses = []
                for start in range(0, len(order), args.batch_size):
                    batch_indices = order[start : start + args.batch_size]
                    batch = _batched(
                        batch_indices,
                        candidates,
                        current,
                        future,
                        device,
                        args.future_cache,
                    )
                    scores = model(*batch[:-1])
                    loss = _loss(scores, batch[-1])
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    losses.append(float(loss.detach()))
                val_scores, val_targets = _evaluate(
                    model,
                    val_indices,
                    candidates,
                    current,
                    future,
                    device,
                    args.batch_size,
                    args.future_cache,
                )
                margin, value = _best_margin(val_scores, val_targets)
                print(
                    f"{mode} fold {outer_fold} epoch {epoch:02d} "
                    f"loss {np.mean(losses):.4f} val {value:.4f} margin {margin:g}",
                    flush=True,
                )
                if value > best_value + 1e-5:
                    best_value = value
                    best_margin = margin
                    best_state = copy.deepcopy(model.state_dict())
                    stale = 0
                else:
                    stale += 1
                    if stale >= args.patience:
                        break
            model.load_state_dict(best_state)
            test_scores, test_targets = _evaluate(
                model,
                test_indices,
                candidates,
                current,
                future,
                device,
                args.batch_size,
                args.future_cache,
            )
            selected = _policy_selection(
                test_scores,
                baseline_index=0,
                margin=best_margin,
            )
            predictions[test_indices] = selected
            outer_scores[test_indices] = test_scores
            fold_metrics = {
                "outer_fold": outer_fold,
                "validation_fold": val_fold,
                "best_margin": best_margin,
                "validation_rfs": best_value,
                "test_rfs": float(
                    test_targets[np.arange(len(test_targets)), selected].mean()
                ),
                "switches": int((selected != 0).sum()),
                "scenes": int(len(test_targets)),
            }
            fold_rows.append(fold_metrics)
            print(json.dumps(fold_metrics), flush=True)
        targets = candidates["candidate_rfs"]
        selected_rfs = targets[np.arange(len(targets)), predictions]
        baseline_rfs = targets[:, 0]
        differences = selected_rfs - baseline_rfs
        rng = np.random.default_rng(args.seed + 900 + mode_index)
        bootstrap = differences[
            rng.integers(0, len(differences), (20000, len(differences)))
        ].mean(1)
        report["modes"][mode] = {
            "folds": fold_rows,
            "top1_rfs": float(selected_rfs.mean()),
            "baseline_rfs": float(baseline_rfs.mean()),
            "mean_candidate_rfs": float(targets.mean(1).mean()),
            "oracle_rfs": float(targets.max(1).mean()),
            "delta_vs_baseline": float(differences.mean()),
            "ci95_vs_baseline": [
                float(value) for value in np.quantile(bootstrap, [0.025, 0.975])
            ],
            "wins": int((differences > 0).sum()),
            "losses": int((differences < 0).sum()),
            "ties": int((differences == 0).sum()),
            "switches": int((predictions != 0).sum()),
        }
        all_predictions[mode] = predictions
        all_scores[mode] = outer_scores
    np.savez_compressed(
        args.output / "predictions.npz",
        tokens=candidates["tokens"],
        scene_tokens=candidates["scene_tokens"],
        candidate_rfs=candidates["candidate_rfs"],
        folds=scene_folds,
        **all_predictions,
        **{f"{mode}_scores": all_scores[mode] for mode in args.modes},
    )
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
