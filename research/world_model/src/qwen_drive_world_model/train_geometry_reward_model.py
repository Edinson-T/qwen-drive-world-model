"""Train a current-BEV RM with explicit logged-future occupancy geometry."""

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
from .train_reward_model import MARGINS as BASE_MARGINS


MODES = (
    "current",
    "logged_geometry",
    "predicted_geometry",
    "logged_relative_geometry",
    "predicted_relative_geometry",
)
GEOMETRY_MARGINS = (*BASE_MARGINS, 1.25, 1.5)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--predicted-data", type=Path, required=True)
    parser.add_argument("--bev-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    return parser.parse_args()


class GeometryRewardModel(nn.Module):
    def __init__(self, mode: str, hidden: int):
        super().__init__()
        self.mode = mode
        self.current = nn.Sequential(
            nn.Linear(32 * 4 * 8, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.trajectory = nn.Sequential(
            nn.Linear(61, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.anchor = nn.Embedding(22, hidden)
        self.geometry = nn.Sequential(
            nn.Linear(30, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, trajectory, anchor, speed, current, geometry):
        batch, candidates = trajectory.shape[:2]
        trajectory = trajectory.clone()
        trajectory[..., :2] /= 50.0
        value = self.trajectory(
            torch.cat(
                [trajectory.flatten(2), speed[:, None, None].expand(-1, candidates, 1)],
                -1,
            )
        )
        value = value + self.anchor(anchor + 1)
        value = value + self.current(current.flatten(1))[:, None]
        if self.mode != "current":
            value = value + self.geometry(geometry)
        return self.head(value).squeeze(-1)


def _compress(latent):
    tensor = torch.from_numpy(latent.astype(np.float32))
    tensor = F.adaptive_avg_pool2d(
        tensor.reshape(32, 4, 16, 32).mean(1).unsqueeze(0),
        (4, 8),
    )[0]
    return tensor.numpy().astype(np.float16)


def _batch(indices, candidates, current, geometry, device):
    trajectories = torch.from_numpy(candidates["trajectories"][indices].astype(np.float32)).to(device)
    anchors = torch.from_numpy(candidates["anchor_ids"][indices].astype(np.int64)).to(device)
    speed = torch.from_numpy(candidates["initial_speed"][indices].astype(np.float32)).to(device)
    cur = torch.from_numpy(current[indices].astype(np.float32)).to(device)
    geom = torch.from_numpy(geometry[indices].astype(np.float32)).to(device)
    targets = torch.from_numpy(candidates["candidate_rfs"][indices].astype(np.float32)).to(device)
    return trajectories, anchors, speed, cur, geom, targets


def _best(scores, targets):
    result = (float("-inf"), GEOMETRY_MARGINS[0])
    for margin in GEOMETRY_MARGINS:
        selected = _policy_selection(scores, 0, margin)
        value = float(targets[np.arange(len(targets)), selected].mean())
        result = max(result, (value, margin))
    return result[1], result[0]


def _delta_loss(scores, targets):
    target_delta = targets[:, 1:] - targets[:, :1]
    predicted_delta = scores[:, 1:] - scores[:, :1]
    weight = 1.0 + target_delta.abs()
    regression = (
        F.smooth_l1_loss(predicted_delta, target_delta / 2.0, reduction="none")
        * weight
    ).mean()
    target_weights = F.softmax(target_delta / 0.5, dim=1)
    ranking = -(
        target_weights * F.log_softmax(predicted_delta / 0.2, dim=1)
    ).sum(1).mean()
    return regression + 0.25 * ranking


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'protocol.json').write_text(json.dumps(
        {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}, indent=2)+'\n')
    device = torch.device(args.device)
    candidates = np.load(args.data / "candidates.npz")
    logged_geometry = np.load(args.data / "geometry.npz")["geometry"]
    predicted_geometry = np.load(args.predicted_data / "geometry.npz")["geometry"]
    # Features must refer to precisely the same records and candidate order.
    with np.load(args.predicted_data / "candidates.npz") as predicted_candidates:
        for key in ("tokens", "scene_tokens", "trajectories", "anchor_ids", "candidate_rfs"):
            np.testing.assert_array_equal(candidates[key], predicted_candidates[key], err_msg=f"predicted geometry mismatch: {key}")
    if logged_geometry.shape != predicted_geometry.shape or logged_geometry.shape[:2] != candidates['trajectories'].shape[:2]:
        raise ValueError('geometry shape/candidate mismatch')
    current = []
    for token in candidates["tokens"]:
        with np.load(args.bev_cache / f"{str(token)}.npz") as payload:
            current.append(_compress(payload["z_t"]))
    current = np.asarray(current)
    folds = _scene_folds(np.asarray(candidates["scene_tokens"]), args.folds, 0)
    torch.set_num_threads(4)
    report = {"modes": {}, "geometry_inputs": {
        "logged": str(args.data), "predicted": str(args.predicted_data),
        "note": "Logged geometry may contain QD teacher predictions; directory provenance determines whether it is GT."},
        "seed": args.seed, "fold_protocol": "Scene-group nested cross-validation; validation chooses checkpoint and baseline-guard margin"}
    predictions = {}
    for mode in args.modes:
        geometry = (
            predicted_geometry
            if mode in ("predicted_geometry", "predicted_relative_geometry")
            else logged_geometry
        )
        if mode in ("logged_relative_geometry", "predicted_relative_geometry"):
            geometry = geometry - geometry[:, :1]
        output = np.zeros(len(folds), dtype=np.int64)
        output_scores = np.zeros(
            (len(folds), candidates["trajectories"].shape[1]),
            dtype=np.float32,
        )
        fold_rows = []
        for fold in range(args.folds):
            remaining = [value for value in range(args.folds) if value != fold]
            val_fold = remaining[fold % len(remaining)]
            train = np.flatnonzero((folds != fold) & (folds != val_fold))
            val = np.flatnonzero(folds == val_fold)
            test = np.flatnonzero(folds == fold)
            torch.manual_seed(args.seed + fold)
            model = GeometryRewardModel(mode, args.hidden).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
            best_state, best_margin, best_value, stale = None, 0.0, -1e9, 0
            best_epoch = -1
            for epoch in range(args.epochs):
                model.train()
                order = np.random.default_rng(fold).permutation(train)
                for start in range(0, len(order), args.batch_size):
                    batch = _batch(order[start : start + args.batch_size], candidates, current, geometry, device)
                    loss = _delta_loss(model(*batch[:-1]), batch[-1])
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                model.eval()
                with torch.inference_mode():
                    batch = _batch(val, candidates, current, geometry, device)
                    scores = model(*batch[:-1]).cpu().numpy()
                margin, value = _best(scores, batch[-1].cpu().numpy())
                if value > best_value + 1e-5:
                    best_state = copy.deepcopy(model.state_dict())
                    best_margin, best_value, stale = margin, value, 0
                    best_epoch = epoch
                else:
                    stale += 1
                    if stale >= 6:
                        break
            model.load_state_dict(best_state)
            checkpoint_dir = args.output / 'checkpoints' / mode
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(dict(model=best_state, mode=mode, hidden=args.hidden,
                            fold=fold, validation_fold=val_fold, margin=best_margin,
                            selected_epoch=best_epoch, seed=args.seed+fold,
                            train_tokens=candidates['tokens'][train].tolist(),
                            validation_tokens=candidates['tokens'][val].tolist(),
                            test_tokens=candidates['tokens'][test].tolist()),
                       checkpoint_dir / f'fold{fold}.pt')
            model.eval()
            with torch.inference_mode():
                batch = _batch(test, candidates, current, geometry, device)
                scores = model(*batch[:-1]).cpu().numpy()
            targets = batch[-1].cpu().numpy()
            output[test] = _policy_selection(scores, 0, best_margin)
            output_scores[test] = scores
            fold_rows.append(
                {
                    "fold": fold,
                    "validation_fold": val_fold,
                    "margin": best_margin,
                    "selected_epoch": best_epoch,
                }
            )
            print(mode, fold, best_value, best_margin, flush=True)
        labels = candidates["candidate_rfs"]
        chosen = labels[np.arange(len(labels)), output]
        plain = labels[np.arange(len(labels)), output_scores.argmax(1)]
        report["modes"][mode] = {
            "top1_rfs": float(chosen.mean()),
            "baseline_rfs": float(labels[:, 0].mean()),
            "oracle_rfs": float(labels.max(1).mean()),
            "switches": int((output != 0).sum()),
            "mean_regret": float((labels.max(1)-chosen).mean()),
            "plain_argmax_rfs": float(plain.mean()),
            "plain_argmax_regret": float((labels.max(1)-plain).mean()),
            "folds": fold_rows,
        }
        predictions[mode] = output
        predictions[f"{mode}_scores"] = output_scores
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(args.output / "predictions.npz", folds=folds, **predictions)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
