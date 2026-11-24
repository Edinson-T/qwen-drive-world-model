"""Train lightweight listwise selectors over fixed RL+K20 candidates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


MODES = (
    "base",
    "current",
    "independent",
    "shared",
    "independent_max",
    "shared_max",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    return parser.parse_args()


def _mode_features(data, mode: str) -> np.ndarray:
    if mode == "base":
        return data["base"]
    if mode == "current":
        return np.concatenate([data["base"], data["current"]], axis=-1)
    if mode == "independent":
        return np.concatenate(
            [data["base"], data["current"], data["independent"]],
            axis=-1,
        )
    if mode == "shared":
        return np.concatenate(
            [data["base"], data["current"], data["shared"]],
            axis=-1,
        )
    if mode == "independent_max":
        return np.concatenate(
            [
                data["base"],
                data["current"],
                data["independent"][..., (1, 4, 7, 10, 13)],
            ],
            axis=-1,
        )
    if mode == "shared_max":
        return np.concatenate(
            [
                data["base"],
                data["current"],
                data["shared"][..., (1, 4, 7, 10, 13)],
            ],
            axis=-1,
        )
    raise ValueError(mode)


class Scorer(nn.Module):
    def __init__(self, input_dim: int, hidden: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


def _loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    normalized_target = target / 10.0
    regression = F.smooth_l1_loss(prediction, normalized_target)
    target_weights = F.softmax(target / 0.5, dim=1)
    ranking = -(
        target_weights * F.log_softmax(prediction / 0.1, dim=1)
    ).sum(1).mean()
    return regression + 0.25 * ranking


def _metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    selected = prediction.argmax(1)
    selected_rfs = target[np.arange(len(target)), selected]
    oracle = target.max(1)
    best = target.argmax(1)
    return {
        "top1_rfs": float(selected_rfs.mean()),
        "mean_candidate_rfs": float(target.mean(1).mean()),
        "oracle_rfs": float(oracle.mean()),
        "rfs_gap": float((oracle - selected_rfs).mean()),
        "first_candidate_rfs": float(target[:, 0].mean()),
        "top_oracle_accuracy": float((selected == best).mean()),
    }


def _train_mode(
    features: np.ndarray,
    targets: np.ndarray,
    split: np.ndarray,
    mode: str,
    args: argparse.Namespace,
) -> tuple[dict, dict]:
    train_indices = np.flatnonzero(split == "train")
    val_indices = np.flatnonzero(split == "val")
    test_indices = np.flatnonzero(split == "test")
    values = _mode_features(features, mode)
    feature_mean = values[train_indices].mean((0, 1), keepdims=True)
    feature_std = values[train_indices].std((0, 1), keepdims=True).clip(min=1e-5)
    normalized = (values - feature_mean) / feature_std
    model = Scorer(normalized.shape[-1], args.hidden)
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    x_train = torch.from_numpy(normalized[train_indices]).float()
    y_train = torch.from_numpy(targets[train_indices]).float()
    x_val = torch.from_numpy(normalized[val_indices]).float()
    y_val = torch.from_numpy(targets[val_indices]).float()
    best_loss = math.inf
    best_state = None
    best_val = None
    patience = 25
    stale = 0
    for epoch in range(args.epochs):
        model.train()
        optimiser.zero_grad(set_to_none=True)
        prediction = model(x_train)
        loss = _loss(prediction, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        model.eval()
        with torch.no_grad():
            val_prediction = model(x_val)
            val_loss = _loss(val_prediction, y_val)
            val_metrics = _metrics(val_prediction.numpy(), y_val.numpy())
        if float(val_loss) < best_loss - 1e-6:
            best_loss = float(val_loss)
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            best_val = val_metrics
            stale = 0
        else:
            stale += 1
        if epoch % 25 == 0 or epoch == args.epochs - 1:
            print(
                f"{mode} epoch {epoch:03d} train {float(loss):.5f} "
                f"val {float(val_loss):.5f} RFS {val_metrics['top1_rfs']:.4f}",
                flush=True,
            )
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError(f"{mode}: no checkpoint selected")
    model.load_state_dict(best_state)
    model.eval()
    x_test = torch.from_numpy(normalized[test_indices]).float()
    y_test = torch.from_numpy(targets[test_indices]).float()
    with torch.no_grad():
        test_prediction = model(x_test).numpy()
    test_metrics = _metrics(test_prediction, y_test.numpy())
    return {
        "mode": mode,
        "feature_dim": int(normalized.shape[-1]),
        "best_val_loss": best_loss,
        "validation": best_val,
        "test": test_metrics,
    }, {
        "state_dict": best_state,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "input_dim": int(normalized.shape[-1]),
        "hidden": args.hidden,
    }


def _train_linear_mode(
    features: dict[str, np.ndarray],
    targets: np.ndarray,
    split: np.ndarray,
    mode: str,
) -> tuple[dict, dict]:
    train_indices = np.flatnonzero(split == "train")
    val_indices = np.flatnonzero(split == "val")
    test_indices = np.flatnonzero(split == "test")
    values = _mode_features(features, mode)
    mean = values[train_indices].mean((0, 1), keepdims=True)
    std = values[train_indices].std((0, 1), keepdims=True).clip(min=1e-5)
    normalized = (values - mean) / std
    centered = normalized - normalized.mean(2, keepdims=True)
    centered_target = targets - targets.mean(1, keepdims=True)
    x_train = centered[train_indices].reshape(-1, centered.shape[-1])
    y_train = centered_target[train_indices].reshape(-1)
    best = None
    for alpha in (0.1, 1.0, 10.0, 100.0, 1000.0):
        gram = x_train.T @ x_train + alpha * np.eye(x_train.shape[1])
        weights = np.linalg.solve(gram, x_train.T @ y_train)
        val_score = centered[val_indices] @ weights
        metrics = _metrics(val_score, targets[val_indices])
        row = {"alpha": alpha, "loss": float(np.mean((val_score - centered_target[val_indices]) ** 2)), **metrics}
        if best is None or row["top1_rfs"] > best["metrics"]["top1_rfs"]:
            best = {"alpha": alpha, "weights": weights, "metrics": row}
    test_score = centered[test_indices] @ best["weights"]
    test_metrics = _metrics(test_score, targets[test_indices])
    return {
        "mode": mode,
        "feature_dim": int(values.shape[-1]),
        "best_val_loss": best["metrics"]["loss"],
        "validation": best["metrics"],
        "test": test_metrics,
    }, {
        "feature_mean": mean,
        "feature_std": std,
        "weights": best["weights"],
        "alpha": best["alpha"],
        "input_dim": int(values.shape[-1]),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data = np.load(args.features)
    targets = np.asarray(data["candidate_rfs"], dtype=np.float32)
    split = np.asarray(data["split"])
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol": "scene-disjoint fixed RL direct+anchor K20/N20",
        "scenes": {
            name: int((split == name).sum()) for name in ("train", "val", "test")
        },
        "modes": {},
    }
    for mode in args.modes:
        metrics, checkpoint = _train_mode(data, targets, split, mode, args)
        report["modes"][mode] = metrics
        torch.save(
            {
                **checkpoint,
                "mode": mode,
                "args": vars(args),
            },
            args.output / f"{mode}.pt",
        )
        linear_metrics, linear_checkpoint = _train_linear_mode(
            data,
            targets,
            split,
            mode,
        )
        report["modes"][f"linear_{mode}"] = linear_metrics
        np.savez_compressed(
            args.output / f"linear_{mode}.npz",
            **linear_checkpoint,
        )
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Fixed-candidate selector prototype",
        "",
        "RL Planner + direct anchor, K=20, N=20. Scene-disjoint split.",
        "",
        "| Feature input | Top-1 RFS | Candidate mean | First candidate | Oracle | RFS gap |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode, item in report["modes"].items():
        test = item["test"]
        lines.append(
            f"| {mode} | {test['top1_rfs']:.4f} | {test['mean_candidate_rfs']:.4f} "
            f"| {test['first_candidate_rfs']:.4f} | {test['oracle_rfs']:.4f} "
            f"| {test['rfs_gap']:.4f} |"
        )
    lines += [
        "",
        "Random@20 expectation equals the candidate mean. Oracle uses labels and is analysis-only.",
    ]
    (args.output / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

