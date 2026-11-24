"""Fine-tune Qwen-Drive with a scene-base anchor-residual planner."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.utils.data import DataLoader

from qwen_drive.trajectory import normalize_history, normalize_trajectory

from .anchors import AnchorBank, select_anchor_ids
from .data import (
    AnchorBalanceSampler,
    PlannerDataset,
    StratifiedAnchorSampler,
    load_training_samples,
    speed_bin_indices,
)
from .heads import (
    attach_anchor_heads,
    attach_scene_base_residual_head,
    predict_scene_base_residual_conditioned,
)
from .lora import load_trainable_adapter, save_trainable_adapter, trainable_state_dict
from .runtime import attach_lora, file_sha256, freeze_module, load_released_planner
from .sampling import low_frequency_noise

__all__ = ["main", "parse_args"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="released Qwen-Drive model directory")
    parser.add_argument("--planner", required=True, help="released planner-sft directory")
    parser.add_argument("--anchors", required=True, help="AnchorBank .npz")
    parser.add_argument("--scenes", required=True, help="WOD-E2E training scene JSONL")
    parser.add_argument("--image-root", required=True, help="directory containing frame paths")
    parser.add_argument("--output", required=True, help="adapter output directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--noise-std", type=float, default=0.02)
    parser.add_argument("--noise-modes", type=int, default=6)
    parser.add_argument(
        "--sampler",
        choices=("anchor", "stratified"),
        default="stratified",
    )
    parser.add_argument("--reasoning-planning", action="store_true")
    parser.add_argument("--candidate-count", type=int, default=None)
    parser.add_argument("--flow-time", type=float, default=0.9)
    parser.add_argument("--trajectory-weight", type=float, default=12.0)
    parser.add_argument("--regression-weight", type=float, default=8.0)
    parser.add_argument("--classifier-weight", type=float, default=1.0)
    parser.add_argument("--ranking-weight", type=float, default=5.0)
    parser.add_argument("--ranking-temperature", type=float, default=0.02)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--max-xy-residual-m", type=float, default=12.0)
    parser.add_argument("--max-heading-residual-rad", type=float, default=0.35)
    parser.add_argument("--mode-offset-scale", type=float, default=0.045)
    parser.add_argument("--min-mode-offset-scale", type=float, default=0.04)
    parser.add_argument("--max-mode-offset-scale", type=float, default=0.08)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--resume", default=None, help="adapter checkpoint to resume")
    parser.add_argument("--skip-base-hash", action="store_true")
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="use torchrun/NCCL across all visible GPUs",
    )
    return parser.parse_args(argv)


def _move_inputs(inputs: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }


def _tile(tensor: torch.Tensor, count: int) -> torch.Tensor:
    return tensor.repeat_interleave(count, dim=0)


def _guard_output(output: Path, model: Path, planner: Path) -> None:
    output = output.resolve()
    for base in (model.resolve(), planner.resolve()):
        if output == base or base in output.parents:
            raise ValueError(f"refusing to write adapters inside base checkpoint {base}")


def _save_checkpoint(
    path: Path,
    model,
    optimizer,
    *,
    epoch: int,
    global_step: int,
    args: argparse.Namespace,
    anchors_hash: str,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": "qwen-drive-base-residual-v2",
        "epoch": str(epoch),
        "global_step": str(global_step),
        "anchors_sha256": anchors_hash,
        "base_model": str(Path(args.model).resolve()),
        "base_planner": str(Path(args.planner).resolve()),
        "candidate_count": "" if args.candidate_count is None else str(args.candidate_count),
    }
    save_trainable_adapter(
        model,
        path / "adapter_model.safetensors",
        metadata=metadata,
    )
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
        },
        path / "training_state.pt",
    )
    shutil.copy2(Path(args.anchors).resolve(), path / "anchors.npz")
    manifest = {
        **metadata,
        "candidate_count": args.candidate_count,
        "loss_mode": "base_residual",
        "lora_rank": args.lora_rank,
        "use_lora": args.lora_rank > 0,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "noise_std": args.noise_std,
        "noise_modes": args.noise_modes,
        "flow_time": args.flow_time,
        "trajectory_weight": args.trajectory_weight,
        "regression_weight": args.regression_weight,
        "classifier_weight": args.classifier_weight,
        "ranking_weight": args.ranking_weight,
        "ranking_temperature": args.ranking_temperature,
        "focal_gamma": args.focal_gamma,
        "focal_alpha": args.focal_alpha,
        "max_xy_residual_m": args.max_xy_residual_m,
        "max_heading_residual_rad": args.max_heading_residual_rad,
        "mode_offset_scale": float(
            model.scene_base_residual_head.mode_offset_scale.detach()
        ),
        "min_mode_offset_scale": args.min_mode_offset_scale,
        "max_mode_offset_scale": args.max_mode_offset_scale,
        "base_planner_sha256": None
        if args.skip_base_hash
        else file_sha256(Path(args.planner) / "model.safetensors"),
    }
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2))
    os.sync()


def _candidate_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    scores: torch.Tensor,
    gt_anchor_id: int,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Nearest-anchor regression, focal classification, and candidate ranking."""
    target = target.expand_as(prediction)
    per_candidate = F.smooth_l1_loss(
        prediction,
        target,
        beta=0.05,
        reduction="none",
    ).mean(dim=(1, 2))
    regression = args.regression_weight * F.l1_loss(
        prediction[gt_anchor_id],
        target[0],
    )
    classes = torch.zeros_like(scores)
    classes[gt_anchor_id] = 1.0
    probabilities = scores.sigmoid()
    pt = classes * probabilities + (1 - classes) * (1 - probabilities)
    alpha_t = args.focal_alpha * classes + (1 - args.focal_alpha) * (1 - classes)
    focal = (
        alpha_t
        * (1 - pt).pow(args.focal_gamma)
        * F.binary_cross_entropy_with_logits(
            scores,
            classes,
            reduction="none",
        )
    ).mean()
    ranking_target = torch.softmax(
        -per_candidate.detach() / args.ranking_temperature,
        dim=0,
    )
    ranking = -(ranking_target * torch.log_softmax(scores, dim=0)).sum()
    classification = (
        args.classifier_weight * focal + args.ranking_weight * ranking
    )
    loss = args.trajectory_weight * (regression + classification)
    return loss, {
        "trajectory_loss": regression.detach(),
        "classification_loss": classification.detach(),
        "ranking_loss": ranking.detach(),
    }


def _all_reduce_gradients(parameters: list[torch.nn.Parameter]) -> None:
    """Synchronize one fixed-shape bucket so both ranks issue identical collectives."""
    payloads = []
    dtypes = set()
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            gradient = torch.zeros_like(parameter)
            parameter.grad = gradient
        dtypes.add(gradient.dtype)
        payloads.append(gradient.reshape(-1))
    if len(dtypes) != 1:
        raise RuntimeError(f"gradient bucket contains mixed dtypes: {sorted(dtypes)}")
    bucket = torch.cat(payloads)
    dist.all_reduce(bucket, op=dist.ReduceOp.SUM)
    bucket.div_(dist.get_world_size())
    offset = 0
    for parameter in parameters:
        gradient = parameter.grad
        count = parameter.numel()
        gradient.copy_(bucket[offset : offset + count].view_as(gradient))
        offset += count


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not 0 <= args.flow_time <= 1:
        raise ValueError("--flow-time must be in [0, 1]")
    if args.ranking_temperature <= 0:
        raise ValueError("--ranking-temperature must be positive")
    if args.lora_rank < 0:
        raise ValueError("--lora-rank must be non-negative")
    if args.candidate_count is not None and args.candidate_count <= 0:
        raise ValueError("--candidate-count must be positive")
    output = Path(args.output).expanduser().resolve()
    args.anchors = str(Path(args.anchors).expanduser().resolve())
    args.scenes = str(Path(args.scenes).expanduser().resolve())
    args.image_root = str(Path(args.image_root).expanduser().resolve())
    _guard_output(output, Path(args.model), Path(args.planner))
    distributed = args.distributed
    if distributed:
        if not dist.is_available():
            raise RuntimeError("torch.distributed is unavailable")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(backend="nccl", device_id=device)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
        device = torch.device(args.device)
    is_main = rank == 0
    torch.manual_seed(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    anchors = AnchorBank.load(args.anchors)
    model = load_released_planner(
        args.model,
        args.planner,
        device=device,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    if args.lora_rank > 0:
        attach_lora(
            model,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )
    else:
        freeze_module(model)
    model.planning_expert.noise_modes = args.noise_modes
    attach_anchor_heads(model, anchors.num_anchors)
    attach_scene_base_residual_head(
        model,
        max_xy_residual_m=args.max_xy_residual_m,
        max_heading_residual_rad=args.max_heading_residual_rad,
        mode_offset_scale=args.mode_offset_scale,
        min_mode_offset_scale=args.min_mode_offset_scale,
        max_mode_offset_scale=args.max_mode_offset_scale,
    )
    if args.resume:
        load_trainable_adapter(model, Path(args.resume) / "adapter_model.safetensors")

    samples = load_training_samples(
        args.scenes,
        args.image_root,
        num_history_points=model.config.num_history_points,
        limit=args.limit,
    )
    dataset = PlannerDataset(
        samples,
        model.processor,
        anchors,
        model.config.trajectory_scale,
        with_reasoning=args.reasoning_planning,
    )
    if args.sampler == "stratified":
        sampler = StratifiedAnchorSampler(
            dataset.anchor_ids,
            dataset.nav_commands,
            dataset.initial_speeds,
            seed=args.seed,
            rank=rank,
            world_size=world_size,
        )
    else:
        sampler = AnchorBalanceSampler(
            dataset.anchor_ids,
            seed=args.seed,
            rank=rank,
            world_size=world_size,
        )
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=lambda batch: batch[0],
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    start_epoch = 0
    global_step = 0
    state_path = output / "training_state.pt"
    if args.resume and state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["epoch"])
        global_step = int(state["global_step"])

    anchors_hash = file_sha256(args.anchors)
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "train_metrics.jsonl"
    metrics_handle = (
        metrics_path.open("a" if args.resume else "w") if is_main else None
    )
    if is_main:
        print(f"loaded {len(samples)} scenes, {anchors.num_anchors} anchors")
        print(f"trainable={len(trainable)} tensors; metrics={metrics_path}")
    if world_size > 1:
        dist.barrier()
    running: dict[str, float] = {}
    anchor_center_np = anchors.normalized.mean(axis=0)

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        model.vlm.eval()
        model.planning_expert.train()
        model.anchor_heads.train()
        model.scene_base_residual_head.train()
        optimizer.zero_grad(set_to_none=True)
        for batch_index, item in enumerate(loader):
            inputs = _move_inputs(item["inputs"], device)
            with torch.no_grad():
                if args.reasoning_planning:
                    scene_cache, position_anchor, _ = (
                        model._prefill_with_reasoning(
                            inputs,
                            model.config.max_reasoning_tokens,
                        )
                    )
                else:
                    scene_cache, position_anchor = model._prefill(inputs)
            scale = model.trajectory_scale(device)
            history = normalize_history(inputs["history"].float(), scale)
            target = normalize_trajectory(item["future"].to(device), scale)
            global_gt_anchor_id = int(item["anchor_id"])
            if args.candidate_count is None:
                selected_anchor_ids = np.arange(
                    anchors.num_anchors,
                    dtype=np.int64,
                )
            else:
                speed_bin = int(
                    speed_bin_indices(
                        np.asarray([item["initial_speed"]], dtype=np.float32)
                    )[0]
                )
                selected_anchor_ids = select_anchor_ids(
                    anchors,
                    args.candidate_count,
                    nav_command=int(inputs["nav_command"][0]),
                    speed_bin=speed_bin,
                    required=global_gt_anchor_id,
                )
            anchor_ids = torch.from_numpy(selected_anchor_ids).to(device)
            candidate_count = len(selected_anchor_ids)
            positive_anchor_index = int(
                np.flatnonzero(selected_anchor_ids == global_gt_anchor_id)[0]
            )
            anchor = torch.from_numpy(anchors.normalized).to(device)
            anchor_center = torch.from_numpy(anchor_center_np).to(device)
            candidate_anchors = anchor[anchor_ids]
            centered_anchors = (
                candidate_anchors.float() - anchor_center.float().unsqueeze(0)
            )
            noise = torch.stack(
                [
                    low_frequency_noise(
                        candidate_anchors.shape[1:],
                        std=args.noise_std,
                        modes=args.noise_modes,
                        generator=generator,
                        device=device,
                    )
                    for _ in range(candidate_count)
                ]
            )
            initial = candidate_anchors.float() + noise
            flow_time = torch.full(
                (candidate_count,),
                args.flow_time,
                device=device,
                dtype=torch.float32,
            )
            candidate_history = model.planning_expert.encode_history(
                _tile(history, candidate_count),
                _tile(inputs["nav_command"], candidate_count),
                _tile(inputs["history_velocity"].float(), candidate_count),
                _tile(inputs["history_acceleration"].float(), candidate_count),
            )
            base_history = model.planning_expert.encode_history(
                history,
                inputs["nav_command"],
                inputs["history_velocity"].float(),
                inputs["history_acceleration"].float(),
            )
            base = model.planning_expert.predict_endpoint(
                anchor_center.float().unsqueeze(0),
                flow_time[:1],
                base_history,
                scene_cache,
                position_anchor,
                inputs["nav_command"],
                inputs["ego_status"].float(),
            )
            prediction, scores, _ = predict_scene_base_residual_conditioned(
                model.planning_expert,
                model.anchor_heads,
                model.scene_base_residual_head,
                initial,
                flow_time,
                candidate_history,
                scene_cache,
                position_anchor.expand(-1, candidate_count),
                base,
                centered_anchors,
                anchor_ids,
                _tile(inputs["nav_command"], candidate_count),
                _tile(inputs["ego_status"].float(), candidate_count),
                scale,
            )
            prediction = prediction.clamp(-1.0, 1.0)
            loss, components = _candidate_loss(
                prediction,
                target,
                scores,
                positive_anchor_index,
                args,
            )
            values = {
                "loss": float(loss.detach()),
                **{name: float(value) for name, value in components.items()},
                "mode_offset_scale": float(
                    model.scene_base_residual_head.mode_offset_scale.detach()
                ),
            }
            for name, value in values.items():
                running[name] = running.get(name, 0.0) + value
            running["_count"] = running.get("_count", 0.0) + 1

            (loss / args.gradient_accumulation).backward()
            if (batch_index + 1) % args.gradient_accumulation == 0:
                if world_size > 1:
                    _all_reduce_gradients(trainable)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % args.log_every == 0:
                    count = max(1, int(running.pop("_count", 0)))
                    averaged = {
                        name: value / count
                        for name, value in running.items()
                        if not name.startswith("_")
                    }
                    record = {
                        "epoch": epoch,
                        "step": global_step,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        **averaged,
                    }
                    if is_main:
                        metrics_handle.write(json.dumps(record) + "\n")
                        metrics_handle.flush()
                        print(
                            f"step={global_step} loss={averaged['loss']:.4f} "
                            f"traj={averaged['trajectory_loss']:.4f} "
                            f"rank={averaged['ranking_loss']:.4f} "
                            f"scale={averaged['mode_offset_scale']:.5f}"
                        )
                    running = {}
                if args.max_steps is not None and global_step >= args.max_steps:
                    break
                if args.checkpoint_every and global_step % args.checkpoint_every == 0:
                    if is_main:
                        _save_checkpoint(
                            output / "checkpoints" / f"step_{global_step:06d}",
                            model,
                            optimizer,
                            epoch=epoch,
                            global_step=global_step,
                            args=args,
                            anchors_hash=anchors_hash,
                        )
                    if world_size > 1:
                        dist.barrier()
        if is_main:
            _save_checkpoint(
                output,
                model,
                optimizer,
                epoch=epoch + 1,
                global_step=global_step,
                args=args,
                anchors_hash=anchors_hash,
            )
        if world_size > 1:
            dist.barrier()
        if args.max_steps is not None and global_step >= args.max_steps:
            break

    if is_main:
        metrics_handle.close()
        print(f"wrote base-residual adapter to {output}")
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
