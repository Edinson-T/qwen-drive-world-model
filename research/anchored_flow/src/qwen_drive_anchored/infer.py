"""Run one scene-base anchor-residual planner over WOD-E2E scenes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from qwen_drive import InferenceMode
from qwen_drive.benchmarks import read_scene_file
from qwen_drive.trajectory import denormalize_trajectory, normalize_history

from .anchors import AnchorBank
from .heads import attach_anchor_heads, attach_scene_base_residual_head
from .lora import load_trainable_adapter
from .runtime import attach_lora, file_sha256, freeze_module, load_released_planner
from .sampling import sample_anchor2_control, sample_base_residual

__all__ = ["main", "parse_args"]


class SceneInputs(Dataset):
    def __init__(self, samples, processor, with_reasoning: bool) -> None:
        self.samples = samples
        self.processor = processor
        self.with_reasoning = with_reasoning

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        return {
            "inputs": self.processor(
                sample.scene,
                with_reasoning=self.with_reasoning,
                device="cpu",
            ),
            "token": sample.token,
            "scene_token": sample.scene_token,
            "future_trajectory": sample.future_trajectory,
            "future_valid": sample.future_valid,
            "preference_trajectories": sample.preference_trajectories,
            "preference_scores": sample.preference_scores,
            "initial_speed": sample.initial_speed,
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--planner", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--control", choices=("none", "anchor2"), default="none")
    parser.add_argument("--anchors", required=True)
    parser.add_argument("--scenes", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--mode",
        choices=("direct_planning", "reasoning_planning"),
        default="direct_planning",
    )
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-base-hash", action="store_true")
    return parser.parse_args(argv)


def _load_manifest(adapter: Path, anchors: Path, planner: Path, skip_hash: bool) -> dict:
    manifest_path = adapter / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"adapter manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("loss_mode") != "base_residual":
        raise ValueError(
            f"expected a base_residual adapter, got {manifest.get('loss_mode')!r}"
        )
    expected_anchors = manifest.get("anchors_sha256")
    if expected_anchors and expected_anchors != file_sha256(anchors):
        raise ValueError("anchor bank hash does not match the adapter manifest")
    expected_base = manifest.get("base_planner_sha256")
    if expected_base and not skip_hash:
        actual = file_sha256(planner / "model.safetensors")
        if actual != expected_base:
            raise ValueError("base planner hash does not match the adapter manifest")
    return manifest


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.skip < 0 or (args.limit is not None and args.limit <= 0):
        raise ValueError("--skip must be non-negative and --limit must be positive")
    device = torch.device(args.device)
    if args.control == "none" and args.adapter is None:
        raise ValueError("--adapter is required unless --control anchor2 is used")
    adapter = Path(args.adapter) if args.adapter else None
    anchors_path = Path(args.anchors)
    planner = Path(args.planner)
    manifest = (
        _load_manifest(adapter, anchors_path, planner, args.skip_base_hash)
        if adapter is not None
        else {}
    )
    if args.num_samples is None:
        args.num_samples = int(manifest.get("candidate_count") or 20)
    anchors = AnchorBank.load(anchors_path)
    model = load_released_planner(
        args.model,
        args.planner,
        device=device,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.planning_expert.noise_modes = int(manifest.get("noise_modes", 6))
    if adapter is not None:
        if int(manifest.get("lora_rank", 0)) > 0:
            attach_lora(
                model,
                rank=int(manifest["lora_rank"]),
                alpha=float(manifest["lora_alpha"]),
                dropout=0.0,
            )
        else:
            freeze_module(model)
        attach_anchor_heads(model, anchors.num_anchors)
        attach_scene_base_residual_head(
            model,
            max_xy_residual_m=float(manifest.get("max_xy_residual_m", 12.0)),
            max_heading_residual_rad=float(
                manifest.get("max_heading_residual_rad", 0.35)
            ),
            mode_offset_scale=float(manifest.get("mode_offset_scale", 0.045)),
            min_mode_offset_scale=float(
                manifest.get("min_mode_offset_scale", 0.04)
            ),
            max_mode_offset_scale=float(
                manifest.get("max_mode_offset_scale", 0.08)
            ),
        )
        load_trainable_adapter(model, adapter / "adapter_model.safetensors")
    model.eval()

    read_limit = None if args.limit is None else args.skip + args.limit
    samples = list(
        read_scene_file(
            args.scenes,
            image_root=args.image_root,
            num_history_points=model.config.num_history_points,
            limit=read_limit,
        )
    )[args.skip :]
    dataset = SceneInputs(
        samples,
        model.processor,
        with_reasoning=args.mode == InferenceMode.REASONING_PLANNING.value,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda batch: batch[0],
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    scale = model.trajectory_scale(device)
    flow_time = (
        0.8
        if args.control == "anchor2"
        else float(manifest.get("flow_time", manifest.get("t_max", 0.9)))
    )
    noise_std = float(manifest.get("noise_std", 0.02))

    with output.open("w") as handle:
        for position, item in enumerate(loader, start=1):
            inputs = _move_inputs(item["inputs"], device)
            with torch.no_grad():
                if args.mode == InferenceMode.REASONING_PLANNING.value:
                    scene_cache, anchor_position, reasoning = (
                        model._prefill_with_reasoning(
                            inputs,
                            model.config.max_reasoning_tokens,
                        )
                    )
                else:
                    scene_cache, anchor_position = model._prefill(inputs)
                    reasoning = None
                common = {
                    "expert": model.planning_expert,
                    "anchor_bank": anchors,
                    "scene_cache": scene_cache,
                    "position_anchor": anchor_position,
                    "history": normalize_history(inputs["history"].float(), scale),
                    "history_velocity": inputs["history_velocity"].float(),
                    "history_acceleration": inputs["history_acceleration"].float(),
                    "nav_command": inputs["nav_command"],
                    "ego_status": inputs["ego_status"].float(),
                    "num_samples": args.num_samples,
                    "noise_std": noise_std,
                    "seed": args.seed,
                    "flow_time": flow_time,
                    "device": device,
                }
                if args.control == "anchor2":
                    normalized, anchor_ids = sample_anchor2_control(**common)
                    scores = None
                else:
                    normalized, anchor_ids, scores = sample_base_residual(
                        heads=model.anchor_heads,
                        residual_head=model.scene_base_residual_head,
                        scale=scale,
                        initial_speed=item["initial_speed"],
                        **common,
                    )
                trajectories = denormalize_trajectory(
                    normalized,
                    scale,
                ).cpu().numpy()
            record = {
                "token": item["token"],
                "scene_token": item["scene_token"],
                "trajectories": trajectories.round(5).tolist(),
                "init": "anchor",
                "anchor_ids": anchor_ids.tolist(),
                "candidate_scores": None if scores is None else scores.tolist(),
                "selected_index": None if scores is None else int(scores.argmax()),
                "reasoning": reasoning,
                "future_trajectory": None
                if item["future_trajectory"] is None
                else item["future_trajectory"].round(5).tolist(),
                "future_valid": None
                if item["future_valid"] is None
                else item["future_valid"].tolist(),
                "preference_trajectories": [
                    trajectory.round(5).tolist()
                    for trajectory in item["preference_trajectories"]
                ],
                "preference_scores": item["preference_scores"],
                "initial_speed": item["initial_speed"],
            }
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            if position % 25 == 0:
                print(f"{position} scenes", flush=True)
    print(f"wrote {output}")


def _move_inputs(inputs: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }


if __name__ == "__main__":
    main()
