#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fractal/qwen-drive
PY=/home/fractal/miniconda3/envs/qwen-drive/bin/python
OUT="$ROOT/outputs/anchored_flow"
PREP="${PREP:-/run/media/fractal/Expansion/qwen_drive_prepared/train}"
RUN_NAME="${1:-k20_base_residual_full1}"
RUN_DIR="$OUT/$RUN_NAME"
EVAL_DIR="$RUN_DIR/eval_full"

cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$ROOT/src:$ROOT/research/anchored_flow/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

"$PY" -m qwen_drive_anchored.train \
  --model "$ROOT/Qwen-Drive-1.0-4B" \
  --planner "$ROOT/Qwen-Drive-1.0-4B/planner-sft" \
  --anchors "$OUT/anchors_k20.npz" \
  --scenes "$PREP/scenes.jsonl" \
  --image-root "$PREP" \
  --output "$RUN_DIR" \
  --device cuda \
  --epochs 1 \
  --learning-rate 1e-5 \
  --lora-rank 16 \
  --lora-alpha 32 \
  --noise-std 0.02 \
  --noise-modes 6 \
  --flow-time 0.9 \
  --max-xy-residual-m 12 \
  --max-heading-residual-rad 0.35 \
  --mode-offset-scale 0.045 \
  --min-mode-offset-scale 0.04 \
  --max-mode-offset-scale 0.08 \
  --classifier-weight 1 \
  --ranking-weight 5 \
  --ranking-temperature 0.02 \
  --gradient-accumulation 8 \
  --checkpoint-every 500 \
  --num-workers 8

mkdir -p "$EVAL_DIR"
"$PY" -m qwen_drive_anchored.infer \
  --model "$ROOT/Qwen-Drive-1.0-4B" \
  --planner "$ROOT/Qwen-Drive-1.0-4B/planner-sft" \
  --adapter "$RUN_DIR" \
  --anchors "$OUT/anchors_k20.npz" \
  --scenes "$ROOT/outputs/waymo_val/scenes.jsonl" \
  --image-root "$ROOT/outputs/waymo_val/frames_root" \
  --output "$EVAL_DIR/predictions.jsonl" \
  --num-samples 20 \
  --limit 478 \
  --num-workers 4 \
  --skip-base-hash

"$PY" -m qwen_drive_anchored.evaluate_variants \
  --run "base-residual-full=$EVAL_DIR/predictions.jsonl" \
  --anchors "$OUT/anchors_k20.npz" \
  --output-dir "$EVAL_DIR/evaluation"
