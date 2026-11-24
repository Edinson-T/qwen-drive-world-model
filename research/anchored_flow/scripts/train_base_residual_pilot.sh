#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fractal/qwen-drive
PY=/home/fractal/miniconda3/envs/qwen-drive/bin/python
OUT="$ROOT/outputs/anchored_flow"
PREP="${PREP:-/run/media/fractal/Expansion/qwen_drive_prepared/train}"
RUN_NAME="${1:-k20_base_residual_pilot200}"

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
  --output "$OUT/$RUN_NAME" \
  --device cuda \
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
  --max-steps 200 \
  --gradient-accumulation 8 \
  --checkpoint-every 50 \
  --num-workers 8

bash "$ROOT/research/anchored_flow/scripts/evaluate_checkpoint_sweep.sh" "$RUN_NAME"
