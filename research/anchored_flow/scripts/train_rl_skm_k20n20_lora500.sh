#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fractal/qwen-drive
PY=/home/fractal/miniconda3/envs/qwen-drive/bin/python
OUT="$ROOT/outputs/anchored_flow"
PREP="${PREP:-/run/media/fractal/Expansion/qwen_drive_prepared/train}"
RUN_NAME="${1:-k20_rl_skm_n20_lora500_lr2e6}"
RUN_DIR="$OUT/$RUN_NAME"
ANCHORS="$OUT/anchors_k20_skm.npz"

cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$ROOT/src:$ROOT/research/anchored_flow/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

"$PY" -m qwen_drive_anchored.train \
  --model "$ROOT/Qwen-Drive-1.0-4B" \
  --planner "$ROOT/Qwen-Drive-1.0-4B/planner-rl" \
  --anchors "$ANCHORS" \
  --scenes "$PREP/scenes.jsonl" \
  --image-root "$PREP" \
  --output "$RUN_DIR" \
  --device cuda \
  --epochs 1 \
  --learning-rate 2e-6 \
  --lora-rank 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --noise-std 0.02 \
  --noise-modes 6 \
  --flow-time 0.9 \
  --max-xy-residual-m 12 \
  --max-heading-residual-rad 0.35 \
  --mode-offset-scale 0.045 \
  --min-mode-offset-scale 0.04 \
  --max-mode-offset-scale 0.08 \
  --classifier-weight 0.1 \
  --ranking-weight 0 \
  --ranking-temperature 0.02 \
  --sampler stratified \
  --candidate-count 20 \
  --reasoning-planning \
  --max-steps 500 \
  --log-every 5 \
  --gradient-accumulation 8 \
  --checkpoint-every 100 \
  --num-workers 16

bash "$ROOT/research/anchored_flow/scripts/evaluate_checkpoint_sweep.sh" \
  "$RUN_NAME" reasoning_planning
