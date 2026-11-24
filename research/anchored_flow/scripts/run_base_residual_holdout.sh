#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fractal/qwen-drive
PY=/home/fractal/miniconda3/envs/qwen-drive/bin/python
OUT="$ROOT/outputs/anchored_flow"
RUN_NAME="${1:-k20_base_residual_pilot200}"
RUN_DIR="$OUT/$RUN_NAME"
HOLDOUT_DIR="$RUN_DIR/holdout"

cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$ROOT/src:$ROOT/research/anchored_flow/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p "$HOLDOUT_DIR"

"$PY" -m qwen_drive_anchored.infer \
  --model "$ROOT/Qwen-Drive-1.0-4B" \
  --planner "$ROOT/Qwen-Drive-1.0-4B/planner-sft" \
  --adapter "$RUN_DIR" \
  --anchors "$OUT/anchors_k20.npz" \
  --scenes "$ROOT/outputs/waymo_val/scenes.jsonl" \
  --image-root "$ROOT/outputs/waymo_val/frames_root" \
  --output "$HOLDOUT_DIR/predictions.jsonl" \
  --num-samples 20 \
  --skip 48 \
  --limit 430 \
  --num-workers 4 \
  --skip-base-hash

"$PY" -m qwen_drive_anchored.evaluate_variants \
  --run "base-residual-holdout=$HOLDOUT_DIR/predictions.jsonl" \
  --anchors "$OUT/anchors_k20.npz" \
  --output-dir "$HOLDOUT_DIR/evaluation"
