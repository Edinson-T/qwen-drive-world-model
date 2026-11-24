#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fractal/qwen-drive
PY=/home/fractal/miniconda3/envs/qwen-drive/bin/python
OUT="$ROOT/outputs/anchored_flow"
RESULT_DIR="$OUT/anchor2_holdout"

cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$ROOT/src:$ROOT/research/anchored_flow/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p "$RESULT_DIR"

"$PY" -m qwen_drive_anchored.infer \
  --model "$ROOT/Qwen-Drive-1.0-4B" \
  --planner "$ROOT/Qwen-Drive-1.0-4B/planner-sft" \
  --control anchor2 \
  --anchors "$OUT/anchors_k20.npz" \
  --scenes "$ROOT/outputs/waymo_val/scenes.jsonl" \
  --image-root "$ROOT/outputs/waymo_val/frames_root" \
  --output "$RESULT_DIR/predictions.jsonl" \
  --num-samples 20 \
  --skip 48 \
  --limit 430 \
  --num-workers 4

"$PY" -m qwen_drive_anchored.evaluate_variants \
  --run "anchor2-holdout=$RESULT_DIR/predictions.jsonl" \
  --anchors "$OUT/anchors_k20.npz" \
  --output-dir "$RESULT_DIR/evaluation"
