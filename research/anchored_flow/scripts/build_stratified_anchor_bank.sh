#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fractal/qwen-drive
PY=/home/fractal/miniconda3/envs/qwen-drive/bin/python
PREP="${PREP:-/run/media/fractal/Expansion/qwen_drive_prepared/train}"
OUT="$ROOT/outputs/anchored_flow/anchors_k20_skm.npz"

cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/research/anchored_flow/src"

"$PY" -m qwen_drive_anchored.anchors \
  --scenes "$PREP/scenes.jsonl" \
  --output "$OUT" \
  --num-anchors 20 \
  --method stratified-medoids
