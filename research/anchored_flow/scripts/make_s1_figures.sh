#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fractal/qwen-drive
PY=/home/fractal/miniconda3/envs/qwen-drive/bin/python
OUT="$ROOT/outputs/visualizations/anchored_flow/paper_figures"
ANALYSIS="${1:-$ROOT/outputs/anchored_flow/k20_base_residual_pilot200/holdout/predictions.jsonl}"
BASELINE="${2:-$ROOT/outputs/waymo_val/sft_1sample.jsonl}"
SUMMARY="${3:-$ROOT/outputs/anchored_flow/holdout_comparison/summary.json}"

cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/research/anchored_flow/src"

"$PY" -m qwen_drive_anchored.paper_figures qualitative \
  --analysis "$ANALYSIS" \
  --baseline "$BASELINE" \
  --scenes "$ROOT/outputs/waymo_val/scenes.jsonl" \
  --image-root "$ROOT/outputs/waymo_val/frames_root" \
  --output-dir "$OUT" \
  --layout s1 \
  --selection none \
  --max-speed 30 \
  --max-oracle-ade 4

"$PY" -m qwen_drive_anchored.paper_figures metrics \
  --run "Holdout=$SUMMARY" \
  --output "$OUT/metrics.png"
