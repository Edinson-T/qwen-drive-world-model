#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fractal/qwen-drive
PY=/home/fractal/miniconda3/envs/qwen-drive/bin/python
TORCHRUN=/home/fractal/miniconda3/envs/qwen-drive/bin/torchrun
OUT="$ROOT/outputs/anchored_flow"
PREP="${PREP:-/run/media/fractal/Expansion/qwen_drive_prepared/train}"
RUN_NAME="${1:-k20_sft_skm_ddp1000}"
RUN_DIR="$OUT/$RUN_NAME"
ANCHORS="$OUT/anchors_k20_skm.npz"

cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTHONPATH="$ROOT/src:$ROOT/research/anchored_flow/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

if [ ! -f "$ANCHORS" ]; then
  bash "$ROOT/research/anchored_flow/scripts/build_stratified_anchor_bank.sh"
fi

"$TORCHRUN" \
  --standalone \
  --nproc_per_node=2 \
  -m qwen_drive_anchored.train \
  --model "$ROOT/Qwen-Drive-1.0-4B" \
  --planner "$ROOT/Qwen-Drive-1.0-4B/planner-sft" \
  --anchors "$ANCHORS" \
  --scenes "$PREP/scenes.jsonl" \
  --image-root "$PREP" \
  --output "$RUN_DIR" \
  --device cuda \
  --distributed \
  --epochs 1 \
  --learning-rate 1e-5 \
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
  --classifier-weight 1 \
  --ranking-weight 5 \
  --ranking-temperature 0.02 \
  --sampler stratified \
  --max-steps 1000 \
  --log-every 5 \
  --gradient-accumulation 4 \
  --checkpoint-every 250 \
  --num-workers 4

bash "$ROOT/research/anchored_flow/scripts/evaluate_checkpoint_sweep.sh" \
  "$RUN_NAME" direct_planning
