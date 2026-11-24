#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fractal/qwen-drive
PY=/home/fractal/miniconda3/envs/qwen-drive/bin/python
OUT="$ROOT/outputs/anchored_flow"
RUN_NAME="${1:-k20_base_residual_pilot200}"
MODE="${2:-direct_planning}"
RUN_DIR="$OUT/$RUN_NAME"
EVAL_DIR="$RUN_DIR/eval_48"

cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$ROOT/src:$ROOT/research/anchored_flow/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

mapfile -t checkpoints < <(
  find "$RUN_DIR/checkpoints" -mindepth 1 -maxdepth 1 -type d -name 'step_*' | sort
)
if [ "${#checkpoints[@]}" -eq 0 ] && [ ! -f "$RUN_DIR/adapter_model.safetensors" ]; then
  echo "no checkpoint history found under $RUN_DIR/checkpoints" >&2
  exit 1
fi

mkdir -p "$EVAL_DIR"
run_args=()
declare -A seen_steps=()
run_checkpoint() {
  local step=$1
  local checkpoint=$2
  if [ -n "${seen_steps[$step]:-}" ]; then
    return
  fi
  seen_steps[$step]=1
  predictions="$EVAL_DIR/val_step_${step}.jsonl"
  checkpoint_anchors="$checkpoint/anchors.npz"
  if [ ! -f "$checkpoint_anchors" ]; then
    checkpoint_anchors="$RUN_DIR/anchors.npz"
  fi
  echo "evaluating checkpoint step $step"
  "$PY" -m qwen_drive_anchored.infer \
    --model "$ROOT/Qwen-Drive-1.0-4B" \
    --planner "$ROOT/Qwen-Drive-1.0-4B/planner-sft" \
    --adapter "$checkpoint" \
    --anchors "$checkpoint_anchors" \
    --scenes "$ROOT/outputs/waymo_val/scenes.jsonl" \
    --image-root "$ROOT/outputs/waymo_val/frames_root" \
    --output "$predictions" \
    --mode "$MODE" \
    --num-samples 20 \
    --limit 48 \
    --num-workers 4 \
    --skip-base-hash
  run_args+=(--run "step-$step=$predictions")
}

for checkpoint in "${checkpoints[@]}"; do
  step=$(basename "$checkpoint" | sed 's/^step_0*//')
  run_checkpoint "$step" "$checkpoint"
done

if [ -f "$RUN_DIR/adapter_model.safetensors" ]; then
  final_step=$(
    "$PY" -c \
      'import json,sys; print(json.load(open(sys.argv[1]))["global_step"])' \
      "$RUN_DIR/manifest.json"
  )
  run_checkpoint "$final_step" "$RUN_DIR"
fi

"$PY" -m qwen_drive_anchored.evaluate_variants \
  "${run_args[@]}" \
  --anchors "$RUN_DIR/anchors.npz" \
  --output-dir "$EVAL_DIR/evaluation"

