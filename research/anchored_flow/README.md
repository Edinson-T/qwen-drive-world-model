# Scene-base anchor-residual planner

Detailed Chinese method and loss documentation:

- [METHOD_CN.md](METHOD_CN.md)

Build the stratified-medoid anchor bank:

```bash
bash research/anchored_flow/scripts/build_stratified_anchor_bank.sh
```

Run the 1,000-step planner-SFT SKM pilot:

```bash
CUDA_VISIBLE_DEVICES=0 \
bash research/anchored_flow/scripts/train_sft_skm_pilot.sh
```

Two-GPU DDP variants for the matched KM/SKM bank ablation:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
bash research/anchored_flow/scripts/train_sft_km_stratified_pilot_ddp.sh

CUDA_VISIBLE_DEVICES=0,1 \
bash research/anchored_flow/scripts/train_sft_skm_pilot_ddp.sh
```

Run a zero-training bank/candidate-count probe from an existing prediction
file:

```bash
PYTHONPATH="$PWD/src:$PWD/research/anchored_flow/src" \
/home/fractal/miniconda3/envs/qwen-drive/bin/python \
  -m qwen_drive_anchored.probe_anchor_count \
  --predictions EXISTING_PREDICTIONS.jsonl \
  --scenes outputs/waymo_val/scenes.jsonl \
  --output-dir outputs/anchored_flow/anchor_count_probe \
  --bank K20N20=outputs/anchored_flow/anchors_k20_skm.npz:20 \
  --bank K64N48=outputs/anchored_flow/anchors_k64_skm.npz:48
```

Run the 1,000-step planner-rl stratified pilot with frozen planner weights and
trainable anchor/residual heads:

```bash
CUDA_VISIBLE_DEVICES=0 \
bash research/anchored_flow/scripts/train_rl_stratified_pilot.sh
```

This package contains one active experiment:

```text
base = Qwen Planner(scene, neutral trajectory)
candidate_k = base + lambda * centered_anchor_k + bounded_residual_k
```

The VLM and base planner stay frozen. LoRA, anchor conditioning, residual
parameters, and the confidence head are stored separately in the adapter.

## Environment

```bash
cd /home/fractal/qwen-drive
export PYTHONPATH="$PWD/src:$PWD/research/anchored_flow/src"
```

## Build anchors

```bash
python -m qwen_drive_anchored.anchors \
  --scenes /run/media/fractal/Expansion/qwen_drive_prepared/train/scenes.jsonl \
  --output outputs/anchored_flow/anchors_k20.npz \
  --num-anchors 20
```

Create data diagnostics with:

```bash
python -m qwen_drive_anchored.visualize_anchors \
  --anchors outputs/anchored_flow/anchors_k20.npz \
  --train-scenes /run/media/fractal/Expansion/qwen_drive_prepared/train/scenes.jsonl \
  --val-scenes outputs/waymo_val/scenes.jsonl \
  --output-dir outputs/visualizations/anchored_flow/anchor_banks/km
```

## Train

```bash
CUDA_VISIBLE_DEVICES=0 \
bash research/anchored_flow/scripts/train_base_residual_pilot.sh
```

The default pilot trains 200 optimizer steps, saves checkpoints every 50 steps,
and evaluates them on the fixed 48-scene development set.

The pilot is not the final generator. After its metrics are frozen, run one
complete epoch before S2:

```bash
CUDA_VISIBLE_DEVICES=0 \
bash research/anchored_flow/scripts/train_base_residual_full.sh
```

This trains on all available training windows, saves checkpoints every 500
optimizer steps, and evaluates the final adapter on all 478 validation scenes.

The loss contains:

- L1 regression for the GT-nearest anchor candidate;
- focal classification over all anchors;
- soft ranking over generated candidate errors.

Mode-offset scale is bounded in `[0.04, 0.08]`, so the fixed anchor geometry
cannot silently collapse to a single scene mode.

## Evaluate

Re-run the 48-scene checkpoint sweep:

```bash
CUDA_VISIBLE_DEVICES=0 \
bash research/anchored_flow/scripts/evaluate_checkpoint_sweep.sh \
  k20_base_residual_pilot200
```

Run the remaining validation scenes after generator selection is frozen:

```bash
CUDA_VISIBLE_DEVICES=0 \
bash research/anchored_flow/scripts/run_base_residual_holdout.sh \
  k20_base_residual_pilot200
```

The holdout uses scenes 49 onward from the 478-scene validation file.

Generate the same-split two-step anchor control before comparing first/mean ADE:

```bash
CUDA_VISIBLE_DEVICES=0 \
bash research/anchored_flow/scripts/run_anchor2_holdout_baseline.sh
```

## Tests

```bash
PYTHONPATH="$PWD/src:$PWD/research/anchored_flow/src" \
/home/fractal/miniconda3/envs/qwen-drive/bin/python \
  -m unittest discover -s research/anchored_flow/tests -v
```

## Paper figures

Generate the current S1 qualitative figure and holdout metric bars:

```bash
bash research/anchored_flow/scripts/make_s1_figures.sh
```

For the final 2x2 figure after WM/reward integration, the prediction record must
contain `candidate_scores` and `selected_index`. Run:

```bash
PYTHONPATH="$PWD/src:$PWD/research/anchored_flow/src" \
/home/fractal/miniconda3/envs/qwen-drive/bin/python \
  -m qwen_drive_anchored.paper_figures qualitative \
  --analysis FINAL_PREDICTIONS.jsonl \
  --baseline outputs/waymo_val/sft_1sample.jsonl \
  --scenes outputs/waymo_val/scenes.jsonl \
  --image-root outputs/waymo_val/frames_root \
  --output-dir outputs/visualizations/anchored_flow/paper_figures \
  --layout full \
  --selection auto \
  --selection-label "WoTE reward"
```

The qualitative command writes PNG, PDF, and a JSON manifest containing the
chosen token, indices, ADE values, and source files.
