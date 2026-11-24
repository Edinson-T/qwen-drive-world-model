# QD BEV World Model v0

> 2026-11-19 新实例状态分支：[INSTANCE_MOTION_V8.md](INSTANCE_MOTION_V8.md)。新取128场景独立验证：IoU=0.8902（两种子均值），同集 GT 速度 CV=0.8492。使用精确历史框、16帧关联和完整源对象范围；不是原 V4 latent 输入不变时的结果，尚未迁移 QD/RFS。

> 当前版本和中文语义 BEV：[LATENT_MOTION_V4.md](LATENT_MOTION_V4.md)。增加宽空间运动分支和训练期速度辅助监督，
> 新验证子集 WM IoU=0.5352，仍落后于 GT 速度 CV=0.8537，尚未迁移 QD/RFS。

> 最新改进及模型入口：[LATENT_MOTION_V3.md](LATENT_MOTION_V3.md)。已完成细节编码、几何残差和数据扩充，
> 在新官方验证子集上占用 IoU=0.4677，ego-warp=0.3838；仍未接入 QD/RFS。

> 新的 Motion latent WM 独立训练主线与首轮结果见
> [LATENT_MOTION_V2.md](LATENT_MOTION_V2.md)。已完成四帧历史、统一坐标、
> 冻结 latent teacher、1s 预测及独立官方 validation 子集评估；首轮尚未超过 ego-warp。
> 下文保留旧 v0 原型的历史用法，不作为新实验的运行入口。

This package is a feasibility prototype for:

```text
current compact BEV latent + full 5 s candidate trajectory
    -> deterministic one-step world model
    -> future compact BEV latent
```

The first training target is `t + 0.5 s`, but every dataset record retains the
full 5 s trajectory and caches target latents at `t + 0.5 s`, `t + 1 s`,
`t + 2 s`, `t + 3 s` and `t + 5 s`. This keeps all horizons used for trajectory
diagnosis and the 3 s / 5 s Waymo RFS horizons available without changing the
first world-model objective.

`--target-horizon` can be set to `0.5`, `1`, `2`, `3`, or `5` after the
corresponding latents have been exported. The action encoder always receives the
complete 5 s candidate trajectory.

## Coordinate protocol

The perception teacher returns each BEV in that frame's ego coordinates. Version
0 therefore uses **future ego-frame prediction**:

- `z_t` is in the t-frame ego coordinates;
- `z_target` is in the t+delta-frame ego coordinates;
- the full 5 s trajectory and its relative future pose are explicit action
  conditions for the world model.

The predicted latent must not be interpreted as a collision map expressed in the
current ego frame until a future-to-current metric warp is implemented.

## Data preparation

Set the paths:

```bash
cd /home/fractal/qwen-drive
export PYTHONPATH="$PWD/src:$PWD/research/world_model/src:$PWD/research/anchored_flow/src:$PWD/tools/waymo_e2e"
```

Pack the 12 existing comparison scenes into real 8-camera current/future frame
sets. The raw WOD-E2E validation data is available on this machine:

```bash
python -m qwen_drive_world_model.pack_waymo \
  --raw-root /mnt/hdd/waymo_e2e/waymo_open_dataset_end_to_end_camera_v_1_0_0 \
  --output outputs/world_model/pilot/frames \
  --extra-per-scene 1 \
  --overwrite
```

With the current comparison manifests this creates 36 real pairs from 12 scenes
(27 train / 9 validation).

For a larger scene-level pilot, use the labeled benchmark scene file directly:

```bash
python -m qwen_drive_world_model.pack_waymo \
  --raw-root /mnt/hdd/waymo_e2e/waymo_open_dataset_end_to_end_camera_v_1_0_0 \
  --scenes outputs/waymo_val/scenes.jsonl \
  --output outputs/world_model/train200/frames \
  --all-scenes \
  --limit-pairs 200 \
  --val-scenes 40 \
  --overwrite
```

This produces 160 train and 40 validation scenes, with no scene crossing the
split.

### Official training split

Build the reusable training scene/frame index once:

```bash
python -m qwen_drive_world_model.build_index \
  --data-root /mnt/hdd/waymo_e2e/waymo_open_dataset_end_to_end_camera_v_1_0_0 \
  --split training \
  --workers 8 \
  --output outputs/waymo_train/scenes.training.index.json.gz
```

Then pack official training scenes without a `scenes.jsonl`:

```bash
python -m qwen_drive_world_model.pack_waymo \
  --raw-root /mnt/hdd/waymo_e2e/waymo_open_dataset_end_to_end_camera_v_1_0_0 \
  --index outputs/waymo_train/scenes.training.index.json.gz \
  --output outputs/world_model/train1000/frames \
  --all-index-scenes \
  --limit-pairs 1000 \
  --val-scenes 0 \
  --overwrite
```

Keep validation separate from training. Pass
`--val-data-dir outputs/world_model/train200/dataset` to `train` when using the
official training split, so the 40 validation scenes remain held out.

This writes:

```text
outputs/world_model/pilot/frames/
  pairs.jsonl
  manifest.json
  frames/<pair>/{current,future_0p5s,future_1s,future_2s,future_3s,future_5s}/...
```

The default frame offsets are `+5`, `+10`, `+20`, `+30`, and `+50` camera frames
at 10 Hz.
Pairs are split by scene, not by adjacent frame.

Important: the WOD-E2E pack contains camera images and trajectories, not LiDAR
labels. `pack_waymo` writes an empty `gt.npz`, and the exported `map_*`,
`occ_*` and `boxes_*` fields are outputs from the frozen QD Perception teacher.
They are camera-only pseudo-targets, not LiDAR ground truth. A true GT BEV
requires building pairs directly from Waymo Perception segments with LiDAR,
3D labels, map features and per-frame pose.

## Export frozen perception latents

The perception head is frozen. Its optional latent output is compressed before
caching; the full `200 x 200 x 256` tensor is never persisted.

```bash
python -m qwen_drive_world_model.export_latents \
  --pairs outputs/world_model/pilot/frames/pairs.jsonl \
  --output outputs/world_model/pilot/dataset \
  --vlm Qwen-Drive-1.0-4B \
  --perception Qwen-Drive-1.0-4B/perception \
  --latent-height 16 \
  --latent-width 32 \
  --latent-channels 128
```

Use `--device cuda` with a working NVIDIA driver. The released perception head
calls the CUDA-only UVTR `voxel_pool_depth` operator, so this export cannot run
on a CPU-only host. The world-model training and unit tests remain CPU-runnable.

For two 32 GB GPUs, run one shard per GPU and then aggregate:

```bash
export CUDA_HOME="$CONDA_PREFIX"
export CC="$CUDA_HOME/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CUDA_HOME/bin/x86_64-conda-linux-gnu-g++"
export NVCC_PREPEND_FLAGS="-ccbin $CXX"
export CPATH="$CUDA_HOME/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=12.0

CUDA_VISIBLE_DEVICES=0 \
python -m qwen_drive_world_model.export_latents \
  --pairs outputs/world_model/pilot/frames/pairs.jsonl \
  --output outputs/world_model/pilot/dataset \
  --device cuda --num-shards 2 --shard-index 0 &

CUDA_VISIBLE_DEVICES=1 \
python -m qwen_drive_world_model.export_latents \
  --pairs outputs/world_model/pilot/frames/pairs.jsonl \
  --output outputs/world_model/pilot/dataset \
  --device cuda --num-shards 2 --shard-index 1 &

wait

python -m qwen_drive_world_model.export_latents \
  --pairs outputs/world_model/pilot/frames/pairs.jsonl \
  --output outputs/world_model/pilot/dataset \
  --aggregate-only
```

Each pair caches:

- `z_t`, `z_0p5s`, `z_1s`, `z_2s`, `z_3s`, `z_5s` from `bev_embed`;
- `f_t`, `f_0p5s`, `f_1s`, `f_2s`, `f_3s`, `f_5s` from `bev_ego_feat_2d`;
- decoded map and occupancy outputs for optional diagnostics;
- the complete 5 s trajectory, per-horizon trajectory windows, ego status and
  relative poses.

## Mini-batch training

First overfit a few real pairs:

```bash
python -m qwen_drive_world_model.train \
  --data-dir outputs/world_model/pilot/dataset \
  --output-dir outputs/world_model/pilot/train_overfit \
  --target-horizon 0.5 \
  --batch-size 2 \
  --steps 100 \
  --overfit
```

Then run the normal pilot:

```bash
python -m qwen_drive_world_model.train \
  --data-dir outputs/world_model/pilot/dataset \
  --val-data-dir outputs/world_model/pilot/dataset \
  --output-dir outputs/world_model/pilot/train_pilot \
  --target-horizon 0.5 \
  --batch-size 4 \
  --steps 300 \
  --early-stop-patience 5 \
  --early-stop-min-delta 1e-4
```

Outputs include:

```text
best.pt
latest.pt
metrics.json
diagnostics.json
training_curves.png
visualizations/
  quad_*.png
  copy_*.png
  actions_*.png
```

The training loop logs train/validation latent SmoothL1, cosine similarity,
copy-current metrics, action sensitivity, elapsed time, and CUDA peak memory.

Early stopping uses validation loss. The default patience is five validation
checks, so a run stops after five checks without a decrease of at least
`1e-4`. Use `--early-stop-patience 0` to disable it. `best.pt` always preserves
the best validation checkpoint, and the final diagnostic images are regenerated
from that best checkpoint.

## Horizon comparison

After training separate runs for each horizon, combine them with:

```bash
python -m qwen_drive_world_model.summarize_sweep \
  --run "h0.5=outputs/world_model/train200/horizon_sweep/h0p5" \
  --run "h1=outputs/world_model/train200/horizon_sweep/h1" \
  --run "h2=outputs/world_model/train200/horizon_sweep/h2" \
  --run "h3=outputs/world_model/train200/horizon_sweep/h3" \
  --run "h5=outputs/world_model/train200/horizon_sweep/h5" \
  --output outputs/world_model/train200/horizon_sweep/summary
```

This writes `summary.json` and `horizon_comparison.png`.

## BEV map decoder and comparison figures

The world model predicts a compact latent, not semantic class labels. A small
latent-to-map decoder is trained against the teacher's 6-class BEV maps:

```bash
python -m qwen_drive_world_model.train_map_decoder \
  --data-dir outputs/world_model/train781/dataset \
  --val-data-dir outputs/world_model/train200/dataset \
  --wm-checkpoint outputs/world_model/train781/formal_h1/best.pt \
  --output-dir outputs/world_model/train781/map_decoder_h1 \
  --target-horizon 1
```

Generate five validation comparisons with:

```bash
python -m qwen_drive_world_model.generate_bev_comparisons \
  --val-data-dir outputs/world_model/train200/dataset \
  --pairs outputs/world_model/train200/frames/pairs.jsonl \
  --annotations-cache outputs/world_model/train781/bev_annotations/cache/pair_cache \
  --h1-run outputs/world_model/train781/formal_h1 \
  --h1-decoder outputs/world_model/train781/map_decoder_h1 \
  --h3-run outputs/world_model/train781/formal_h3 \
  --h3-decoder outputs/world_model/train781/map_decoder_h3 \
  --output outputs/world_model/train781/bev_comparisons \
  --num-figures 5
```

## Waymo Perception GT training

For a physical-world target rather than camera-teacher distillation, build
windows directly from Waymo Open Dataset Perception segments:

```bash
python -m qwen_drive_world_model.build_perception_gt_dataset \
  --data-dir /home/fractal/datasets/waymo_perception \
  --output outputs/world_model/perception_gt_v1 \
  --max-frames 200 \
  --anchors-per-segment 40 \
  --val-scenes 4
```

This packs the official 3D LiDAR labels, map features and ego poses into
future-ego-frame windows. Each sample contains:

- 200x400 map-feature BEV and LiDAR-box occupancy targets at every horizon;
- a compact 9-channel GT BEV (`8` map classes + occupancy);
- up to 64 future 3D boxes with vehicle/pedestrian/cyclist/sign classes;
- the complete 5 s ego trajectory derived from frame poses.

Train the multi-task GT world model with:

```bash
python -m qwen_drive_world_model.train_perception_gt \
  --data-dir outputs/world_model/perception_gt_v1 \
  --output-dir outputs/world_model/perception_gt_v1/train_h1 \
  --target-horizon 1 \
  --batch-size 8 \
  --steps 1000
```

The model predicts future semantic map, obstacle occupancy, and object boxes.
The box head uses DETR-style object queries and Hungarian set matching rather
than fitting boxes from the occupancy raster after the fact.

Generate five validation comparisons with three cameras and all three
prediction horizons:

```bash
python -m qwen_drive_world_model.generate_perception_gt_comparisons \
  --data-dir outputs/world_model/perception_gt_v1 \
  --raw-data-dir /home/fractal/datasets/waymo_perception \
  --h1-run outputs/world_model/perception_gt_v1/train_h1_v2 \
  --h3-run outputs/world_model/perception_gt_v1/train_h3 \
  --h5-run outputs/world_model/perception_gt_v1/train_h5 \
  --output outputs/world_model/perception_gt_v1/bev_comparisons
```

Each figure contains `FRONT_LEFT`, `FRONT`, `FRONT_RIGHT`, followed by
prediction/GT pairs at 1 s, 3 s and 5 s. The BEV panels overlay semantic map,
dynamic occupancy, predicted or GT 3D boxes, and the logged ego trajectory.

## Waymo Motion GT World Model

Build action-conditioned GT windows from Waymo Motion v1.3.1:

```bash
python -m qwen_drive_world_model.build_motion_gt_dataset \
  --data-dir /mnt/hdd/waymo_motion_v1_3_1/training \
  --output outputs/world_model/motion_gt_v1 \
  --max-scenarios 1000 \
  --anchors-per-scenario 8 \
  --val-every 10
```

The Motion WM input is `[13,50,100]`:

- 8 semantic map channels;
- 1 2D box-footprint occupancy channel;
- 4 box-derived height-slice channels.

It predicts the existing 9-channel future target (map + 2D occupancy) and 3D
object boxes. The QD map grid is `x[-30,30]m`, `y[-15,15]m`, `0.15m/cell`;
high-resolution rasters are `200x400` and model rasters are `50x100`.

```bash
python -m qwen_drive_world_model.train_motion_gt \
  --data-dir outputs/world_model/motion_gt_v1 \
  --output-dir outputs/world_model/motion_gt_v1/train_h1 \
  --target-horizon 1 \
  --init-checkpoint outputs/world_model/perception_gt_v1/train_h1_v2/best.pt
```

Warm starting preserves the old 9-channel BEV path, adds a zero-initialized
height projection, and resets the object classification output because Motion
class 3 is `other` rather than Perception class 3 `sign`.

The first Motion pilot uses 1,000 scenarios and 8 anchors/scenario, producing
7,200 train / 800 validation windows. The official Motion validation split is
kept separately as held-out test data. Metrics are stored in
`outputs/world_model/motion_gt_v1/VALIDATION_METRICS.md`.

Train one shared model for all five horizons:

```bash
python -m qwen_drive_world_model.train_shared_motion_gt \
  --data-dir outputs/world_model/motion_gt_v1 \
  --test-data-dir outputs/world_model/motion_gt_test_v1 \
  --output-dir outputs/world_model/motion_gt_v1/train_shared \
  --init-checkpoint outputs/world_model/motion_gt_v1/train_h1/best.pt
```

The shared model encodes the current BEV and trajectory once, conditions a
shared decoder with a horizon embedding and relative ego pose, and predicts all
five horizons in parallel. Comparison against the independent checkpoints is in
`outputs/world_model/motion_gt_v1/SHARED_VS_INDEPENDENT.md`.

For the occupancy-optimized run, use `--occupancy-horizon-weights 2 2 2 1 1`.
Threshold calibration, PR curves and ego-corridor recall are documented in
`outputs/world_model/motion_gt_v1/OCCUPANCY_OPTIMIZATION.md`.

Box-head calibration uses increased no-object weighting, focal classification,
and class-aware NMS. Results are documented in
`outputs/world_model/motion_gt_v1/BOX_OPTIMIZATION.md`.

The first fixed-candidate selector prototype uses frozen RL Planner + direct
anchor K20/N20, official per-candidate RFS labels, and scene-disjoint
train/validation/test splits. See `outputs/selector_prototype/README.md`.

The selector was also evaluated on all 478 validation candidate records with
current QD BEV and five-horizon WM features. After adding the frozen RL base as
candidate 0, RL base scores `8.4432`; the best baseline-guarded WM selector
scores `8.4514`, with a confidence interval that still crosses zero. The current
evidence therefore does not show a deployment gain over RL base. See
`outputs/selector_fullval/README.md`.

A full candidate-level Reward Model was then trained with trajectory-only,
current-QD-BEV and current-plus-future-QD-BEV inputs. Current BEV scores
`8.4491`; adding future BEV drops to `8.4007`. The paired future-minus-current
effect is `-0.0483 RFS` with a 95% CI of `[-0.0959, -0.0062]`. See
`outputs/selector_fullval/reward_model_qd_v1/README.md`.

## Tests

```bash
PYTHONPATH="$PWD/src:$PWD/research/world_model/src" \
  /home/fractal/miniconda3/envs/qwen-drive/bin/python \
  -m unittest discover -s research/world_model/tests -v
```

## Scope

Not included in v0:

- planning-expert or VLM training;
- reward-model training;
- WoTE weight transfer;
- geometry expert;
- multi-step recurrent rollout;
- full-dataset training;
- metric warping from future ego coordinates into the current ego frame.
