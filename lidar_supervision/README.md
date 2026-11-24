# LiDAR supervision sandbox

Standalone scripts that validate the LiDAR → camera → depth-target pipeline on
the **bundled demo frames**. They import nothing from `src/qwen_drive` or
`src/qwen_drive_perception`, never load model weights, and write only under
`outputs/`.

## Why

Before training anything on Waymo Perception, confirm three things:

1. the projection is correct (points land on the right objects),
2. the target generation is correct (coverage and depth ranges are sane),
3. a small head can overfit a batch (loss goes to ~0).

Only then is it worth downloading Perception segments.

## Commands

```bash
conda activate qwen-drive
cd /home/fractal/qwen-drive
export PYTHONPATH=/home/fractal/qwen-drive
export CUDA_VISIBLE_DEVICES=0

# 1. Render projections (numpy + pillow only, no GPU needed)
python -m lidar_supervision.verify_targets \
    --data data/demo/perception --out outputs/lidar_debug

# 2. Random-tensor health check
python -m lidar_supervision.train_geometry --smoke --steps 5

# 3. Overfit the demo frames
python -m lidar_supervision.train_geometry \
    --data data/demo/perception --steps 300 --overfit 2 \
    --image-size 256 144 --batch-size 2 --out outputs/lidar_smoke
```

## What to look for

- `verify_targets`: overlays in `outputs/lidar_debug/`; points must hug the road,
  cars and buildings. Per-camera coverage should be roughly 0.1–5%.
- `--smoke`: finite loss, positive gradient norm.
- `--overfit 2`: with a fixed pair of samples the loss should fall to well
  below 0.05 within a few hundred steps. If it plateaus high, the target or the
  mask is wrong.

Each run writes two files:

- `head.pt` — weights plus a `config` block and the final metrics;
- `train_config.json` — the same configuration in readable form (command line,
  image/target size, steps, learning rate, data path, final loss).

## What this does **not** prove

The demo frames are six nuScenes/nuPlan samples. They validate the pipeline and
that the head can learn; they say nothing about generalisation or about whether
geometry supervision helps planning. Those need Waymo Perception segments and a
separate planning ablation.

## Real run on Waymo Perception

```bash
# 1. fetch ~20 segments (~20 GB)
mkdir -p /home/fractal/datasets/waymo_perception
cd /home/fractal/datasets/waymo_perception
gsutil -m cp $(gsutil ls gs://waymo_open_dataset_v_1_4_2/individual_files/training/ | head -20) .

# 2. decode LiDAR range images into compact depth targets (~50 MB/segment)
cd /home/fractal/qwen-drive
python -m lidar_supervision.prepare_waymo \
    --data /home/fractal/datasets/waymo_perception \
    --out outputs/waymo_perception_targets \
    --image-size 256 144 --cameras 1 2 3 --max-frames 200

# 3. train with a segment-level split (last 5 segments held out)
python -m lidar_supervision.train_waymo \
    --data outputs/waymo_perception_targets \
    --out outputs/waymo_geom --val-segments 5 \
    --steps 3000 --batch-size 16 --lr 3e-4 --amp
```

`prepare_waymo` reads the **official per-cell camera projection** embedded in
each range image, so no extrinsic convention is assumed. `train_waymo` reports
per-segment AbsRel and δ<1.25 next to a constant-depth baseline; the held-out
segments are the only meaningful number.
