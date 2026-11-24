# WOD-E2E scene builder

Rebuilds the Waymo Open Dataset end-to-end (WOD-E2E) **validation** benchmark
scene file from the released raw tfrecords, so the planning metrics in
`docs/evaluation.md` can actually be reproduced.

This directory is deliberately separate from `src/qwen_drive/`: it is a data
preparation tool for the released benchmark, not part of the model library, and
it is not imported by any runtime code.

## Why this exists

The repository ships the benchmark *scene files* as artifacts of the training
pipeline, not as sources, and there is no builder under `scripts/`. Without that
file you can score predictions but you cannot generate them, because
`run_planning.py` needs the JSONL scene file plus its extracted camera frames.

Only the validation split is reproducible:

* the public **validation** shards carry both the ground-truth future and the
  human rater trajectories/scores, so ADE, FDE and **RFS** can be computed
  locally;
* the public **test** shards carry neither. `test_sequence_frames_for_submission.json`
  lists the 1505 test frames to predict, and the metrics for them only exist on
  the official leaderboard. The test numbers in the report cannot be recomputed
  locally.

## What it builds

For every validation scene that carries a rater label (479 scenes in the
`v1.0.0` release, matching the count in the technical report):

1. the four camera timestamps the model reads — `t-1.5s .. t-0s` at 2 Hz — which
   are frame indices `n-15, n-10, n-5, n` of the same scene, because the released
   camera sequence runs at 10 Hz;
2. the three front cameras (`FRONT`, `FRONT_LEFT`, `FRONT_RIGHT`) for each of
   those timestamps;
3. the 10 Hz ego history/future fields, resampled from the raw 4 Hz states with
   the repository's own `scripts/interpolate_waymo_trajectory.py`;
4. the rater trajectories and scores, and the verbatim instruction text.

## Requirements

Run it with the released environment (the `qwen-drive` conda env), which needs
`numpy`, `scipy` and `Pillow`:

```bash
/home/fractal/miniconda3/envs/qwen-drive/bin/python tools/waymo_e2e/build_scenes.py ...
```

No TensorFlow and no `waymo-open-dataset` package are required: `e2e_proto.py`
decodes the protobuf wire format directly.

## Usage

```bash
python tools/waymo_e2e/build_scenes.py \
    --data-root /home/fractal/datasets/waymo_e2e/waymo_open_dataset_end_to_end_camera_v_1_0_0 \
    --output outputs/waymo_val/scenes.jsonl \
    --image-root outputs/waymo_val/frames_root
```

The first run scans all `val_*.tfrecord-*` shards (~106k records, a few minutes)
and caches an index next to the output as
`outputs/waymo_val/scenes.jsonl.index.json.gz`. Later runs reuse it; pass
`--rebuild-index` to rescan. Useful flags:

| Flag | Effect |
| --- | --- |
| `--tokens HASH [HASH ...]` | build only the given scene tokens (hashes) |
| `--limit N` | stop after N scenes |
| `--split val` | shard prefix to read (only `val` is scorable) |
| `--missing-history skip\|nearest` | what to do when one history frame is absent (1 scene lacks the exact `n-15` frame) |

Output layout follows `docs/data.md`:

```
outputs/waymo_val/scenes.jsonl
outputs/waymo_val/frames_root/val/<scene_token>/<CAMERA>/<frame_index>.jpg
```

## Scoring the result

```bash
# trajectories -> predictions file
python scripts/run_planning.py \
    --model Qwen-Drive-1.0-4B --planner Qwen-Drive-1.0-4B/planner-sft \
    --scenes outputs/waymo_val/scenes.jsonl --image-root outputs/waymo_val/frames_root \
    --output outputs/waymo_val/predictions.jsonl \
    --mode reasoning_planning --num-samples 6

# ADE/FDE/HeadingMAE, offline
python scripts/eval_waymo.py --predictions outputs/waymo_val/predictions.jsonl

# add RFS (needs the official rater-feedback utils, which are numpy-only)
export WAYMO_OPEN_DATASET_SRC=/path/to/waymo-open-dataset/src
python scripts/eval_waymo.py --predictions outputs/waymo_val/predictions.jsonl --rater-feedback
```

`run_planning.py` copies the ground truth and the preference trajectories into
the predictions file, which is why `eval_waymo.py` needs nothing but that file.
With several samples per scene it keeps the candidate closest to the
highest-rated rater trajectory, so the reported number is an oracle bound.

## Correctness check

The bundled `data/demo/planning_scenes.jsonl` is a slice of this validation
split, so the builder can be validated against it field by field:

```bash
python tools/waymo_e2e/verify_demo.py \
    --data-root /home/fractal/datasets/waymo_e2e/waymo_open_dataset_end_to_end_camera_v_1_0_0 \
    --index outputs/waymo_val/scenes.jsonl.index.json.gz
```

On the reference release all four demo scenes reproduce exactly: every
trajectory, velocity and acceleration field matches with `max|Δ| = 0`, the rater
scores and the instruction text are identical, and the extracted JPEGs are
pixel-identical to the demo frames.

## Files

| File | Role |
| --- | --- |
| `e2e_proto.py` | dependency-free `E2EDFrame` tfrecord/protobuf reader |
| `build_scenes.py` | index the shards, gather history frames, write the scene file |
| `verify_demo.py` | rebuild the bundled demo scenes and diff them against the shipped file |
