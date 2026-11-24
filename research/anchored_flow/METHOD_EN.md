# Anchor-Conditioned Trajectory Generation

> Version: 2026-11-09  
> This update adds a common `N=20` evaluation for QD-SFT, QD-RL, KM, SKM,
> and RL+anchor on all 478 WOD-E2E validation scenes. It also explains RFS,
> ADE, DiffusionDriveV2 Div, Mean candidate RFS, Oracle RFS, and RFS gap.
> A later SFT direct+anchor test uses the same anchor bank and offset scale as
> RL direct+anchor.
> We did not change Qwen-Drive's original `src/`, `scripts/`, `docs/`, or
> model weights.

This document describes the generator used by the current code. It does not
include the removed WTA, plain DD, recursive endpoint, or absolute-anchor
residual experiments.

## Terms used in this guide

- QD: Qwen-Drive.
- Ego vehicle: the vehicle controlled by the model.
- GT (ground truth): the real future path recorded in the data.
- BEV (bird's-eye view): a top-down view of the road.
- WM (world model): a model that predicts future scene states.
- RM (reward model): a model that gives a score to a candidate path.
- Top-1: the one path chosen for the vehicle to drive.
- Oracle: a best-possible result made with answer labels. It is not deployable.

## 1. What the driving model must output

At the current time, the model receives:

- multi-frame images from three front cameras;
- the ego vehicle's pose, speed, and acceleration from the last 1.5 seconds;
- a navigation command, such as go straight, turn left, or turn right;
- the current ego state.

The model must predict the ego vehicle's path for the next 5 seconds. The path
has 50 waypoints at 10 Hz. Each waypoint contains:

```text
x: forward distance
y: left or right offset
heading: vehicle direction
```

All paths use the current ego frame:

```text
(0, 0) = current vehicle position
+x     = in front of the vehicle
+y     = to the left of the vehicle
```

So the output is a 3D line. It shows where the vehicle may go in the next
5 seconds and how it may turn.

## 2. What is an anchor

An anchor is a typical future route learned from the training data.

The training set has 37,643 future paths. Saving every path is too expensive.
We use K-Means to group similar paths into 20 representative paths. Each
representative path is called an anchor.

```text
37,643 training paths
  -> K-Means clustering
  -> 20 typical path centers
  -> each center is one anchor
```

The anchor bank is built offline from the training set. The anchors do not move
when we train the Planner. The model learns how to adjust each anchor for the
current scene.

Current anchor bank:

```text
file: outputs/anchored_flow/anchors_k20.npz
count: K=20
each anchor: 50 waypoints
```

## 3. How the anchors are built

First, paths are normalized with the Qwen-Drive scale:

```text
x_norm = x / 165 m
y_norm = y / 25 m
heading_norm = heading / 1.5703125 rad
```

Each `50 x 3` path is flattened into a vector of length 150. We run
deterministic K-Means++ on the 37,643 vectors. After clustering, the 20 centers
are converted back to meters and radians.

Each anchor stores how many training windows it covers. The current 20 anchors
have very different counts. The smallest has only 124 samples. The largest has
more than 10,000.

## 4. Why an anchor cannot be the final path

A global K-Means anchor does not know the current scene. A high-speed straight
anchor cannot work for every low-speed intersection. So an anchor cannot be the
final output by itself.

Using an anchor plus a very small correction is also not enough. In an earlier
test, direct anchor output had:

```text
first preference ADE: 9.337 m
mean preference ADE: 15.985 m
```

The old custom ADE Headroom was high in this test. However, most candidates were
not usable. This was fake diversity made from bad paths. It is also why we no
longer use Headroom as a main paper metric.

The current method uses:

```text
scene base path
+ a small amount of anchor mode difference
+ a learned residual with a small limit
```

## 5. Scene base: first predict one path for the scene

First, we compute the mean of the 20 anchors. We use this mean path as a neutral
input to the frozen Qwen-Drive Planner:

```text
base = Planner(scene, neutral_trajectory)
```

The base path already contains scene information from the cameras, history,
navigation command, and ego state. It keeps all candidates suitable for the
current scene.

The base is not the output of one anchor. It is the common start for all modes.

## 6. Centered anchor: extract the direction difference

First, compute the mean of the 20 anchors:

```text
anchor_center = mean(anchor_0 ... anchor_19)
```

Then compute:

```text
centered_anchor_k = anchor_k - anchor_center
```

`centered_anchor_k` is no longer a full route. It only describes how anchor `k`
differs from the average anchor. For example, it may be more left, more right,
turn earlier, or travel farther.

## 7. Candidate path formula

Candidate `k` is:

```text
candidate_k =
    base
    + lambda * centered_anchor_k
    + delta_k(scene, anchor_k)
```

Each term:

- `base`: the shared scene path;
- `lambda * centered_anchor_k`: the mode difference from anchor `k`;
- `delta_k`: a small scene-dependent correction learned by the network.

`lambda` is trainable, but it must stay in this range:

```text
0.04 <= lambda <= 0.08
initial value = 0.045
```

We keep a lower bound. If `lambda` becomes 0, all 20 paths become the same base
path. This is called mode collapse: the model gives many samples, but they are
almost the same.

Early experiments used `modes@1m` as a quick check. It only looked at the
endpoint after 5 seconds. If a new endpoint was less than 1 meter from an
existing endpoint, it was put in the same group.

This rule was defined by this project. It is not a WOD-E2E metric. It is not a
DiffusionDrive metric. The result also depends on candidate order. We no longer
use it for the main evaluation. We use DiffusionDriveV2 `Div` instead.

The old endpoint results remain only as tuning background:

```text
lambda 0.035 -> about 4 endpoint groups
lambda 0.040 -> about 5 endpoint groups
lambda 0.045 -> about 5 endpoint groups
lambda 0.060 -> about 7 endpoint groups
```

Training keeps `lambda` close to 0.045. Its main job is to stop the anchor
differences from shrinking to zero. Section 16 defines the official diversity
metric used in this project.

## 8. Limits on the learned residual

`delta_k` comes from a linear head with zero initialization. At the start of
training:

```text
delta_k = 0
```

So the model does not destroy the anchor prior at the start.

The residual has these limits:

```text
XY correction at each waypoint <= 12 m
heading correction at each waypoint <= 0.35 rad
```

The limit uses a smooth `tanh`, not a hard cut. The gradient is still kept after
the value reaches the limit.

## 9. How anchor identity enters the Transformer

Full architecture diagram:

```text
outputs/visualizations/anchored_flow/architecture/planner_base_residual_architecture.png
outputs/visualizations/anchored_flow/architecture/planner_base_residual_architecture.pdf
```

Each candidate also receives:

- all waypoint coordinates of `anchor_k`;
- an embedding for the anchor ID;
- the current flow time `t=0.9`;
- scene K/V features;
- history pose, speed, and acceleration;
- navigation command and ego status.

The anchor ID embedding is projected and added to the adaptive layer norm
condition:

```text
condition = time + navigation + ego + anchor_embedding
```

The full anchor shape goes directly into the waypoint input and Fourier
features.

## 10. Why we add low-frequency noise

Each anchor gets a small amount of low-frequency noise:

```text
noise_std = 0.02
noise_modes = 6
```

This is not independent random noise at every waypoint. The noise is smooth and
uses six low-frequency cosine directions. This avoids a zig-zag path. It also
keeps different candidates from becoming exactly the same.

## 11. How training labels are assigned

For each training scene, compute the mean XY distance between the GT future path
and each of the 20 anchors:

```text
error_k = mean_t ||GT_t - anchor_k,t||_2
gt_anchor_id = argmin_k error_k
```

This anchor is called the GT-nearest anchor.

Only this path receives the direct GT path regression loss. All 20 candidates
take part in the classification and ranking losses.

## 12. Total loss

Current total loss:

```text
total =
    trajectory_weight * (
        regression_weight * L1_positive
        + classifier_weight * focal
        + ranking_weight * ranking
    )
```

Current weights:

```text
trajectory_weight = 12
regression_weight = 8
classifier_weight = 1
ranking_weight = 5
```

So the full expression is:

```text
total = 12 * (8 * L1_positive + 1 * focal + 5 * ranking)
```

### 12.1 Positive path L1

```text
L1_positive = mean |candidate_(gt_anchor_id) - GT|
```

This loss only updates the path from the GT-nearest anchor. It makes the best
mode match the real future path as closely as possible.

### 12.2 Focal classification

The score head outputs 20 logits. The GT-nearest anchor is the positive class.
The other 19 anchors are negative classes. Focal loss uses:

```text
alpha = 0.25
gamma = 2.0
```

Focal loss gives less weight to easy negative samples. Training focuses more on
anchors that are hard to separate.

### 12.3 Candidate ranking

For each candidate, compute a smooth L1 error `e_k`. Then compute:

```text
p_k = softmax(-e_k / 0.02)
```

This is a soft target for the best candidate. The score logits must learn this
target:

```text
ranking = -sum_k p_k * log_softmax(score)_k
```

The goal is to make the score head rank path quality. It should not only predict
the nearest anchor class.

### 12.4 Why total loss stays near 180

The random baseline of `ranking` is about:

```text
ln(20) = 2.9957
```

The ranking part alone adds this value to the total loss:

```text
12 * 5 * 2.9957 = 179.74
```

So the total loss stays near 180. The useful values to watch are:

- positive path L1;
- focal loss;
- ranking loss;
- first and median ADE;
- DDv2 Div, Mean candidate RFS, and RFS gap.

In training, ranking loss stayed near 2.99. This means it did not learn to rank
candidate quality. The final selector should be replaced by WoTE's future-BEV
reward model.

## 13. Training setup

Current full training run:

```text
training windows: 37,643
epochs: 1
gradient accumulation: 8
optimizer steps: 4,705
learning rate: 1e-5
AdamW weight decay: 1e-4
gradient clip: 1.0
LoRA rank: 16
LoRA alpha: 32
```

The VLM and original Planner are frozen. The trainable parts are:

- Planner LoRA;
- anchor embedding and score head;
- scene-base residual head;
- mode offset scale.

## 14. Current checkpoints and candidate sets

A checkpoint is saved every 500 steps. In the 48-scene sweep, KM step 500 had
the best generator results. The full 478-scene test also showed that step 500
was slightly better than step 4705.

Current main systems:

| Name | Meaning |
| --- | --- |
| QD-SFT-reasoning | Original Qwen-Drive SFT Planner with reasoning |
| QD-RL-reasoning | Original Qwen-Drive RL Planner with reasoning |
| KM-step500 | Global K-Means anchors + BaseResidual, step 500 |
| SKM-step250 | Stratified K-Means anchors + BaseResidual, step 250 |
| SFT direct+anchor | Frozen SFT base plus 20 centered anchors, with no new Planner flow |
| RL direct+anchor | Frozen RL base plus 20 centered anchors, with no new Planner flow |

Frozen trained checkpoints:

```text
KM:
outputs/anchored_flow/k20_base_residual_full1/checkpoints/step_000500

SKM:
outputs/anchored_flow/k20_sft_skm_pilot1000/checkpoints/step_000250
```

The two direct+anchor methods do not train a new residual head. They use one
frozen SFT or RL path as the base:

```text
candidate_k =
    base
    + 0.045 * centered_anchor_k
```

The anchor difference is added after the Planner step. The Planner does not
rewrite the candidates again. This isolates the effect of the SFT and RL bases.

## 15. Common evaluation setup

Earlier experiments had three problems:

1. QD results used different candidate counts, such as 1, 6, and 16;
2. some QD-SFT results used direct planning and some used reasoning planning;
3. some results used old custom metrics and could not be compared with
   DiffusionDrive.

The current formal test uses:

```text
validation set: all 478 WOD-E2E validation scenes
candidate count: N=20
planning mode: reasoning_planning for QD
time grid: Waymo official 5 s, 4 Hz, 20 positions
```

The repository's `docs/evaluation.md` has a conflict about the time grid. It
says the reported numbers use the normalized grid, but the published QD values
match the absolute 4 Hz grid. We tested this:

```text
published QD RFS: 7.95 / 8.45
published QD ADE: 2.31 / 1.27
```

These values match the absolute 4 Hz grid. So this document and the common
evaluation script use the absolute 4 Hz grid.

The QD `N=20` files were generated again with the original inference script:

```text
outputs/waymo_val/sft_20sample_reasoning_fullval.jsonl
outputs/waymo_val/rl_20sample_reasoning_fullval.jsonl
```

The QD `RFS/ADE` values in the table use one separate reasoning sample. This is
the deployable one-path result. The 20 samples are only used to measure
diversity and the Oracle limit. We do not report the best of 20 as a deployable
result.

How each method chooses the path it will execute:

| Method | How the final path is chosen |
| --- | --- |
| QD-SFT-reasoning | one separate reasoning sample |
| QD-RL-reasoning | one separate reasoning sample |
| KM-step500 | highest score among 20 candidates |
| SKM-step250 | highest score among 20 candidates |
| SFT direct+anchor | no learned selector; current test uses candidate 0 |
| RL direct+anchor | no learned selector; current test uses candidate 0 |

Common evaluation code:

```text
research/anchored_flow/src/qwen_drive_anchored/evaluate_paper_metrics.py
```

Full reports:

```text
outputs/anchored_flow/paper_metrics_n20_reasoning_2026-11-08/summary.md
outputs/anchored_flow/paper_metrics_n20_reasoning_2026-11-08/per_scene.csv

outputs/anchored_flow/paper_metrics_sft_vs_rl_direct_anchor_2026-11-09/summary.md
outputs/anchored_flow/paper_metrics_sft_vs_rl_direct_anchor_2026-11-09/per_scene.csv
```

## 16. What each metric means and how it is computed

### 16.1 RFS: the main human-preference score

RFS means **Rater Feedback Score**. It ranges from 0 to 10. Higher is better.

RFS does not only compare a prediction with the recorded driver path. A scene
may have more than one safe and reasonable path. Waymo asks human raters to
score several candidate paths. The metric checks whether the model path is close
to one of these reasonable paths.

How it is computed:

```text
1. Resample the model path from 50 waypoints at 10 Hz
   to 20 waypoints at 4 Hz.
2. At 3 s and 5 s, compute the lateral and longitudinal difference
   between the model path and each human-rated path.
3. Scale the allowed error with the initial speed.
4. If the model path enters the allowed area, called the trust region,
   for a human-rated path, treat it as a reasonable driving choice.
5. Take the score from the best matching rated path.
   A larger error gives more score decay.
6. If the path is outside every allowed area, the score is clipped to at least 4.
7. Average the scene scores over all validation scenes.
```

Formula:

```text
RFS_scene = official_rater_feedback(
    prediction,
    human_preference_trajectories,
    human_preference_scores,
    initial_speed
)

RFS = mean(RFS_scene over all validation scenes)
```

The official tool uses speed-dependent limits at 3 s and 5 s:

```text
base limits: [1.0 m, 1.8 m]
lateral multiplier: 1
longitudinal multiplier: 4
final scale by initial speed: 0.5 to 1.0
```

In simple words, a slow vehicle gets a smaller allowed error. A fast vehicle
gets a larger allowed error. The model does not need to copy the recorded path
exactly. It can choose another path that humans rate as reasonable.

### 16.2 ADE: how close the path is to the real future

ADE means **Average Displacement Error**. The unit is meters. Lower is better.

For one selected prediction and one GT path:

```text
ADE =
  mean over all waypoints
  sqrt((x_pred - x_gt)^2 + (y_pred - y_gt)^2)
```

For example, if there are 20 test points, compute the 2D distance at each point
and take the mean.

RFS checks whether the path is a reasonable human choice. ADE checks how close
it is to the exact recorded future. A different safe path can have a high RFS
and a higher ADE. WOD-E2E uses RFS as the main score. ADE is used for path
accuracy and as a tie-breaker.

### 16.3 DDv2 Div: how different the candidates are

`Div` is a path-diversity metric defined by DiffusionDriveV2. It does not count
endpoint groups. It compares all candidates at every future time.

For one scene and one future time `t`:

```text
Div_raw(t) =
  mean 2D distance over all candidate pairs

scale(t) =
  mean distance from each candidate waypoint to the ego origin

Div(t) =
  min(1, Div_raw(t) / (scale(t) + 1e-6))
```

Then:

```text
Div_scene = 100% * mean(Div(t) over all waypoints)
Div = mean(Div_scene over all validation scenes)
```

Simple meaning:

- if all 20 paths are almost the same, Div is close to 0%;
- if paths split into straight, left, right, and different speeds, Div goes up;
- a high Div only means the paths are different;
- a high Div does not mean every path is safe or good;
- one crazy path can raise Div, so we must also check RFS, ADE, and Oracle RFS.

We must use the same candidate count for a fair comparison. The formal table
uses `N=20`, so QD, KM, SKM, and RL+anchor can be compared.

### 16.4 Mean candidate RFS: the average quality of 20 paths

Run the official RFS on each candidate. Then average the 20 scores:

```text
Mean candidate RFS =
  mean(RFS of candidate 1 ... candidate 20)
```

This is not the path the model will choose. It only shows the average quality of
the whole candidate set. One model may find one great path, while the other 19
are bad. In that case, Mean candidate RFS goes down.

### 16.5 Oracle RFS: the best possible score in the candidate set

For each scene, compute official RFS for all 20 candidates and keep the highest
one:

```text
Oracle RFS_scene =
  max(RFS_candidate_1 ... RFS_candidate_20)

Oracle RFS =
  mean(Oracle RFS_scene over all validation scenes)
```

Oracle RFS means: what score could we reach if we already knew the best path?
It uses validation human scores, so it leaks labels. It is only an upper bound.
It is **not a deployable result**.

### 16.6 RFS gap: how far the selector is from the upper bound

```text
RFS gap = Oracle RFS - RFS of the selected path
```

Lower is better. It uses RFS score units.

| Pattern | Meaning |
| --- | --- |
| small RFS gap | the selector is close to the best candidate |
| large RFS gap and high Oracle | good paths exist, but the selector cannot find them |
| small RFS gap and low Oracle | the candidate set itself lacks good paths |

This is clearer than the old custom `Headroom`. It uses the official RFS
directly and does not use ADE as a proxy for RFS.

## 17. Common results on all 478 scenes

| Method | N | RFS up | ADE down | DDv2 Div up | Mean candidate RFS up | Oracle RFS up | RFS gap down |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| QD-SFT-reasoning | 20 | 7.9548 | 2.3167 | 3.96% | 7.9615 | 8.0928 | 0.1380 |
| SFT direct+anchor | 20 | 7.7928 | 2.3657 | 27.33% | 7.7210 | 8.7703 | 0.9775 |
| QD-RL-reasoning | 20 | 8.4432 | 1.2807 | 3.25% | 8.4510 | 8.5433 | 0.1001 |
| KM-step500 | 20 | 7.4854 | 2.5139 | 20.81% | 7.6236 | 8.6552 | 1.1698 |
| SKM-step250 | 20 | 7.5319 | 2.6305 | 21.85% | 7.6686 | 8.6606 | 1.1287 |
| RL direct+anchor | 20 | 8.2372 | 1.4260 | 27.43% | 8.1610 | 8.9623 | 0.7251 |

Important notes:

- QD `RFS/ADE` comes from one deployable reasoning sample;
- QD `Div/Mean/Oracle` uses the same model and reasoning mode, but comes from a
  separate set of 20 samples;
- QD's actual one-path result and 20-candidate test set come from two sampling
  runs. So QD's RFS gap also has a small random sampling difference. KM, SKM,
  and RL+anchor compute this gap inside one candidate set;
- QD-RL used validation RFS as a reward. Its validation score is in-sample. It
  is not a clean test on unseen data;
- KM/SKM RFS uses argmax from the candidate score head;
- SFT/RL direct+anchor have no selector. Their current RFS uses candidate 0;
- Oracle RFS and RFS gap use validation labels. They are for analysis only.

## 18. What the results mean

### 18.1 Original QD has low candidate diversity

QD-SFT and QD-RL have Div of only `3.96%` and `3.25%`. The model uses independent
noise for 20 samples, but the final paths are still very close.

QD-RL has an RFS that is `0.4884` higher than QD-SFT. Its ADE is also much lower.
RL improves path quality and human-preference alignment. However, RL does not
fix mode collapse. Its Div is even lower.

### 18.2 SFT direct+anchor shows that Planner updates cause the collapse

SFT direct+anchor has Div `27.33%`. RL direct+anchor has Div `27.43%`.
Their candidate-to-candidate distances are almost exactly the same. The mean
difference is about `8 x 10^-17 m`. Both use the same anchor bank and add the
same offset after the base.

This shows that the SFT base can keep the anchor differences. The earlier SFT
anchor-init result had lower diversity because all 20 paths were updated by the
SFT Planner from `t=0.8` to `1.0`. The Planner pulled different starts toward
similar paths.

The weak point of the SFT base is path quality:

```text
SFT direct+anchor Oracle RFS: 8.7703
RL  direct+anchor Oracle RFS: 8.9623
```

Both actual RFS values are lower than their single-path baselines. This happens
because they use candidate 0 and have no selector.

### 18.3 KM and SKM make different candidates, but the selector is still bad

KM and SKM raise Div to `20.81%` and `21.85%`. This means the anchor prior stops
all candidates from collapsing to the same path.

But their actual RFS is only `7.4854` and `7.5319`. This is even lower than the
original QD-SFT RFS of `7.9548`. Their RFS gaps are large: `1.1698` and `1.1287`.

In simple words:

```text
The candidate set now has different driving choices.
However, the score head often selects a bad path.
```

SKM compared with KM:

```text
RFS: +0.0465
Div: +1.04 percentage points
ADE: 0.1166 m worse
Oracle RFS: +0.0054
```

This is a small gain. It is not enough to prove that SKM is a decisive
improvement. SKM mainly adds navigation and speed grouping and uses real
medoid paths to cover rare scenes. A medoid is a real training path near the
center of a group. A full anchor ablation should come after the world-model main
line.

### 18.4 RL direct+anchor has the best coverage, but no good selector

RL+anchor has Div `27.43%`. This is about 8.4 times the QD-RL value. Its Oracle
RFS is `8.9623`. This means very good human-preferred paths do exist in the
20-candidate set.

However, the current system only takes candidate 0. Its RFS is `8.2372`. This is
lower than the single-path QD-RL result of `8.4432`. The RFS gap is `0.7251`.

This means:

```text
The generator already gives useful candidates.
The main bottleneck is now candidate selection.
We need a selector that understands the future scene.
```

## 19. Flow times and measured speed

Different experiments start at different flow times:

| Method | Flow prediction times | Candidate flow steps |
| --- | --- | ---: |
| Original QD | query at `0.0, 0.1, ..., 0.9`, then update to `1.0` | 10 |
| SFT anchor init (A2) | `0.8 -> 0.9 -> 1.0` | 2 |
| Current BaseResidual | base and candidate queries at `t=0.9`; no explicit Euler loop | 1 candidate endpoint query |
| Direct base+anchor | no new flow; add the anchor offset | 0 |

Nominal flow steps:

```text
original QD: 10
A2 anchor init: 2
BaseResidual: one candidate endpoint query at t=0.9
direct anchor: 0 extra candidate flow steps
```

BaseResidual does not start at `t=0.8`. It does not run two Euler updates. It
predicts the scene base at `t=0.9`, makes one batched candidate query at the
same time, and lets the residual head output the final path. It has two network
forwards, base and candidate, but only one candidate endpoint query.

So the candidate query in BaseResidual uses about 10 times fewer calls than
original QD. A2 uses five times fewer. Total delay will not drop by the same
amount. Camera encoding, the VLM, reasoning, and data loading still take most
of the time.

`Direct base+anchor` has no flow interval of its own. Its base usually comes
from an already generated QD or frozen Planner path. The candidate step is:

```text
candidate_k = base + 0.045 * centered_anchor_k
```

So the full RL direct+anchor pipeline is:

```text
RL base: original QD flow, t=0.0 ... 0.9 for 10 steps, final update to 1.0
anchor expansion: no flow, vector addition only
```

It is not correct to write it as `t=0.9 -> 1.0` or as a two-step flow that
starts at `t=0.8`.

We measured original QD on the same GPU and the same 16 RL reasoning scenes:

| Candidate count | Total time | Time per scene |
| ---: | ---: | ---: |
| 1 | 13.8 s | 0.86 s |
| 5 | 15.7 s | 0.98 s |
| 20 | 24.9 s | 1.55 s |

Results:

- `1 -> 5`: about 14% slower per scene;
- `5 -> 20`: about 58% slower;
- `1 -> 20`: about 80% slower, or `1/1.8` of the one-path speed.

The VLM runs once per scene. The expert rollouts are batched, so time does not
grow linearly with candidate count. More candidates still use more expert work,
memory, and batch capacity.

Direct anchor expansion uses almost no GPU. It took about 3.2 s to read, expand,
and write all 478 scenes with 20 candidates. That is about 6.7 ms per scene,
including file I/O. In an online system, it is one QD base path plus a small
vector operation.

The future BEV WM is different. If each candidate needs its own future BEV
prediction, WM cost will grow with candidate count. Shared scene features and
batched candidate heads can reduce this cost. The 20-to-5 compression should
only be accepted after measuring both Oracle RFS loss and WM delay.

## 20. Final path selection design

The current simple score head is only a control. It is not the final paper
selector.

The main paper path should be:

```text
20 candidate paths
  -> action-conditioned BEV world model
  -> predict future BEV for each candidate
  -> reward model scores each future BEV
  -> choose the highest reward
  -> final path
```

The WM (world model) and RM (reward model) need control experiments. We must
show that WM+RM selects well. We must also show that future BEV is needed:

| Control | Goal |
| --- | --- |
| random selection | lowest baseline |
| fixed first candidate | no-selector baseline |
| trajectory-only learned selector | no future BEV |
| reward without future BEV | test whether current state is enough |
| action-conditioned BEV WM + reward | full method |

If the full method does not beat the trajectory-only selector or the reward
without future BEV, we cannot say the world model helps the final choice.

## 21. Current conclusion and next steps

The current results show:

- original QD has clear mode collapse on WOD-E2E;
- anchor-conditioned BaseResidual greatly improves candidate diversity;
- SFT/RL direct+anchor show that the later Planner update causes the collapse;
- RL direct+anchor gives the highest Oracle RFS with 20 candidates;
- KM/SKM simple score heads cannot turn candidate coverage into a reliable
  top-1 result;
- SFT/RL direct+anchor also have no learned selector, so their current RFS is
  far below the candidate upper bound.

The next steps are:

1. freeze the current candidate generator and the `N=20` test setup;
2. implement the action-conditioned BEV world model;
3. train the reward model with future BEV;
4. compare with random, first, trajectory-only selector, and reward without
   future BEV;
5. only after WM+RM top-1 RFS clearly beats the baselines should we run more
   anchor-count, KM/SKM, and SFT/RL ablations.
