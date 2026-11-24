# Scene-base residual status

Updated: 2026-11-09

## Active method

The active planner is:

```text
base = planner(scene, neutral trajectory)
candidate_k = base + lambda * centered_anchor_k + bounded_residual_k
```

Only `base_residual` remains in the codebase. WTA, plain DD, recursive
endpoint, Gaussian-start, and absolute-anchor residual training paths were
removed after negative or unsafe results.

## Canonical N=20 evaluation

The current formal evaluation uses all 478 WOD-E2E validation scenes, 20
candidates per scene, the Waymo absolute 4 Hz grid, and reasoning mode for QD.
`modes@1m`, Pair FDE, and ADE Headroom below are old development diagnostics.
The current main table uses DDv2 Div and official RFS:

| Method | N | RFS | ADE | DDv2 Div | Mean candidate RFS | Oracle RFS | RFS gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| QD-SFT-reasoning | 20 | 7.9548 | 2.3167 | 3.96% | 7.9615 | 8.0928 | 0.1380 |
| SFT direct+anchor | 20 | 7.7928 | 2.3657 | 27.33% | 7.7210 | 8.7703 | 0.9775 |
| QD-RL-reasoning | 20 | 8.4432 | 1.2807 | 3.25% | 8.4510 | 8.5433 | 0.1001 |
| KM-step500 | 20 | 7.4854 | 2.5139 | 20.81% | 7.6236 | 8.6552 | 1.1698 |
| SKM-step250 | 20 | 7.5319 | 2.6305 | 21.85% | 7.6686 | 8.6606 | 1.1287 |
| RL direct+anchor | 20 | 8.2372 | 1.4260 | 27.43% | 8.1610 | 8.9623 | 0.7251 |

The direct-anchor rows use the same SKM K=20 bank and `scale=0.045`. Their
candidate-to-candidate spread is identical, so SFT can preserve anchor
diversity when the Planner does not rewrite the candidates. The earlier SFT
anchor-init result collapsed because the SFT Planner updated all candidates
from `t=0.8` to `1.0`.

Oracle RFS and RFS gap use validation rater labels and are analysis-only.
QD-RL validation scores are in-sample because validation RFS was used as its
training reward.

Full reports:

```text
outputs/anchored_flow/paper_metrics_n20_reasoning_2026-11-08/summary.md
outputs/anchored_flow/paper_metrics_sft_vs_rl_direct_anchor_2026-11-09/summary.md
```

## Flow time and speed

`Original QD` uses ten flow predictions at `t=0.0 ... 0.9` and moves to
`t=1.0`. SFT anchor init (A2) uses `t=0.8 -> 0.9 -> 1.0`. The current
BaseResidual starts directly at `t=0.9`: it makes one scene-base query and one
batched candidate query, then the residual head outputs the final path. It has
no explicit `0.9 -> 1.0` Euler loop. Direct base+anchor adds the anchor offset
after the base and has no flow interval of its own.

On 16 identical RL reasoning scenes:

| Candidate count | Time per scene |
| ---: | ---: |
| 1 | 0.86 s |
| 5 | 0.98 s |
| 20 | 1.55 s |

The VLM runs once per scene. Going from 1 to 20 candidates is about 1.8 times
slower, not 20 times slower. Direct anchor expansion itself is negligible; the
future WM cost still needs separate measurement.

## 48-scene development result

Selected checkpoint: `k20_base_residual_pilot200`, step 200.

| Metric | Value |
| --- | ---: |
| modes@1m | 5.000 |
| Pair FDE | 1.829 m |
| Headroom | 48.1% |
| First preference ADE | 2.433 m |
| Mean preference ADE | 2.483 m |
| Oracle preference ADE | 1.435 m |
| Selected preference ADE | 2.548 m |
| Low-diversity scenes | 8% |
| Headroom > 10% | 98% |

The generator passes the development-set coverage gate. First and mean ADE are
within the predefined 5% guardrail relative to A2 (2.421/2.431 m).

### 430-scene holdout

The generator result generalises:

| Metric | Value |
| --- | ---: |
| modes@1m | 5.000 |
| Pair FDE | 1.829 m |
| Headroom | 40.7% |
| First preference ADE | 2.972 m |
| Mean preference ADE | 3.064 m |
| Oracle preference ADE | 2.132 m |
| Selected preference ADE | 3.061 m |
| Low-diversity scenes | 6% |
| Headroom > 10% | 96% |

Coverage and diversity clearly hold on the larger split. Selection remains
broken: selected ADE is effectively equal to mean ADE.

### Same-split A2 comparison

| Metric | A2 | BaseResidual | Delta |
| --- | ---: | ---: | ---: |
| modes@1m | 3.181 | 5.000 | +1.819 |
| Headroom | 23.1% | 40.7% | +17.6 pp |
| First preference ADE | 2.966 m | 2.972 m | +0.2% |
| Mean preference ADE | 2.962 m | 3.064 m | +3.4% |
| Oracle preference ADE | 2.506 m | 2.132 m | -0.374 m |
| First GT ADE | 2.246 m | 2.335 m | +4.0% |
| Medoid GT ADE | 2.235 m | 2.312 m | +3.4% |
| Oracle GT ADE | 1.846 m | 1.781 m | -0.065 m |

The larger candidate set also improves physical tail behavior: maximum speed
drops from 38.75 to 33.23 m/s and maximum acceleration from 169.21 to
129.04 m/s^2. The S1 generator gate passes on the holdout.

### Full-epoch generator

The one-epoch run completed at step 4,705 and was evaluated on all 478
validation scenes:

| Metric | Released QD | BaseResidual full |
| --- | ---: | ---: |
| modes@1m | 1.000 | 5.000 |
| Pair FDE | n/a | 1.826 m |
| Headroom | 0% | 40.4% |
| First preference ADE | 2.902 m | 2.936 m |
| Mean preference ADE | 2.902 m | 3.027 m |
| Oracle preference ADE | 2.902 m | 2.097 m |
| First GT ADE | 2.250 m | 2.427 m |
| Medoid GT ADE | 2.250 m | 2.404 m |
| Oracle GT ADE | 2.250 m | 1.879 m |
| Selected preference ADE | n/a | 2.951 m |

The full run preserves the pilot's generator properties: 6% low-diversity
scenes, 96% of scenes with more than 10% headroom, maximum speed 33.05 m/s,
and maximum acceleration 129.04 m/s^2.

The final full epoch is not automatically the best checkpoint. On the held-out
430 scenes, the 200-step pilot has slightly better preference ADE and Oracle
ADE than the final full epoch. Full-run checkpoints must be swept on the 48
scene development split, followed by one complete 478-scene evaluation of the
selected checkpoint. The simple score head still provides essentially no top-1
gain and is retained only as a control.

### Selected generator checkpoint

Step 500 and step 4705 were evaluated on all 478 validation scenes:

| Metric | Step 500 | Step 4705 |
| --- | ---: | ---: |
| modes@1m | 5.000 | 5.000 |
| Headroom | 40.5% | 40.4% |
| First preference ADE | **2.922 m** | 2.936 m |
| Mean preference ADE | **3.014 m** | 3.027 m |
| Oracle preference ADE | **2.088 m** | 2.097 m |
| First GT ADE | **2.414 m** | 2.427 m |
| Mean GT ADE | **2.587 m** | 2.597 m |
| Oracle GT ADE | **1.875 m** | 1.879 m |

Step 500 is selected and frozen as the S1 generator. Step 4705 has a lower
`selected_ade_pref`, but that comes only from the temporary simple score head,
which will be replaced by the WoTE BEV world model and reward model.

### WOD-E2E RFS

The official Waymo rater-feedback implementation is vendored under
`third_party/waymo-open-dataset/src`. On all 478 validation scenes:

| Selection | RFS | RFS within trust region |
| --- | ---: | ---: |
| Released Qwen-Drive SFT | 7.9548 | 67.78% |
| Step500, first candidate | 7.9675 | 67.15% |
| Step500, simple score argmax | 7.5010 | 53.97% |
| Step500, Oracle by preference label | 8.2797 | 68.83% |

The candidate set has a meaningful Oracle gain of +0.3249 RFS over the released
planner. However, the temporary simple score head degrades RFS to 7.5010. These
results strongly support replacing that head with the WoTE future-BEV reward
model and prohibit reporting Oracle RFS as a deployable result.

### RL-base anchor probe

An inference-only probe added centered anchors directly to the released
`planner-rl` trajectory without training a new adapter:

| Candidate scale | modes@1m | Headroom | Oracle preference ADE | Oracle RFS |
| ---: | ---: | ---: | ---: | ---: |
| RL base only | 1.000 | 0% | 2.846 m | 8.4432 |
| RL + anchors, lambda 0.04 | 5.000 | 44.7% | 2.055 m | **8.6477** |
| RL + anchors, lambda 0.05 | 6.000 | 49.4% | 1.943 m | 8.6442 |

This is an Oracle-only diagnostic using the preference label to select a
candidate. It nevertheless shows that RL + anchor candidates are a stronger
starting point than SFT + anchors and justifies retraining the anchor residual
with `planner-rl` under the reasoning condition.

### Stratified medoid anchor bank

The low-cost anchor-bank replacement groups training windows by navigation
command and initial-speed bin, guarantees one medoid for every nonempty
stratum, and uses real training trajectories instead of cluster means.

Generated bank:

```text
outputs/anchored_flow/anchors_k20_skm.npz
```

It contains 12 straight, 4 left-turn, and 4 right-turn anchors. Every nonempty
navigation/speed stratum is represented. Validation nearest-anchor coverage at
1 m is 93.5%, lower than the old global bank's 97.5%, so the new bank trades
some common-case geometric fit for explicit command/speed and tail coverage.

An inference-only planner-rl probe with lambda 0.04 produced:

| Bank | modes@1m | Headroom | Oracle preference ADE | Oracle RFS |
| --- | ---: | ---: | ---: | ---: |
| Old global K-Means | 5.0 | 44.7% | 2.055 m | 8.6477 |
| Stratified medoids | 5.0 | 44.7% | 2.054 m | 8.6564 |

The tail-aware bank gives only a small immediate RFS increase before retraining,
so a 1,000-step stratified planner-rl pilot is the next appropriate test.

### SKM SFT 1,000-step pilot

The best SKM checkpoint on 48 scenes was step 250. It was evaluated on all 478
validation scenes:

| Metric | KM step500 | SKM step250 |
| --- | ---: | ---: |
| modes@1m | 5.000 | 5.000 |
| Pair FDE | 1.829 m | 1.783 m |
| Headroom | 40.5% | 41.1% |
| First preference ADE | 2.922 m | 2.995 m |
| Mean preference ADE | 3.014 m | 3.004 m |
| Oracle preference ADE | 2.088 m | 2.066 m |
| First GT ADE | 2.414 m | 2.325 m |
| Mean GT ADE | 2.587 m | 2.566 m |
| Oracle GT ADE | 1.875 m | 1.841 m |
| Oracle RFS | 8.2797 | **8.2992** |

SKM improves most generator and Oracle metrics slightly, but fixed-first RFS
falls from 7.9675 to 7.8924. The +0.0195 Oracle RFS gain is too small to treat
as conclusive without multiple seeds. SKM does not change modes because K=20,
N=20, and lambda remain fixed.

### Zero-training bank/candidate-count probe

Using the SKM250 candidate mean as a fixed scene base, larger banks were tested
without retraining a residual head:

| Setting | modes@1m | Headroom | Oracle pref ADE | Oracle RFS |
| --- | ---: | ---: | ---: | ---: |
| K20, N20 | 5.00 | 41.1% | 2.066 m | 8.2992 |
| K32, N20 | 4.91 | 40.4% | 2.072 m | 8.2917 |
| K64, N20 | 2.48 | 24.4% | 2.418 m | 8.1782 |
| K32, N32 | 9.00 | 42.8% | 2.020 m | 8.2990 |
| K64, N32 | 4.33 | 32.5% | 2.250 m | 8.2772 |
| K64, N48 | 7.62 | 43.5% | 2.002 m | **8.3729** |
| K64, N64 | 10.00 | 44.2% | 1.981 m | 8.3473 |

Increasing K without increasing N does not help and can collapse diversity.
K64 with N48 gives the best observed Oracle RFS, but requires 2.4x candidate
world-model work. K32N32 adds modes without an RFS gain. K/N selection should be
finalized before full WM training, or the WM should be trained with variable
candidate counts.

### RL-base candidate-count probe

The same zero-training probe was repeated with the released `planner-rl`
trajectory as scene base:

| Setting | modes@1m | Headroom | Oracle pref ADE | Oracle RFS |
| --- | ---: | ---: | ---: | ---: |
| RL only | 1.00 | 0% | 2.846 m | 8.4432 |
| RL + K20N20 | 5.00 | 47.1% | 1.998 m | **8.6436** |
| RL + K64N48 | 7.62 | 49.7% | 1.926 m | 8.6413 |

K64N48 improves geometric Oracle ADE but does not improve Oracle RFS over
K20N20 while requiring 2.4x candidate WM work. This resolves the generator
choice for the initial WM: freeze the RL planner plus K20 anchored candidates;
keep SFT/SKM and larger K/N as later ablations.

The requested K64N40 setting was also probed before training:

| Setting | modes@1m | Headroom | Oracle pref ADE | Oracle RFS |
| --- | ---: | ---: | ---: | ---: |
| RL + K20N20 | 5.00 | 47.1% | 1.998 m | **8.6436** |
| RL + K64N40 | 6.53 | 49.3% | 1.932 m | 8.6252 |

K64N40 gains geometric modes and Oracle ADE but loses RFS, so training it before
WM is not justified for the mainline. It remains available as a later ablation.

### Statistical selector controls

On the RL K20N20 candidate set, simple non-WM selectors produce:

| Selector | RFS |
| --- | ---: |
| Random candidate | 8.1422 |
| Longest endpoint | 7.4736 |
| Candidate medoid / median endpoint | 8.4250 |
| RL base | 8.4432 |
| Oracle | 8.6436 |

Simple statistics do not exploit the available headroom. However, the Oracle
gap over RL is only 0.2004 RFS. A post-hoc WM/RM must therefore beat a
trajectory-only learned selector and no-future-feature reward, not merely the
current hand-written rules. If the WM never influences candidate generation,
RFS is mathematically capped by the Oracle value of the fixed candidate set.

The repository evaluator's "Oracle" chooses the candidate closest to the best
preference trajectory by ADE. Computing the official RFS independently for
every candidate gives a more accurate candidate-set ceiling:

| Candidate set | Mean candidate RFS | True RFS Oracle |
| --- | ---: | ---: |
| RL + K20N20 | 8.1640 | **8.9651** |
| RL + K64N40 | 8.2130 | 8.9577 |

The validation labels have a per-scene best-score mean of 9.6004, so an average
RFS of 10 is not attainable. The practical target is to close the gap from the
current 8.9651 candidate ceiling toward the 9.6004 rater-label ceiling.

This 8.9651 number must not be reported as a deployable result:

- choosing the maximum candidate per scene uses validation rater labels directly;
- the released planner-rl was reward-optimized with WOD-E2E validation RFS and
  is explicitly documented as in-sample on validation;
- the SKM anchor bank itself was fitted only on the training split and does not
  directly use validation preferences.

Thus 8.9651 is the upper bound of an in-sample, label-selected candidate set.
The final clean number must come from the official test leaderboard or from a
reward/selector trained without those evaluation labels.

The gain comes primarily from the fixed bounded anchor offset: Pair FDE remains
1.829 m and mode-offset scale changes only from 0.0450008 to 0.0450025 during
training. This is a successful structural prior, not evidence that the Qwen
planner has learned multimodal generation end to end.

## Blocker

The selector is still the blocker. The candidate set has a high Oracle RFS,
but the KM/SKM score head makes the actual result worse. SFT and RL
direct+anchor also use the first candidate, so neither turns candidate coverage
into a deployable top-1 gain.

## Next

1. Freeze the `N=20` candidate generator and evaluation protocol.
2. Implement the action-conditioned BEV world model.
3. Train the reward model with future BEV.
4. Compare against random, first-candidate, trajectory-only selector, and
   reward without future BEV.
5. Measure the quality/compute trade-off before adopting 20-to-5 candidate
   compression.

## Compatibility

The refactor preserves the existing checkpoint state keys:

```text
anchor_heads.*
scene_base_residual_head.residual.*
scene_base_residual_head.raw_mode_offset_scale
```

Existing step-200 weights remain loadable by the current code.
