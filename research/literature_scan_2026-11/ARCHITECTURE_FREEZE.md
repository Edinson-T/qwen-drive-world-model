# Architecture freeze

**Freeze ID:** `qwen-drive-arch-freeze-2026-11-04`

**Effective date:** 2026-11-04

**Review trigger:** earlier of (a) completion of the S1 anchored-flow pilot, or
(b) 2026-12-18. The review may retain the freeze for another bounded period,
but it cannot silently change the mainline.

## Frozen thesis contract

The mainline is frozen as:

```text
S0: released Qwen-Drive flow planner baseline
        |
        v
S1: motion-anchor initialization + truncated flow
        |
        +-- fail --> diagnose / try one pre-registered alternative
        |
        +-- pass
               |
               v
S2: action-conditioned compact BEV world model + critic
        |
        v
WOD-E2E main table and ablations
```

Geometry is not part of S1 or S2. It remains an optional representation
extension after the generator-and-selector story is demonstrated.

## Hypotheses

**H1: generator bottleneck.** The released planner's 16 independent Gaussian
samples collapse to approximately one endpoint mode at 1 m on WOD-E2E val.

**H2: anchored-flow remedy.** Motion anchors plus a truncated flow schedule
increase candidate coverage and selection headroom without degrading oracle
trajectory quality.

**H3: deployable selection.** A candidate-conditioned compact BEV world model
and critic convert coverage headroom into top-1 gains over random and
heuristic selection.

**H4: optional geometry.** Dense/geometric supervision may improve the BEV
representation, but is not required for H1--H3.

## Frozen S1 experiment

S1 contains exactly four runs:

1. released planner, standard Gaussian, 10 Euler steps;
2. anchor initialization, no adaptation, 2 Euler steps;
3. anchor initialization + planner LoRA, 2 Euler steps;
4. anchor initialization + planner LoRA, 4 Euler steps.

The first anchor-bank size is 20. A 64-anchor run is a post-pass ablation, not
part of the initial architecture search.

Report for every run:

- `modes@1m`;
- pairwise ADE and FDE;
- endpoint diversity ratio;
- oracle ADE;
- first-candidate and medoid ADE;
- selection headroom relative to the best-rated human trajectory;
- anchor usage histogram;
- RFS and ADE on the fixed WOD-E2E validation split.

Passing requires:

```text
modes@1m >= 5
selection headroom >= 20%
oracle ADE does not regress relative to S0
first/medoid ADE does not show a material collapse
camera-only inference remains valid
```

The headroom criterion alone is not sufficient: adding bad candidates can
inflate headroom. Oracle and typical-candidate quality are mandatory
guardrails.

## Frozen S2 experiment

S2 starts only after S1 passes. Its architecture is:

```text
scene representation z_t
  + candidate trajectory a_i
  -> compact BEV/occupancy future latent z'_i
  -> quality score q_i
  -> select argmax_i q_i
```

The critic gradient may shape the future representation, following WAC's
ablation. Dense pointmaps and video-frame generation are excluded. A simple
learned score head, random selection, and a rule/heuristic selector are
required controls.

Passing S2 requires top-1 ADE to be at least 15% below random selection and the
selected trajectory to improve WOD-E2E RFS without unacceptable ADE regression.

## New-paper admission rule

A new paper can enter the active plan only through one of these roles:

1. **stronger baseline**: the paper invalidates an existing comparison;
2. **component replacement**: it solves a measured failure with comparable or
   lower integration cost;
3. **risk reference**: it threatens novelty or identifies a confound.

The default action is `defer`. Adding a module because it is recent, popular, or
architecturally attractive is not an admissible reason.

Every proposed architectural change must state:

- which frozen hypothesis or measured failure it addresses;
- the exact baseline and ablation;
- expected metric movement;
- data and GPU cost;
- novelty overlap with DiffusionDrive, DriveAnchor, WAC, WoTE, DriveSuprim, or
  DriveWLA;
- rollback path.

## Reopening conditions

The architecture may be reopened only if one of the following occurs:

1. S1 fails after the pre-registered anchored-flow runs, with no meaningful
   increase in mode coverage;
2. a paper published before the cutoff directly invalidates the novelty or
   baseline of the current method;
3. WOD-E2E rules, splits, or inference inputs materially change;
4. the frozen method cannot run within the available 2 x 32 GB budget and a
   documented alternative has evidence of comparable strength.

A new paper does not reopen the architecture merely by reporting a higher
benchmark score on a different dataset.

## Source and checkpoint isolation

- `src/qwen_drive/` remains byte-identical to the baseline tag.
- New code lives under `research/`.
- Released VLM and planner checkpoints are opened read-only.
- Training saves adapters and new heads, not a copy of the released base
  planner.
- The adapter manifest records base source hash, base planner hash, config, and
  anchor-bank hash.
- A compatibility sampler must reproduce the original sampler in
  standard-Gaussian, 10-step mode.

