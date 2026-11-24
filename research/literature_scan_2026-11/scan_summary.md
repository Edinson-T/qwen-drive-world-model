# Scan summary

## Decision

Keep the current thesis narrative and freeze the architecture until the S1
anchored-flow pilot produces evidence.

The retained literature does not justify adding geometry, MoE, goal points, or
a new video world model now. It does justify four focused actions:

1. implement anchored/truncated flow as the primary S1 intervention;
2. evaluate diversity and deployable selection separately;
3. preserve WAC's critic-gradient experiment when S2 starts;
4. treat DriveAnchor and DriveSuprim as the strongest novelty and baseline
   threats.

## Evidence that directly supports S1

- DiffusionDrive names mode collapse, reports a diversity increase from 11% to
  74%, and uses 20 K-Means anchors with two truncated diffusion steps.
- MGF independently argues that a symmetric standard Gaussian prior suppresses
  minority trajectory modes; its mixed Gaussian prior uses K=8 clusters.
- DriveTransformer reports a multi-mode versus single-mode planning ablation,
  supporting the need to model alternatives explicitly.
- Qwen-Drive itself reports a small average/minimum ADE gap on PAI-AV and calls
  its candidates "concentrated around similar motions."

## Evidence that directly supports S2

- WoTE passes each candidate through a recurrent BEV world model before reward
  scoring and reports 88.3 PDMS with 18.7 ms candidate evaluation latency.
- WAC reports 93.4 PDMS; detaching critic gradients from future features drops
  PDMS to 92.3. This is the clearest ablation for preserving critic-to-world
  gradient flow.
- DriveWLA reports a 4.6-point PDMS drop when the World Expert is removed and a
  2.8-point drop when it is frozen during imitation training.

These results support the S2 hypothesis, but they do not prove that Qwen-Drive's
collapsed candidates can be fixed by a selector. That is why S2 remains blocked
on S1 coverage.

## Strongest threats

### DriveAnchor

DriveAnchor already uses recorded trajectories as locally perturbed flow starts
and studies low-budget decoding. The thesis cannot claim novelty for the general
idea of anchor-initialized flow. The defensible claim is narrower: diagnose and
repair mode collapse in a VLM-conditioned flow planner, then test whether the
recovered coverage produces real WOD-E2E selection gains under camera-only
inference.

### DriveSuprim

DriveSuprim shows that precise selection among near-optimal candidates matters.
If a simple learned score head recovers most of the gain, replacing it with a
more complex world model will not be justified. S2 must therefore include a
plain score-head control, not only random and WM baselines.

### DriveWLA and WAC

Both papers provide evidence for future/world representations in planning. The
contribution cannot be described as "attach a world model to Qwen-Drive."
The contribution must connect the diagnosis, the generator repair, and the
conversion of coverage into deployable selection.

## Explicit non-actions

- Do not add DriveMoE or action-skill experts before S1 fails.
- Do not add GoalFlow's goal vocabulary as a second generator.
- Do not attach DVGT-2, VLGA, or GeoWAM to the planner.
- Do not train a video world model.
- Do not replace the released planner with DriveWLA or Alpamayo-R1.
- Do not claim that Qwen-Drive first identified mode collapse; cite
  DiffusionDrive for the term and Qwen-Drive for the concentration observation.

## Next experimental checkpoint

Run the four S1 configurations and produce one table containing diversity,
oracle quality, typical-candidate quality, and selection headroom. The result
decides whether the project proceeds to S2 or uses one pre-registered fallback.

