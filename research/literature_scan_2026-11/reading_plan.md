# Deep-reading order

Read nine papers in this order. Each read must end with one concrete experiment
or one explicit rejection.

1. **DiffusionDrive** — extract the exact anchored-Gaussian and truncated
   schedule; map DDPM time to Qwen-Drive's flow time.
2. **WAC** — extract candidate-conditioned critic, asymmetric attention, and
   the critic-gradient ablation; draft the S2 control matrix.
3. **WoTE** — extract the compact BEV state, action-conditioned rollout, reward
   model, and candidate-count/latency ablations.
4. **DriveAnchor** — identify overlap with anchored flow and design the novelty
   and baseline comparison.
5. **DriveSuprim** — extract the oracle top-K study and hard-negative selection
   lessons; define the simple-selector baseline.
6. **MGF** — formalize why a multimodal prior is preferable to a standard
   Gaussian prior and choose diversity metrics.
7. **Qwen-Drive** — document the released sampler, flow parameterization, and
   the paper's qualitative candidate-concentration statement.
8. **DriveWLA** — extract world-expert and critic-gradient-free ablations; use
   it as a strong comparison, not an architecture template.
9. **GoalFlow** — assess goal-conditioned flow as the main pre-registered
   fallback if motion anchors do not create useful coverage.

After these nine, skim the remaining matrix entries only for baseline numbers,
compute constraints, and review-risk language.

