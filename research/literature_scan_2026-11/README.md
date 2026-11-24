# Qwen-Drive bounded literature scan, 2026-11-04

This directory records a deliberately bounded literature review for the thesis
line:

```text
diagnose flow-planner mode collapse
  -> anchored / truncated flow
  -> action-conditioned compact BEV world model + critic
  -> WOD-E2E evaluation
```

The scan closes on **2026-11-04**. The title/abstract candidate pool contains 32
papers; 18 are retained for structured extraction, with a hard limit of 20.
Nine are scheduled for deep reading. Papers published or revised after the
cutoff do not change the frozen architecture unless one of the reopening
conditions in `ARCHITECTURE_FREEZE.md` is met.

## Deliverables

- `ARCHITECTURE_FREEZE.md`: current architecture contract, gates, and reopening
  conditions.
- `paper_matrix.tsv`: structured extraction for all 18 retained papers.
- `scan_summary.md`: decisions and implications for the thesis.
- `reading_plan.md`: the 9 papers to read deeply, in order, and why.
- `screening_log.md`: candidate exclusions and their reasons.

## Scope and limitations

- The scan emphasizes trajectory diversity, candidate scoring, action-conditioned
  world models, and VLA planners.
- The local PDFs are the primary evidence source. Supplementary PDFs were
  downloaded to `cache/` from the arXiv URLs recorded in `paper_matrix.tsv`.
- A preprint is not treated as peer-reviewed evidence merely because it is
  available on arXiv.
- Numbers are copied from the authors' papers. They have not been independently
  reproduced unless explicitly stated.
- "Camera-only" refers to inference input. Training-time LiDAR or privileged
  supervision is recorded separately when the paper states it.
- This scan does not modify `src/qwen_drive/` or any released checkpoint.

