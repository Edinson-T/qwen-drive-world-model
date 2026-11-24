# Screening log

## Bounded search protocol

- Cutoff: 2026-11-04.
- Seed set: Qwen-Drive, DiffusionDrive, MGF, DriveAnchor, WoTE, WAC, DriveWLA,
  GeoWAM, and the local related-work PDFs.
- Expansion: backward/forward citations and targeted searches for autonomous
  driving VLA, world models, multimodal trajectory generation, flow matching,
  and candidate selection.
- Candidate pool: 32 papers after de-duplication.
- Hard cap: 20 deep extractions. Final retained count: 18.
- Peer-reviewed work and high-impact preprints are marked separately.
- A title/abstract-only exclusion is not treated as a technical conclusion.

## Retained

The 18 retained papers are listed in `paper_matrix.tsv`. Priority was based on
one of three direct links:

- a measured failure mode matching H1/H2/H3;
- a component needed for the frozen S1/S2 experiment;
- a credible reviewer baseline or novelty threat.

## Screened out

The following were kept in the candidate log but not deep-extracted in this
round. Reasons are title/abstract level only:

- DriveVLA-W0, DriveWAM, SimWAM, DriveVA, DriveWorld-VLA: future/video world
  modeling, often larger than the current budget and not directly about
  candidate coverage.
- WCog-VLA, LatentVLA, AutoVLA, Drive-JEPA: VLA or reasoning alternatives that
  do not directly answer the generator-collapse/selection question.
- World4Drive, Epona: useful world-model comparisons but secondary to WoTE/WAC
  for the frozen S2 design.
- Hydra-MDP++, Drive-R1, Senna-2: highly related selector or reasoning follow-ups
  that should be checked at the final baseline freeze.
- DriveLM, ORION, OccWorld: downloaded while constructing the candidate pool,
  but omitted from the final 18 due to the hard cap. They remain useful for
  later related-work work, VQA baselines, or alternative world-state designs.

## Reopen rule for this log

Do not extend the scan merely because new preprints appear. Re-run only when
the architecture freeze is reopened under its recorded conditions, or when the
thesis enters its final baseline-freeze phase.

