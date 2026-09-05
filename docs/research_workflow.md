# Research workflow and evidence ledger

The Git repository holds versioned source, protocols, small results and writing
evidence. Cloud discussions generate questions and reviews; their decisions must
be recorded here before they become executable assumptions. Raw data and model
weights are separate artifacts referenced by immutable URLs and checksums.

## One iteration

1. Record a concrete question, falsifiable hypothesis, target estimand and failure
   criterion in a dated protocol. Separate exploration from confirmation.
2. Search primary papers and official package repositories. Record the exact
   source, dataset/version, license, input requirements and intended use in
   `qir_sources.md`. A promising package is not evidence that its weights or
   assumptions transfer to binary ecological rasters.
3. Download public data reproducibly and verify hashes. Document units, labels,
   missingness, retrospective covariates and acquisition times in a data card.
4. Compile events with censoring, lock source/validation/external splits and
   preprocess from source only. Never tune against external labels.
5. Implement a minimal model beside a strong common baseline. Verify leakage,
   censoring and invariance with focused tests; run a small end-to-end fixture.
6. Run the registered comparison. Save configuration, source/input hashes,
   predictions, metrics, calibration and failure cases. Commit selected small
   artifacts under `results_summary`; retain negative outcomes.
7. Write only claims supported by those files, with explicit limitations. A
   literature citation supports background; it cannot replace the project's own
   measurement. Record a stop/continue decision before the next iteration.

## Current writing evidence

- Historical v3.5–v3.7: real-data spatial-proxy and recovery audit. Consult
  [the archive provenance](archive/README.md) for exact recovered files and
  regenerated figures before citing an exact code version.
- QIR phase one: `results_summary/qir_phase1/REPORT.md` supports a retrospective
  two-site pilot, its negative/mixed outcome and implementation limitations.
- Unsupported: general tipping-point prediction; consistent temporal-model gain;
  transfer to another ecosystem; formal uncertainty coverage; operational
  historical warning. These remain questions, not paper conclusions.

## Immediate next protocol

The fixed-cohort spline comparison is recorded in
`results_summary/qir_phase1/historical_baseline/REPORT.md`. It exposed a failure:
Hellegat calibration pressure lies entirely outside the source training range,
and the fitted calibration offset reaches its search boundary. This comparison
cannot establish learned-representation superiority.

Preserve that failed comparison. First design a new exploratory spatial split
using source covariates to provide reasonable training/calibration overlap,
without selecting on target outcomes. Check overlap, optimization and calibration
boundaries before comparing models across multiple fixed seeds. Report both
directions and both proper losses, with block-level uncertainty and dependence
sensitivity. Previously inspected targets cannot become fresh confirmatory
holdouts. Audit time-varying covariate availability before calling any result
prospective. The QIR residual has not yet been trained on the spline baseline.

Only after this gate, select a second system with observed repeated disturbance
and recovery histories, retained failures to recover, and regional holdouts.
The source register contains candidates, not an already validated second dataset.
