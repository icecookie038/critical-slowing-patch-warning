# Phase-one pilot record — 2026-09-05

Status: implementation and first real-data pilot completed; scientific efficacy
unconfirmed. The synthetic fixture is an engineering check, not ecological evidence.

The retrospective marsh pilot contains 230 loss episodes and 676 observed risk
intervals. Whole sites are external tests in both directions. Source training and
calibration use disjoint blocks with contained image footprints. There are 8
event-bearing Hellegat blocks and 10 Paulina blocks. Seed 2026, 25 epochs and the
geometry-dependent pilot configuration were fixed before viewing these metrics.

After source-only intercept calibration, on all external intervals without mask
degradation, Hellegat-to-Paulina baseline Brier/log loss are 0.091386/0.314547;
QIR gives 0.091386/0.314547. Paulina-to-Hellegat baseline gives 0.116988/0.403959;
QIR gives 0.115988/0.400544. A single-date spatial residual is worse in the first
direction (0.101510/0.351859) and better in the reverse (0.103953/0.331968).
These are observed-interval, equally block-weighted metrics, not integrated
survival scores. Full uncalibrated, calibrated, support-subset and degradation
results are retained in `marsh/metrics.csv`.

Decision: no stable two-direction benefit from temporal QIR has been established.
Do not select a favorable direction, metric or post-hoc architecture as proof of
transferability. The temporal residual in the first direction effectively produces
the same ranking and calibrated predictions as M0. Fixed-epoch optimization and
the small neural M0 need stronger convergence and baseline checks.

Source calibration does not guarantee improvement on the external site. The
support diagnostic accepts 100% of Paulina when trained on Hellegat and about
49.84% of Hellegat when trained on Paulina. This is a source-covariate distance
rule, not a validated uncertainty bound or coverage guarantee.

The publication's static masks and 2005–2010 inundation map make this a
retrospective association pilot. The code excludes future outcome imagery, but
that alone cannot establish historical availability of all covariates. Only
first observed recovery is identified across long acquisition gaps. Censoring and
hidden transitions require further investigation.

Verification: 15 focused tests passed; the final synthetic smoke experiment
completed all four models in both directions; the real-data pilot also completed
all eight fits. Run manifests preserve input/source SHA256 hashes, configuration,
versions, split IDs and training losses. `dirty_worktree: true` records that the
experiment ran before its source commit; the per-file hashes identify the actual
code. Raw data and binary model checkpoints remain outside version control.

Next: restore the historical spline model and evaluate on identical event IDs,
add source-only convergence and multi-seed checks, then audit an independent
system for repeated 2-D observations and valid recovery/censoring definitions.
No second-system result, calibrated event-time coverage or publication-ready
novelty claim exists yet.
