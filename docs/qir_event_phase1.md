# QIR Event Lite phase one

This branch starts the quality-aware irregular raster event-time direction discussed on
2026-09-04. Its deliverable is an executable pilot with a traceable protocol. It does
not establish a new general-purpose early warning indicator, ecological tipping-point
prediction, or cross-system improvement.

## What is implemented

- A typed binary raster interface with explicit masks, timestamps, environmental
  arrays, site names and physical pixel size.
- Loss episodes from observed adjacent 1-to-0 transitions, followed until first
  observed recovery, end of observation or first missing endpoint. Right-censored
  episodes remain in the ledger. A loss on the final date has no follow-up exposure.
- No bridging of missing outcomes. Hidden multiple transitions inside an acquisition
  gap are not identifiable; the endpoint is first **observed** recovery.
- Prediction histories terminate at the start of each risk interval. The interval
  endpoint is used only for the recovery label, never for image input or quality.
- Centered patches defined in metres; out-of-bounds and missing pixels have separate
  masks. Patches are sampled onto a fixed 16-by-16 grid with nearest-neighbour sampling.
  This is a lossy pilot representation, not a validated multiscale resampling method.
- A small masked CNN, quality-dependent scale gate, forward-only GRU with log time
  gaps, and additive hazard residual. No batch normalization across examples or dates.
- Complementary log-log interval likelihood with actual elapsed years:
  `p = 1 - exp(-delta_t * exp(eta))`. Exposure is evaluated in log space with numerical
  bounds; extreme log integrated hazards outside [-30,20] saturate intentionally.
- Environmental/waiting-time MLP (M0), quality-only residual (MQ), single-date spatial
  residual (M2), and causal temporal residual (QIR). The M0 component is frozen when
  each residual is trained. This small MLP is **not** the historical v3.7 spline model.
- Whole-site external evaluation in both directions; disjoint source training and
  calibration blocks. Entire image footprints stay inside assigned blocks.
- Equal total weight per block for fitting and metrics; fixed epochs, seed and
  architecture without target-label tuning.
- A source-calibration-only hazard intercept, kNN extrapolation diagnostic, and
  separate supported/unsupported metrics. These provide no conformal coverage guarantee.
- A fixed 25% random history-mask stress test. Outcome labels remain unchanged.
- Run manifests, input and source hashes, split IDs, predictions, episode ledgers and
  model/scaler checkpoints. Checkpoints and raw data stay outside Git.

## Data and availability boundaries

The real pilot uses the original van Belzen et al. data bundle at
https://zenodo.org/records/4998258 (DOI 10.5061/dryad.7174h), associated with
https://doi.org/10.1038/ncomms15811.

Published ZIP MD5: `eb545ba9e8e68e41c7f2988157e447c7`.
The adapter verifies it before reading any raster. The source TIFFs lack a meaningful
geotransform; 0.25 m pixel spacing follows the author's MATLAB `dx=0.25`. Hellegat uses
mask band 1 equal to zero; Paulina uses band 1 greater than zero, following the supplied
`StatisticalResilienceIndicators.m`. The pilot retains every second source pixel
(0.5 m regular subsampling), not area-averaged coarse graining.

**Retrospective, not an operational backtest:** the publication's inundation map is
derived from water levels measured in 2005-2010 and its study mask is static. Those
products are not verified as available at the early historical prediction dates.
The adapter repeats the covariate as a retrospective site attribute. Future outcome
images are excluded by code, but this does not make every covariate operationally
time-safe. A prospective analysis needs dated environmental and quality products.

The small Hellegat extent cannot provide four separated source blocks with 128 m
image footprints under this buffer design. Before viewing model results we fixed the
real pilot to 16/32/64 m footprints within 128 m blocks, 32 sampled coordinates per
block, seed 2026, and 25 epochs. Full-scale experiments must revisit spatial correlation
range using source data, not choose a successful scale on the external test site.

The pilot contains only 8 and 10 eligible event-bearing blocks across the two sites.
It cannot support strong uncertainty, transfer or second-system claims. All reported
losses are block-weighted **observed-interval** metrics. They are not integrated
survival Brier scores or uncensored event-time accuracy.

## Reproduce

From the repository root, in a dedicated Python environment:

```bash
python -m pip install -r requirements-qir.txt
python -m pytest tests/test_qir_event.py -q
python -m src.qir_event.experiment --output results/qir_smoke
python -m src.qir_event.marsh_adapter /path/to/marsh_ncomms.zip --output data/processed/qir_marsh
python -m src.qir_event.experiment --config configs/qir_marsh_pilot.json --input data/processed/qir_marsh/Hellegat.npz data/processed/qir_marsh/Paulina.npz --output results/qir_marsh_pilot
```

Without `--input`, the experiment creates synthetic engineering fixtures only; it
does not train a surrogate simulator for deployment. Source-only fitting occurs
separately in each direction. The tested environment is recorded in
`requirements-qir-tested.txt`; the small CSV/JSON run summaries are in
`results_summary/qir_phase1/`.

## Next milestones and gates

1. **Archive and audit historical v3.7.** Preserve recovered source and negative
   results, identify the original commit, run the historical tests. Restore data and
   result provenance separately from the new prototype.
2. **Strong common baseline.** Reuse the historical spline cloglog model and compare
   all models on identical buffered event IDs and source-only preprocessing. Add
   convergence diagnostics, multiple source-only seeds and block bootstrap. More
   baseline optimization is not a claim of spatial representation improvement.
3. **Data availability audit.** Establish timestamped environment/masks and separate
   label quality from historical input quality. Characterize informative missingness
   and interval censoring. Keep retrospective and prospective estimands separate.
4. **Independent system.** Obtain repeated 2-D disturbance/recovery observations,
   defined persistent recovery, un-recovered objects and whole-region holdouts.
   Landslide polygon inventories alone are insufficient. See the source register.
5. **Quality training.** Add temporal deletion, registration and resolution operators,
   training-only augmentation and multi-seed ensembles. Test empirical uncertainty
   against errors; lower quality does not mathematically guarantee higher entropy
   for every individual example, so do not force that as a universal law.
6. **Calibration and transfer.** Add source-validation-selected support rules and
   spatial dependence sensitivity. Compare coverage/error against rejection fraction.
   Claim finite-sample coverage only under justified assumptions and a tested method.
7. **Paper.** Each numerical claim must identify an immutable input version, run,
   model and result file. Freeze external evaluation choices before a confirmatory
   rerun and before seeing the second-system labels.

Stop adding complexity if the learned residual does not improve both Brier and log
loss on independently held-out data relative to a properly fitted common baseline.
Retain negative results. New direction is active; scientific validation is unfinished.

