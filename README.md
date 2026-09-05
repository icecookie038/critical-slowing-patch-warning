# Ecological slowing-down benchmarks with spatial patch explanation

**科研入口：** [当前工作与下一步](docs/research_start_here_zh.md) ·
[历史归档与校验](docs/archive/README.md) ·
[QIR 原型运行](README-QIR.md) ·
[论文内部草稿](manuscript/qir_pilot_draft_zh.md)。
以下保留 v3.7 研究背景；QIR 先导试验及其基线校准问题请以科研入口中的最新记录为准。

This repository studies ecological critical slowing down using **direct recovery
after disturbance** as the primary resilience measurement. Vegetation patches
are retained only as a secondary description of where slow recovery is spatially
organized.

The current real-data route replaces the earlier claim that patch metrics can
serve as general-purpose critical-slowing-down detectors. Version 3.6 adds three
published experimental datasets and audits observation precision. Version 3.7
adds a leakage-safe, right-censored event-level survival model for the tidal
marshes. The spatial SEIR and synthetic vegetation experiments remain as legacy
method development, not as the main empirical evidence.

## Current research question (v3.7)

1. Does the right-censored recovery hazard decline along an increasing ecological
   stress gradient?
2. Which observation-design limitations cause published early-warning methods to
   lose detection performance?
3. After controlling for stress, is the remaining local recovery field spatially
   consistent across analysis scales?
4. Can patch configuration describe that field? Patch variables are not used to
   define a CSD label, alarm, or composite PWSI score.
5. Do patch covariates measured before an outcome improve event-level recovery
   probabilities in whole-site external validation?

## Verified real-data result

The analysis reproduces the two tidal-marsh sites in van Belzen et al. (2017):

| Site | Recovery bins | Slope per inundation percentage point (day^-1) | R² | P |
| --- | ---: | ---: | ---: | ---: |
| Hellegat | 35 | -3.58054e-05 | 0.8278 | 3.75e-14 |
| Paulina | 12 | -2.33593e-05 | 0.5950 | 0.00330 |

The maximum relative difference from the published recovery-rate files is below
`3.4e-05`. Stress-adjusted local recovery fields are positively consistent across
32, 64, and 128 m grids. At the pre-specified 64 m scale, however, patch features
do **not** improve held-out prediction in both site-to-site transfer directions.
Their independent contribution is therefore not established, and their role
remains descriptive.

## Published-data precision benchmark

Three additional published datasets are now included:

- Clements & Ozgul (2016): a daily predator–prey collapse experiment;
- Dai et al. (2015): 48 parallel yeast populations under two deteriorating
  environmental drivers;
- Rindi et al. (2018): a two-year, 5 × 30-cell macroalgal field experiment.

The benchmark shows that sampling precision contributes to failures but is not
their sole cause. The Clements trait composite falls from a 62.5% signal rate at
daily sampling to 6.25% at two-day sampling. In the macroalgal maps, retaining
only 25% of cells leaves spatial CV highly similar to the full map (median rank
similarity 0.973) but destabilizes component density (0.128). In the salt marsh,
the native 0.25 m maps do not rescue the cross-site patch increment; an
exploratory 5 m × 128 m candidate is model-sensitive and has only 10/15
independent blocks, so it is reserved for pre-registered validation on new data.

## Time-safe event-level result

Every observed vegetation loss is followed through the irregular aerial-survey
intervals until first recovery or right censoring. Identical event histories are
collapsed into a grouped binomial likelihood, which retains 1,488,244 Hellegat
and 944,293 Paulina person-intervals. Patch variables use only the image before
loss or the image available at the start of a risk interval.

On common inundation support, adding pre-loss or time-varying patch variables
worsens both Brier loss and logarithmic loss in both site-transfer directions.
Time-varying patches can improve interval ranking AUC, but probability accuracy
and calibration deteriorate. The independent patch increment remains **not
established**; patches are retained as descriptive spatial explanation only.

## Reproduce the analysis

Install dependencies:

```bash
pip install -r requirements.txt
```

Download and checksum-verify the public data bundle:

```bash
python scripts/download_tidal_marsh_data.py
```

Download and checksum-verify the published v3.6 benchmark data:

```bash
python scripts/download_published_csd_benchmarks.py
```

Run the complete real-data analysis:

```bash
python scripts/analysis/run_v3_5_local_recovery_spatial_explanation.py
```

Run the published-data benchmark and the locked salt-marsh precision audit:

```bash
python scripts/analysis/run_v3_6_published_csd_benchmark.py
python scripts/analysis/run_v3_6_salt_marsh_precision_audit.py
```

Run the time-safe event-level survival analysis:

```bash
python scripts/analysis/run_v3_7_time_safe_event_survival.py
```

Run tests:

```bash
python -m unittest discover -s tests -v
```

Generated raw data and results are intentionally excluded from Git. Default
outputs are written to `results/v3_5_local_recovery_spatial_explanation/` and
`results/v3_6_published_csd_benchmark/`; v3.7 outputs are written to
`results/v3_7_time_safe_event_survival/`.

## Analysis design

- Irregular observation intervals are retained.
- Unrecovered losses are right-censored rather than discarded.
- Recovery hazards are estimated from recovered-event counts and exposure time.
- The stress response is fitted with a regularized spline Poisson rate model.
- Spatial blocks, rather than pixels, define held-out validation groups.
- Cross-site transfer is evaluated in both directions without fitting on the test
  site.
- Patch associations are tested only after stress adjustment, with block-level
  permutation tests and within-site/scale false-discovery-rate correction.
- The pre-specified primary block width is 64 m; 32 and 128 m are sensitivity
  analyses.
- Event-level patch covariates use only maps available at or before the risk
  interval; later recovery maps cannot enter predictors.
- Complementary-log-log hazards use the actual interval length as an offset.
- Spatial-block bootstrap intervals preserve within-block dependence.

## Scientific scope

The present result supports a **local-to-landscape direct-recovery framework**.
It does not show that patch morphology predicts all ecological transitions, nor
does it turn spatial structure into a universal early-warning indicator. The
time-safe event model is now complete. Generalization beyond the published
benchmark still requires a new independent ecological disturbance-recovery
dataset with high-frequency two-dimensional observations.

The current decision and manuscript guidance are documented in
[`docs/v3_7_time_safe_event_survival.md`](docs/v3_7_time_safe_event_survival.md).

## Data and reference

- van Belzen, J. et al. (2017), [Vegetation recovery in tidal marshes reveals
  critical slowing down under increased inundation](https://doi.org/10.1038/ncomms15811),
  *Nature Communications* 8, 15811.
- Public data bundle: [Dryad DOI 10.5061/dryad.7174h (Zenodo mirror)](https://zenodo.org/records/4998258).
- Full v3.6 source registry and checksums:
  [`docs/data_sources/published_csd_benchmarks.md`](docs/data_sources/published_csd_benchmarks.md).
