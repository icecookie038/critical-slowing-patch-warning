# Experiment Plan v0.8: Dynamic Patch Indicators

## 1. Version Goal

The goal of v0.8 is to introduce and validate a new set of dynamic patch indicators for early warning of ecological critical transitions.

This version does not modify the v0.1–v0.7 classic patch baseline. It does not retrain machine learning models. It focuses only on indicator implementation and pilot visualization.

The v0.8 indicators include:

```text
PDSI: Patch-size Distribution Shift Index
DPCI: Dominant Patch Change / Growth Index
DPCR: Dominant Patch Collapse Rate
FAI: Fragmentation Acceleration Index
PSII: Patch-Structure Instability Index
```

---

## 2. Version Position

The current project version sequence is:

```text
v0.7 = classic patch baseline
v0.8 = dynamic patch indicators
v0.9 = feature group comparison
v1.0 = traditional early-warning signal baseline
v1.1 = lead time and feature importance
```

The v0.8 stage is an indicator development stage. It prepares the dynamic indicators for v0.9, where they will be used in machine learning feature-group experiments.

---

## 3. Files Added in v0.8

The main files added in v0.8 are:

```text
src/features/dynamic_patch_indicators.py
scripts/run_dynamic_indicator_pilot.py
docs/indicators_dynamic_patch.md
docs/experiment_plan_v0.8_dynamic_patch.md
```

The pilot outputs are:

```text
results_summary/dynamic_patch_pilot_metrics.csv
figures/dynamic_patch_pilot/dynamic_indicators_trend.png
figures/dynamic_patch_pilot/patch_state_summary.png
```

---

## 4. What v0.8 Does

v0.8 performs the following tasks:

```text
1. Implement dynamic patch indicators.
2. Verify that the indicators can be computed from spatial patch sequences.
3. Generate a pilot trend figure.
4. Check whether the indicators show reasonable temporal behavior.
5. Fix numerical instability in PSII normalization.
6. Document mathematical definitions and implementation rules.
```

---

## 5. What v0.8 Does Not Do

v0.8 does not perform the following tasks:

```text
1. It does not modify v0.1–v0.7 classic patch baseline.
2. It does not modify the real SEIR simulation module.
3. It does not modify the original data generation pipeline.
4. It does not retrain machine learning models.
5. It does not compare AUC, AUPRC, F1, or lead time.
6. It does not replace the old patch indicators.
```

Machine learning experiments will start in v0.9.

---

## 6. Indicator-Level Validation

The v0.8 pilot checks whether the dynamic indicators behave consistently with the expected SEIR expansion process.

Expected behavior in SEIR infection expansion:

```text
PDSI:
should respond to temporal shifts in patch-size distribution.

DPCI:
should respond to growth of the dominant infected patch.

DPCR:
should remain low in SEIR expansion because the infected dominant patch usually grows rather than collapses.

FAI:
should respond when fragmentation accelerates.

PSII:
should summarize instability from PDSI, DPCI, and FAI without numerical explosion.
```

---

## 7. Pilot Script

The pilot script is:

```text
scripts/run_dynamic_indicator_pilot.py
```

The default command is:

```powershell
.\.venv\Scripts\python.exe -u scripts\run_dynamic_indicator_pilot.py
```

The script currently uses a synthetic SEIR-like expanding patch sequence for testing.

This synthetic sequence is only used for indicator verification and visualization. It is not the formal SEIR training dataset.

---

## 8. Pilot Outputs

The pilot script generates:

```text
results_summary/dynamic_patch_pilot_metrics.csv
figures/dynamic_patch_pilot/dynamic_indicators_trend.png
figures/dynamic_patch_pilot/patch_state_summary.png
```

The CSV file stores raw indicator values.

The trend figure shows:

```text
PDSI
DPCI
DPCR
FAI
PSII
critical time reference line
```

The patch-state summary figure shows:

```text
patch count
largest patch area
total patch area
largest patch ratio
```

---

## 9. PSII Normalization Rule

PSII must be normalized using only the baseline or training period.

The whole time series must not be used for PSII z-score fitting, because doing so would introduce information leakage.

The implemented PSII rule includes:

```text
1. baseline-only mean and standard deviation
2. minimum standard deviation to avoid numerical explosion
3. positive z-score accumulation
4. z-score clipping to avoid domination by one component
```

This is important for later v0.9 machine learning experiments.

---

## 10. SEIR Dynamic Feature Definition

For SEIR infection expansion, the recommended dynamic feature group is:

```text
dynamic_patch_only = PDSI + DPCI + FAI + PSII
```

DPCR is computed for completeness but is not used as the main SEIR dynamic indicator.

Reason:

```text
SEIR infected patches usually expand.
DPCI captures dominant infected patch growth.
DPCR is more suitable for collapse or degradation systems.
```

---

## 11. Future v0.9 Extension

In v0.9, the dynamic indicators will be connected to machine learning feature-group experiments.

The planned feature groups are:

```text
image_only
classic_patch_only
dynamic_patch_only
classic_dynamic_patch
full
```

The planned experimental scale is:

```text
5 seeds × 5 feature groups = 25 training runs
```

The target comparison table will include:

```text
AUC
AUPRC
F1
Lead time
mean ± std across seeds
```

---

## 12. Success Criteria for v0.8

v0.8 is considered successful if:

```text
1. dynamic_patch_indicators.py runs without errors.
2. run_dynamic_indicator_pilot.py runs without errors.
3. dynamic_patch_pilot_metrics.csv is generated.
4. dynamic_indicators_trend.png is generated.
5. patch_state_summary.png is generated.
6. PSII no longer shows numerical explosion.
7. DPCI is more active than DPCR in SEIR expansion.
8. The mathematical definitions are documented.
```

---

## 13. Current Status

Current v0.8 status:

```text
Dynamic indicator implementation: completed
Pilot script: completed
Pilot figure: completed
PSII numerical stabilization: completed
Documentation: in progress
Machine learning training: not started
```

The next version after v0.8 is v0.9 feature group comparison.