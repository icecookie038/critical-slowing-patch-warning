# Experiment Plan v0.9: SEIR Feature Group Comparison

## 1. Version Goal

The goal of v0.9 is to evaluate whether the newly introduced dynamic patch indicators improve early-warning prediction under the SEIR spatial spreading model.

This version compares five feature groups:

```text
image_only
classic_patch_only
dynamic_patch_only
classic_dynamic_patch
full
```

The v0.9 experiments are based on the same SEIR simulation framework used in v0.7. The data generation model is not replaced in this version.

---

## 2. Background

Versions v0.1–v0.7 established the classic patch baseline.

The original workflow was:

```text
spatial infection field
→ binary infected region
→ connected infected patches
→ patch feature extraction
→ critical transition risk prediction
```

Version v0.8 introduced the first-generation dynamic patch indicators:

```text
PDSI: Patch-size Distribution Shift Index
DPCI: Dominant Patch Change / Growth Index
DPCR: Dominant Patch Collapse Rate
FAI: Fragmentation Acceleration Index
PSII: Patch-Structure Instability Index
```

For SEIR infection expansion, the main dynamic indicators used in v0.9 are:

```text
PDSI
DPCI
FAI
PSII
```

DPCR is computed for completeness, but it is not used as the main SEIR expansion feature because SEIR infected patches usually expand rather than collapse.

---

## 3. Feature Groups

### 3.1 image_only

This group uses only spatial image sequences.

It tests whether the spatial infection field alone is sufficient for early-warning prediction.

### 3.2 classic_patch_only

This group uses the classic patch feature sequence from the v0.1–v0.7 baseline.

It represents the main classic patch baseline.

### 3.3 dynamic_patch_only

This group uses only the v0.8 dynamic patch indicators.

For SEIR expansion, the dynamic features are:

```text
PDSI
DPCI
FAI
PSII
```

This group tests whether the new dynamic patch indicators alone can predict critical transition risk.

### 3.4 classic_dynamic_patch

This group combines the classic patch features and the dynamic patch indicators.

It tests whether dynamic patch indicators provide additional predictive information beyond the classic patch baseline.

### 3.5 full

This group uses spatial image sequences, classic patch features, and dynamic patch indicators together.

It represents the most complete feature setting in v0.9.

---

## 4. Experiment Scale

The v0.9 experiment uses:

```text
5 seeds × 5 feature groups = 25 runs
```

Seeds:

```text
42
123
2026
3407
7777
```

Feature groups:

```text
image_only
classic_patch_only
dynamic_patch_only
classic_dynamic_patch
full
```

Main experimental setting:

```text
num_sims = 300
grid = 64
sim_steps = 200
horizon = 30
reg_weight = 0.0
weight_decay = 0.001
```

Small pilot runs such as:

```text
sims20_L32_steps80_h20
```

are excluded from the v0.9 summary tables.

---

## 5. Implementation

The feature group definitions are implemented in:

```text
src/features/feature_groups.py
```

The deep learning model is trained by:

```text
scripts/train_deep_model.py
```

The v0.9 batch runner is:

```text
scripts/run_seir_feature_groups.py
```

The result collection script is:

```text
scripts/collect_feature_group_results.py
```

The paper table export script is:

```text
scripts/export_feature_group_tables.py
```

---

## 6. Output Files

The raw training outputs are stored under:

```text
scripts/results/
```

These raw result folders are not committed to GitHub.

The collected result summaries are:

```text
results_summary/feature_group_summary_all.csv
results_summary/feature_group_summary_grouped.csv
```

The exported paper tables are:

```text
docs/paper_tables/feature_group_comparison_table.csv
docs/paper_tables/feature_group_comparison_table.md
docs/paper_tables/feature_group_comparison_table.tex
```

---

## 7. Main Results

The v0.9 feature group comparison shows the following overall pattern:

```text
classic_patch_only ≈ classic_dynamic_patch ≈ full > image_only >> dynamic_patch_only
```

The first-generation dynamic patch indicators alone show limited standalone predictive performance.

However, combining dynamic patch indicators with classic patch features maintains performance comparable to the classic patch baseline and the full model.

This suggests that the classic patch features already capture most of the discriminative state information in the current SEIR setting.

---

## 8. Interpretation

The v0.8 dynamic patch indicators mainly describe short-term temporal changes in patch structure, including:

```text
patch-size distribution shift
dominant patch growth
fragmentation acceleration
composite structural instability
```

These indicators provide interpretable temporal-structural information, but they do not substantially improve AUC over the classic patch baseline in the current SEIR setting.

A likely reason is that the current SEIR simulation produces relatively smooth spatial expansion, while the classic patch features already contain strong state information such as patch area, dominant patch structure, spatial aggregation, and temporal trend.

Therefore, the v0.8 dynamic indicators are more suitable for mechanistic interpretation and potential intervention-window analysis than for replacing the classic patch features as standalone predictors.

---

## 9. Current Conclusion

Version v0.9 successfully connects dynamic patch indicators to the deep learning pipeline and completes a five-group feature comparison.

The main conclusion is:

```text
Dynamic patch v1 alone is not sufficient to outperform the classic patch baseline.
Classic + dynamic patch features maintain comparable performance to the full model.
The first-generation dynamic patch indicators provide additional interpretability but limited standalone predictive gain in the current SEIR setting.
```

---

## 10. Next Diagnostic Direction

The next optional diagnostic version is:

```text
v0.9.1 = dynamic patch v2 indicators
```

The planned second-generation dynamic patch indicators are:

```text
BCI: Boundary Complexity Index
FPV: Front Propagation Velocity
PMR: Patch Merging Rate
GCGR: Giant Component Growth Rate
PPI: Percolation Proximity Index
```

These indicators are more directly related to spatial spreading, front propagation, patch merging, and connectivity transition.

They may be more suitable for improving predictive performance in SEIR-like spreading systems.

---

## 11. Version Position

The version relationship is:

```text
v0.7 = classic patch baseline
v0.8 = dynamic patch indicators v1
v0.9 = feature group comparison
v0.9.1 = dynamic patch indicators v2 diagnostic
v1.0 = traditional early-warning signal baseline
v1.1 = lead time and feature importance
```

The current v0.9 version should be closed before starting v0.9.1.