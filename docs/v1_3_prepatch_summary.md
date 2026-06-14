# v1.3 Pre-patch Spatial Organization Indicators Summary

## 1. Purpose

The purpose of v1.3 is to incorporate pre-patch spatial organization indicators suggested by the advisor.

The main idea is to detect early spatial organization before visible patch structures become dominant. This extends the paper from visible patch-based warning to a broader spatial organization early-warning framework.

The v1.3 indicators are designed to capture:

```text
1. local synchronization
2. spatial connectivity
3. boundary rigidity
4. dominant mode locking
```

This stage builds on the v1.2 event-based label-fix dataset.

---

## 2. Files Added

Two new files were added:

```text
prepatch_indicators.py
build_v1_3_prepatch_dataset.py
```

### prepatch_indicators.py

This file computes the first version of pre-patch spatial organization indicators.

Implemented indicators:

```text
local_neighbor_corr_mean
local_neighbor_corr_max
sync_edge_ratio
moran_i
geary_c
high_state_component_ratio
gradient_entropy
boundary_sharpness
gradient_top10_mean
svd_mode1_energy_ratio
svd_spectral_gap
svd_mode1_ac1
dominant_mode_stability
dominant_mode_localization
```

The current implementation contains 14 pre-patch indicators.

### build_v1_3_prepatch_dataset.py

This script reads the v1.2 dataset and generates v1.3 datasets:

```text
data/processed/v1_3_prepatch/seir_v1_3_prepatch_h30_seed42.npz
data/processed/v1_3_prepatch/seir_v1_3_prepatch_only_h30_seed42.npz
data/processed/v1_3_prepatch/seir_v1_3_visible_prepatch_h30_seed42.npz
```

The `.npz` files are generated data and should not be committed to GitHub.

---

## 3. Dataset Design

Three dataset variants were generated.

### 3.1 Full prepatch dataset

```text
seir_v1_3_prepatch_h30_seed42.npz
```

This file keeps:

```text
X_img
X_patch_original
X_prepatch
X_patch_combined
y_risk
y_remaining
sim_id
time_idx
critical_time
```

### 3.2 Prepatch-only dataset

```text
seir_v1_3_prepatch_only_h30_seed42.npz
```

In this file:

```text
X_patch = X_prepatch
```

This allows `train_patch_baselines.py` to directly train tabular models using only pre-patch indicators.

### 3.3 Visible patch + prepatch dataset

```text
seir_v1_3_visible_prepatch_h30_seed42.npz
```

In this file:

```text
X_patch = concat(original visible patch indicators, repeated prepatch indicators)
```

This allows direct comparison between visible patch features and visible patch + prepatch features.

---

## 4. Classification Results

### 4.1 Prepatch-only

The prepatch-only dataset achieved strong classification performance.

The best models reached approximately:

```text
AUC ≈ 0.93–0.94
F1 ≈ 0.83–0.84
```

This indicates that pre-patch spatial organization indicators alone contain useful early-warning information.

### 4.2 Visible patch + prepatch

The visible patch + prepatch dataset achieved stronger classification performance.

The best result was approximately:

```text
RandomForest:
AUC ≈ 0.963
AUPRC ≈ 0.927
F1 ≈ 0.865
```

This is slightly higher than the previous v1.2 visible patch RandomForest baseline.

---

## 5. Strict First-Alarm Results

All models in the comparison achieved:

```text
valid_window_alarm_rate = 1.0
miss_rate = 0.0
late_alarm_rate = 0.0
```

Therefore, the main comparison is based on:

```text
valid_first_alarm_lead_mean
valid_first_alarm_lead_median
prewindow_alarm_rate
```

---

## 6. Overall Comparison

| Group | Model | Valid alarm rate | Miss rate | Lead mean | Lead median | Lead Q25 | Lead Q75 | Pre-window alarm rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| v1.2 visible patch stats | HistGradientBoosting | 1.0 | 0.0 | 28.48 | 30.0 | 27.5 | 30.0 | 0.613 |
| v1.3 visible prepatch | HistGradientBoosting | 1.0 | 0.0 | 28.03 | 29.0 | 27.5 | 30.0 | 0.548 |
| v1.3 visible prepatch | ExtraTrees | 1.0 | 0.0 | 27.74 | 30.0 | 27.0 | 30.0 | 0.484 |
| v1.3 visible prepatch | LogisticRegression | 1.0 | 0.0 | 27.61 | 29.0 | 27.5 | 30.0 | 0.581 |
| v1.3 prepatch only | LogisticRegression | 1.0 | 0.0 | 27.32 | 29.0 | 26.5 | 30.0 | 0.581 |
| v1.2 visible patch stats | MLP | 1.0 | 0.0 | 27.32 | 30.0 | 26.5 | 30.0 | 0.581 |
| v1.2 visible patch stats | ExtraTrees | 1.0 | 0.0 | 27.03 | 29.0 | 26.0 | 30.0 | 0.419 |
| v1.3 prepatch only | ExtraTrees | 1.0 | 0.0 | 26.87 | 29.0 | 24.0 | 30.0 | 0.484 |
| v1.3 prepatch only | RandomForest | 1.0 | 0.0 | 26.74 | 28.0 | 24.0 | 30.0 | 0.484 |
| v1.3 prepatch only | MLP | 1.0 | 0.0 | 26.68 | 28.0 | 24.0 | 30.0 | 0.419 |
| v1.3 prepatch only | HistGradientBoosting | 1.0 | 0.0 | 26.65 | 28.0 | 23.0 | 30.0 | 0.484 |
| v1.3 visible prepatch | RandomForest | 1.0 | 0.0 | 26.48 | 28.0 | 26.0 | 30.0 | 0.290 |
| v1.2 visible patch stats | RandomForest | 1.0 | 0.0 | 26.32 | 28.0 | 25.5 | 30.0 | 0.290 |
| v1.2 visible patch stats | LogisticRegression | 1.0 | 0.0 | 26.26 | 28.0 | 23.0 | 30.0 | 0.419 |
| v1.3 visible prepatch | MLP | 1.0 | 0.0 | 25.10 | 27.0 | 24.0 | 30.0 | 0.387 |

---

## 7. Interpretation

The v1.3 results support the advisor's direction.

The key finding is not that pre-patch indicators overwhelmingly outperform visible patch indicators. Instead, the important result is that pre-patch indicators can independently provide early-warning information before visible patch structures dominate.

The prepatch-only models achieved valid-window alarm rate = 1.0 and miss rate = 0.0, showing that spatial organization signals before visible patch formation are predictive.

The visible patch + prepatch setting produced a small but meaningful improvement for RandomForest:

```text
v1.2 visible patch stats + RandomForest:
LeadMean ≈ 26.32
PrewindowRate ≈ 0.290

v1.3 visible prepatch + RandomForest:
LeadMean ≈ 26.48
PrewindowRate ≈ 0.290
```

This means that adding prepatch indicators slightly improved valid-window lead time without increasing pre-window false alarms.

For HistGradientBoosting, visible prepatch reduced the pre-window alarm rate compared with the v1.2 visible patch stats model:

```text
v1.2 visible patch stats + HistGradientBoosting:
LeadMean ≈ 28.48
PrewindowRate ≈ 0.613

v1.3 visible prepatch + HistGradientBoosting:
LeadMean ≈ 28.03
PrewindowRate ≈ 0.548
```

Thus, the prepatch indicators can improve the interpretability of the warning process and may help stabilize the trade-off between early alarms and pre-window false alarms.

---

## 8. Current Conclusion

The v1.3 results suggest that:

```text
1. Pre-patch spatial organization indicators are independently predictive.
2. Visible patch + prepatch indicators provide a more complete spatial warning chain.
3. The improvement over v1.2 is not large, but the interpretation is stronger.
4. The paper should not claim that prepatch indicators strongly outperform all baselines.
5. The stronger claim is that early warning can be explained as a transition from pre-patch spatial organization to visible patch structure.
```

Recommended paper framing:

```text
From pre-patch spatial organization to visible patch structure:
interpretable spatial indicators for early warning of ecological critical transitions.
```

---

## 9. Next Steps

Immediate next steps:

```text
1. Commit v1.3 prepatch code and summary.
2. Decide whether to refine the prepatch indicator set.
3. Consider building a PWSI composite index in v1.4.
4. Later perform multi-seed robustness.
5. Later perform horizon robustness under h15 / h30 / h60.
```

Files to commit:

```text
prepatch_indicators.py
build_v1_3_prepatch_dataset.py
docs/v1_3_prepatch_summary.md
```

Files not to commit:

```text
data/processed/v1_3_prepatch/
results_baselines/
results/
*.npz
```