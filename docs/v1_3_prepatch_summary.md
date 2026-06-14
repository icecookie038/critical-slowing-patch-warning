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
## 10. Persistent-alarm Sensitivity Analysis

A persistent-alarm sensitivity analysis was conducted by increasing the alarm persistence requirement from `persistent_k = 2` to `persistent_k = 3` and `persistent_k = 4`.

The purpose of this analysis was to test whether requiring consecutive alarms can reduce pre-window false alarms while preserving valid-window warning performance.

Across all tested settings, the models maintained:

```text
valid_window_alarm_rate = 1.0
miss_rate = 0.0
late_alarm_rate = 0.0
```

This indicates that increasing the alarm persistence requirement did not cause missed event simulations under the current seed42 h30 setting.

### 10.1 Prepatch-only results

For the `v1_3_prepatch_only` setting, increasing `persistent_k` reduced pre-window alarms for several models.

Representative results:

| Setting | Model | persistent_k | Lead mean | Lead median | Pre-window alarm rate |
|---|---|---:|---:|---:|---:|
| prepatch_only | MLP | 3 | 26.68 | 28.0 | 0.355 |
| prepatch_only | MLP | 4 | 26.10 | 28.0 | 0.323 |
| prepatch_only | HistGradientBoosting | 3 | 26.45 | 28.0 | 0.419 |
| prepatch_only | HistGradientBoosting | 4 | 26.45 | 28.0 | 0.387 |
| prepatch_only | ExtraTrees | 3 | 26.74 | 29.0 | 0.452 |
| prepatch_only | ExtraTrees | 4 | 26.26 | 28.0 | 0.452 |

These results show that stricter persistence rules can reduce overly early alarms in prepatch-only models, although very strict settings may slightly reduce lead time.

### 10.2 Visible patch + prepatch results

The most useful sensitivity result was observed in the `v1_3_visible_prepatch` setting.

Representative results:

| Setting | Model | persistent_k | Lead mean | Lead median | Pre-window alarm rate |
|---|---|---:|---:|---:|---:|
| visible_prepatch | HistGradientBoosting | 2 | 28.03 | 29.0 | 0.548 |
| visible_prepatch | HistGradientBoosting | 3 | 28.03 | 29.0 | 0.419 |
| visible_prepatch | HistGradientBoosting | 4 | 27.61 | 28.0 | 0.387 |
| visible_prepatch | ExtraTrees | 3 | 27.74 | 30.0 | 0.387 |
| visible_prepatch | ExtraTrees | 4 | 27.74 | 30.0 | 0.355 |
| visible_prepatch | RandomForest | 3 | 26.48 | 28.0 | 0.258 |
| visible_prepatch | RandomForest | 4 | 25.81 | 28.0 | 0.258 |
| visible_prepatch | MLP | 3 | 25.00 | 27.0 | 0.290 |
| visible_prepatch | MLP | 4 | 24.84 | 27.0 | 0.290 |

For HistGradientBoosting, increasing `persistent_k` from 2 to 3 reduced the pre-window alarm rate from approximately 0.548 to 0.419 while preserving the same mean valid-window lead time of approximately 28.03.

For RandomForest, increasing `persistent_k` from 2 to 3 reduced the pre-window alarm rate from approximately 0.290 to 0.258 while preserving the mean valid-window lead time of approximately 26.48.

This suggests that a moderate persistent-alarm rule can reduce over-early alarms without sacrificing useful warning lead time.

### 10.3 Interpretation

The persistent-alarm sensitivity analysis supports the use of a stricter alarm rule for first-alarm evaluation.

The main conclusion is:

```text
A persistent-alarm rule can reduce pre-window false alarms while preserving valid-window warning performance.
```

For the current v1.3 results, `persistent_k = 3` provides the best trade-off between early warning and false-alarm control.

Recommended setting:

```text
Main analysis: persistent_k = 3
Sensitivity analysis: persistent_k = 2 and persistent_k = 4
```

This makes the first-alarm evaluation more conservative and more suitable for ecological early-warning interpretation.   
## 11. Feature Importance and Conservative Prepatch Set

A feature-importance analysis was conducted to identify which pre-patch indicators contributed most strongly to the warning task.

RandomForest and ExtraTrees were used to estimate both impurity-based importance and permutation importance. The group-level permutation importance showed that the most informative category was local synchronization.

Group-level importance:

| Group | Group permutation importance | Group impurity importance | Number of features |
|---|---:|---:|---:|
| local_synchronization | 0.0935 | 0.2656 | 3 |
| boundary_rigidity | 0.0527 | 0.2594 | 3 |
| spatial_connectivity | 0.0507 | 0.2207 | 3 |
| dominant_mode_locking | 0.0327 | 0.2544 | 5 |

The most important individual features were:

| Rank | Feature | Group | Permutation importance |
|---:|---|---|---:|
| 1 | local_neighbor_corr_mean | local_synchronization | 0.0433 |
| 2 | sync_edge_ratio | local_synchronization | 0.0389 |
| 3 | gradient_entropy | boundary_rigidity | 0.0274 |
| 4 | svd_mode1_ac1 | dominant_mode_locking | 0.0196 |
| 5 | geary_c | spatial_connectivity | 0.0177 |
| 6 | moran_i | spatial_connectivity | 0.0166 |
| 7 | high_state_component_ratio | spatial_connectivity | 0.0164 |
| 8 | gradient_top10_mean | boundary_rigidity | 0.0133 |

This result suggests that the pre-patch warning signal is mainly driven by local synchronization, followed by boundary rigidity and spatial connectivity. Dominant-mode locking contributed less overall, although `svd_mode1_ac1` remained informative.

Based on the feature-importance ranking, a conservative prepatch feature set was constructed using the top-ranked indicators:

```text
local_neighbor_corr_mean
sync_edge_ratio
gradient_entropy
svd_mode1_ac1
geary_c
moran_i
high_state_component_ratio
gradient_top10_mean
```

Two conservative datasets were generated:

```text
seir_v1_3_conservative_prepatch_only_h30_seed42.npz
seir_v1_3_visible_conservative_prepatch_h30_seed42.npz
```

### 11.1 Conservative prepatch-only results

Under `persistent_k = 3`, the conservative prepatch-only setting achieved valid-window alarms for all event simulations.

Representative results:

| Model | Valid alarm rate | Miss rate | Lead mean | Lead median | Lead Q25 | Lead Q75 | Pre-window alarm rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| RandomForest | 1.0 | 0.0 | 26.84 | 30.0 | 22.5 | 30.0 | 0.484 |
| ExtraTrees | 1.0 | 0.0 | 26.61 | 30.0 | 23.0 | 30.0 | 0.419 |
| LogisticRegression | 1.0 | 0.0 | 26.39 | 29.0 | 22.5 | 30.0 | 0.387 |
| MLP | 1.0 | 0.0 | 26.29 | 28.0 | 22.0 | 30.0 | 0.355 |
| HistGradientBoosting | 1.0 | 0.0 | 26.23 | 27.0 | 22.0 | 30.0 | 0.452 |

The conservative prepatch-only setting preserved warning ability, but did not uniformly improve over the full prepatch feature set.

### 11.2 Visible patch + conservative prepatch results

Under `persistent_k = 3`, the visible patch + conservative prepatch setting produced the following representative results:

| Model | Valid alarm rate | Miss rate | Lead mean | Lead median | Lead Q25 | Lead Q75 | Pre-window alarm rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| HistGradientBoosting | 1.0 | 0.0 | 28.13 | 29.0 | 27.5 | 30.0 | 0.452 |
| ExtraTrees | 1.0 | 0.0 | 27.32 | 29.0 | 26.0 | 30.0 | 0.323 |
| RandomForest | 1.0 | 0.0 | 26.58 | 29.0 | 25.5 | 30.0 | 0.290 |
| LogisticRegression | 1.0 | 0.0 | 26.42 | 28.0 | 24.0 | 30.0 | 0.452 |
| MLP | 1.0 | 0.0 | 25.94 | 28.0 | 24.0 | 30.0 | 0.355 |

The conservative set did not uniformly outperform the full prepatch feature set. However, it provided a useful robustness check, showing that a reduced set of high-importance prepatch indicators can still preserve valid-window warning ability.

The main conclusion is:

```text
The full prepatch feature set should remain the main v1.3 result, while the conservative prepatch set should be reported as a robustness or ablation analysis.
```

### 11.3 Interpretation

The feature-importance and conservative-set analyses support three conclusions:

```text
1. Local synchronization is the dominant pre-patch warning signal.
2. Boundary rigidity and spatial connectivity provide additional useful information.
3. A reduced high-importance prepatch set can preserve valid-window warning ability, but it does not consistently outperform the full prepatch feature set.
```

Therefore, the main paper should emphasize the full prepatch indicator framework, while the conservative feature set can be reported as supplementary robustness evidence.