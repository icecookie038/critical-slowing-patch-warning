# v2.2 Synthetic Robustness and Threshold Sensitivity Plan

## 1. Purpose

The purpose of v2.2 is to strengthen the robustness evidence before moving to real remote-sensing validation.

The previous versions have established the main framework:

```text
v1.2: event-based label correction
v1.3: prepatch spatial organization indicators
v1.4: PWSI construction
v2.0: vegetation CA cross-system validation
```

The next step is to test whether the proposed prepatch indicators and PWSI remain stable under different alarm thresholds, window lengths, and synchronization thresholds.

## 2. Overall Strategy

The current strategy is:

```text
Step 1: PWSI threshold sensitivity
Step 2: window length sensitivity
Step 3: correlation threshold sensitivity
Step 4: optional spatial interpretation experiments
Step 5: real forest disturbance remote-sensing case study
```

Real remote-sensing validation will be conducted after the synthetic robustness experiments.

## 3. Priority Experiment 1: PWSI Threshold Sensitivity

This is the first priority.

The goal is to test whether PWSI warning performance depends on a single manually selected alarm threshold.

Thresholds to test:

```text
mean + 2σ
mean + 2.5σ
mean + 3σ
95% quantile
99% quantile
```

Datasets:

```text
SEIR seed42
vegetation CA seed42
```

Main input:

```text
PWSI_equal time series
```

Metrics:

```text
Valid-window alarm rate
Miss rate
Lead time
Pre-window alarm rate
Late alarm rate
First alarm time
```

Expected conclusion:

```text
PWSI should remain informative under a reasonable range of alarm thresholds.
```

This experiment is important because it shows that the warning performance of PWSI is not caused by one arbitrary threshold.

## 4. Priority Experiment 2: Window Length Sensitivity

This is the second priority.

The goal is to test whether prepatch indicators and PWSI are sensitive to the sliding-window length.

Window lengths to test:

```text
5
10
15
20
```

Main settings:

```text
prepatch_only
visible_prepatch
visible patch + PWSI_equal
```

Representative models:

```text
LogisticRegression
RandomForest
```

Datasets:

```text
SEIR seed42
vegetation CA seed42
```

Metrics:

```text
AUC
AUPRC
F1
Lead time
Miss rate
Pre-window alarm rate
Late alarm rate
```

Expected conclusion:

```text
The proposed framework should remain stable under reasonable window-length choices.
```

This experiment is important because prepatch indicators are computed from sliding windows. Therefore, the paper needs to show that the results are not caused by a single selected window length.

## 5. Priority Experiment 3: Correlation Threshold Sensitivity

This is the third priority.

The goal is to test whether the local synchronization signal depends on a single correlation threshold.

Correlation thresholds to test:

```text
0.4
0.5
0.6
0.7
0.8
```

Main indicators:

```text
sync_edge_ratio
local_neighbor_corr_mean
local_neighbor_corr_max
PWSI_equal
```

Datasets:

```text
SEIR seed42
vegetation CA seed42
```

Metrics:

```text
Lead time
Miss rate
Pre-window alarm rate
Late alarm rate
```

Expected conclusion:

```text
The local synchronization signal should not be caused by a single arbitrary correlation threshold.
```

This experiment is important because local synchronization is one of the core mechanisms behind prepatch spatial organization.

## 6. Optional Experiment 1: Spatial Correlation Length

This experiment is optional.

The goal is to strengthen the theoretical interpretation of spatial connectivity before patch formation.

Possible indicator:

```text
spatial_correlation_length
```

Basic idea:

```text
Compute spatial autocorrelation C(r) at different distances.
Fit the decay of C(r) over distance.
Estimate the spatial correlation length ξ.
```

Expected interpretation:

```text
If spatial correlation length increases before the event, it suggests that the influence range of local perturbations is expanding before the critical transition.
```

This experiment can strengthen the mechanism of spatial connectivity expansion, but it is not the first priority.

## 7. Optional Experiment 2: Patch Compactness

This experiment is optional.

The goal is to strengthen the interpretation of boundary rigidity and patch morphology.

Possible indicators:

```text
patch compactness
perimeter-area ratio
perimeter² / area
area / perimeter
```

Expected interpretation:

```text
If patch compactness changes before the event, it suggests that the system is moving from fragmented and unstable structures toward more organized patch morphology.
```

This experiment can be used as an additional explanatory analysis.

## 8. Optional Experiment 3: SVD High-loading Region Overlap

This experiment is optional.

The goal is to test whether the dominant spatial mode can identify future high-risk regions.

Possible metrics:

```text
IoU
Dice coefficient
Overlap ratio
```

Basic idea:

```text
Extract high-loading regions from the first SVD spatial mode.
Compare these regions with future event or degraded regions.
```

Expected interpretation:

```text
If the high-loading regions of the dominant spatial mode overlap with future event regions, it suggests that the spatial blueprint of future degradation is already partially visible before the transition.
```

This experiment can improve spatial interpretability, but it is not required before the main robustness experiments.

## 9. Experiments Not Prioritized at This Stage

The following experiments are not prioritized at this stage:

```text
Transformer
larger CNN-RNN models
additional deep learning architectures
complex ensemble models
```

The reason is that the current paper focuses on interpretable ecological indicators rather than model complexity.

If needed, XGBoost or LightGBM can be added later as supplementary machine-learning baselines, but they should not replace the main indicator-based framework.

## 10. Current Priority

The immediate next task is:

```text
Build and run analyze_pwsi_threshold_sensitivity.py
```

The first version should focus on:

```text
SEIR seed42 PWSI_equal
vegetation CA seed42 PWSI_equal
```

The expected output should include:

```text
threshold_name
threshold_value
valid_window_alarm_rate
miss_rate
lead_mean
lead_median
prewindow_alarm_rate
late_alarm_rate
```

After this experiment is completed, the results will be summarized in:

```text
docs/v2_2_synthetic_robustness_summary.md
```
