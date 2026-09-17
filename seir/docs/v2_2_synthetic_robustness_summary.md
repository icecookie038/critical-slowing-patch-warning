# v2.2 Synthetic Robustness Summary

## 1. Purpose

The purpose of v2.2 is to test whether the proposed PWSI and prepatch-based early-warning framework remain stable under different robustness settings before moving to real remote-sensing validation.

The first completed robustness experiment focuses on PWSI threshold sensitivity.

This experiment evaluates whether the warning performance of PWSI_equal depends on a single manually selected alarm threshold.

## 2. PWSI Threshold Sensitivity Experiment

Five threshold rules were tested:

```text
mean + 2σ
mean + 2.5σ
mean + 3σ
95% quantile
99% quantile
```

The experiment was conducted on two systems:

```text
SEIR seed42
vegetation CA seed42
```

Main input:

```text
PWSI_equal time series
```

The evaluated metrics include:

```text
Valid-window alarm rate
Miss rate
Lead mean
Lead median
Pre-window alarm rate
Late alarm rate
```

The main purpose of this experiment is to verify whether PWSI_equal remains informative under a reasonable range of alarm thresholds.

## 3. SEIR seed42 Results

The SEIR seed42 threshold sensitivity results are shown below.

| Threshold rule | Direction | Valid-window alarm rate | Miss rate | Lead mean | Pre-window alarm rate | Late alarm rate |
| -------------- | --------- | ----------------------: | --------: | --------: | --------------------: | --------------: |
| q95            | high      |                   0.283 |     0.717 |     25.61 |                 0.717 |           0.000 |
| mean + 2σ      | high      |                   0.811 |     0.189 |     23.93 |                 0.165 |           0.000 |
| q99            | high      |                   0.496 |     0.504 |     23.54 |                 0.504 |           0.000 |
| mean + 2.5σ    | high      |                   0.953 |     0.047 |     21.92 |                 0.024 |           0.000 |
| mean + 3σ      | high      |                   0.921 |     0.079 |     19.90 |                 0.016 |           0.000 |

In the SEIR system, the mean + 2.5σ rule produced the highest valid-window alarm rate and a low pre-window alarm rate.

The mean + 2σ rule produced earlier warnings, but its pre-window alarm rate was higher than that of mean + 2.5σ.

Therefore, in SEIR seed42:

```text
mean + 2.5σ is the most conservative and stable threshold rule.
mean + 2σ provides earlier warnings but allows more pre-window alarms.
```

## 4. Vegetation CA seed42 Results

The vegetation CA seed42 threshold sensitivity results are shown below.

| Threshold rule | Direction | Valid-window alarm rate | Miss rate | Lead mean | Pre-window alarm rate | Late alarm rate |
| -------------- | --------- | ----------------------: | --------: | --------: | --------------------: | --------------: |
| q95            | high      |                   0.250 |     0.750 |     27.48 |                 0.750 |           0.000 |
| q99            | high      |                   0.830 |     0.170 |     26.46 |                 0.170 |           0.000 |
| mean + 2σ      | high      |                   0.865 |     0.135 |     24.22 |                 0.130 |           0.000 |
| mean + 2.5σ    | high      |                   0.865 |     0.135 |     17.41 |                 0.005 |           0.000 |
| mean + 3σ      | high      |                   0.510 |     0.490 |     13.75 |                 0.000 |           0.000 |

In the vegetation CA system, the mean + 2σ rule provided the most balanced performance.

It achieved a high valid-window alarm rate, a moderate lead time, and a controlled pre-window alarm rate.

The mean + 2.5σ rule reduced pre-window alarms, but it also shortened the mean lead time. The mean + 3σ rule was too conservative and reduced the valid-window alarm rate.

Therefore, in vegetation CA seed42:

```text
mean + 2σ is the most balanced threshold rule.
mean + 2.5σ is more conservative but may delay warnings.
mean + 3σ is too conservative for vegetation CA.
```

## 5. Cross-system Comparison

Across both SEIR and vegetation CA systems, the mean + 2σ rule showed the most balanced cross-system performance.

The key comparison is:

```text
SEIR seed42, mean + 2σ:
Valid-window alarm rate = 0.811
Pre-window alarm rate = 0.165
Lead mean = 23.93
Late alarm rate = 0.000

Vegetation CA seed42, mean + 2σ:
Valid-window alarm rate = 0.865
Pre-window alarm rate = 0.130
Lead mean = 24.22
Late alarm rate = 0.000
```

The mean + 2.5σ rule can be retained as a conservative sensitivity setting because it reduces pre-window alarms, especially in the SEIR system.

The quantile-based rules were less stable across systems. In particular, q95 produced many pre-window alarms, while q99 showed inconsistent behavior between SEIR and vegetation CA.

## 6. Interpretation of Miss Rate

The miss rate in this analysis should be interpreted together with the pre-window alarm rate.

Some non-valid warnings were not complete failures to alarm. Instead, they were caused by alarms occurring earlier than the valid warning window.

This means that a high miss rate may include two different cases:

```text
1. The model or index did not alarm before the event.
2. The model or index alarmed too early, before the valid warning window.
```

Therefore, the threshold sensitivity results should not be interpreted using miss rate alone.

The more appropriate interpretation should consider:

```text
Valid-window alarm rate
Pre-window alarm rate
Late alarm rate
Lead mean
```

## 7. Main Threshold Selection

Based on the cross-system comparison, the recommended main threshold rule is:

```text
mean + 2σ
```

The reason is that mean + 2σ provides the most balanced performance across both SEIR and vegetation CA.

It maintains relatively high valid-window alarm rates while avoiding excessive pre-window alarms.

The conservative sensitivity threshold is:

```text
mean + 2.5σ
```

This threshold can be used as a stricter warning rule, especially when the goal is to reduce pre-window alarms.

The following rules are not recommended as the main rule:

```text
q95
q99
mean + 3σ
```

Reasons:

```text
q95 is too sensitive and produces excessive pre-window alarms.
q99 is not stable across systems.
mean + 3σ is too conservative and may reduce useful lead time.
```

## 8. Main Conclusion

The PWSI threshold sensitivity experiment shows that PWSI_equal does not rely on a single arbitrary threshold.

The main conclusion is:

```text
PWSI_equal remains informative under multiple threshold rules. The mean + 2σ rule provides the most balanced cross-system performance across SEIR and vegetation CA, while mean + 2.5σ provides a more conservative warning rule.
```

This result supports the robustness of PWSI_equal as a compact and interpretable early-warning index.

## 9. Role in the Main Paper

This experiment can be used as a robustness analysis in the final paper.

It supports the following statement:

```text
The proposed PWSI is not sensitive to a single manually selected alarm threshold. Across SEIR and vegetation CA systems, PWSI maintained useful warning performance under multiple threshold rules, with mean + 2σ providing the most balanced cross-system behavior.
```

This strengthens the ecological indicator interpretation of PWSI because it shows that PWSI can be used not only as a machine-learning feature, but also as a rule-based warning index.

## 10. Next Step

The next experiment should be window length sensitivity.

The goal is to test whether prepatch indicators and PWSI remain stable under different sliding-window lengths.

Suggested window lengths:

```text
5
10
15
20
```

Recommended test settings:

```text
prepatch_only
visible_prepatch
visible patch + PWSI_equal
```

Recommended representative models:

```text
LogisticRegression
RandomForest
```

Datasets:

```text
SEIR seed42
vegetation CA seed42
```

Expected conclusion:

```text
The proposed prepatch and PWSI framework should remain stable under reasonable sliding-window length choices.
```
## 11. Window Length Sensitivity Experiment: Vegetation CA

The second robustness experiment focuses on window length sensitivity.

The goal of this experiment is to test whether the proposed prepatch indicators and PWSI framework depend on a single selected sliding-window length.

Four window lengths were tested:

```text
5
10
15
20
```

The experiment was conducted on the vegetation CA system using seed42.

Three settings were evaluated:

```text
visible_patch
visible_prepatch
visible patch + PWSI_equal
```

Two representative models were tested:

```text
LogisticRegression
RandomForest
```

The evaluated metrics include:

```text
Valid-window alarm rate
Miss rate
Lead mean
Lead median
Pre-window alarm rate
Late alarm rate
```

## 12. Vegetation CA Window Length Sensitivity Results

| Window | Setting            | Model              | Valid alarm | Miss rate | Lead mean | Lead median | Pre-window alarm rate | Late alarm rate |
| -----: | ------------------ | ------------------ | ----------: | --------: | --------: | ----------: | --------------------: | --------------: |
|      5 | visible_patch      | LogisticRegression |       1.000 |     0.000 |     26.32 |        28.0 |                  0.14 |           0.000 |
|     10 | visible_patch      | LogisticRegression |       1.000 |     0.000 |     26.72 |        29.0 |                  0.14 |           0.000 |
|     15 | visible_patch      | LogisticRegression |       1.000 |     0.000 |     27.20 |        29.0 |                  0.20 |           0.000 |
|     20 | visible_patch      | LogisticRegression |       1.000 |     0.000 |     26.96 |        28.0 |                  0.16 |           0.000 |
|      5 | visible_patch      | RandomForest       |       1.000 |     0.000 |     26.38 |        27.5 |                  0.14 |           0.000 |
|     10 | visible_patch      | RandomForest       |       1.000 |     0.000 |     26.22 |        27.5 |                  0.14 |           0.000 |
|     15 | visible_patch      | RandomForest       |       1.000 |     0.000 |     26.28 |        27.0 |                  0.14 |           0.000 |
|     20 | visible_patch      | RandomForest       |       1.000 |     0.000 |     26.54 |        29.0 |                  0.18 |           0.000 |
|      5 | visible_prepatch   | LogisticRegression |       1.000 |     0.000 |     29.30 |        30.0 |                  0.26 |           0.000 |
|     10 | visible_prepatch   | LogisticRegression |       1.000 |     0.000 |     29.12 |        30.0 |                  0.10 |           0.000 |
|     15 | visible_prepatch   | LogisticRegression |       1.000 |     0.000 |     28.92 |        30.0 |                  0.10 |           0.000 |
|     20 | visible_prepatch   | LogisticRegression |       1.000 |     0.000 |     29.38 |        30.0 |                  0.22 |           0.000 |
|      5 | visible_prepatch   | RandomForest       |       1.000 |     0.000 |     28.68 |        30.0 |                  0.20 |           0.000 |
|     10 | visible_prepatch   | RandomForest       |       1.000 |     0.000 |     28.56 |        29.0 |                  0.12 |           0.000 |
|     15 | visible_prepatch   | RandomForest       |       1.000 |     0.000 |     28.84 |        30.0 |                  0.16 |           0.000 |
|     20 | visible_prepatch   | RandomForest       |       1.000 |     0.000 |     28.82 |        30.0 |                  0.18 |           0.000 |
|      5 | visible_pwsi_equal | LogisticRegression |       1.000 |     0.000 |     27.46 |        28.0 |                  0.16 |           0.000 |
|     10 | visible_pwsi_equal | LogisticRegression |       1.000 |     0.000 |     27.46 |        28.5 |                  0.16 |           0.000 |
|     15 | visible_pwsi_equal | LogisticRegression |       1.000 |     0.000 |     27.02 |        28.0 |                  0.14 |           0.000 |
|     20 | visible_pwsi_equal | LogisticRegression |       1.000 |     0.000 |     26.66 |        27.0 |                  0.08 |           0.000 |
|      5 | visible_pwsi_equal | RandomForest       |       1.000 |     0.000 |     26.70 |        27.0 |                  0.14 |           0.000 |
|     10 | visible_pwsi_equal | RandomForest       |       1.000 |     0.000 |     26.12 |        27.0 |                  0.14 |           0.000 |
|     15 | visible_pwsi_equal | RandomForest       |       1.000 |     0.000 |     26.20 |        27.0 |                  0.14 |           0.000 |
|     20 | visible_pwsi_equal | RandomForest       |       1.000 |     0.000 |     26.36 |        27.0 |                  0.16 |           0.000 |

## 13. Window Length Sensitivity Interpretation

The vegetation CA window length sensitivity experiment shows that the proposed framework remains stable across different sliding-window lengths.

Across window lengths of 5, 10, 15, and 20, all tested settings achieved:

```text
Valid-window alarm rate = 1.000
Miss rate = 0.000
Late alarm rate = 0.000
```

This indicates that the warning performance is not caused by a single selected window length.

The visible_prepatch setting consistently achieved the longest lead time. Its mean lead time remained close to 29 across different window lengths and models.

The visible_patch baseline also remained stable, with mean lead time generally between 26 and 27.

The visible patch + PWSI_equal setting provided a stable compact representation. Although its lead time was slightly lower than that of full visible_prepatch, it remained robust across all tested window lengths.

## 14. Main Conclusion of Window Length Sensitivity

The main conclusion is:

```text
The prepatch and PWSI framework remains robust under different sliding-window lengths in the vegetation CA system. Visible_prepatch consistently provides the strongest early-warning lead time, while visible patch + PWSI_equal provides a stable and compact applied representation.
```

This result strengthens the robustness evidence of the proposed framework before moving to real remote-sensing validation.

## 15. Correlation Threshold Sensitivity Experiment

The third robustness experiment focuses on the sensitivity of the local synchronization indicator to different correlation thresholds.

The goal of this experiment is to test whether the local synchronization signal depends on a single manually selected correlation threshold.

Five correlation thresholds were tested:

```text
0.4
0.5
0.6
0.7
0.8
```

The experiment was conducted on the vegetation CA seed42 dataset.

The tested indicator was:

```text
sync_edge_ratio
```

The alarm rule was:

```text
mean + 2σ
```

The evaluated metrics include:

```text
Valid-window alarm rate
Miss rate
Lead mean
Lead median
Pre-window alarm rate
Late alarm rate
```

## 16. Vegetation CA Correlation Threshold Sensitivity Results

| Correlation threshold | Threshold rule | Valid alarm | Miss rate | Lead mean | Lead median | Pre-window alarm rate |
| --------------------: | -------------- | ----------: | --------: | --------: | ----------: | --------------------: |
|                   0.4 | mean + 2σ      |       0.425 |     0.575 |     25.78 |        28.0 |                 0.430 |
|                   0.5 | mean + 2σ      |       0.395 |     0.605 |     25.87 |        28.0 |                 0.505 |
|                   0.6 | mean + 2σ      |       0.350 |     0.650 |     26.47 |        29.0 |                 0.590 |
|                   0.7 | mean + 2σ      |       0.265 |     0.735 |     27.25 |        29.0 |                 0.715 |
|                   0.8 | mean + 2σ      |       0.120 |     0.880 |     28.75 |        29.0 |                 0.875 |

## 17. Correlation Threshold Sensitivity Interpretation

The single local synchronization indicator showed clear sensitivity to the selected correlation threshold.

As the correlation threshold increased from 0.4 to 0.8, the valid-window alarm rate decreased from 0.425 to 0.120, while the pre-window alarm rate increased from 0.430 to 0.875.

This indicates that sync_edge_ratio can capture early local synchronization signals, but when it is used alone as a hard-threshold warning indicator, it tends to generate many pre-window alarms.

Therefore, sync_edge_ratio should not be used as the final standalone warning rule.

Instead, it is more appropriate to interpret local synchronization as one component of the broader prepatch spatial organization framework.

## 18. Main Conclusion of Correlation Threshold Sensitivity

The main conclusion is:

```text
The local synchronization signal is an important precursor component, but a single threshold-based sync_edge_ratio indicator is not sufficiently stable as an independent warning rule. This result supports the use of multi-indicator integration through the full prepatch framework and PWSI rather than relying on one local synchronization threshold alone.
```

This result strengthens the methodological motivation for PWSI.

It shows that the proposed framework should not depend on any single local synchronization threshold. Instead, local synchronization should be combined with spatial connectivity, boundary rigidity, and dominant mode locking to form a more stable and interpretable warning index.

## 19. Updated v2.2 Robustness Conclusion

After completing PWSI threshold sensitivity, window length sensitivity, and correlation threshold sensitivity, the v2.2 synthetic robustness experiments support the following conclusions:

```text
1. PWSI_equal does not rely on a single arbitrary alarm threshold.
2. The prepatch and PWSI framework remains stable under different sliding-window lengths in the vegetation CA system.
3. A single hard-threshold local synchronization indicator is sensitive to the selected correlation threshold and should not be used alone as the final warning rule.
4. The full prepatch framework and PWSI are better justified as multi-indicator integrated warning representations.
```

Overall, v2.2 strengthens the robustness evidence of the proposed framework before moving to real remote-sensing validation.

The results suggest that the proposed framework should be interpreted as an integrated spatial early-warning indicator system rather than a single-threshold local synchronization detector.

## 20. Next Step

The next stage is:

```text
v2.3 Synthetic robustness final summary and paper-level figure/table consolidation
```

The purpose of v2.3 is to organize the existing experimental results into paper-level figures, tables, and result summaries.

The next tasks include:

```text
1. Consolidate SEIR, vegetation CA, PWSI, and robustness results.
2. Build a final main result table.
3. Build a final robustness table.
4. Prepare paper-level figure structure.
5. Decide which results should be placed in the main text and which should be placed in supplementary materials.
```

After v2.3, the project can move to:

```text
v2.4 real forest disturbance remote-sensing case study
```
