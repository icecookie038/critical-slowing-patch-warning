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
