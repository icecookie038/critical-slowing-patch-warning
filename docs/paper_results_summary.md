# Paper Results Summary

## 1. Current Experiment Chain

The current project has completed the following experimental versions:

```text
v0.7   classic patch baseline
v0.8   dynamic patch indicators v1
v0.9   feature group comparison
v0.9.1 dynamic patch indicators v2 diagnostic
v1.0   traditional EWS baseline
v1.1   lead-time proxy and feature importance
```

The current results provide a complete SEIR-based evidence chain for a patch-based early-warning framework.

---

## 2. Main Method Comparison

The current method comparison table is:

| Method | AUC | AUPRC | F1 | Accuracy | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| Traditional EWS | 0.836 ± 0.042 | 0.781 ± 0.062 | 0.722 ± 0.050 | 0.760 ± 0.045 | 0.675 ± 0.064 | 0.777 ± 0.036 |
| Image only | 0.846 ± 0.032 | 0.803 ± 0.046 | 0.732 ± 0.041 | 0.767 ± 0.030 | 0.678 ± 0.044 | 0.797 ± 0.041 |
| Classic patch | 0.867 ± 0.030 | 0.836 ± 0.034 | 0.749 ± 0.042 | 0.786 ± 0.038 | 0.714 ± 0.090 | 0.797 ± 0.050 |
| Dynamic patch v1 | 0.538 ± 0.017 | 0.443 ± 0.016 | 0.572 ± 0.022 | 0.401 ± 0.022 | 0.401 ± 0.022 | 1.000 ± 0.000 |
| Classic + dynamic patch | 0.868 ± 0.028 | 0.835 ± 0.032 | 0.750 ± 0.036 | 0.787 ± 0.028 | 0.709 ± 0.051 | 0.803 ± 0.075 |
| Full | 0.869 ± 0.027 | 0.836 ± 0.033 | 0.751 ± 0.036 | 0.791 ± 0.033 | 0.725 ± 0.084 | 0.787 ± 0.053 |

---

## 3. Main Interpretation

The strongest result is not that dynamic patch indicators significantly improve AUC.

The strongest result is:

```text
Classic patch-structure features provide a strong and compact representation for spatial SEIR critical-transition risk prediction.
```

The classic patch model outperforms traditional EWS and image-only baselines in AUC, AUPRC, F1, and accuracy.

The full model performs only slightly better than the classic patch model, which suggests that classic patch features already capture most of the discriminative spatial-structural information.

---

## 4. Traditional EWS Baseline

Traditional EWS is effective:

```text
AUC = 0.836 ± 0.042
AUPRC = 0.781 ± 0.062
F1 = 0.722 ± 0.050
```

However, it is weaker than classic patch features:

```text
Traditional EWS AUC = 0.836 ± 0.042
Classic patch AUC   = 0.867 ± 0.030
```

This supports the conclusion that patch-structure features are more suitable than traditional scalar EWS indicators for spatial spreading systems.

---

## 5. Dynamic Patch Indicators

Dynamic patch v1 alone has weak predictive performance:

```text
Dynamic patch v1 AUC = 0.538 ± 0.017
```

It has very high recall but low precision:

```text
Recall = 1.000 ± 0.000
Precision = 0.401 ± 0.022
```

This means dynamic patch v1 is highly sensitive but noisy. It tends to issue many alarms, including many false alarms.

Therefore, dynamic patch indicators should not be presented as strong standalone predictors in the current SEIR setting.

Their better role is:

```text
mechanistic interpretation
sensitive warning signal
temporal-structural explanation
```

---

## 6. Dynamic Patch v2 Diagnostic

Dynamic patch v2 was tested as a diagnostic extension.

The v2 indicators included:

```text
BCI   Boundary Complexity Index
FPV   Front Propagation Velocity
PMR   Patch Merging Rate
GCGR  Giant Component Growth Rate
PPI   Percolation Proximity Index
```

The seed=42 diagnostic result was:

```text
dynamic_patch_v2_only:
AUC   = 0.5629
AUPRC = 0.4530
F1    = 0.5447

classic_dynamic_v2_patch:
AUC   = 0.8665
AUPRC = 0.8220
F1    = 0.7340
```

Dynamic patch v2 only slightly improved over dynamic patch v1 under the current SEIR setting.

Therefore, patch v2 is kept as a diagnostic and supplementary result, but it is not included in the main 5-seed method comparison table.

---

## 7. Lead-Time Proxy Result

The current prediction files contain:

```text
y_remaining_true
y_remaining_pred
y_risk_true
risk_prob
```

They do not contain:

```text
simulation ID
time index
critical time
```

Therefore, v1.1 reports a sample-level lead-time proxy, not a strict first-alarm lead time.

The lead-time proxy is defined as:

```text
the average remaining time to critical transition among correctly predicted risk samples
```

The main results are:

| Method | Alarm recall | Alarm precision | Mean lead time | Median lead time |
|---|---:|---:|---:|---:|
| Traditional EWS | 0.78 ± 0.04 | 0.67 ± 0.06 | 13.81 ± 0.30 | 13.40 ± 0.55 |
| Image only | 0.80 ± 0.04 | 0.68 ± 0.04 | 13.89 ± 0.23 | 13.20 ± 0.45 |
| Classic patch | 0.80 ± 0.05 | 0.71 ± 0.09 | 14.09 ± 0.28 | 13.60 ± 0.55 |
| Dynamic patch v1 | 1.00 ± 0.00 | 0.40 ± 0.02 | 15.18 ± 0.13 | 15.00 ± 0.00 |
| Classic + dynamic patch | 0.80 ± 0.07 | 0.71 ± 0.05 | 14.14 ± 0.20 | 13.40 ± 0.55 |
| Full | 0.79 ± 0.05 | 0.73 ± 0.08 | 14.02 ± 0.26 | 13.40 ± 0.55 |

Dynamic patch v1 has the largest lead-time proxy, but its precision is low. Therefore, it should be interpreted as an early but noisy signal.

Classic patch, classic + dynamic patch, and full models provide more balanced early-warning performance.

---

## 8. Feature Importance Result

The feature-value diagnostic importance analysis identifies which classic patch features separate risk and non-risk samples most strongly.

The top features are:

```text
patch_area_cv
centroid_y
largest_patch_area
patch_area_std
largest_patch_ratio
gini
mean_patch_area
aggregation_index
centroid_x
perimeter
fragmentation
edge_density
patch_count
perimeter_area_ratio
spread_x
```

The strongest features are related to:

```text
patch-size heterogeneity
dominant patch formation
spatial aggregation
centroid displacement
boundary and perimeter structure
fragmentation
```

This supports the interpretation that spatial critical-transition risk is reflected in structural changes of infected patches.

---

## 9. Current Paper Main Claim

The paper should not claim:

```text
Dynamic patch indicators significantly improve predictive performance.
```

The paper should claim:

```text
Patch-structure features provide a strong and interpretable early-warning representation for spatial critical transitions.
```

A more accurate main conclusion is:

```text
Classic patch indicators outperform traditional EWS and image-only baselines in spatial SEIR risk prediction. Dynamic patch indicators are sensitive to temporal structural changes but are noisy as standalone predictors. Together, the results suggest that patch-based early-warning signals provide both predictive strength and spatial-structural interpretability.
```

---

## 10. Current Limitation

The current SEIR model produces relatively smooth spatial expansion.

This may explain why dynamic patch indicators do not substantially improve prediction. Many dynamic structural changes are already captured by classic patch features under this setting.

The current lead-time analysis is also only a proxy because strict simulation-level alarm timing metadata is not yet saved.

---

## 11. Next Experiments

The next technical versions should be:

```text
v1.2 strict lead-time metadata
v1.3 extended SEIR model
v1.4 spatial stochastic cellular automaton
```

The extended SEIR model should include:

```text
spatial heterogeneity
stochastic local transmission
heterogeneous recovery
spatial barriers
longer temporal windows
```

The spatial cellular automaton model should be used as a second data-generating system to test whether dynamic patch indicators become more useful in systems with patch birth, merging, fragmentation, and percolation-like transitions.