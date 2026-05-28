# Experiment Plan v1.1: Lead-Time Proxy and Feature Importance

## 1. Version Goal

The goal of v1.1 is to extend the method comparison beyond standard classification metrics by adding:

```text
1. lead-time proxy analysis
2. feature-value diagnostic importance
```

This version does not retrain new models. It analyzes existing validation outputs from previous experiments.

---

## 2. Lead-Time Proxy Analysis

The prediction files currently contain:

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

Therefore, v1.1 reports a sample-level lead-time proxy rather than a strict first-alarm lead time.

The lead-time proxy is defined as:

```text
the average remaining time to critical transition among correctly predicted risk samples.
```

This measures how early the model tends to issue correct risk alarms at the sample level.

---

## 3. Lead-Time Proxy Results

The lead-time proxy analysis collected:

```text
6 methods × 5 seeds = 30 rows
```

The analyzed methods were:

```text
Traditional EWS
Image only
Classic patch
Dynamic patch v1
Classic + dynamic patch
Full
```

The generated files are:

```text
results_summary/lead_time_proxy_summary_all.csv
results_summary/lead_time_proxy_summary_grouped.csv

docs/paper_tables/lead_time_proxy_table.csv
docs/paper_tables/lead_time_proxy_table.md
docs/paper_tables/lead_time_proxy_table.tex
```

The main result is that classic patch, classic + dynamic patch, and full models provide stable lead-time proxy values while maintaining relatively high alarm precision.

Dynamic patch v1 shows a larger lead-time proxy, but its alarm precision is much lower. Therefore, it should be interpreted as a sensitive but noisy early-warning signal rather than a strong standalone predictor.

---

## 4. Feature-Value Diagnostic Importance

The feature importance analysis focuses on classic patch features.

It estimates how strongly each feature separates risk and non-risk samples using:

```text
absolute Cohen's d
single-feature AUC
absolute mean difference
```

This analysis is not SHAP or permutation importance. It is a diagnostic analysis of feature-value separation.

---

## 5. Top Important Patch Features

The top-ranked classic patch features include:

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

The generated files are:

```text
results_summary/feature_importance_summary_all.csv
results_summary/feature_importance_summary_grouped.csv

docs/paper_tables/feature_importance_table.csv
docs/paper_tables/feature_importance_table.md
docs/paper_tables/feature_importance_table.tex
```

---

## 6. Interpretation

The most discriminative patch features are related to:

```text
patch-size heterogeneity
dominant patch formation
spatial aggregation
centroid displacement
boundary and perimeter structure
fragmentation
```

This supports the central interpretation that spatial critical-transition risk is reflected in changes in patch structure rather than only in scalar abundance or traditional temporal EWS signals.

---

## 7. Main Conclusion

The v1.1 analysis supports the following conclusion:

```text
Classic patch features provide not only strong predictive performance but also interpretable spatial-structural evidence for early warning.
```

The lead-time proxy result further suggests that patch-based methods can provide stable early-warning signals before critical transition, while dynamic patch indicators may act as sensitive but noisy warning signals.

---

## 8. Limitation

The lead-time result in v1.1 is a proxy rather than strict first-alarm lead time because the current prediction files do not store simulation ID, time index, or critical time.

A future version should modify the training pipeline to save sample-level temporal metadata:

```text
simulation ID
time index
critical time
remaining time
```

This would allow strict trajectory-level first-alarm lead-time analysis.

---

## 9. Version Position

The version sequence is now:

```text
v0.7   classic patch baseline
v0.8   dynamic patch indicators v1
v0.9   feature group comparison
v0.9.1 dynamic patch v2 diagnostic
v1.0   traditional EWS baseline
v1.1   lead-time proxy and feature importance
```

The next possible version is:

```text
v1.2 = strict lead-time metadata or extended spatial model
```