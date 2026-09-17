# Experiment Plan v1.0: Traditional EWS Baseline

## 1. Version Goal

The goal of v1.0 is to compare the proposed patch-based early-warning framework with traditional early-warning signals.

The key question is:

```text
Are patch-structure features more effective than traditional early-warning signals for spatial SEIR critical-transition risk prediction?
```

This version does not modify the SEIR data generation model.

---

## 2. Motivation

Previous versions showed that:

```text
v0.7 = classic patch baseline
v0.8 = dynamic patch indicators v1
v0.9 = feature group comparison
v0.9.1 = dynamic patch v2 diagnostic
```

The v0.9 results showed that classic patch features perform strongly, while first-generation dynamic patch indicators have limited standalone predictive performance.

The v0.9.1 diagnostic further showed that second-generation dynamic patch indicators provide mechanistic interpretation but do not substantially improve prediction under the current SEIR setting.

Therefore, v1.0 introduces a traditional EWS baseline to determine whether the patch-based method provides advantages over standard early-warning indicators.

---

## 3. Traditional EWS Indicators

The traditional EWS feature set includes:

```text
total_infected
temporal_variance
temporal_ac1
temporal_skewness
return_rate
kendall_trend
spatial_variance
moran_i
spatial_skewness
patch_size_slope
```

These indicators represent common temporal and spatial early-warning signals, including variance, autocorrelation, trend, spatial clustering, and patch-size distribution slope.

---

## 4. Implementation

The traditional EWS indicators are implemented in:

```text
src/features/traditional_ews.py
```

The training pipeline is extended in:

```text
scripts/train_deep_model.py
```

The result collection script is:

```text
scripts/collect_traditional_ews_results.py
```

The method comparison table is exported by:

```text
scripts/export_method_comparison_tables.py
```

---

## 5. Experiment Setting

The v1.0 traditional EWS baseline uses the same SEIR setting as v0.9:

```text
num_sims = 300
grid = 64
sim_steps = 200
horizon = 30
```

The five seeds are:

```text
42
123
2026
3407
7777
```

The tested feature group is:

```text
traditional_ews_only
```

---

## 6. Result Files

The collected traditional EWS summaries are saved to:

```text
results_summary/traditional_ews_summary_all.csv
results_summary/traditional_ews_summary_grouped.csv
```

The final method comparison table is saved to:

```text
docs/paper_tables/method_comparison_table.csv
docs/paper_tables/method_comparison_table.md
docs/paper_tables/method_comparison_table.tex
```

---

## 7. Main Results

The traditional EWS baseline achieved:

```text
AUC       = 0.836 ± 0.042
AUPRC     = 0.781 ± 0.062
F1        = 0.722 ± 0.050
Accuracy  = 0.760 ± 0.045
Precision = 0.675 ± 0.064
Recall    = 0.777 ± 0.036
```

The full method comparison table shows:

```text
Traditional EWS         AUC = 0.836 ± 0.042
Image only              AUC = 0.846 ± 0.032
Classic patch           AUC = 0.867 ± 0.030
Dynamic patch v1        AUC = 0.538 ± 0.017
Classic + dynamic patch AUC = 0.868 ± 0.028
Full                    AUC = 0.869 ± 0.027
```

---

## 8. Interpretation

The traditional EWS baseline is effective and performs above random level.

However, classic patch features outperform traditional EWS in AUC, AUPRC, F1, and accuracy.

This suggests that patch-structure features provide a stronger and more compact representation for spatial SEIR critical-transition risk prediction.

The result also shows that the full model performs only slightly better than classic patch features alone, indicating that classic patch features already capture most of the discriminative spatial-structural information.

---

## 9. Main Conclusion

The v1.0 result supports the following conclusion:

```text
Traditional EWS indicators are useful but weaker than classic patch-structure features in the current spatial SEIR setting.
```

Therefore, the main paper direction should not be:

```text
Dynamic patch indicators significantly improve AUC.
```

Instead, the stronger and more accurate direction is:

```text
Patch-structure features provide a strong early-warning representation for spatial critical transitions.
Dynamic patch indicators provide additional temporal-structural interpretation.
Traditional EWS indicators provide a useful but weaker baseline.
```

---

## 10. Version Position

The version sequence is now:

```text
v0.7   classic patch baseline
v0.8   dynamic patch indicators v1
v0.9   feature group comparison
v0.9.1 dynamic patch v2 diagnostic
v1.0   traditional EWS baseline
v1.1   lead time and feature importance
```

The next version should focus on:

```text
v1.1 = lead time analysis + feature importance
```

Lead time is not included in v1.0 because it requires sample-level temporal metadata such as simulation ID, time index, remaining time, and critical time.