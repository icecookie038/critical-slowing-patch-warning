# Paper Outline

## Tentative Title

Patch-structure features as interpretable early-warning signals for spatial critical transitions in SEIR spreading systems

中文暂定题目：

面向空间临界转变风险预警的斑块结构特征：基于 SEIR 空间传播模型的可解释早期预警框架

---

## 1. Core Paper Claim

The main claim of this paper is:

```text
Patch-structure features provide a strong and interpretable early-warning representation for spatial critical transitions.
```

The paper should not claim:

```text
Dynamic patch indicators significantly improve AUC.
```

The correct interpretation is:

```text
Classic patch indicators provide the strongest compact structural representation.
Dynamic patch indicators provide sensitive temporal-structural information, but they are noisy as standalone predictors.
Traditional EWS indicators are useful but weaker than patch-structure features.
```

---

## 2. Paper Type

This paper is positioned as a computational ecology / ecological informatics study.

Target journal direction:

```text
Ecological Informatics
```

The current version is mainly based on SEIR simulation experiments.

Future extensions may include:

```text
extended SEIR
spatial stochastic cellular automaton
vegetation degradation model
remote sensing / forest pest case study
```

---

## 3. Main Evidence Chain

The current evidence chain is:

```text
1. SEIR spatial spreading simulation
2. Classic patch feature extraction
3. Dynamic patch indicator development
4. Feature group comparison
5. Traditional EWS baseline comparison
6. Lead-time proxy analysis
7. Feature-value diagnostic importance
```

Completed versions:

```text
v0.7   classic patch baseline
v0.8   dynamic patch indicators v1
v0.9   feature group comparison
v0.9.1 dynamic patch indicators v2 diagnostic
v1.0   traditional EWS baseline
v1.1   lead-time proxy and feature importance
```

---

## 4. Proposed Paper Structure

# 1. Introduction

## 1.1 Background

Ecological systems may undergo abrupt critical transitions. Early-warning signals are important for detecting rising risk before regime shifts, outbreaks, or spatial spreading events.

Traditional early-warning signals usually focus on scalar time-series indicators such as variance, autocorrelation, skewness, and return rate.

However, many ecological transitions are spatial processes. Spatial spreading, aggregation, fragmentation, and patch dominance may contain additional warning information that cannot be fully captured by scalar indicators.

## 1.2 Research Gap

Most traditional EWS approaches do not fully describe patch-level spatial structure.

Existing spatial indicators often focus on static spatial states, while fewer studies compare patch-structure features, dynamic patch indicators, image-based models, and traditional EWS baselines under the same prediction task.

There is still a need to evaluate whether patch-structure features can provide effective and interpretable early-warning information for spatial critical-transition prediction.

## 1.3 Research Aim

This paper aims to evaluate whether patch-structure features can provide effective and interpretable early-warning information for spatial critical-transition risk prediction.

The central question is:

```text
Can patch-structure features outperform traditional early-warning signals in spatial SEIR critical-transition risk prediction?
```

## 1.4 Contributions

Main contributions:

```text
1. A patch-based early-warning framework for spatial SEIR critical-transition prediction.
2. A systematic comparison among traditional EWS, image-only, classic patch, dynamic patch, and full feature groups.
3. A set of dynamic patch indicators for temporal structural interpretation.
4. Lead-time proxy and feature-value diagnostic importance analyses.
5. Evidence that classic patch features outperform traditional EWS and provide interpretable spatial warning information.
```

---

# 2. Methods

## 2.1 Spatial SEIR Simulation

Describe the spatial SEIR spreading model.

Key elements:

```text
grid-based spatial domain
susceptible, exposed, infected, recovered states
local spreading process
critical transition / outbreak risk definition
multiple random seeds
```

Important setting:

```text
num_sims = 300
grid = 64
sim_steps = 200
horizon = 30
seeds = 42, 123, 2026, 3407, 7777
```

The current SEIR model is used as the baseline data-generating system.

## 2.2 Patch Extraction

Patch extraction workflow:

```text
spatial infection field
→ binary infected region
→ connected infected patches
→ patch feature extraction
```

The infected region is extracted from the spatial infection field using a threshold. Connected components are then identified as infection patches.

## 2.3 Classic Patch Features

Classic patch features include:

```text
patch count
total patch area
largest patch area
mean patch area
largest patch ratio
patch area variation
Gini
edge density
aggregation
centroid
spread
perimeter
compactness
fragmentation
patch density
```

These features describe the current spatial structure of infected patches.

## 2.4 Dynamic Patch Indicators

Dynamic patch v1 includes:

```text
PDSI: Patch-size Distribution Shift Index
DPCI: Dominant Patch Change / Growth Index
FAI: Fragmentation Acceleration Index
PSII: Patch-Structure Instability Index
```

Dynamic patch v2 diagnostic includes:

```text
BCI: Boundary Complexity Index
FPV: Front Propagation Velocity
PMR: Patch Merging Rate
GCGR: Giant Component Growth Rate
PPI: Percolation Proximity Index
```

Dynamic patch v2 is treated as diagnostic rather than a full formal comparison because it was only tested on seed=42 and did not show sufficient performance gain.

## 2.5 Traditional EWS Baseline

Traditional EWS features include:

```text
total infected
temporal variance
temporal AC1
temporal skewness
return rate
Kendall trend
spatial variance
Moran's I
spatial skewness
patch-size slope
```

These indicators represent common temporal and spatial early-warning signals.

## 2.6 Prediction Model

The prediction model uses spatial and patch-based inputs.

Main components:

```text
CNN branch for spatial image features
GRU branch for temporal sequence learning
patch feature input branch
risk classification output
```

Feature groups:

```text
traditional_ews_only
image_only
classic_patch_only
dynamic_patch_only
classic_dynamic_patch
full
```

## 2.7 Evaluation Metrics

Classification metrics:

```text
AUC
AUPRC
F1
Accuracy
Precision
Recall
```

Additional analyses:

```text
lead-time proxy
feature-value diagnostic importance
```

The lead-time result is currently a proxy because the prediction files do not store simulation ID, time index, or critical time.

---

# 3. Results

## 3.1 Method Comparison

Main result:

```text
Classic patch, classic + dynamic patch, and full models perform best.
Traditional EWS is effective but weaker.
Dynamic patch v1 alone is weak and noisy.
```

Main table:

```text
docs/paper_tables/method_comparison_table.md
```

Key AUC results:

```text
Traditional EWS:         AUC = 0.836 ± 0.042
Image only:              AUC = 0.846 ± 0.032
Classic patch:           AUC = 0.867 ± 0.030
Dynamic patch v1:        AUC = 0.538 ± 0.017
Classic + dynamic patch: AUC = 0.868 ± 0.028
Full:                    AUC = 0.869 ± 0.027
```

## 3.2 Comparison with Traditional EWS

Traditional EWS:

```text
AUC = 0.836 ± 0.042
```

Classic patch:

```text
AUC = 0.867 ± 0.030
```

Interpretation:

```text
Patch-structure features outperform traditional EWS in the current spatial SEIR setting.
```

## 3.3 Role of Dynamic Patch Indicators

Dynamic patch v1:

```text
AUC = 0.538 ± 0.017
Recall = 1.000 ± 0.000
Precision = 0.401 ± 0.022
```

Interpretation:

```text
Dynamic patch v1 is sensitive but noisy.
It is useful for temporal-structural interpretation but not as a standalone predictor.
```

Dynamic patch v2 diagnostic:

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

Interpretation:

```text
Dynamic patch v2 only slightly improves over dynamic patch v1 under the current SEIR setting.
It is kept as a diagnostic result rather than a full formal method comparison.
```

## 3.4 Lead-Time Proxy

Main table:

```text
docs/paper_tables/lead_time_proxy_table.md
```

Interpretation:

```text
Classic patch and full models provide stable early-warning signals with reasonable precision.
Dynamic patch v1 has earlier alarms but much lower precision.
```

Important note:

```text
This is a lead-time proxy, not strict first-alarm lead time.
```

## 3.5 Feature Importance

Main table:

```text
docs/paper_tables/feature_importance_table.md
```

Top features:

```text
patch_area_cv
centroid_y
largest_patch_area
patch_area_std
largest_patch_ratio
gini
mean_patch_area
aggregation_index
```

Interpretation:

```text
Risk is reflected in patch-size heterogeneity, dominant patch formation, spatial aggregation, and centroid displacement.
```

---

# 4. Discussion

## 4.1 Why Patch Features Work

Patch features directly describe spatial structure, including patch size, dominance, aggregation, fragmentation, and spatial displacement.

These properties are more suitable for spatial spreading systems than scalar temporal EWS alone.

## 4.2 Why Dynamic Patch Indicators Are Noisy

Dynamic patch indicators focus on temporal structural change.

In the current SEIR model, spatial spreading is relatively smooth, so classic patch features already capture most discriminative information.

Dynamic indicators may become more useful in systems with stronger stochasticity, patch merging, fragmentation, and percolation-like transitions.

## 4.3 Comparison with Traditional EWS

Traditional EWS is useful but less spatially explicit.

Patch features provide a stronger representation for spatial SEIR critical-transition risk prediction.

## 4.4 Limitations

Current limitations:

```text
1. SEIR simulation is relatively smooth.
2. Lead time is currently a proxy, not strict first-alarm lead time.
3. Dynamic patch v2 was only tested diagnostically.
4. No external remote-sensing or ecological case has been included yet.
```

## 4.5 Future Work

Future versions:

```text
v1.2 strict lead-time metadata
v1.3 extended SEIR
v1.4 spatial stochastic cellular automaton
v1.5 vegetation degradation or remote-sensing validation
```

---

# 5. Conclusion

Main conclusion:

```text
Patch-structure features provide a strong, compact, and interpretable early-warning representation for spatial critical transitions.
```

Secondary conclusion:

```text
Traditional EWS indicators are useful but weaker than patch features.
Dynamic patch indicators provide sensitive temporal-structural information but are noisy as standalone predictors under the current SEIR setting.
```

---

## 6. Current Tables

Main tables:

```text
docs/paper_tables/method_comparison_table.md
docs/paper_tables/lead_time_proxy_table.md
docs/paper_tables/feature_importance_table.md
docs/paper_tables/robustness_comparison_table.md
```

---

## 7. Current Main Figures To Prepare Later

Planned figures:

```text
Figure 1. Overall framework
Figure 2. SEIR spatial spreading examples
Figure 3. Patch extraction workflow
Figure 4. Dynamic patch indicators
Figure 5. Method comparison
Figure 6. Traditional EWS comparison
Figure 7. Lead-time proxy
Figure 8. Feature importance
```

---

## 8. Next Technical Version

The next technical version should be:

```text
v1.2 strict lead-time metadata
```

Purpose:

```text
Save simulation ID, time index, critical time, remaining time, true risk label, and predicted risk probability for each validation sample.
```

This will allow strict trajectory-level first-alarm lead-time analysis.