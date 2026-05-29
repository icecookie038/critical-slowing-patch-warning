# Draft Discussion Section

## 4. Discussion

### 4.1 Main findings

This study evaluated a patch-based early-warning framework for spatial critical-transition risk prediction using a spatial SEIR simulation system.

The main finding is that classic patch-structure features provide a strong and interpretable representation for early warning. Compared with traditional early-warning signals and image-only inputs, classic patch features achieved stronger and more stable predictive performance.

The main method comparison showed:

```text
Traditional EWS:         AUC = 0.836 ± 0.042
Image only:              AUC = 0.846 ± 0.032
Classic patch:           AUC = 0.867 ± 0.030
Dynamic patch v1:        AUC = 0.538 ± 0.017
Classic + dynamic patch: AUC = 0.868 ± 0.028
Full:                    AUC = 0.869 ± 0.027
```

These results suggest that patch-structure features can capture important spatial warning information in the SEIR spreading process.

The full model achieved the highest AUC, but its improvement over the classic patch model was small. This indicates that classic patch features already contain most of the discriminative spatial-structural information under the current SEIR setting.

Therefore, the central conclusion is:

```text
Patch-structure features provide a strong, compact, and interpretable early-warning representation for spatial critical transitions.
```

---

### 4.2 Why patch-structure features outperform traditional EWS

Traditional early-warning signals are widely used to detect critical slowing down and rising systemic risk. They usually focus on scalar time-series properties, such as variance, autocorrelation, skewness, return rate, and trend.

In this study, traditional EWS indicators were effective:

```text
Traditional EWS AUC = 0.836 ± 0.042
```

However, classic patch features performed better:

```text
Classic patch AUC = 0.867 ± 0.030
```

This improvement is meaningful because spatial spreading systems contain information that cannot be fully represented by scalar temporal indicators.

Patch features directly describe:

```text
patch size
dominant patch formation
patch-size heterogeneity
spatial aggregation
centroid displacement
perimeter and boundary structure
fragmentation
patch density
```

These properties are closely related to how local infections or disturbances expand into larger spatial clusters.

Traditional EWS can detect changes in overall system behavior, but patch features describe how the spatial structure of the infected region changes. This makes patch features more suitable for spatial risk prediction.

---

### 4.3 Why classic patch features are strong

The feature-value diagnostic importance analysis showed that the most discriminative classic patch features include:

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

These features mainly describe four types of spatial information.

First, patch-size heterogeneity is important. Features such as `patch_area_cv`, `patch_area_std`, and `gini` indicate whether the infected region is dominated by uneven patch-size distributions.

Second, dominant patch formation is important. Features such as `largest_patch_area` and `largest_patch_ratio` describe whether one large infected cluster is becoming dominant.

Third, spatial organization is important. Features such as `aggregation_index`, `fragmentation`, and `edge_density` reflect whether infected cells are spatially aggregated or fragmented.

Fourth, spatial displacement is important. Features such as `centroid_x`, `centroid_y`, `spread_x`, and `spread_y` describe the spatial movement and expansion of the infected region.

Together, these results suggest that early-warning information is encoded not only in the amount of infection, but also in the spatial organization of infection patches.

---

### 4.4 Role of dynamic patch indicators

The dynamic patch indicators were designed to describe temporal changes in patch structure.

The first-generation dynamic patch indicators included:

```text
PDSI: Patch-size Distribution Shift Index
DPCI: Dominant Patch Change / Growth Index
FAI: Fragmentation Acceleration Index
PSII: Patch-Structure Instability Index
```

However, dynamic patch v1 alone showed weak predictive performance:

```text
AUC = 0.538 ± 0.017
AUPRC = 0.443 ± 0.016
Precision = 0.401 ± 0.022
Recall = 1.000 ± 0.000
```

This result indicates that dynamic patch v1 is highly sensitive but noisy. It tends to detect many risk samples, but it also produces many false alarms.

Therefore, dynamic patch indicators should not be presented as strong standalone predictors under the current SEIR setting.

Their more appropriate role is:

```text
temporal-structural interpretation
sensitive early warning signal
mechanistic description of patch evolution
```

The second-generation dynamic patch indicators were also tested diagnostically. They included:

```text
BCI: Boundary Complexity Index
FPV: Front Propagation Velocity
PMR: Patch Merging Rate
GCGR: Giant Component Growth Rate
PPI: Percolation Proximity Index
```

The seed=42 diagnostic result showed:

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

The v2 indicators slightly improved over v1, but the improvement was not large enough to justify a full five-seed expansion under the current SEIR setting.

This suggests that the limited contribution of dynamic patch indicators may be caused by the relatively smooth spatial expansion produced by the current SEIR model.

---

### 4.5 Why dynamic indicators may become more useful in complex spatial systems

Although dynamic patch indicators did not strongly improve prediction in the current SEIR setting, this does not mean they are useless.

The current SEIR model produces relatively smooth spatial spreading. In this setting, classic patch features already capture most of the important spatial state information.

Dynamic indicators may become more useful in systems with stronger temporal and spatial complexity, such as:

```text
patch birth
patch merging
patch fragmentation
local stochastic outbreaks
spatial barriers
heterogeneous transmission
percolation-like transitions
abrupt local collapses
```

These processes are not strongly represented in the current smooth SEIR setting.

Therefore, future experiments should test dynamic patch indicators in more complex data-generating systems, especially:

```text
extended SEIR with spatial heterogeneity and stochasticity
spatial stochastic cellular automaton
vegetation degradation model
remote-sensing or forest pest case study
```

In such systems, dynamic indicators may better capture early structural instability before large-scale transition.

---

### 4.6 Lead-time proxy interpretation

The lead-time proxy analysis evaluated whether different models could issue correct risk alarms before the transition.

The current prediction files contain:

```text
y_remaining_true
y_remaining_pred
y_risk_true
risk_prob
```

However, they do not contain:

```text
simulation ID
time index
critical time
```

Therefore, the current analysis reports a sample-level lead-time proxy, not a strict first-alarm lead time.

The lead-time proxy is defined as:

```text
the average remaining time to critical transition among correctly predicted risk samples
```

The lead-time proxy results showed that classic patch, classic + dynamic patch, and full models provided stable early-warning signals while maintaining reasonable alarm precision.

Dynamic patch v1 showed the largest mean lead-time proxy, but its alarm precision was low. This means it tends to warn earlier, but at the cost of more false alarms.

Therefore, dynamic patch indicators may be useful as sensitive warning signals, but they should be combined with stronger structural features or decision thresholds before being used as practical predictors.

---

### 4.7 Practical implication for intervention timing

One important motivation of early-warning analysis is to support intervention timing.

In the current results, patch-based methods provide two types of useful information.

First, they provide risk probabilities that can indicate whether a system is approaching a high-risk transition window.

Second, the patch features themselves provide interpretable spatial information about where and how the system is changing.

For example:

```text
increase in largest patch area
increase in largest patch ratio
increase in patch-size heterogeneity
increase in aggregation
shift in centroid location
change in fragmentation
```

These changes can help identify whether risk is driven by the formation of a dominant cluster, spatial aggregation, or spatial spread.

This means patch-based early warning may support both temporal and spatial intervention decisions.

Temporal intervention question:

```text
When should intervention be triggered?
```

Spatial intervention question:

```text
Which patch structure or spatial region should be monitored?
```

The current study mainly addresses temporal risk prediction. Future work should further connect patch indicators with spatial intervention strategies.

---

### 4.8 Limitations

This study has several limitations.

First, the current SEIR model is relatively smooth. This may reduce the additional value of dynamic patch indicators. More complex spatial dynamics may be needed to fully evaluate dynamic patch indicators.

Second, the current lead-time analysis is only a proxy. Strict first-alarm lead-time analysis requires simulation-level metadata, including simulation ID, time index, critical time, and remaining time.

Third, the current study is based on simulation data. Although the SEIR model provides a controlled environment, external validation using ecological or remote-sensing data is still needed.

Fourth, dynamic patch v2 was only tested as a diagnostic experiment with seed=42. It was not included in the main five-seed comparison because it did not show sufficient improvement under the current setting.

Fifth, the current feature importance analysis is a feature-value diagnostic analysis. It measures risk/non-risk separation, but it is not equivalent to SHAP, permutation importance, or causal feature attribution.

---

### 4.9 Future work

Future work should proceed in several directions.

First, strict lead-time metadata should be added to the training and prediction pipeline. Each validation sample should store:

```text
simulation ID
time index
critical time
remaining time
true risk label
predicted risk probability
```

This will allow strict trajectory-level first-alarm lead-time analysis.

Second, the SEIR model should be extended. An extended SEIR model should include:

```text
spatially heterogeneous transmission rate
spatially heterogeneous recovery rate
local stochastic perturbation
spatial barriers
heterogeneous diffusion
longer temporal windows
```

This would test whether dynamic patch indicators become more useful in more complex spatial spreading processes.

Third, a spatial stochastic cellular automaton model should be introduced as a second data-generating system. Such a model can naturally generate:

```text
patch birth
patch merging
patch fragmentation
local collapse
percolation-like spreading
abrupt spatial transitions
```

This would provide a better test environment for dynamic patch indicators.

Fourth, the framework should be evaluated using ecological or remote-sensing case studies, such as vegetation degradation, forest pest outbreaks, or disease spread in spatial landscapes.

---

### 4.10 Overall interpretation

The current evidence suggests that patch-structure features are a strong and interpretable representation for spatial early warning.

The strongest paper claim should be:

```text
Patch-structure features outperform traditional EWS baselines and provide interpretable spatial warning information in a spatial SEIR critical-transition prediction task.
```

The paper should not overclaim that dynamic patch indicators significantly improve predictive performance under the current setting.

A more accurate interpretation is:

```text
Classic patch features dominate predictive performance.
Dynamic patch indicators provide sensitive but noisy temporal-structural warning information.
Traditional EWS indicators are useful but weaker than patch-structure features.
```

Thus, the proposed patch-based framework provides both predictive strength and spatial-structural interpretability.

---

## 5. Conclusion

This study demonstrates that patch-structure features can serve as effective early-warning indicators for spatial critical-transition risk prediction.

Compared with traditional EWS, patch features provide stronger predictive performance and more direct spatial interpretation.

Although dynamic patch indicators do not substantially improve AUC under the current SEIR setting, they offer useful temporal-structural information and may become more valuable in more complex spatial systems.

Overall, the results support the development of patch-based early-warning frameworks for spatial ecological transitions, spreading systems, and spatial risk prediction.