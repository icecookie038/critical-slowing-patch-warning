# v2.0 Vegetation Cellular Automaton Validation Summary

## 1. Purpose

The purpose of v2.0 is to test whether the proposed pre-patch spatial organization indicators and PWSI can generalize beyond the SEIR spatial spreading model.

In v1.2–v1.4, the framework was validated under a synthetic SEIR spatial spreading system. However, SEIR is primarily a spreading process rather than a direct ecological degradation process. Therefore, v2.0 introduces a second ecological spatial transition system:

```text
Spatial Cellular Automaton Vegetation Degradation Model
```

The goal is to evaluate whether the same early-warning framework remains effective in a vegetation degradation system.

## 2. Vegetation Cellular Automaton Model

The vegetation cellular automaton model simulates a spatial degradation process in which each grid cell has a vegetation state:

```text
V(i, j, t) ∈ [0, 1]
```

where higher values represent healthier vegetation and lower values represent degraded vegetation.

The model includes the following ecological mechanisms:

```text
1. Gradually increasing external stress
2. Local positive feedback from neighboring vegetation
3. Spatial spread of degradation pressure
4. Spatial heterogeneity
5. Stochastic local perturbation
```

The model produces a transition process from healthy vegetation to local degradation, patch expansion, and eventually system-level degradation.

## 3. Event-based Degradation Label

The degradation event was defined using observable system-level degradation conditions.

The event time was defined as the first time when:

```text
vegetation_cover <= theta_cover
and
degraded_area >= theta_degraded
```

for several consecutive time steps.

The warning label followed the same event-based logic as v1.2:

```text
y_h = 1 if 0 < critical_time - t <= horizon
y_h = 0 otherwise
```

The main setting was:

```text
horizon = 30
persistent_k = 3
seed = 42
```

## 4. Dataset Construction

A formal vegetation CA dataset was generated using 200 simulations.

The initial dataset included direct event-defining variables such as vegetation cover and degraded area in the visible patch feature set. This produced nearly perfect performance and was identified as a potential label-leakage risk.

To avoid direct leakage, a clean version of the vegetation CA dataset was constructed. In the clean version, direct event-defining variables were removed from the model input features.

The clean dataset retained spatial-structure indicators such as:

```text
std_vegetation
var_vegetation
largest_degraded_patch_ratio
degraded_patch_count_ratio
mean_degraded_patch_area_ratio
largest_patch_to_degraded_area
degraded_edge_density
spatial_aggregation
neighbor_autocorr
gradient_mean
gradient_top10_mean
gradient_entropy
```

The direct event variables were retained only as metadata, not as model input features.

## 5. Clean Vegetation CA Dataset Summary

The clean vegetation CA dataset was successfully generated.

Dataset summary:

```text
Valid simulations: 200 / 200
X_img: (17750, 10, 1, 64, 64)
X_patch: (17750, 10, 12)
y_remaining: (17750,)
y_risk: (17750,)
risk ratio: 0.3380
mean critical_time: 100.32
min critical_time: 63
max critical_time: 141
```

This dataset provides a second ecological spatial transition system for cross-system validation.

## 6. Clean Visible Patch Baseline

The clean visible patch baseline was first evaluated to confirm that spatial patch indicators remain useful after removing direct event-defining variables.

Classification results showed strong but more realistic performance than the leakage-prone version:

```text
AUC ≈ 0.975–0.983
AUPRC ≈ 0.967–0.975
F1 ≈ 0.904–0.922
```

Strict first-alarm results:

| Model                | Valid alarm rate | Miss rate | Lead mean | Lead median | Pre-window alarm rate | Late alarm rate |
| -------------------- | ---------------: | --------: | --------: | ----------: | --------------------: | --------------: |
| LogisticRegression   |              1.0 |       0.0 |     26.72 |        29.0 |                  0.14 |             0.0 |
| ExtraTrees           |              1.0 |       0.0 |     26.42 |        28.0 |                  0.14 |             0.0 |
| RandomForest         |              1.0 |       0.0 |     26.22 |        27.5 |                  0.14 |             0.0 |
| MLP                  |              1.0 |       0.0 |     25.80 |        26.5 |                  0.14 |             0.0 |
| HistGradientBoosting |              1.0 |       0.0 |     25.56 |        26.5 |                  0.12 |             0.0 |

These results show that clean visible patch indicators provide effective early-warning signals without directly using event-defining variables.

## 7. Vegetation CA Prepatch Validation

The v1.3 prepatch indicator framework was then transferred to the vegetation CA dataset.

Two settings were tested:

```text
prepatch_only
visible_prepatch
```

### 7.1 Prepatch-only Results

Classification results showed strong performance:

```text
AUC ≈ 0.995–0.997
AUPRC ≈ 0.990–0.994
F1 ≈ 0.945–0.961
```

Strict first-alarm results:

| Model                | Valid alarm rate | Miss rate | Lead mean | Lead median | Pre-window alarm rate | Late alarm rate |
| -------------------- | ---------------: | --------: | --------: | ----------: | --------------------: | --------------: |
| HistGradientBoosting |              1.0 |       0.0 |     29.16 |        30.0 |                  0.42 |             0.0 |
| RandomForest         |              1.0 |       0.0 |     28.82 |        30.0 |                  0.36 |             0.0 |
| ExtraTrees           |              1.0 |       0.0 |     28.76 |        30.0 |                  0.34 |             0.0 |
| LogisticRegression   |              1.0 |       0.0 |     28.70 |        30.0 |                  0.14 |             0.0 |
| MLP                  |              1.0 |       0.0 |     28.30 |        29.5 |                  0.22 |             0.0 |

The prepatch-only results show that pre-patch spatial organization indicators can independently provide early-warning information in the vegetation CA system.

### 7.2 Visible Patch + Prepatch Results

Classification results were also strong:

```text
AUC ≈ 0.997–0.998
AUPRC ≈ 0.995–0.996
F1 ≈ 0.963–0.967
```

Strict first-alarm results:

| Model                | Valid alarm rate | Miss rate | Lead mean | Lead median | Pre-window alarm rate | Late alarm rate |
| -------------------- | ---------------: | --------: | --------: | ----------: | --------------------: | --------------: |
| LogisticRegression   |              1.0 |       0.0 |     29.12 |        30.0 |                  0.10 |             0.0 |
| MLP                  |              1.0 |       0.0 |     28.94 |        30.0 |                  0.16 |             0.0 |
| HistGradientBoosting |              1.0 |       0.0 |     28.80 |        30.0 |                  0.18 |             0.0 |
| ExtraTrees           |              1.0 |       0.0 |     28.70 |        29.5 |                  0.16 |             0.0 |
| RandomForest         |              1.0 |       0.0 |     28.56 |        29.0 |                  0.12 |             0.0 |

The visible_prepatch setting preserved high lead time while reducing pre-window alarm rates compared with several prepatch-only models.

## 8. Vegetation CA PWSI Validation

The v1.4 PWSI framework was also transferred to the vegetation CA system.

Two settings were tested:

```text
PWSI_equal only
visible patch + PWSI_equal
```

### 8.1 PWSI_equal Only

Strict first-alarm results:

| Model                | Valid alarm rate | Miss rate | Lead mean | Lead median | Pre-window alarm rate | Late alarm rate |
| -------------------- | ---------------: | --------: | --------: | ----------: | --------------------: | --------------: |
| HistGradientBoosting |             0.96 |      0.04 |     26.73 |        30.0 |                  0.54 |             0.0 |
| ExtraTrees           |             0.96 |      0.04 |     26.63 |        30.0 |                  0.50 |             0.0 |
| MLP                  |             0.96 |      0.04 |     26.58 |        30.0 |                  0.52 |             0.0 |
| LogisticRegression   |             0.96 |      0.04 |     26.38 |        30.0 |                  0.50 |             0.0 |
| RandomForest         |             0.94 |      0.06 |     25.79 |        28.0 |                  0.40 |             0.0 |

PWSI_equal alone retained early-warning ability in the vegetation CA system, but some information loss was observed after compressing the full prepatch framework into a single index.

### 8.2 Visible Patch + PWSI_equal

Strict first-alarm results:

| Model                | Valid alarm rate | Miss rate | Lead mean | Lead median | Pre-window alarm rate | Late alarm rate |
| -------------------- | ---------------: | --------: | --------: | ----------: | --------------------: | --------------: |
| LogisticRegression   |              1.0 |       0.0 |     27.46 |        28.5 |                  0.16 |             0.0 |
| ExtraTrees           |              1.0 |       0.0 |     27.10 |        28.5 |                  0.20 |             0.0 |
| MLP                  |              1.0 |       0.0 |     26.76 |        27.5 |                  0.24 |             0.0 |
| HistGradientBoosting |              1.0 |       0.0 |     26.42 |        27.0 |                  0.20 |             0.0 |
| RandomForest         |              1.0 |       0.0 |     26.12 |        27.0 |                  0.14 |             0.0 |

The visible patch + PWSI_equal setting was more stable than PWSI_equal alone. It achieved zero miss rate across all tested models and maintained moderate-to-high lead time with controlled pre-window alarm rates.

## 9. Cross-system Interpretation

The vegetation CA results provide cross-system synthetic evidence beyond the SEIR spatial spreading model.

The main findings are:

```text
1. Clean visible patch indicators remain effective after removing direct event-defining variables.
2. Prepatch indicators can independently provide early-warning information in vegetation degradation.
3. Visible patch + prepatch provides a strong and stable representation in vegetation CA.
4. PWSI_equal is transferable to vegetation CA, but single-index compression loses some information.
5. Visible patch + PWSI_equal provides a compact and more stable applied warning representation.
```

The results support the claim that the proposed prepatch framework and PWSI are not limited to the SEIR model. They also work in a second ecological spatial transition system.

## 10. Main v2.0 Conclusion

The main conclusion of v2.0 is:

```text
The proposed pre-patch spatial organization indicators and PWSI are not specific to the SEIR spatial spreading model. In a second ecological spatial transition system represented by a vegetation cellular automaton degradation model, prepatch indicators and PWSI also provided early-warning information. Full prepatch indicators achieved stronger mechanism-level performance, while PWSI_equal provided a compact and interpretable ecological warning index.
```

## 11. Implication for the Main Paper

The v2.0 results upgrade the paper from a single-system simulation study to a cross-system synthetic validation study.

The recommended framing is:

```text
From pre-patch spatial organization to visible patch structure:
interpretable spatial indicators and PWSI for early warning of ecological critical transitions across spatial spreading and vegetation degradation systems.
```

The main paper should emphasize:

```text
1. SEIR provides a controlled spatial spreading benchmark.
2. Vegetation CA provides a second ecological degradation benchmark.
3. Prepatch indicators work across both systems.
4. PWSI_equal provides a compact ecological warning index.
5. Real remote-sensing or pine-wilt-disease validation remains the next external validation step.
```

## 12. Limitations

The vegetation CA model is still a synthetic model. It improves ecological relevance compared with SEIR alone, but it does not replace real remote-sensing validation.

The current v2.0 results should therefore be interpreted as cross-system synthetic validation, not as real-world empirical validation.

The next step should be a remote-sensing or pine-wilt-disease case study.

## 13. Next Step

The next stage is:

```text
v2.1 Remote-sensing / pine-wilt-disease case study
```

The purpose of v2.1 is to test whether PWSI and prepatch spatial organization indicators can provide meaningful warning signals in real ecological monitoring data.

A practical minimum goal is:

```text
Use remote-sensing vegetation index or forest disturbance data to compute PWSI trajectories, spatial risk maps, and disturbed-vs-undisturbed comparisons.
```

A stronger goal is:

```text
Use pine-wilt-disease or forest disturbance event data to evaluate first-alarm lead time in real ecological data.
```
