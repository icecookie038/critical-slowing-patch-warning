# Experiment Plan v0.9.1: Dynamic Patch v2 Indicators

## 1. Version Goal

The goal of v0.9.1 is to evaluate whether second-generation dynamic patch indicators can provide stronger predictive information than the first-generation dynamic patch indicators introduced in v0.8.

The v0.9 feature group comparison showed that the first-generation dynamic patch indicators had limited standalone predictive power under the current SEIR setting. Therefore, v0.9.1 introduces a new group of spreading- and connectivity-aware indicators.

This version is a diagnostic version. It is not designed to replace the v0.9 feature group experiment.

---

## 2. Motivation

The v0.8 dynamic patch indicators mainly described short-term changes in patch-size distribution and structural instability:

```text
PDSI: Patch-size Distribution Shift Index
DPCI: Dominant Patch Change / Growth Index
FAI: Fragmentation Acceleration Index
PSII: Patch-Structure Instability Index
```

However, the SEIR spatial spreading process is strongly related to:

```text
front propagation
boundary expansion
patch merging
giant component formation
percolation-like connectivity transition
```

Therefore, v0.9.1 introduces second-generation dynamic indicators that are more directly related to spatial spreading mechanisms.

---

## 3. Dynamic Patch v2 Indicators

The second-generation dynamic patch indicators are:

```text
BCI   Boundary Complexity Index
FPV   Front Propagation Velocity
PMR   Patch Merging Rate
GCGR  Giant Component Growth Rate
PPI   Percolation Proximity Index
```

### 3.1 BCI: Boundary Complexity Index

BCI measures the complexity of infected patch boundaries.

A simple grid-based form is:

```text
BCI(t) = Perimeter(t)^2 / Area(t)
```

A higher BCI indicates a more complex spreading boundary.

### 3.2 FPV: Front Propagation Velocity

FPV measures the expansion speed of the infected front.

Using the equivalent infected radius:

```text
R(t) = sqrt(Area(t) / pi)

FPV(t) = max(0, R(t) - R(t-1))
```

A higher FPV indicates faster spatial expansion.

### 3.3 PMR: Patch Merging Rate

PMR measures whether multiple small infected patches are merging into larger connected patches.

```text
PMR(t) = max(0, PatchCount(t-1) - PatchCount(t))
```

PMR is counted when the total infected area does not shrink.

### 3.4 GCGR: Giant Component Growth Rate

GCGR measures the growth of the dominant connected infected component.

```text
GCR(t) = LargestPatchArea(t) / TotalPatchArea(t)

GCGR(t) = max(0, GCR(t) - GCR(t-1))
```

A higher GCGR indicates that a dominant infected cluster is forming.

### 3.5 PPI: Percolation Proximity Index

PPI measures whether the largest infected patch is approaching system scale.

```text
PPI(t) = LargestPatchArea(t) / GridArea
```

A higher PPI suggests that the infected region is closer to a large-scale connected state.

---

## 4. Implementation

The v2 indicators are implemented in:

```text
src/features/dynamic_patch_v2_indicators.py
```

The training pipeline is extended in:

```text
scripts/train_deep_model.py
```

Two new feature groups are added:

```text
dynamic_patch_v2_only
classic_dynamic_v2_patch
```

Their meanings are:

```text
dynamic_patch_v2_only:
    only uses BCI, FPV, PMR, GCGR, and PPI

classic_dynamic_v2_patch:
    uses classic patch features + dynamic patch v2 indicators
```

The SEIR data generation model is not modified in v0.9.1.

---

## 5. Diagnostic Experiment

A seed=42 diagnostic experiment was conducted using the current SEIR setting:

```text
num_sims = 300
grid = 64
sim_steps = 200
horizon = 30
seed = 42
```

The tested feature groups were:

```text
dynamic_patch_v2_only
classic_dynamic_v2_patch
```

---

## 6. Diagnostic Results

The diagnostic results were:

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

Compared with the v0.9 results, dynamic_patch_v2_only only slightly improves over the first-generation dynamic_patch_only group.

The classic_dynamic_v2_patch group remains comparable to the classic patch baseline, but it does not clearly outperform it.

---

## 7. Interpretation

The v2 indicators are more mechanistically related to spatial spreading than the v1 indicators. However, under the current SEIR configuration, they still do not provide strong standalone predictive power.

The likely reason is that the current SEIR simulation produces relatively smooth spatial expansion. In this setting, the classic patch features already capture most of the discriminative state information, such as:

```text
infected area
largest patch area
largest patch ratio
patch count
spatial aggregation
temporal trend
```

Therefore, the additional v2 dynamic indicators do not substantially improve AUC under the current simulation setting.

---

## 8. Version Conclusion

The v0.9.1 diagnostic result suggests that dynamic patch v2 indicators are valid and interpretable, but they do not provide enough standalone predictive gain under the current SEIR model.

Therefore, v0.9.1 will not be expanded to a full 5-seed or 25-run experiment at this stage.

The current conclusion is:

```text
Dynamic patch v2 indicators are successfully implemented and connected to the training pipeline.
They provide spreading- and connectivity-aware interpretation.
However, they do not substantially improve predictive performance under the current SEIR setting.
```

---

## 9. Next Direction

The next improvement should focus on the data-generating process rather than adding more patch indicators to the current SEIR setting.

Possible next directions are:

```text
1. Extend the SEIR temporal window
2. Increase SEIR spatial complexity
3. Add spatial heterogeneity and stochasticity
4. Introduce a spatial stochastic cellular automaton model
5. Introduce a vegetation degradation model
6. Continue with traditional EWS baseline comparison
```

The recommended next formal version is:

```text
v1.0 = traditional early-warning signal baseline
```

A later model-extension version may include:

```text
extended SEIR
spatial stochastic cellular automaton
vegetation degradation model
```

---

## 10. Version Position

The current version sequence is:

```text
v0.7   classic patch baseline
v0.8   dynamic patch indicators v1
v0.9   feature group comparison
v0.9.1 dynamic patch indicators v2 diagnostic
v1.0   traditional EWS baseline
v1.1   lead time and feature importance
v1.2   extended spatial model or cellular automaton
```

v0.9.1 should be treated as a diagnostic result rather than a full experimental version.