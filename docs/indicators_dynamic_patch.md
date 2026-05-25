# v0.8 Dynamic Patch Indicators

## 1. Purpose

This document defines the dynamic patch indicators introduced in v0.8.

The previous versions v0.1–v0.7 are retained as the classic patch baseline. The v0.8 indicators are not used to replace the classic patch metrics. Instead, they extend the original patch feature system by describing temporal changes in patch structure.

The original classic patch workflow is:

```text
spatial infection field
→ binary infected region
→ connected infected patches
→ patch area distribution
→ patch warning features
→ critical transition risk prediction
```

The v0.8 dynamic patch indicators extend this workflow by measuring how patch structures change over time.

---

## 2. Classic Patch Baseline

The classic patch indicators from v0.1–v0.7 include:

```text
Patch count
Largest patch area
Mean patch area
Largest patch ratio
Gini
JSD
Spatial aggregation
AC1
Variance
Trend
```

These indicators describe the patch state at a given time window.

In contrast, the v0.8 indicators focus on temporal structural changes.

---

## 3. PDSI: Patch-size Distribution Shift Index

PDSI measures the temporal shift of the patch-size distribution.

Let \( P_t \) be the normalized patch-size distribution at time \( t \), and \( P_{t-\tau} \) be the distribution at a previous time step.

The Patch-size Distribution Shift Index is defined as:

\[
PDSI(t) = JSD(P_t, P_{t-\tau})
\]

where \( JSD \) is the Jensen-Shannon distance and \( \tau \) is the temporal lag.

Interpretation:

```text
larger PDSI
= stronger temporal shift in patch-size structure
= stronger spatial structural instability
```

Important implementation rule:

```text
Patch-size bins must be fixed across the whole time series.
They must not be recomputed at each time step.
```

---

## 4. DPCI: Dominant Patch Change / Growth Index

DPCI measures the growth rate of the dominant patch.

Let \( L_t \) be the largest patch area at time \( t \). DPCI is defined as:

\[
DPCI(t) = \frac{\max(0, L_t - L_{t-\tau})}{\max(L_{t-\tau}, 1)}
\]

Interpretation:

```text
larger DPCI
= faster growth of the dominant patch
= stronger expansion of the target spatial state
```

For the SEIR infection-spreading system, infected patches usually expand before or during outbreak transition. Therefore, DPCI is the preferred dominant-patch indicator for SEIR experiments.

---

## 5. DPCR: Dominant Patch Collapse Rate

DPCR measures the collapse rate of the dominant patch.

It is defined as:

\[
DPCR(t) = \frac{\max(0, L_{t-\tau} - L_t)}{\max(L_{t-\tau}, 1)}
\]

Interpretation:

```text
larger DPCR
= faster collapse of the dominant patch
= stronger degradation of the target spatial state
```

Important distinction:

```text
DPCI is mainly used for SEIR infected-patch expansion.
DPCR is mainly used for vegetation degradation or healthy-patch collapse.
```

Therefore, DPCR is not used as the main dominant-patch indicator in SEIR expansion experiments.

---

## 6. FAI: Fragmentation Acceleration Index

FAI measures the acceleration of fragmentation.

First define fragmentation density:

\[
F(t) = \frac{N_t}{A_t + \epsilon}
\]

where:

```text
N_t = number of patches at time t
A_t = total patch area at time t
epsilon = small constant to avoid division by zero
```

Then FAI is defined as:

\[
FAI(t) = \max(0, F(t) - 2F(t-\tau) + F(t-2\tau))
\]

Interpretation:

```text
larger FAI
= faster acceleration of fragmentation
= increasing spatial instability
```

---

## 7. PSII: Patch-Structure Instability Index

PSII is a composite dynamic patch instability index.

For SEIR infection expansion:

\[
PSII(t) = mean\left( z^+(PDSI), z^+(DPCI), z^+(FAI) \right)
\]

For vegetation degradation or healthy-patch collapse:

\[
PSII(t) = mean\left( z^+(PDSI), z^+(DPCR), z^+(FAI) \right)
\]

where \( z^+ \) means the positive part of the standardized z-score:

\[
z^+(x) = \max(0, z(x))
\]

Implementation rules:

```text
1. The z-score mean and standard deviation must be estimated only from the baseline or training period.
2. The whole time series must not be used to fit PSII normalization.
3. A minimum standard deviation is used to avoid numerical explosion when the baseline is too stable.
4. A clipping threshold is used to prevent one component from dominating PSII.
```

---

## 8. Indicator Usage in SEIR Experiments

For SEIR infection expansion, the main dynamic indicators are:

```text
PDSI
DPCI
FAI
PSII
```

DPCR is still computed for completeness, but it is not the main SEIR expansion indicator.

Recommended SEIR dynamic feature group:

```text
dynamic_patch_only = PDSI + DPCI + FAI + PSII
```

---

## 9. Indicator Usage in Vegetation Degradation Experiments

For vegetation degradation or healthy-patch collapse, the main dynamic indicators are:

```text
PDSI
DPCR
FAI
PSII
```

Recommended vegetation degradation dynamic feature group:

```text
dynamic_patch_only = PDSI + DPCR + FAI + PSII
```

---

## 10. Version Position

The version relationship is:

```text
v0.7 = classic patch baseline
v0.8 = dynamic patch indicators
v0.9 = feature group comparison
v1.0 = traditional EWS baseline
v1.1 = lead time and feature importance
```

The v0.8 indicators are implemented in:

```text
src/features/dynamic_patch_indicators.py
```

The v0.8 pilot script is:

```text
scripts/run_dynamic_indicator_pilot.py
```

The pilot outputs are:

```text
results_summary/dynamic_patch_pilot_metrics.csv
figures/dynamic_patch_pilot/dynamic_indicators_trend.png
figures/dynamic_patch_pilot/patch_state_summary.png
```