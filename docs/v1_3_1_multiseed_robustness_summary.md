# v1.3.1 Multi-seed Robustness Summary

## 1. Purpose

The purpose of v1.3.1 is to test whether the v1.3 pre-patch spatial organization indicators are robust across different random seeds.

The main setting is:

```text
horizon = 30
persistent_k = 3
seeds = 42, 123, 2026
```

The tested feature settings are:

```text
visible_patch
prepatch_only
visible_prepatch
```

CNN-GRU was not rerun in this stage, because the purpose of v1.3.1 is to test the robustness of interpretable patch and pre-patch indicators.

## 2. Main Finding

The multi-seed results show that pre-patch indicators are not specific to seed42.

Across seed123 and seed2026, prepatch_only consistently produced valid-window alarms with zero miss rate and zero late-alarm rate in the tested models.

The benefit of combining visible patch and prepatch indicators was model-dependent. In some models, visible_prepatch improved lead time or reduced pre-window false alarms. However, it did not uniformly outperform visible_patch or prepatch_only across all models.

Therefore, the correct interpretation is:

```text
Pre-patch indicators provide robust independent early-warning information across random seeds, while the combined visible-prepatch setting provides model-dependent gains and may reduce pre-window false alarms in selected models.
```

## 3. Seed42 Reference Results

The seed42 results were used as the original v1.3 reference experiment.

Under the main setting of horizon = 30 and persistent_k = 3, the seed42 experiments showed that prepatch_only could independently generate valid-window alarms, while visible_prepatch provided an interpretable combination of early spatial organization and visible patch structure.

The seed42 results supported the initial conclusion that pre-patch spatial organization indicators contain early-warning information before visible patch dominance.

## 4. Seed123 Results

### 4.1 visible_patch

| Model                | Lead mean | Lead median | Pre-window alarm rate |
| -------------------- | --------: | ----------: | --------------------: |
| MLP                  |     27.83 |          30 |                 0.586 |
| HistGradientBoosting |     27.03 |          29 |                 0.483 |
| ExtraTrees           |     26.69 |          29 |                 0.483 |
| RandomForest         |     25.62 |          27 |                 0.414 |
| LogisticRegression   |     25.00 |          27 |                 0.552 |

### 4.2 prepatch_only

| Model                | Lead mean | Lead median | Pre-window alarm rate |
| -------------------- | --------: | ----------: | --------------------: |
| LogisticRegression   |     28.45 |          30 |                 0.621 |
| ExtraTrees           |     27.93 |          30 |                 0.552 |
| MLP                  |     27.38 |          30 |                 0.621 |
| RandomForest         |     26.90 |          28 |                 0.517 |
| HistGradientBoosting |     24.86 |          26 |                 0.483 |

### 4.3 visible_prepatch

| Model                | Lead mean | Lead median | Pre-window alarm rate |
| -------------------- | --------: | ----------: | --------------------: |
| MLP                  |     27.83 |          28 |                 0.586 |
| HistGradientBoosting |     27.76 |          30 |                 0.517 |
| RandomForest         |     26.66 |          28 |                 0.448 |
| LogisticRegression   |     25.66 |          27 |                 0.552 |
| ExtraTrees           |     24.38 |          26 |                 0.414 |

## 5. Seed2026 Results

### 5.1 visible_patch

| Model                | Lead mean | Lead median | Pre-window alarm rate | Miss rate |
| -------------------- | --------: | ----------: | --------------------: | --------: |
| RandomForest         |     26.50 |        28.5 |                 0.438 |     0.000 |
| LogisticRegression   |     26.38 |          27 |                 0.406 |     0.000 |
| MLP                  |     25.71 |          27 |                 0.375 |     0.031 |
| ExtraTrees           |     25.59 |          27 |                 0.281 |     0.000 |
| HistGradientBoosting |     25.00 |          26 |                 0.313 |     0.000 |

### 5.2 prepatch_only

| Model                | Lead mean | Lead median | Pre-window alarm rate | Miss rate |
| -------------------- | --------: | ----------: | --------------------: | --------: |
| ExtraTrees           |     28.09 |          30 |                 0.406 |     0.000 |
| RandomForest         |     27.13 |          30 |                 0.406 |     0.000 |
| LogisticRegression   |     26.78 |          29 |                 0.406 |     0.000 |
| HistGradientBoosting |     26.25 |        28.5 |                 0.281 |     0.000 |
| MLP                  |     25.53 |          28 |                 0.281 |     0.000 |

### 5.3 visible_prepatch

| Model                | Lead mean | Lead median | Pre-window alarm rate | Miss rate |
| -------------------- | --------: | ----------: | --------------------: | --------: |
| RandomForest         |     26.69 |          29 |                 0.406 |     0.000 |
| HistGradientBoosting |     26.44 |          28 |                 0.375 |     0.000 |
| LogisticRegression   |     25.53 |          26 |                 0.281 |     0.000 |
| ExtraTrees           |     25.38 |          27 |                 0.188 |     0.000 |
| MLP                  |     23.94 |          24 |                 0.344 |     0.000 |

## 6. Cross-seed Interpretation

The seed123 and seed2026 experiments confirm that the prepatch indicators are not an artifact of seed42.

The most stable conclusion is not that visible_prepatch always improves performance. Instead, the robust conclusion is:

```text
Pre-patch spatial organization indicators can independently provide early-warning information before visible patch dominance.
```

The visible_prepatch setting should be interpreted as a combined representation of early spatial organization and visible patch structure. Its advantage is model-dependent: in some models it improves lead time, while in others it reduces pre-window false alarms.

## 7. Implication for the Main Paper

The main paper should emphasize:

```text
1. Event-based labels avoid AC1-defined circularity.
2. Prepatch indicators independently predict critical transitions.
3. Visible patch and prepatch indicators form an interpretable transition chain.
4. Persistent-alarm rules reduce premature false alarms.
5. Multi-seed results support robustness.
```

The paper should not claim that visible_prepatch uniformly outperforms visible_patch across all models.

## 8. Current Conclusion

The current v1.3.1 results support the following conclusion:

```text
Pre-patch spatial organization indicators provide robust and interpretable early-warning information across different random seeds. They are not merely seed-specific artifacts. However, their combination with visible patch indicators produces model-dependent gains rather than uniform performance improvement.
```

Therefore, the prepatch indicator framework should remain part of the main paper, while conservative feature selection and full sensitivity tables can be reported as robustness or supplementary evidence.

## 9. Next Step

The next stage is v1.4 PWSI construction.

PWSI should be based on the full prepatch indicator framework, with equal-weight and importance-weighted versions tested separately.

The recommended next version is:

```text
v1.4: Pre-patch Warning Signal Index, PWSI
```
