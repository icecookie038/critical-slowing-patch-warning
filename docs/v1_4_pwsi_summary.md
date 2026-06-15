# v1.4 PWSI Index Construction and Robustness Summary

## 1. Purpose

The purpose of v1.4 is to compress the v1.3 pre-patch spatial organization indicator framework into a compact and interpretable ecological warning index.

The proposed index is:

```text
PWSI = Pre-patch Warning Signal Index
```

The purpose of PWSI is not to replace the full prepatch feature set in predictive modeling. Instead, PWSI aims to provide a compact and interpretable indicator that can be used for ecological monitoring, comparison across systems, and future remote-sensing applications.

## 2. Background

In v1.3, fourteen prepatch indicators were constructed to describe spatial organization before visible patch dominance. These indicators were grouped into four mechanistic categories:

```text
Z_sync  = local synchronization
Z_conn  = spatial connectivity
Z_rigid = boundary rigidity
Z_mode  = dominant mode locking
```

The fourteen original prepatch indicators form the mechanism layer. The four group scores form the interpretation layer. PWSI forms the compact application layer.

The conceptual structure is:

```text
14 prepatch indicators
        ↓
Z_sync, Z_conn, Z_rigid, Z_mode
        ↓
PWSI
```

## 3. PWSI Definition

Two versions of PWSI were constructed.

### 3.1 Equal-weight PWSI

The main version is the equal-weight index:

```text
PWSI_equal = 0.25 * Z_sync
           + 0.25 * Z_conn
           + 0.25 * Z_rigid
           + 0.25 * Z_mode
```

This version was selected as the main PWSI because it is simple, transparent, and does not depend on model-specific feature importance.

### 3.2 Importance-weighted PWSI

A sensitivity version was also constructed using group-level feature importance from v1.3:

```text
PWSI_importance = w_sync  * Z_sync
                + w_conn  * Z_conn
                + w_rigid * Z_rigid
                + w_mode  * Z_mode
```

The group-level permutation importance values were:

| Group   | Group permutation importance |
| ------- | ---------------------------: |
| Z_sync  |                     0.093480 |
| Z_rigid |                     0.052703 |
| Z_conn  |                     0.050721 |
| Z_mode  |                     0.032717 |

After normalization, the approximate weights were:

| Group   | Approximate normalized weight |
| ------- | ----------------------------: |
| Z_sync  |                         0.407 |
| Z_rigid |                         0.230 |
| Z_conn  |                         0.221 |
| Z_mode  |                         0.142 |

PWSI_importance was retained as a sensitivity version, not as the main index.

## 4. Dataset Construction

Two v1.4 dataset types were first generated:

```text
seir_v1_4_pwsi_only_h30_seed42.npz
seir_v1_4_visible_pwsi_h30_seed42.npz
```

The first dataset contained:

```text
Z_sync
Z_conn
Z_rigid
Z_mode
PWSI_equal
PWSI_importance
```

The second dataset combined visible patch indicators with the compact PWSI representation.

However, because the first PWSI dataset still contained six features, a single-index ablation was then conducted.

Four single-index datasets were generated:

```text
seir_v1_4_pwsi_equal_only_h30_seed42.npz
seir_v1_4_pwsi_importance_only_h30_seed42.npz
seir_v1_4_visible_pwsi_equal_h30_seed42.npz
seir_v1_4_visible_pwsi_importance_h30_seed42.npz
```

The single-index ablation was designed to answer whether one PWSI value alone can provide valid early-warning signals.

## 5. Seed42 Results: Compact PWSI Feature Set

### 5.1 PWSI feature set only

This setting used the six compact PWSI-related features:

```text
Z_sync, Z_conn, Z_rigid, Z_mode, PWSI_equal, PWSI_importance
```

| Model                | Valid alarm rate | Miss rate | Lead mean | Lead median | Pre-window alarm rate |
| -------------------- | ---------------: | --------: | --------: | ----------: | --------------------: |
| ExtraTrees           |              1.0 |       0.0 |     25.77 |          27 |                 0.323 |
| RandomForest         |              1.0 |       0.0 |     25.65 |          27 |                 0.290 |
| MLP                  |              1.0 |       0.0 |     25.29 |          25 |                 0.323 |
| LogisticRegression   |              1.0 |       0.0 |     25.26 |          26 |                 0.323 |
| HistGradientBoosting |              1.0 |       0.0 |     25.23 |          25 |                 0.355 |

This result shows that the compact PWSI feature set can independently provide valid-window warnings.

### 5.2 Visible patch + compact PWSI feature set

| Model                | Valid alarm rate | Miss rate | Lead mean | Lead median | Pre-window alarm rate |
| -------------------- | ---------------: | --------: | --------: | ----------: | --------------------: |
| HistGradientBoosting |              1.0 |       0.0 |     28.61 |          30 |                 0.613 |
| ExtraTrees           |              1.0 |       0.0 |     27.55 |          30 |                 0.387 |
| LogisticRegression   |              1.0 |       0.0 |     26.68 |          29 |                 0.387 |
| RandomForest         |              1.0 |       0.0 |     26.55 |          29 |                 0.258 |
| MLP                  |              1.0 |       0.0 |     26.26 |          28 |                 0.484 |

This result shows that combining visible patch indicators with compact PWSI features preserves strong early-warning ability.

## 6. Seed42 Results: Single-index Ablation

### 6.1 PWSI_equal only

| Model                | Valid alarm rate | Miss rate | Lead mean | Lead median | Pre-window alarm rate |
| -------------------- | ---------------: | --------: | --------: | ----------: | --------------------: |
| HistGradientBoosting |              1.0 |       0.0 |     26.52 |          28 |                 0.484 |
| MLP                  |              1.0 |       0.0 |     25.84 |          27 |                 0.355 |
| LogisticRegression   |              1.0 |       0.0 |     25.84 |          27 |                 0.355 |
| ExtraTrees           |              1.0 |       0.0 |     23.65 |          25 |                 0.387 |
| RandomForest         |              1.0 |       0.0 |     23.29 |          25 |                 0.419 |

PWSI_equal alone achieved valid-window alarms with zero miss rate and zero late-alarm rate. This confirms that a single equal-weight PWSI can independently provide early-warning information.

### 6.2 PWSI_importance only

| Model                | Valid alarm rate | Miss rate | Lead mean | Lead median | Pre-window alarm rate |
| -------------------- | ---------------: | --------: | --------: | ----------: | --------------------: |
| LogisticRegression   |              1.0 |       0.0 |     26.13 |          28 |                 0.355 |
| MLP                  |              1.0 |       0.0 |     26.10 |          28 |                 0.355 |
| HistGradientBoosting |              1.0 |       0.0 |     24.58 |          27 |                 0.323 |
| ExtraTrees           |              1.0 |       0.0 |     23.55 |          24 |                 0.258 |
| RandomForest         |              1.0 |       0.0 |     23.32 |          24 |                 0.323 |

PWSI_importance also provided valid-window warnings, but it did not clearly outperform PWSI_equal.

### 6.3 Visible patch + PWSI_equal

| Model                | Valid alarm rate | Miss rate | Lead mean | Lead median | Pre-window alarm rate |
| -------------------- | ---------------: | --------: | --------: | ----------: | --------------------: |
| HistGradientBoosting |              1.0 |       0.0 |     28.32 |          30 |                 0.548 |
| ExtraTrees           |              1.0 |       0.0 |     27.55 |          29 |                 0.387 |
| RandomForest         |              1.0 |       0.0 |     27.00 |          29 |                 0.355 |
| MLP                  |              1.0 |       0.0 |     26.87 |          29 |                 0.516 |
| LogisticRegression   |              1.0 |       0.0 |     26.58 |          28 |                 0.355 |

Visible patch + PWSI_equal preserved strong warning ability while using only one prepatch summary index.

### 6.4 Visible patch + PWSI_importance

| Model                | Valid alarm rate | Miss rate | Lead mean | Lead median | Pre-window alarm rate |
| -------------------- | ---------------: | --------: | --------: | ----------: | --------------------: |
| HistGradientBoosting |              1.0 |       0.0 |     28.29 |          30 |                 0.580 |
| ExtraTrees           |              1.0 |       0.0 |     27.58 |          29 |                 0.452 |
| MLP                  |              1.0 |       0.0 |     26.58 |          28 |                 0.516 |
| LogisticRegression   |              1.0 |       0.0 |     26.55 |          28 |                 0.355 |
| RandomForest         |              1.0 |       0.0 |     25.94 |          28 |                 0.258 |

PWSI_importance did not provide a clear advantage over PWSI_equal. Therefore, PWSI_equal was selected as the main index.

## 7. Multi-seed Robustness of PWSI_equal

After selecting PWSI_equal as the main index, multi-seed robustness was tested using seed123 and seed2026.

The tested settings were:

```text
PWSI_equal only
visible patch + PWSI_equal
```

### 7.1 Seed123: PWSI_equal only

| Model                | Valid alarm rate | Miss rate | Lead mean | Lead median | Pre-window alarm rate |
| -------------------- | ---------------: | --------: | --------: | ----------: | --------------------: |
| MLP                  |              1.0 |       0.0 |     26.52 |          30 |                 0.483 |
| LogisticRegression   |              1.0 |       0.0 |     26.24 |          28 |                 0.483 |
| RandomForest         |              1.0 |       0.0 |     25.14 |          27 |                 0.586 |
| ExtraTrees           |              1.0 |       0.0 |     24.17 |          27 |                 0.517 |
| HistGradientBoosting |              1.0 |       0.0 |     24.07 |          24 |                 0.379 |

### 7.2 Seed123: Visible patch + PWSI_equal

| Model                | Valid alarm rate | Miss rate | Lead mean | Lead median | Pre-window alarm rate |
| -------------------- | ---------------: | --------: | --------: | ----------: | --------------------: |
| ExtraTrees           |              1.0 |       0.0 |     27.17 |          30 |                 0.483 |
| HistGradientBoosting |              1.0 |       0.0 |     27.03 |          29 |                 0.483 |
| MLP                  |              1.0 |       0.0 |     26.48 |          28 |                 0.483 |
| RandomForest         |              1.0 |       0.0 |     26.17 |          29 |                 0.414 |
| LogisticRegression   |              1.0 |       0.0 |     25.10 |          27 |                 0.586 |

### 7.3 Seed2026: PWSI_equal only

| Model                | Valid alarm rate | Miss rate | Lead mean | Lead median | Pre-window alarm rate |
| -------------------- | ---------------: | --------: | --------: | ----------: | --------------------: |
| RandomForest         |              1.0 |       0.0 |     25.81 |          28 |                 0.469 |
| ExtraTrees           |              1.0 |       0.0 |     25.19 |        27.5 |                 0.344 |
| HistGradientBoosting |              1.0 |       0.0 |     24.97 |        26.5 |                 0.438 |
| LogisticRegression   |              1.0 |       0.0 |     24.56 |          25 |                 0.281 |
| MLP                  |              1.0 |       0.0 |     24.47 |        24.5 |                 0.281 |

### 7.4 Seed2026: Visible patch + PWSI_equal

| Model                | Valid alarm rate | Miss rate | Lead mean | Lead median | Pre-window alarm rate |
| -------------------- | ---------------: | --------: | --------: | ----------: | --------------------: |
| ExtraTrees           |              1.0 |       0.0 |     25.78 |          27 |                 0.344 |
| LogisticRegression   |              1.0 |       0.0 |     25.72 |          26 |                 0.281 |
| HistGradientBoosting |              1.0 |       0.0 |     25.53 |        26.5 |                 0.344 |
| RandomForest         |              1.0 |       0.0 |     25.25 |        25.5 |                 0.219 |
| MLP                  |            0.969 |     0.031 |     25.19 |          25 |                 0.344 |

The multi-seed results show that PWSI_equal is not specific to seed42. It preserves valid-window warning ability across different random seeds.

## 8. Interpretation

The PWSI results support four main conclusions.

First, PWSI_equal can independently provide valid-window early-warning signals. Across seed42, seed123, and seed2026, PWSI_equal only achieved valid-window alarms with zero miss rate in most tested models.

Second, PWSI_equal is more suitable than PWSI_importance as the main index. Although PWSI_importance was also effective, it did not clearly outperform PWSI_equal. PWSI_equal is simpler, more transparent, and does not depend on model-specific weights.

Third, visible patch + PWSI_equal provides a compact representation of both visible patch structure and pre-patch spatial organization. This setting generally preserved stable warning ability across seeds.

Fourth, PWSI should not be interpreted as outperforming the full prepatch feature set. The full prepatch feature set remains more informative for detailed predictive modeling and mechanism analysis. PWSI is better interpreted as a compact ecological warning index for monitoring and application.

## 9. Main v1.4 Conclusion

The main conclusion of v1.4 is:

```text
PWSI_equal compresses the full prepatch spatial organization framework into a compact and interpretable early-warning index. It preserves valid-window warning ability across random seeds and can be combined with visible patch indicators to represent both early spatial organization and visible patch structure.
```

Therefore:

```text
Main PWSI index: PWSI_equal
Sensitivity index: PWSI_importance
Mechanism layer: Z_sync, Z_conn, Z_rigid, Z_mode
Main compact application setting: visible patch + PWSI_equal
Full predictive mechanism setting: full prepatch indicators
```

## 10. Implication for the Main Paper

For the main paper, the recommended interpretation is:

```text
The full prepatch indicator framework provides the strongest mechanism-level representation.
PWSI_equal provides a compact and interpretable ecological warning index.
Visible patch + PWSI_equal provides an applied compact representation that combines visible patch structure with pre-patch spatial organization.
```

The paper should not claim that PWSI uniformly outperforms the full prepatch feature set. Instead, it should claim that PWSI compresses the prepatch framework into a simpler index while preserving valid-window early-warning ability.

## 11. Next Step

The next stage is:

```text
v2.0: Spatial cellular automaton vegetation degradation model
```

The purpose of v2.0 is to test whether the prepatch indicator framework and PWSI are useful beyond the SEIR spatial spreading system.

The expected v2.0 goal is:

```text
Validate prepatch indicators and PWSI under a second ecological spatial transition model.
```
