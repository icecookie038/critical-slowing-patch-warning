# v2.3A Patch Formation Process Analysis Summary

## 1. Purpose

The purpose of v2.3A is to use the proposed prepatch indicators to analyze the gradual formation process of spatial patches before critical transition.

This experiment directly follows the idea that spatial patches are not static structures that suddenly appear. Instead, they gradually develop through a prepatch spatial organization process before becoming visible in the system state.

The main question is:

```text
Can the proposed prepatch indicators reveal the gradual formation process of spatial patches before visible patch emergence?
```

## 2. Dataset and Experimental Setting

The analysis was conducted on the vegetation CA system using seed42.

Input datasets:

```text
data/processed/v2_0_vegetation_ca_clean/vegetation_ca_h30_seed42.npz
data/processed/v2_0_vegetation_prepatch/seir_v1_3_prepatch_h30_seed42.npz
data/processed/v2_0_vegetation_pwsi/seir_v1_4_pwsi_only_h30_seed42.npz
```

The vegetation CA system was selected because it is closer to ecological degradation than the SEIR spatial spreading model.

All simulations were aligned by their critical transition time. The x-axis of the trajectory analysis is:

```text
time to critical transition
```

where 0 represents the critical transition time.

## 3. Indicator Groups

The following prepatch indicator groups were analyzed:

```text
Z_sync
Z_conn
Z_rigid
Z_mode
PWSI_equal
visible patch metric
```

Their meanings are:

```text
Z_sync: local synchronization before visible patch emergence
Z_conn: spatial connectivity and aggregation expansion
Z_rigid: boundary rigidity and morphological organization
Z_mode: dominant spatial mode locking
PWSI_equal: integrated prepatch warning signal index
visible patch metric: late-stage visible patch structure
```

## 4. Event-aligned Prepatch Dynamics

The event-aligned trajectory results show that multiple prepatch indicators increase before the critical transition.

The prepatch indicators, especially Z_sync, Z_conn, Z_mode, and PWSI_equal, show sustained increases during the pre-transition period.

This suggests that the system is already undergoing spatial organization before visible patch structures become dominant.

The visible patch metric shows a much sharper late-stage increase near the transition. Therefore, it was plotted separately to avoid compressing the prepatch indicator trajectories.

## 5. First-rise Time Comparison

The first-rise time analysis compares when each indicator first exceeds its early baseline level.

The first-rise rule was:

```text
indicator > baseline mean + 2σ
```

The main results are:

| Indicator            | Detection rate | Median lead to transition |
| -------------------- | -------------: | ------------------------: |
| Z_conn               |          0.995 |                        38 |
| Z_mode               |          0.995 |                        37 |
| Z_rigid              |          0.775 |                        36 |
| Z_sync               |          1.000 |                        36 |
| PWSI_equal           |          1.000 |                        35 |
| Visible patch metric |          1.000 |                        30 |

The first-rise results show that prepatch indicators generally rise earlier than the visible patch metric.

Z_conn, Z_mode, Z_rigid, and Z_sync reached their first-rise points approximately 36–38 time steps before transition, while the visible patch metric rose around 30 time steps before transition.

This indicates that prepatch indicators can capture earlier spatial organization before visible patches become strongly expressed.

## 6. Interpretation

The results support the interpretation that visible patches are late-stage manifestations of earlier prepatch spatial organization.

The observed process can be summarized as:

```text
local synchronization and spatial connectivity begin to increase
→ boundary and morphological organization gradually strengthen
→ dominant spatial modes become more stable
→ PWSI_equal increases as an integrated prepatch signal
→ visible patch metrics rise more sharply at a later stage
→ critical transition occurs
```

This does not imply that all systems must follow a strictly fixed order of Z_sync, Z_conn, Z_rigid, and Z_mode. Instead, the key conclusion is that prepatch indicators as a group rise earlier than visible patch metrics.

Therefore, the proposed prepatch indicators provide a mechanism-level interpretation of patch formation before critical transition.

## 7. Main Conclusion

The main conclusion of this analysis is:

```text
The event-aligned patch formation analysis shows that prepatch indicators rise earlier than visible patch metrics in the vegetation CA system. This suggests that visible patches are late-stage manifestations of earlier prepatch spatial organization.
```

This result directly supports the idea that spatial patches are not sudden static structures, but gradually emerging spatial organizations.

## 8. Role in the Main Paper

This analysis should be used as the mechanism interpretation part of the paper.

It supports the transition from a performance-based statement:

```text
The proposed indicators improve early-warning performance.
```

to a process-based statement:

```text
The proposed indicators reveal the gradual spatial organization process underlying patch formation before critical transition.
```

This is important for Ecological Indicators because it strengthens the ecological meaning and interpretability of PWSI and prepatch indicators.

## 9. Candidate Figures

The following figures were generated:

```text
clean_event_aligned_prepatch_trajectory.png
clean_event_aligned_visible_patch_trajectory.png
clean_first_rise_time_comparison.png
typical_spatial_evolution.png
```

Recommended use:

```text
The prepatch trajectory figure and first-rise time comparison figure should be used as main-text mechanism figures.
The visible patch trajectory and typical spatial evolution figure can be used as supporting or supplementary figures.
```

## 10. Next Step

The next stage is:

```text
v2.3B Paper-level result consolidation
```

The purpose of v2.3B is to organize all existing results into paper-level figures, tables, and manuscript structure.

The next tasks include:

```text
1. Decide which results should be placed in the main text.
2. Decide which results should be placed in supplementary materials.
3. Build the final main result table.
4. Build the final robustness table.
5. Design the final paper figure structure.
6. Prepare for the real forest disturbance remote-sensing validation stage.
```
