# Feature Importance Table

This table reports feature-value diagnostic importance for classic patch features. Importance is estimated by the separation between risk and non-risk samples using absolute Cohen's d and single-feature AUC.

|   Rank | Feature              | Abs Cohen d   | Single-feature AUC   | Abs difference   |
|-------:|:---------------------|:--------------|:---------------------|:-----------------|
|      1 | patch_area_cv        | 1.070 ± 0.075 | 0.774 ± 0.015        | 0.422 ± 0.023    |
|      2 | centroid_y           | 1.069 ± 0.062 | 0.781 ± 0.014        | 0.135 ± 0.005    |
|      3 | largest_patch_area   | 1.028 ± 0.056 | 0.781 ± 0.014        | 2.662 ± 0.112    |
|      4 | patch_area_std       | 1.009 ± 0.067 | 0.773 ± 0.014        | 0.159 ± 0.007    |
|      5 | largest_patch_ratio  | 0.929 ± 0.061 | 0.772 ± 0.014        | 0.326 ± 0.016    |
|      6 | gini                 | 0.870 ± 0.067 | 0.793 ± 0.017        | 0.026 ± 0.001    |
|      7 | mean_patch_area      | 0.847 ± 0.073 | 0.796 ± 0.015        | 0.009 ± 0.000    |
|      8 | aggregation_index    | 0.804 ± 0.061 | 0.799 ± 0.015        | 0.054 ± 0.003    |
|      9 | centroid_x           | 0.691 ± 0.024 | 0.802 ± 0.015        | 0.007 ± 0.001    |
|     10 | perimeter            | 0.678 ± 0.058 | 0.756 ± 0.017        | 0.236 ± 0.018    |
|     11 | fragmentation        | 0.647 ± 0.033 | 0.622 ± 0.004        | 0.056 ± 0.002    |
|     12 | edge_density         | 0.605 ± 0.029 | 0.784 ± 0.017        | 0.001 ± 0.000    |
|     13 | patch_count          | 0.554 ± 0.057 | 0.657 ± 0.012        | 0.139 ± 0.012    |
|     14 | perimeter_area_ratio | 0.516 ± 0.028 | 0.785 ± 0.017        | 0.000 ± 0.000    |
|     15 | spread_x             | 0.504 ± 0.011 | 0.641 ± 0.004        | 0.003 ± 0.000    |
