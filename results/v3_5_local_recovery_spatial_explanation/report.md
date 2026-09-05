# v3.5 local recovery and spatial patch explanation

## Locked positioning

Direct right-censored recovery hazard is the primary resilience/CSD measurement. Vegetation-patch variables are secondary spatial explanations only; they do not define a positive CSD label and are not combined into PWSI.

## Direct recovery reproduction

| site | n_bins | slope_per_percent | intercept | r_squared | p_value | max_relative_replication_error | events_in_analysis_gradient | recoveries_in_analysis_gradient |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hellegat | 35 | -3.58054e-05 | 0.00167569 | 0.82775 | 3.75413e-14 | 3.33568e-05 | 556640 | 440955 |
| Paulina | 12 | -2.33593e-05 | 0.00120171 | 0.594972 | 0.00330419 | 1.16127e-05 | 518729 | 451727 |

## Supported local units

Eligibility: at least 50 losses and 10 recoveries.

| site | block_width_m | supported_units | represented_blocks | recoveries | exposure_years |
| --- | --- | --- | --- | --- | --- |
| Hellegat | 32 | 411 | 112 | 439154 | 4.45979e+06 |
| Hellegat | 64 | 248 | 34 | 440401 | 4.51936e+06 |
| Hellegat | 128 | 137 | 10 | 440789 | 4.54197e+06 |
| Paulina | 32 | 274 | 136 | 451067 | 3.43179e+06 |
| Paulina | 64 | 142 | 46 | 451488 | 3.44124e+06 |
| Paulina | 128 | 73 | 15 | 451694 | 3.45638e+06 |

## Cross-site validation at the primary 64 m scale

Negative percentages are improvements over the stress-only model; positive percentages are worse.

| block_width_m | train_site | test_site | model | alpha | n_units | n_blocks | exposure_years | poisson_deviance | weighted_log_rmse | spearman_rho | spearman_p | deviance_change_vs_stress_percent | log_rmse_change_vs_stress_percent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 64 | Paulina | Hellegat | stress_only | 0.1 | 248 | 34 | 4.51936e+06 | 0.0699905 | 1.18131 | 0.660947 | 1.62304e-32 | 0 | 0 |
| 64 | Paulina | Hellegat | stress_plus_static_patch | 0.1 | 248 | 34 | 4.51936e+06 | 0.0699738 | 1.1794 | 0.658754 | 3.05874e-32 | -0.0239167 | -0.162266 |
| 64 | Paulina | Hellegat | stress_plus_all_patch | 0.1 | 248 | 34 | 4.51936e+06 | 0.0624659 | 1.13076 | 0.675672 | 1.99766e-34 | -10.751 | -4.27927 |
| 64 | Hellegat | Paulina | stress_only | 0.1 | 142 | 46 | 3.44124e+06 | 0.0270366 | 0.451883 | 0.354625 | 1.49062e-05 | 0 | 0 |
| 64 | Hellegat | Paulina | stress_plus_static_patch | 0.1 | 142 | 46 | 3.44124e+06 | 0.0415718 | 0.587274 | 0.383333 | 2.49282e-06 | 53.761 | 29.9616 |
| 64 | Hellegat | Paulina | stress_plus_all_patch | 0.1 | 142 | 46 | 3.44124e+06 | 0.0301889 | 0.510371 | 0.357031 | 1.29145e-05 | 11.6594 | 12.9432 |

## Stress-adjusted patch associations

Associations use spatial-block held-out stress predictions, block-level permutations, and Benjamini-Hochberg correction within site and scale.

| block_width_m | feature | feature_group | hellegat_rho | hellegat_q | paulina_rho | paulina_q | same_direction | both_fdr_below_0p05 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 64 | component_density_mean | static | -0.352689 | 0.0792 | -0.12409 | 0.576 | True | False |
| 64 | coverage_change_rate | dynamic | 0.307257 | 0.129 | -0.1107 | 0.581143 | False | False |
| 64 | coverage_mean | static | 0.605195 | 0.0045 | 0.398212 | 0.09 | True | False |
| 64 | edge_change_rate | dynamic | 0.448128 | 0.018 | 0.0646932 | 0.648 | True | False |
| 64 | edge_density_mean | static | -0.277617 | 0.133714 | -0.0828245 | 0.648 | True | False |
| 64 | fragmentation_change_rate | dynamic | -0.0961333 | 0.633 | -0.251251 | 0.216 | True | False |
| 64 | largest_component_change_rate | dynamic | -0.425646 | 0.018 | -0.248533 | 0.216 | True | False |
| 64 | largest_component_fraction_mean | static | 0.556473 | 0.0045 | 0.355656 | 0.0945 | True | False |
| 64 | turnover_rate | dynamic | 0.0792972 | 0.633 | -0.151156 | 0.576 | False | False |

## Multiscale consistency of local recovery fields

| site | fine_scale_m | coarse_scale_m | n_fine_blocks | spearman_rho | p_value |
| --- | --- | --- | --- | --- | --- |
| Hellegat | 32 | 64 | 112 | 0.67284 | 4.46255e-16 |
| Hellegat | 32 | 128 | 112 | 0.284426 | 0.00237089 |
| Hellegat | 64 | 128 | 34 | 0.433594 | 0.0104133 |
| Paulina | 32 | 64 | 136 | 0.7182 | 7.488e-23 |
| Paulina | 32 | 128 | 136 | 0.582964 | 9.59618e-14 |
| Paulina | 64 | 128 | 46 | 0.756001 | 1.24307e-09 |

## Decision

- Direct recovery replication: **PASS**
- Local recovery data sufficiency: **PASS**
- Independent patch increment: **NOT ESTABLISHED**
- Patch role: **secondary spatial explanation**

The ecological route proceeds through local-to-landscape direct recovery. Patch geometry may describe where slow recovery is spatially organized, but it is promoted beyond context only if it adds held-out cross-site information after inundation control.