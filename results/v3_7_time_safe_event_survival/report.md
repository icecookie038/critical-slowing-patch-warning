# v3.7 time-safe event-level recovery survival analysis

## Locked role of each signal

Direct recovery after an observed vegetation loss is the resilience/CSD measurement. Patch configuration is a secondary spatial covariate only. It is not used to label CSD, choose the stress gradient, or define an alarm.

## Event construction and retention

| site | raw_disturbance_events | retained_disturbance_events | retained_recoveries | retained_person_intervals | grouped_risk_rows | retained_blocks |
| --- | --- | --- | --- | --- | --- | --- |
| Hellegat | 556640 | 556526 | 440841 | 1488244 | 5086 | 34 |
| Paulina | 518729 | 518729 | 451727 | 944293 | 1976 | 50 |

A loss event starts when a previously vegetated pixel is first observed absent. Each later survey interval contributes one at-risk record until first recovery or right censoring. The complementary-log-log model uses log(interval years) as an offset. Identical pixel histories are collapsed to a grouped binomial likelihood without dropping events.

## Temporal leakage audit

- Hellegat: pre_loss_strictly_before_origin=True, current_patch_not_after_risk_start=True, positive_follow_up_intervals=True, events_do_not_exceed_risk_set=True
- Paulina: pre_loss_strictly_before_origin=True, current_patch_not_after_risk_start=True, positive_follow_up_intervals=True, events_do_not_exceed_risk_set=True

The origin-patch model uses only the map immediately before the loss. The time-varying model uses the map at the beginning of each risk interval and its immediately preceding transition. Neither can see the interval outcome or a later image.

## Cross-site external validation

Primary transfer is restricted to common inundation support (0.305–0.415) to prevent stress extrapolation.

| train_site | test_site | model | fit_converged | fit_iterations | person_intervals | recoveries | interval_log_loss | interval_brier | interval_auc | expected_observed_ratio | calibration_slope | log_loss_change_vs_stress_percent | brier_change_vs_stress_percent | auc_change_vs_stress |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Paulina | Hellegat | stress_time | True | 18 | 1220530 | 295430 | 0.599 | 0.20053 | 0.66985 | 1.5836 | 0.66695 | 0 | 0 | 0 |
| Paulina | Hellegat | stress_time_plus_origin_patch | True | 22 | 1220530 | 295430 | 0.70218 | 0.23816 | 0.63547 | 1.913 | 0.5654 | 17.224 | 18.763 | -0.034377 |
| Paulina | Hellegat | stress_time_plus_current_patch | True | 38 | 1220530 | 295430 | 0.69028 | 0.21432 | 0.68857 | 1.6921 | 0.5196 | 15.239 | 6.8738 | 0.018718 |
| Hellegat | Paulina | stress_time | True | 14 | 944293 | 451727 | 0.83793 | 0.28405 | 0.60174 | 0.61952 | 0.29578 | 0 | 0 | 0 |
| Hellegat | Paulina | stress_time_plus_origin_patch | True | 19 | 944293 | 451727 | 0.92614 | 0.31005 | 0.60801 | 0.50962 | 0.27332 | 10.528 | 9.1527 | 0.0062739 |
| Hellegat | Paulina | stress_time_plus_current_patch | True | 23 | 944293 | 451727 | 0.92814 | 0.30539 | 0.64746 | 0.46194 | 0.35769 | 10.766 | 7.5114 | 0.045718 |

Negative loss changes indicate improvement; positive values indicate worse held-out probability predictions.

## Spatial-block bootstrap

| train_site | test_site | model | n_blocks | brier_delta_median | brier_delta_ci_low | brier_delta_ci_high | log_loss_delta_median | log_loss_delta_ci_low | log_loss_delta_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hellegat | Paulina | stress_time_plus_current_patch | 50 | 0.022435 | 0.0016393 | 0.046763 | 0.093038 | 0.013158 | 0.19818 |
| Hellegat | Paulina | stress_time_plus_origin_patch | 50 | 0.026355 | 0.016869 | 0.035947 | 0.087814 | 0.054122 | 0.12772 |
| Paulina | Hellegat | stress_time_plus_current_patch | 25 | 0.013312 | -0.0052238 | 0.028207 | 0.089863 | 0.034013 | 0.13319 |
| Paulina | Hellegat | stress_time_plus_origin_patch | 25 | 0.037345 | 0.015977 | 0.0539 | 0.10323 | 0.051117 | 0.1387 |

The bootstrap resamples complete held-out 64 m blocks and quantifies test-set uncertainty conditional on the model fitted in the other site.

## Decision

- Pre-loss patch increment: **NOT ESTABLISHED**
- Time-varying patch increment: **NOT ESTABLISHED**
- Event-level, right-censored, time-safe recovery model: **COMPLETE**
- Patch role: **descriptive spatial explanation only**

Both patch models worsen Brier loss and logarithmic loss in both site-transfer directions. Time-varying patches improve interval AUC, but better ranking does not compensate for poorer probability accuracy and calibration. Patch geometry therefore cannot be presented as an independent or transferable recovery detector.

## Precision interpretation

Low temporal frequency remains a limitation for fine patch dynamics: the sites contain only 8 and 11 maps. However, the direct event analysis retains more than two million risk intervals on common stress support. The persistent cross-site loss degradation means limited precision is not a sufficient explanation for the failed patch increment. The route can proceed with direct recovery as the main signal and patches as maps of spatial organization; rescuing a universal patch detector is stopped.
