# Research scope and decision gates

## Project boundary

This repository studies tropical-cyclogenesis dynamics. It is separate from the
ecological `critical-slowing-patch-warning` repository because the event
definition, observations, controls, physical covariates, and validation design
are different.

## Current evidence

- v0.1 established a reproducible five-case Himawari-8 processing chain.
- v0.2 added five tracked disturbances that remained below 34 kt.
- Cold-cloud area and its change are candidate signals in this ten-track pilot.
- Static/dynamic organization features have not yet shown stable incremental
  value across disturbances.
- Geographic and seasonal metadata alone separate the current small sample,
  demonstrating residual matching confounding.
- A four-anchor anonymous ERA5 pilot is used only as an early environmental
  confounding check; the full three-hourly environment trajectory remains a
  later replication target.
- In the ten-track pilot, cold-cloud separation becomes sustained only from the
  24-hour cutoff, while organization metrics are not a stable early signal.
- The current literature already contains satellite-IR, convective-core, polar
  structure, and deep-learning genesis classifiers. Prediction accuracy or
  radial representation alone is therefore not a sufficient novelty claim.

## Ordered decision gates

1. **Environment baseline:** add ERA5 850-hPa relative vorticity, 600--850-hPa
   relative humidity, and 200--850-hPa vertical wind shear.
2. **Incremental value:** compare environment-only, environment plus cold-cloud
   area, and environment plus organization features using disturbance-grouped
   validation.
3. **Negative-sample expansion:** objectively track persistent open-ocean cloud
   clusters and match them by month, latitude, and environment.
4. **Representation learning:** train a low-dimensional organization variable
   only if organization adds stable information beyond the environment baseline.
5. **Dynamics and early warning:** apply weak-form system identification and
   critical-slowing tests only after the state variables and controls pass the
   preceding gates.

The v0.1/v0.2 results are feasibility pilots, not publishable predictive
performance and not evidence of critical slowing down.
