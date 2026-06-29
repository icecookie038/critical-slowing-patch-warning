# v2.4 Integrated Patch Dynamics and Precursor Warning

## Purpose

This stage integrates four types of indicators into a single event-aligned patch-formation process:

1. Traditional EWS
2. Prepatch spatial organization
3. Dynamic patch restructuring
4. Visible patch manifestation

The expected mechanism is:

Prepatch spatial organization
-> Dynamic patch restructuring
-> Visible patch manifestation
-> Critical transition

## Main script

scripts/analysis/analyze_integrated_patch_dynamics.py

## First test

Vegetation CA, seed42.

## Main outputs

results/v2_4_integrated_patch_dynamics/vegetation_seed42/

Important files:

- integrated_patch_dynamics_timeseries.csv
- first_rise_all_indicators_summary.csv
- formation_stage_summary.csv
- formation_order_summary.csv
- figures/integrated_patch_dynamics_trajectory.png
- figures/formation_stage_lead_boxplot.png
- figures/formation_order_probability.png

## Interpretation

The key question is not only whether a model predicts transition, but whether different patch indicators rise in an ordered sequence before the transition.

The expected order is:

t_prepatch < t_dynamic_patch < t_visible_patch < t_event

This directly responds to the supervisor's suggestion to study dynamic patch analysis and strengthen precursor early warning.
