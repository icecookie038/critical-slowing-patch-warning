# v2.3A Patch Formation Process Analysis Plan

## 1. Purpose

The purpose of v2.3A is to analyze the gradual formation process of spatial patches before critical transition.

The previous versions have already constructed and validated the main early-warning framework:

```text
v1.2: event-based label correction
v1.3: prepatch spatial organization indicators
v1.4: PWSI construction
v2.0: vegetation CA cross-system validation
v2.2: synthetic robustness and threshold sensitivity
```

The next step is not to add a more complex prediction model, but to use the existing indicators to explain how patches gradually emerge before the critical transition.

This directly follows the research idea that patches are not static structures that suddenly appear, but dynamic structures that develop through a spatial organization process.

## 2. Main Scientific Question

The main question is:

```text
Can the proposed prepatch indicators reveal the gradual formation process of spatial patches before visible patch emergence and system-level transition?
```

Specifically, this analysis aims to test whether the following process can be observed:

```text
local synchronization seeds
→ spatial connectivity expansion
→ boundary rigidity
→ dominant spatial mode locking
→ visible patch formation
→ critical transition
```

## 3. Main Dataset

The first analysis will focus on the vegetation CA system.

Dataset:

```text
vegetation CA seed42
```

Recommended input files:

```text
data/processed/v2_0_vegetation_ca_clean/vegetation_ca_h30_seed42.npz
data/processed/v2_0_vegetation_prepatch/seir_v1_3_prepatch_h30_seed42.npz
data/processed/v2_0_vegetation_pwsi/seir_v1_4_pwsi_only_h30_seed42.npz
```

The vegetation CA system is selected first because it is closer to ecological degradation than the SEIR spatial spreading model.

SEIR can be used later as supplementary evidence if needed.

## 4. Indicator Groups

The patch formation process will be analyzed using four prepatch indicator groups:

```text
Z_sync
Z_conn
Z_rigid
Z_mode
```

Their ecological meanings are:

```text
Z_sync: local synchronization before visible patch emergence
Z_conn: spatial connectivity and aggregation expansion
Z_rigid: boundary rigidity and morphological organization
Z_mode: dominant spatial mode locking
```

PWSI_equal will be used as the integrated process index:

```text
PWSI_equal = 0.25 Z_sync + 0.25 Z_conn + 0.25 Z_rigid + 0.25 Z_mode
```

Visible patch indicators will be used as late-stage patch structure references.

## 5. Experiment 1: Event-aligned Indicator Trajectory

The first experiment is event-aligned trajectory analysis.

For each simulation, all time points will be aligned by critical_time.

The x-axis will be:

```text
time to critical transition
```

For example:

```text
-60, -50, -40, -30, -20, -10, 0
```

The y-axis will show standardized indicator values:

```text
Z_sync
Z_conn
Z_rigid
Z_mode
PWSI_equal
visible patch metric
```

The purpose is to test whether prepatch indicators rise before visible patch indicators become dominant.

Expected interpretation:

```text
If Z_sync and Z_conn increase earlier than visible patch metrics, this suggests that local synchronization and spatial connectivity are early stages of patch formation.
```

## 6. Experiment 2: Indicator First-rise Time

The second experiment is first-rise time comparison.

For each simulation and each indicator, the first-rise time will be defined as the first time when the indicator exceeds its baseline level.

A simple first-rise rule is:

```text
indicator > baseline mean + 2σ
```

The baseline period will be selected from early pre-event time points.

The following first-rise times will be compared:

```text
Z_sync first-rise time
Z_conn first-rise time
Z_rigid first-rise time
Z_mode first-rise time
PWSI_equal first-rise time
visible patch first-rise time
```

The purpose is to test whether prepatch indicators rise earlier than visible patch metrics.

Expected interpretation:

```text
If prepatch indicators show earlier first-rise times than visible patch metrics, this supports the idea that visible patches are late-stage manifestations of earlier spatial organization.
```

## 7. Experiment 3: Typical Simulation Spatial Evolution

The third experiment is a typical simulation visualization.

One representative vegetation CA simulation will be selected.

The following time points will be visualized:

```text
critical_time - 50
critical_time - 40
critical_time - 30
critical_time - 20
critical_time - 10
critical_time
```

The figure should include:

```text
raw vegetation state map
PWSI or prepatch risk map
visible degraded patch map
future event region overlay
```

The purpose is to visually show that prepatch warning signals appear before clear visible patch formation.

## 8. Expected Main Result

The expected main result is:

```text
The temporal evolution of prepatch indicators suggests a progressive spatial organization process before visible patch emergence. Local synchronization and spatial connectivity tend to increase before visible patch metrics become dominant, indicating that visible patches are late-stage manifestations of earlier prepatch spatial organization.
```

## 9. Role in the Main Paper

This experiment will become the mechanism interpretation part of the paper.

It can support the transition from:

```text
The proposed indicators improve early-warning performance.
```

to:

```text
The proposed indicators reveal the gradual spatial organization process underlying patch formation before critical transition.
```

This is important for Ecological Indicators because it strengthens the ecological meaning and interpretability of PWSI and prepatch indicators.

## 10. Current Priority

The immediate next task is:

```text
Build and run analyze_patch_formation_process.py
```

The first version should generate:

```text
1. event-aligned indicator trajectory table
2. first-rise time comparison table
3. figures for event-aligned trajectories
```

After this experiment is completed, the results will be summarized in:

```text
docs/v2_3_patch_formation_process_summary.md
```
