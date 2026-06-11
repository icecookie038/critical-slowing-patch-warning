# v1.2 Metadata-Fix and Strict First-Alarm Summary

## 1. Purpose of This Stage

This stage fixes the prediction output format of `train_patch_baselines.py` and adds a simulation-level strict first-alarm analysis workflow.

The previous prediction files only saved:

```text
y_true
risk_prob
y_remaining
```

This was sufficient for classification metrics, ROC/PR curves, and binned risk plots. However, it was not sufficient for strict early-warning evaluation because first-alarm lead time must be computed by grouping predictions by simulation trajectory.

Therefore, the prediction output was extended to include simulation-level metadata.

The updated prediction files now include:

```text
sample_index
sim_id
time_idx
critical_time
y_remaining
y_true
risk_prob
y_pred
threshold
model
feature_mode
seed
split
```

This enables strict first-alarm lead-time analysis.

---

## 2. Code Changes

### Updated file

```text
train_patch_baselines.py
```

Main changes:

```text
1. load_patch_dataset() now reads time_idx and critical_time.
2. split_dataset() now returns train_idx and val_idx.
3. prediction files now save simulation-level metadata.
```

The script still uses simulation-level splitting when `sim_id` is available, so samples from the same simulation trajectory are not mixed across train and validation sets.

### Added file

```text
analyze_first_alarm.py
```

This script computes simulation-level first-alarm metrics by:

```text
1. loading model prediction files;
2. grouping predictions by sim_id;
3. sorting each simulation by time_idx;
4. identifying the first alarm;
5. computing lead_time = critical_time - first_alarm_time.
```

---

## 3. Strict First-Alarm Definition

For each simulation:

```text
lead_time = critical_time - first_alarm_time
```

For the h30 warning task, a strict valid-window alarm is defined as:

```text
0 < lead_time <= 30
```

The script separates four alarm types:

```text
1. any first alarm:
   the first alarm anywhere in the available trajectory;

2. valid-window alarm:
   the first alarm inside the official warning horizon;

3. pre-window alarm:
   alarm before the h30 warning window;

4. late alarm:
   alarm at or after the event time.
```

This prevents overestimating early-warning performance when the model triggers alarms before the defined h30 warning window.

---

## 4. Initial Result: stats + RandomForest, h30

Prediction file:

```text
results_baselines/v1_2_label_fix_metadata/seed42_stats/RandomForest_predictions.csv
```

### persistent_k = 1

```text
n_simulations = 31
n_event_simulations = 31
any_alarm_rate = 1.0
valid_window_alarm_rate = 1.0
valid_window_miss_rate = 0.0
valid_first_alarm_lead_mean ≈ 26.32
valid_first_alarm_lead_median = 28.0
valid_first_alarm_lead_q25 = 25.5
valid_first_alarm_lead_q75 = 30.0
prewindow_alarm_rate ≈ 0.387
late_alarm_rate = 0.0
```

### persistent_k = 2

```text
n_simulations = 31
n_event_simulations = 31
any_alarm_rate = 1.0
valid_window_alarm_rate = 1.0
valid_window_miss_rate = 0.0
valid_first_alarm_lead_mean ≈ 26.32
valid_first_alarm_lead_median = 28.0
valid_first_alarm_lead_q25 = 25.5
valid_first_alarm_lead_q75 = 30.0
prewindow_alarm_rate ≈ 0.290
late_alarm_rate = 0.0
```

Increasing the persistent alarm requirement from k = 1 to k = 2 reduces the pre-window alarm rate from approximately 0.387 to 0.290.

---

## 5. stats Mode: All Models, h30, persistent_k = 2

The first-alarm analysis was then run for all five tabular models under the `stats` feature mode.

| Model                | Valid-window alarm rate | Miss rate | Valid lead mean | Valid lead median | Pre-window alarm rate |
| -------------------- | ----------------------: | --------: | --------------: | ----------------: | --------------------: |
| HistGradientBoosting |                     1.0 |       0.0 |           28.48 |              30.0 |                 0.613 |
| MLP                  |                     1.0 |       0.0 |           27.32 |              30.0 |                 0.581 |
| ExtraTrees           |                     1.0 |       0.0 |           27.03 |              29.0 |                 0.419 |
| RandomForest         |                     1.0 |       0.0 |           26.32 |              28.0 |                 0.290 |
| LogisticRegression   |                     1.0 |       0.0 |           26.26 |              28.0 |                 0.419 |

All five models achieved:

```text
valid_window_alarm_rate = 1.0
valid_window_miss_rate = 0.0
late_alarm_rate = 0.0
```

HistGradientBoosting achieved the longest valid-window first-alarm lead time:

```text
valid_first_alarm_lead_mean ≈ 28.48
valid_first_alarm_lead_median = 30.0
```

However, HistGradientBoosting also produced the highest pre-window alarm rate:

```text
prewindow_alarm_rate ≈ 0.613
```

RandomForest achieved a slightly shorter but still strong valid-window lead time:

```text
valid_first_alarm_lead_mean ≈ 26.32
valid_first_alarm_lead_median = 28.0
```

and the lowest pre-window alarm rate:

```text
prewindow_alarm_rate ≈ 0.290
```

Thus, RandomForest currently provides the best trade-off between valid early warning and avoiding overly early pre-window alarms.

---

## 6. Interpretation

The v1.2 label-fix experiment shows that patch statistical features remain predictive under the event-based transition label.

The strict first-alarm analysis provides stronger evidence than ordinary sample-level classification metrics because it evaluates whether the model can trigger an alarm at the simulation-trajectory level before the event.

Current conclusion:

```text
Under the v1.2 event-based h30 label, all stats-mode tabular models successfully generated valid-window alarms for all validation event simulations. HistGradientBoosting achieved the longest valid-window lead time, while RandomForest provided the most conservative and stable trade-off due to its lower pre-window alarm rate.
```

This supports using RandomForest as the current main model candidate for v1.2 strict early-warning analysis, while reporting HistGradientBoosting as the maximum-lead model.

---

## 7. Current Status

Completed:

```text
1. v1.2 label-fix dataset generation.
2. Metadata output fix in train_patch_baselines.py.
3. Added analyze_first_alarm.py.
4. Re-ran stats mode with metadata output.
5. Verified prediction files contain sim_id, time_idx, critical_time, and y_remaining.
6. Ran strict first-alarm analysis for stats + RandomForest.
7. Ran strict first-alarm analysis for all stats-mode models under h30 and persistent_k = 2.
```

Not yet completed:

```text
1. Re-run last mode with metadata-fixed prediction output.
2. Run first-alarm analysis for all last-mode models.
3. Compare stats vs last.
4. Decide whether to run flatten mode.
5. Later perform horizon robustness: h15 / h30 / h60.
6. Later perform label robustness: strict / medium / loose event thresholds.
```

---

## 8. Next Steps

Immediate next steps:

```text
1. Push the current code and summary document to GitHub.
2. Re-run last mode using the metadata-fixed train_patch_baselines.py.
3. Run first-alarm analysis for all last-mode models under h30 and persistent_k = 2.
4. Compare stats vs last.
5. Decide whether flatten mode is necessary.
```

Files to commit:

```text
train_patch_baselines.py
analyze_first_alarm.py
docs/v1_2_first_alarm_summary.md
```

Files not to commit:

```text
results_baselines/
data/processed/
*.npz
train_patch_baselines_before_metadata_fix.py
__pycache__/
```
