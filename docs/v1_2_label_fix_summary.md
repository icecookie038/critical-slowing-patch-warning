# v1.2 Label Fix Summary

## 1. Why label fixing is needed

The previous labeling strategy used AC1 / variance / trend to detect the critical point. However, AC1 is an early-warning feature and should not be used as the ground-truth transition label. Otherwise, the model may learn an AC1-defined target rather than a real transition event.

In v1.2, AC1 is retained as a traditional early-warning feature, but the transition time is redefined using an observable event in the spatial SEIR process.

## 2. New transition definition

The observable transition time `t_event` is defined as the first time point at which:

```text
I_total_ratio >= theta_I
and
infected_area_ratio >= theta_A
```

for `persistent_k` consecutive time steps.

The early-warning label is then defined by the prediction horizon:

```text
y_h = 1 if 0 < t_event - t <= h
y_h = 0 otherwise
```

## 3. Threshold tests

### A. Strict label

```text
theta_I = 0.05
theta_A = 0.05
persistent_k = 3
Valid simulations = 10 / 30
```

### B. Medium label

```text
theta_I = 0.03
theta_A = 0.03
persistent_k = 3
Valid simulations = 13 / 30
```

### C. Loose label

```text
theta_I = 0.02
theta_A = 0.03
persistent_k = 3
Valid simulations = 14 / 30
```

## 4. Final decision

The medium label setting is selected as the main labeling strategy:

```text
theta_I = 0.03
theta_A = 0.03
persistent_k = 3
```

The strict and loose settings will be used for label robustness analysis.

## 5. Notes

AC1 is not used to define the ground-truth transition time. It is only used as a traditional early-warning feature.
