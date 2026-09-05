# Historical spline follow-up — exploratory

The recovered v3.7 pressure/waiting-time spline estimator was fitted on the
original QIR pilot's exact input files, event IDs and block partitions. Input and
cohort-source SHA256 checks and a comparison against the saved interval labels
passed. Fractional trial weights give each training block equal total weight;
this adapts the historical estimator to the pilot protocol and is not a rerun
of the original full-resolution v3.7 experiment. Spline settings were not
selected against the target labels. Both numerical fits converged.

The important finding is a source-partition failure: Hellegat training pressure
ranges from 0.19894 to 0.39349, while its calibration blocks range from 0.40851
to 0.45052, entirely outside the training range. The scalar calibration fit
reaches its upper search boundary (+5). On Paulina, the calibrated spline has
Brier 0.567441 and log loss 4.875537, versus uncalibrated 0.092546/0.468763.
That boundary solution and severe degradation are diagnostic failures, not
evidence that a learned image representation is superior to a sound baseline.

In the reverse direction the calibrated spline gives Brier 0.117152 and log
loss 1.031084. Paulina's narrower environmental range does not cover much of
the Hellegat test range. Full comparisons, predictions, coefficients,
convergence and range diagnostics are saved beside this report.

Decision: stop interpreting model rankings as representational gains on this
small split. Design the next source training/calibration partition using
source covariates and spatial separation, audit covariate overlap before
fitting, and evaluate calibration with explicit extrapolation diagnostics.
Retain this failed comparison. Any revised split is a new exploratory protocol;
the previously viewed target does not become a fresh confirmatory test.

The existing QIR residual still uses its neural M0; it has not been retrained
on top of the spline. Source-only multi-seed/convergence work, quality training
and an independent system remain necessary. This follow-up provides no general
uncertainty coverage or second-system result.

Reproduce after installing `requirements-research-tests.txt` and rebuilding the
pilot NPZ inputs: `python -m src.qir_event.compare_historical_baseline`.
