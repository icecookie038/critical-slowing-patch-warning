"""Time-safe event-level recovery survival models for tidal marshes.

The published aerial photographs observe recovery only at irregular survey
times.  We therefore use a complementary-log-log discrete-time hazard model
with the interval duration as an offset.  Pixel events are retained in the
risk-set construction and then collapsed only when their complete covariate
history is identical; this is likelihood-equivalent to fitting one row per
pixel and interval.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import optimize
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from src.real_data.tidal_marsh import (
    TidalMarshData,
    _inundation_bins,
    analysis_bin_edges,
    iter_recovery_events,
    snapshot_patch_context,
)


ORIGIN_PATCH_FEATURES = [
    "origin_coverage",
    "origin_component_density",
    "origin_largest_component_fraction",
    "origin_edge_density",
]

CURRENT_PATCH_FEATURES = [
    "current_coverage",
    "current_component_density",
    "current_largest_component_fraction",
    "current_edge_density",
]

HISTORY_PATCH_FEATURES = [
    "history_turnover_rate",
    "history_coverage_change_rate",
    "history_fragmentation_change_rate",
    "history_edge_change_rate",
    "history_largest_component_change_rate",
]

SURVIVAL_MODEL_FEATURES: dict[str, list[str]] = {
    "stress_time": [],
    "stress_time_plus_origin_patch": ORIGIN_PATCH_FEATURES,
    "stress_time_plus_current_patch": CURRENT_PATCH_FEATURES
    + HISTORY_PATCH_FEATURES,
}


@dataclass(frozen=True)
class SurvivalModelConfig:
    stress_knots: int = 4
    stress_degree: int = 2
    max_iter: int = 200
    tolerance: float = 1e-8
    ridge_alpha: float = 1e-4


@dataclass(frozen=True)
class CloglogFit:
    params: np.ndarray
    converged: bool
    iterations: int


@dataclass
class GroupedCloglogModel:
    preprocessing: ColumnTransformer
    result: CloglogFit
    patch_features: tuple[str, ...]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        columns = ["inundation_fraction", "log_elapsed_years_start", *self.patch_features]
        transformed = self.preprocessing.transform(frame[columns])
        design = _add_intercept(transformed)
        offset = np.log(frame["interval_years"].to_numpy(dtype=float))
        predicted = _cloglog_probability(design @ self.result.params + offset)
        return np.clip(predicted, 1e-9, 1.0 - 1e-9)

    def coefficients(self) -> pd.DataFrame:
        names = ["intercept", *self.preprocessing.get_feature_names_out().tolist()]
        values = np.asarray(self.result.params, dtype=float)
        return pd.DataFrame({"term": names, "coefficient": values})


def _rename_patch_columns(
    context: pd.DataFrame,
    index_name: str,
    prefix: str,
    include_dynamic: bool,
) -> pd.DataFrame:
    static = [
        "coverage",
        "component_density",
        "largest_component_fraction",
        "edge_density",
    ]
    dynamic = [
        "turnover_rate",
        "coverage_change_rate",
        "fragmentation_change_rate",
        "edge_change_rate",
        "largest_component_change_rate",
    ]
    selected = ["site", "block_width_m", "block_id", "snapshot_index", *static]
    if include_dynamic:
        selected.extend(dynamic)
    renamed = context[selected].rename(columns={"snapshot_index": index_name}).copy()
    renamed = renamed.rename(
        columns={
            name: f"{prefix}_{name}"
            for name in [*static, *(dynamic if include_dynamic else [])]
        }
    )
    return renamed


def build_discrete_recovery_risk_sets(
    data: TidalMarshData,
    block_width_m: float = 64.0,
    patch_resolution_m: float = 1.0,
    min_valid_patch_cells: int = 16,
) -> tuple[pd.DataFrame, dict[str, int | float | str]]:
    """Build irregular-interval recovery risk sets with time-safe covariates."""
    site = data.site
    block_pixels_float = block_width_m / site.pixel_size_m
    block_pixels = int(round(block_pixels_float))
    if not np.isclose(block_pixels_float, block_pixels):
        raise ValueError("block_width_m must align with source pixels")

    height, width = data.inundation.shape
    n_block_rows = ceil(height / block_pixels)
    n_block_cols = ceil(width / block_pixels)
    edges = analysis_bin_edges(site)
    n_bins = len(edges) - 1
    years = np.asarray(site.years, dtype=float)
    chunks: list[pd.DataFrame] = []
    raw_events = 0
    raw_recoveries = 0
    raw_person_intervals = 0

    for batch in iter_recovery_events(data.vegetation, site.years):
        inundation = data.inundation[batch.rows, batch.cols]
        bin_index = _inundation_bins(inundation, edges)
        valid_gradient = bin_index >= 0
        raw_events += int(valid_gradient.sum())
        raw_recoveries += int((batch.recovered & valid_gradient).sum())
        block_id = (
            (batch.rows // block_pixels) * n_block_cols
            + (batch.cols // block_pixels)
        )

        for recovery_index in range(batch.disturbance_index + 1, len(years)):
            cutoff = years[recovery_index] - years[batch.disturbance_index]
            at_risk = valid_gradient & (batch.duration_years >= cutoff)
            if not at_risk.any():
                continue
            recovered_now = (
                at_risk
                & batch.recovered
                & np.isclose(batch.duration_years, cutoff)
            )
            keys = block_id[at_risk] * n_bins + bin_index[at_risk]
            risk_counts = np.bincount(
                keys, minlength=n_block_rows * n_block_cols * n_bins
            )
            event_counts = np.bincount(
                keys,
                weights=recovered_now[at_risk].astype(np.int64),
                minlength=len(risk_counts),
            ).astype(np.int64)
            occupied = np.flatnonzero(risk_counts > 0)
            occupied_blocks = occupied // n_bins
            occupied_bins = occupied % n_bins
            raw_person_intervals += int(risk_counts[occupied].sum())
            chunks.append(
                pd.DataFrame(
                    {
                        "site": site.name,
                        "block_width_m": float(block_width_m),
                        "block_id": occupied_blocks,
                        "block_row": occupied_blocks // n_block_cols,
                        "block_col": occupied_blocks % n_block_cols,
                        "origin_index": batch.disturbance_index,
                        "origin_year": years[batch.disturbance_index],
                        "pre_loss_index": batch.disturbance_index - 1,
                        "risk_start_index": recovery_index - 1,
                        "risk_start_year": years[recovery_index - 1],
                        "risk_end_index": recovery_index,
                        "risk_end_year": years[recovery_index],
                        "interval_years": years[recovery_index]
                        - years[recovery_index - 1],
                        "elapsed_years_start": years[recovery_index - 1]
                        - years[batch.disturbance_index],
                        "inundation_bin": occupied_bins,
                        "inundation_fraction": (
                            edges[occupied_bins] + edges[occupied_bins + 1]
                        )
                        / 2.0,
                        "n_at_risk": risk_counts[occupied].astype(np.int64),
                        "n_recovered": event_counts[occupied],
                    }
                )
            )

    if not chunks:
        raise ValueError(f"No recovery risk sets were found for {site.name}")
    risk_sets = pd.concat(chunks, ignore_index=True)
    risk_sets["log_elapsed_years_start"] = np.log1p(
        risk_sets["elapsed_years_start"].to_numpy(dtype=float)
    )

    context = snapshot_patch_context(
        data,
        block_width_m=block_width_m,
        patch_resolution_m=patch_resolution_m,
        min_valid_cells=min_valid_patch_cells,
    )
    origin = _rename_patch_columns(
        context, "pre_loss_index", "origin", include_dynamic=False
    )
    current = _rename_patch_columns(
        context, "risk_start_index", "current", include_dynamic=True
    ).rename(
        columns={
            name: name.replace("current_", "history_", 1)
            for name in [
                "current_turnover_rate",
                "current_coverage_change_rate",
                "current_fragmentation_change_rate",
                "current_edge_change_rate",
                "current_largest_component_change_rate",
            ]
        }
    )
    merge_keys_origin = [
        "site",
        "block_width_m",
        "block_id",
        "pre_loss_index",
    ]
    merge_keys_current = [
        "site",
        "block_width_m",
        "block_id",
        "risk_start_index",
    ]
    merged = risk_sets.merge(
        origin,
        on=merge_keys_origin,
        how="inner",
        validate="many_to_one",
    ).merge(
        current,
        on=merge_keys_current,
        how="inner",
        validate="many_to_one",
    )
    merged["event_fraction"] = merged["n_recovered"] / merged["n_at_risk"]
    merged["risk_row_id"] = np.arange(len(merged), dtype=np.int64)

    first_intervals = merged["risk_start_index"] == merged["origin_index"]
    summary: dict[str, int | float | str] = {
        "site": site.name,
        "block_width_m": float(block_width_m),
        "patch_resolution_m": float(patch_resolution_m),
        "raw_disturbance_events": raw_events,
        "retained_disturbance_events": int(
            merged.loc[first_intervals, "n_at_risk"].sum()
        ),
        "raw_recoveries": raw_recoveries,
        "retained_recoveries": int(merged["n_recovered"].sum()),
        "raw_person_intervals": raw_person_intervals,
        "retained_person_intervals": int(merged["n_at_risk"].sum()),
        "grouped_risk_rows": int(len(merged)),
        "retained_blocks": int(merged["block_id"].nunique()),
    }
    return merged, summary


def restrict_to_common_stress_support(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, tuple[float, float]]:
    """Restrict two-site transfer to the observed inundation overlap."""
    sites = sorted(frame["site"].unique())
    if len(sites) != 2:
        raise ValueError(f"Exactly two sites are required, got {sites}")
    lower = max(float(frame.loc[frame["site"] == site, "inundation_fraction"].min()) for site in sites)
    upper = min(float(frame.loc[frame["site"] == site, "inundation_fraction"].max()) for site in sites)
    if lower >= upper:
        raise ValueError("Sites do not share a common inundation range")
    supported = frame.loc[
        (frame["inundation_fraction"] >= lower)
        & (frame["inundation_fraction"] <= upper)
    ].copy()
    return supported.reset_index(drop=True), (lower, upper)


def _build_preprocessing(
    patch_features: Sequence[str],
    config: SurvivalModelConfig,
) -> ColumnTransformer:
    stress = Pipeline(
        [
            (
                "spline",
                SplineTransformer(
                    n_knots=config.stress_knots,
                    degree=config.stress_degree,
                    include_bias=False,
                    extrapolation="linear",
                ),
            ),
            ("scale", StandardScaler()),
        ]
    )
    transformers: list[tuple[str, object, list[str]]] = [
        ("stress", stress, ["inundation_fraction"]),
        ("elapsed", StandardScaler(), ["log_elapsed_years_start"]),
    ]
    if patch_features:
        patch = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]
        )
        transformers.append(("patch", patch, list(patch_features)))
    return ColumnTransformer(transformers, remainder="drop")


def _add_intercept(design: np.ndarray) -> np.ndarray:
    values = np.asarray(design, dtype=float)
    return np.column_stack([np.ones(len(values), dtype=float), values])


def _cloglog_probability(linear_predictor: np.ndarray) -> np.ndarray:
    cumulative_hazard = np.exp(np.clip(linear_predictor, -30.0, 30.0))
    probability = -np.expm1(-cumulative_hazard)
    return np.clip(probability, 1e-12, 1.0 - 1e-12)


def _fit_cloglog_likelihood(
    design: np.ndarray,
    events: np.ndarray,
    trials: np.ndarray,
    offset: np.ndarray,
    max_iter: int,
    tolerance: float,
    ridge_alpha: float,
) -> CloglogFit:
    """Optimize the grouped binomial complementary-log-log likelihood."""
    design = np.asarray(design, dtype=float)
    events = np.asarray(events, dtype=float)
    trials = np.asarray(trials, dtype=float)
    offset = np.asarray(offset, dtype=float)
    non_events = trials - events
    total_weight = float(trials.sum())
    exposure = float(np.sum(trials * np.exp(offset)))
    initial_rate = max(float(events.sum()) / exposure, 1e-8)
    initial = np.zeros(design.shape[1], dtype=float)
    initial[0] = np.log(initial_rate)

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        eta = design @ parameters + offset
        cumulative_hazard = np.exp(np.clip(eta, -30.0, 30.0))
        probability = np.clip(
            -np.expm1(-cumulative_hazard), 1e-12, 1.0 - 1e-12
        )
        log_likelihood = events * np.log(probability) - non_events * cumulative_hazard
        penalty = 0.5 * ridge_alpha * float(np.dot(parameters[1:], parameters[1:]))
        loss = -float(log_likelihood.sum()) / total_weight + penalty

        event_derivative = np.zeros_like(cumulative_hazard)
        moderate = cumulative_hazard < 50.0
        event_derivative[moderate] = cumulative_hazard[moderate] / np.expm1(
            cumulative_hazard[moderate]
        )
        derivative_eta = (
            non_events * cumulative_hazard - events * event_derivative
        ) / total_weight
        gradient = design.T @ derivative_eta
        gradient[1:] += ridge_alpha * parameters[1:]
        return loss, gradient

    result = optimize.minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": max_iter, "ftol": tolerance, "gtol": tolerance},
    )
    if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
        raise RuntimeError("Complementary-log-log likelihood did not produce a finite fit")
    return CloglogFit(
        params=np.asarray(result.x, dtype=float),
        converged=bool(result.success),
        iterations=int(result.nit),
    )


def fit_grouped_cloglog(
    frame: pd.DataFrame,
    patch_features: Sequence[str] = (),
    config: SurvivalModelConfig = SurvivalModelConfig(),
) -> GroupedCloglogModel:
    """Fit a grouped discrete-time proportional-hazard likelihood."""
    features = ["inundation_fraction", "log_elapsed_years_start", *patch_features]
    preprocessing = _build_preprocessing(patch_features, config)
    transformed = preprocessing.fit_transform(frame[features])
    design = _add_intercept(transformed)
    trials = frame["n_at_risk"].to_numpy(dtype=float)
    events = frame["n_recovered"].to_numpy(dtype=float)
    offset = np.log(frame["interval_years"].to_numpy(dtype=float))
    result = _fit_cloglog_likelihood(
        design,
        events,
        trials,
        offset,
        max_iter=config.max_iter,
        tolerance=config.tolerance,
        ridge_alpha=config.ridge_alpha,
    )
    return GroupedCloglogModel(preprocessing, result, tuple(patch_features))


def evaluate_grouped_predictions(
    frame: pd.DataFrame,
    predicted: np.ndarray,
    include_calibration: bool = True,
) -> dict[str, float | int]:
    """Evaluate interval discrimination, proper loss, and calibration."""
    events = frame["n_recovered"].to_numpy(dtype=float)
    trials = frame["n_at_risk"].to_numpy(dtype=float)
    non_events = trials - events
    predicted = np.clip(np.asarray(predicted, dtype=float), 1e-9, 1.0 - 1e-9)
    total = float(trials.sum())
    log_loss = -float(
        np.sum(events * np.log(predicted) + non_events * np.log1p(-predicted))
        / total
    )
    brier = float(
        np.sum(events * (1.0 - predicted) ** 2 + non_events * predicted**2)
        / total
    )
    labels = np.concatenate([np.ones(len(frame)), np.zeros(len(frame))])
    scores = np.concatenate([predicted, predicted])
    weights = np.concatenate([events, non_events])
    positive_weight = weights > 0
    auc = float(
        roc_auc_score(
            labels[positive_weight],
            scores[positive_weight],
            sample_weight=weights[positive_weight],
        )
    )

    metrics: dict[str, float | int] = {
        "n_grouped_rows": int(len(frame)),
        "n_blocks": int(frame["block_id"].nunique()),
        "person_intervals": int(total),
        "recoveries": int(events.sum()),
        "interval_log_loss": log_loss,
        "interval_brier": brier,
        "interval_auc": auc,
        "expected_observed_ratio": float(np.sum(predicted * trials) / events.sum()),
    }
    if include_calibration:
        cloglog_prediction = np.log(-np.log1p(-predicted))
        calibration = _fit_cloglog_likelihood(
            _add_intercept(cloglog_prediction[:, None]),
            events,
            trials,
            np.zeros(len(frame), dtype=float),
            max_iter=100,
            tolerance=1e-8,
            ridge_alpha=0.0,
        )
        metrics["calibration_intercept"] = float(calibration.params[0])
        metrics["calibration_slope"] = float(calibration.params[1])
    return metrics


def cross_site_survival_comparison(
    frame: pd.DataFrame,
    config: SurvivalModelConfig = SurvivalModelConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit on one complete site and evaluate on the other site."""
    sites = sorted(frame["site"].unique())
    if len(sites) != 2:
        raise ValueError(f"Exactly two sites are required, got {sites}")
    metric_records: list[dict[str, float | int | str]] = []
    prediction_records: list[pd.DataFrame] = []
    coefficient_records: list[pd.DataFrame] = []

    for test_site in sites:
        train = frame.loc[frame["site"] != test_site].copy()
        test = frame.loc[frame["site"] == test_site].copy()
        train_site = str(train["site"].iloc[0])
        for model_name, patch_features in SURVIVAL_MODEL_FEATURES.items():
            model = fit_grouped_cloglog(train, patch_features, config)
            predicted = model.predict(test)
            metrics = evaluate_grouped_predictions(test, predicted)
            metric_records.append(
                {
                    "train_site": train_site,
                    "test_site": test_site,
                    "model": model_name,
                    "fit_converged": model.result.converged,
                    "fit_iterations": model.result.iterations,
                    **metrics,
                }
            )
            prediction = test[
                [
                    "risk_row_id",
                    "site",
                    "block_id",
                    "origin_index",
                    "risk_start_index",
                    "risk_end_index",
                    "interval_years",
                    "inundation_fraction",
                    "n_at_risk",
                    "n_recovered",
                ]
            ].copy()
            prediction["train_site"] = train_site
            prediction["model"] = model_name
            prediction["predicted_interval_probability"] = predicted
            prediction_records.append(prediction)
            coefficients = model.coefficients()
            coefficients["train_site"] = train_site
            coefficients["test_site"] = test_site
            coefficients["model"] = model_name
            coefficient_records.append(coefficients)

    metrics = pd.DataFrame.from_records(metric_records)
    predictions = pd.concat(prediction_records, ignore_index=True)
    coefficients = pd.concat(coefficient_records, ignore_index=True)
    baselines = metrics.loc[metrics["model"] == "stress_time"].set_index(
        ["train_site", "test_site"]
    )
    metrics["log_loss_change_vs_stress_percent"] = np.nan
    metrics["brier_change_vs_stress_percent"] = np.nan
    metrics["auc_change_vs_stress"] = np.nan
    for index, row in metrics.iterrows():
        baseline = baselines.loc[(row["train_site"], row["test_site"])]
        metrics.loc[index, "log_loss_change_vs_stress_percent"] = (
            (row["interval_log_loss"] - baseline["interval_log_loss"])
            / baseline["interval_log_loss"]
            * 100.0
        )
        metrics.loc[index, "brier_change_vs_stress_percent"] = (
            (row["interval_brier"] - baseline["interval_brier"])
            / baseline["interval_brier"]
            * 100.0
        )
        metrics.loc[index, "auc_change_vs_stress"] = (
            row["interval_auc"] - baseline["interval_auc"]
        )
    return metrics, predictions, coefficients


def block_bootstrap_model_differences(
    predictions: pd.DataFrame,
    n_bootstrap: int = 499,
    seed: int = 20260901,
) -> pd.DataFrame:
    """Bootstrap held-out spatial blocks without refitting the train-site model."""
    records: list[dict[str, float | int | str]] = []
    generator = np.random.default_rng(seed)
    for (train_site, test_site), direction in predictions.groupby(
        ["train_site", "site"], sort=True
    ):
        baseline = direction.loc[direction["model"] == "stress_time"].set_index(
            "risk_row_id"
        )
        blocks = np.sort(baseline["block_id"].unique())
        for model_name in sorted(set(direction["model"]) - {"stress_time"}):
            candidate = direction.loc[direction["model"] == model_name].set_index(
                "risk_row_id"
            )
            joined = baseline[
                ["block_id", "n_at_risk", "n_recovered", "predicted_interval_probability"]
            ].rename(
                columns={"predicted_interval_probability": "baseline_prediction"}
            ).join(
                candidate[["predicted_interval_probability"]].rename(
                    columns={"predicted_interval_probability": "candidate_prediction"}
                ),
                how="inner",
                validate="one_to_one",
            )
            indices_by_block = {
                block: np.flatnonzero(joined["block_id"].to_numpy() == block)
                for block in blocks
            }
            deltas: list[tuple[float, float, float]] = []
            for _ in range(n_bootstrap):
                sampled = generator.choice(blocks, size=len(blocks), replace=True)
                indices = np.concatenate([indices_by_block[block] for block in sampled])
                sample = joined.iloc[indices]
                baseline_metrics = evaluate_grouped_predictions(
                    sample.rename(
                        columns={"baseline_prediction": "_prediction"}
                    ),
                    sample["baseline_prediction"].to_numpy(),
                    include_calibration=False,
                )
                candidate_metrics = evaluate_grouped_predictions(
                    sample.rename(
                        columns={"candidate_prediction": "_prediction"}
                    ),
                    sample["candidate_prediction"].to_numpy(),
                    include_calibration=False,
                )
                deltas.append(
                    (
                        float(candidate_metrics["interval_log_loss"])
                        - float(baseline_metrics["interval_log_loss"]),
                        float(candidate_metrics["interval_brier"])
                        - float(baseline_metrics["interval_brier"]),
                        float(candidate_metrics["interval_auc"])
                        - float(baseline_metrics["interval_auc"]),
                    )
                )
            values = np.asarray(deltas)
            record: dict[str, float | int | str] = {
                "train_site": str(train_site),
                "test_site": str(test_site),
                "model": model_name,
                "n_blocks": int(len(blocks)),
                "n_bootstrap": int(n_bootstrap),
            }
            for column, metric in enumerate(["log_loss", "brier", "auc"]):
                record[f"{metric}_delta_median"] = float(np.median(values[:, column]))
                record[f"{metric}_delta_ci_low"] = float(
                    np.quantile(values[:, column], 0.025)
                )
                record[f"{metric}_delta_ci_high"] = float(
                    np.quantile(values[:, column], 0.975)
                )
            records.append(record)
    return pd.DataFrame.from_records(records)


def leakage_audit(frame: pd.DataFrame) -> Mapping[str, bool]:
    """Return explicit temporal-order checks used by the analysis report."""
    return {
        "pre_loss_strictly_before_origin": bool(
            (frame["pre_loss_index"] < frame["origin_index"]).all()
        ),
        "current_patch_not_after_risk_start": bool(
            (frame["risk_start_index"] < frame["risk_end_index"]).all()
        ),
        "positive_follow_up_intervals": bool((frame["interval_years"] > 0).all()),
        "events_do_not_exceed_risk_set": bool(
            (frame["n_recovered"] <= frame["n_at_risk"]).all()
        ),
    }
