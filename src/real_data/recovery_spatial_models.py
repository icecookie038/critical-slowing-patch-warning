"""Models for direct recovery hazards and secondary patch explanations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from src.real_data.tidal_marsh import (
    PATCH_DYNAMIC_FEATURES,
    PATCH_STATIC_FEATURES,
    filter_supported_units,
)


MODEL_FEATURES: dict[str, list[str]] = {
    "stress_only": [],
    "stress_plus_static_patch": PATCH_STATIC_FEATURES,
    "stress_plus_all_patch": PATCH_STATIC_FEATURES + PATCH_DYNAMIC_FEATURES,
}


@dataclass(frozen=True)
class RecoveryModelConfig:
    alpha: float = 0.1
    n_knots: int = 4
    degree: int = 2
    max_iter: int = 2_000


def merge_recovery_and_patch(
    units: pd.DataFrame,
    patch_context: pd.DataFrame,
    min_losses: int = 50,
    min_recoveries: int = 10,
) -> pd.DataFrame:
    supported = filter_supported_units(units, min_losses, min_recoveries)
    patch_columns = [
        "site",
        "block_width_m",
        "block_id",
        "patch_resolution_m",
        "valid_patch_cells",
        *PATCH_STATIC_FEATURES,
        *PATCH_DYNAMIC_FEATURES,
    ]
    merged = supported.merge(
        patch_context[patch_columns],
        on=["site", "block_width_m", "block_id"],
        how="inner",
        validate="many_to_one",
    )
    return merged.reset_index(drop=True)


def build_recovery_model(
    patch_features: Sequence[str],
    config: RecoveryModelConfig = RecoveryModelConfig(),
) -> Pipeline:
    stress_pipeline = Pipeline(
        [
            (
                "spline",
                SplineTransformer(
                    n_knots=config.n_knots,
                    degree=config.degree,
                    include_bias=False,
                    extrapolation="linear",
                ),
            ),
            ("scale", StandardScaler()),
        ]
    )
    transformers: list[tuple[str, object, list[str]]] = [
        ("stress", stress_pipeline, ["inundation_fraction"])
    ]
    if patch_features:
        patch_pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]
        )
        transformers.append(("patch", patch_pipeline, list(patch_features)))

    preprocessing = ColumnTransformer(transformers, remainder="drop")
    return Pipeline(
        [
            ("preprocess", preprocessing),
            (
                "poisson",
                PoissonRegressor(
                    alpha=config.alpha,
                    fit_intercept=True,
                    max_iter=config.max_iter,
                    tol=1e-8,
                ),
            ),
        ]
    )


def _fit_weights(frame: pd.DataFrame) -> np.ndarray:
    weights = frame["exposure_years"].to_numpy(dtype=float)
    return weights / weights.mean()


def _weighted_log_rmse(
    observed: np.ndarray,
    predicted: np.ndarray,
    weights: np.ndarray,
) -> float:
    observed = np.clip(np.asarray(observed, dtype=float), 1e-12, None)
    predicted = np.clip(np.asarray(predicted, dtype=float), 1e-12, None)
    weights = np.asarray(weights, dtype=float)
    return float(
        np.sqrt(np.average((np.log(observed) - np.log(predicted)) ** 2, weights=weights))
    )


def evaluate_recovery_predictions(
    frame: pd.DataFrame,
    predicted: np.ndarray,
) -> dict[str, float]:
    observed = frame["hazard_per_year"].to_numpy(dtype=float)
    weights = frame["exposure_years"].to_numpy(dtype=float)
    predicted = np.clip(np.asarray(predicted, dtype=float), 1e-12, None)
    rank = stats.spearmanr(observed, predicted)
    return {
        "n_units": int(len(frame)),
        "n_blocks": int(frame["block_id"].nunique()),
        "exposure_years": float(weights.sum()),
        "poisson_deviance": float(
            mean_poisson_deviance(observed, predicted, sample_weight=weights)
        ),
        "weighted_log_rmse": _weighted_log_rmse(observed, predicted, weights),
        "spearman_rho": float(rank.statistic),
        "spearman_p": float(rank.pvalue),
    }


def cross_site_model_comparison(
    frame: pd.DataFrame,
    config: RecoveryModelConfig = RecoveryModelConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on one complete site and evaluate on the other site."""
    sites = sorted(frame["site"].unique())
    if len(sites) != 2:
        raise ValueError(f"Exactly two sites are required, got {sites}")

    metric_records: list[dict[str, float | str]] = []
    prediction_records: list[pd.DataFrame] = []
    for test_site in sites:
        train = frame.loc[frame["site"] != test_site].copy()
        test = frame.loc[frame["site"] == test_site].copy()
        train_site = str(train["site"].iloc[0])
        for model_name, patch_features in MODEL_FEATURES.items():
            model = build_recovery_model(patch_features, config)
            model.fit(
                train[["inundation_fraction", *patch_features]],
                train["hazard_per_year"],
                poisson__sample_weight=_fit_weights(train),
            )
            predicted = model.predict(
                test[["inundation_fraction", *patch_features]]
            )
            metrics = evaluate_recovery_predictions(test, predicted)
            metric_records.append(
                {
                    "train_site": train_site,
                    "test_site": test_site,
                    "model": model_name,
                    "alpha": config.alpha,
                    **metrics,
                }
            )
            prediction = test[
                [
                    "site",
                    "block_width_m",
                    "block_id",
                    "block_row",
                    "block_col",
                    "inundation_fraction",
                    "n_losses",
                    "n_recovered",
                    "exposure_years",
                    "hazard_per_year",
                ]
            ].copy()
            prediction["train_site"] = train_site
            prediction["model"] = model_name
            prediction["predicted_hazard_per_year"] = predicted
            prediction_records.append(prediction)

    metrics = pd.DataFrame.from_records(metric_records)
    predictions = pd.concat(prediction_records, ignore_index=True)
    return metrics, predictions


def add_increment_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    result = metrics.copy()
    baselines = result.loc[result["model"] == "stress_only"].set_index(
        ["train_site", "test_site"]
    )
    result["deviance_change_vs_stress_percent"] = np.nan
    result["log_rmse_change_vs_stress_percent"] = np.nan
    for index, row in result.iterrows():
        key = (row["train_site"], row["test_site"])
        baseline = baselines.loc[key]
        result.loc[index, "deviance_change_vs_stress_percent"] = (
            (row["poisson_deviance"] - baseline["poisson_deviance"])
            / baseline["poisson_deviance"]
            * 100.0
        )
        result.loc[index, "log_rmse_change_vs_stress_percent"] = (
            (row["weighted_log_rmse"] - baseline["weighted_log_rmse"])
            / baseline["weighted_log_rmse"]
            * 100.0
        )
    return result


def out_of_fold_stress_predictions(
    site_frame: pd.DataFrame,
    config: RecoveryModelConfig = RecoveryModelConfig(),
    n_splits: int = 5,
) -> np.ndarray:
    """Predict direct recovery from stress with spatial blocks held out."""
    groups = site_frame["block_id"].to_numpy()
    unique_groups = np.unique(groups)
    splits = min(n_splits, len(unique_groups))
    if splits < 2:
        raise ValueError("At least two spatial blocks are required")
    predictions = np.full(len(site_frame), np.nan, dtype=float)
    group_kfold = GroupKFold(n_splits=splits)
    for train_index, test_index in group_kfold.split(site_frame, groups=groups):
        train = site_frame.iloc[train_index]
        test = site_frame.iloc[test_index]
        model = build_recovery_model([], config)
        model.fit(
            train[["inundation_fraction"]],
            train["hazard_per_year"],
            poisson__sample_weight=_fit_weights(train),
        )
        predictions[test_index] = model.predict(test[["inundation_fraction"]])
    if not np.isfinite(predictions).all():
        raise RuntimeError("Out-of-fold predictions are incomplete")
    return predictions


def block_recovery_field(
    site_frame: pd.DataFrame,
    predicted_stress_hazard: np.ndarray,
) -> pd.DataFrame:
    """Aggregate unit hazards into a stress-adjusted local recovery field."""
    frame = site_frame.copy()
    frame["predicted_stress_hazard_per_year"] = np.asarray(
        predicted_stress_hazard, dtype=float
    )
    frame["expected_recoveries"] = (
        frame["predicted_stress_hazard_per_year"] * frame["exposure_years"]
    )

    aggregations: dict[str, str] = {
        "n_losses": "sum",
        "n_recovered": "sum",
        "exposure_years": "sum",
        "expected_recoveries": "sum",
        "center_y_m": "first",
        "center_x_m": "first",
        "block_row": "first",
        "block_col": "first",
        "patch_resolution_m": "first",
        "valid_patch_cells": "first",
    }
    aggregations.update({name: "first" for name in PATCH_STATIC_FEATURES})
    aggregations.update({name: "first" for name in PATCH_DYNAMIC_FEATURES})
    field = (
        frame.groupby(["site", "block_width_m", "block_id"], as_index=False)
        .agg(aggregations)
        .copy()
    )
    field["observed_hazard_per_year"] = (
        field["n_recovered"] / field["exposure_years"]
    )
    field["expected_stress_hazard_per_year"] = (
        field["expected_recoveries"] / field["exposure_years"]
    )
    field["log_hazard_residual"] = np.log(
        field["observed_hazard_per_year"].clip(lower=1e-12)
    ) - np.log(field["expected_stress_hazard_per_year"].clip(lower=1e-12))
    threshold = field["log_hazard_residual"].quantile(0.25)
    field["slow_recovery_q25"] = field["log_hazard_residual"] <= threshold
    return field


def permutation_spearman(
    x: Iterable[float],
    y: Iterable[float],
    n_permutations: int = 999,
    seed: int = 20260822,
) -> tuple[float, float]:
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    finite = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[finite]
    y_arr = y_arr[finite]
    if len(x_arr) < 4 or np.unique(x_arr).size < 2 or np.unique(y_arr).size < 2:
        return np.nan, np.nan
    observed = float(stats.spearmanr(x_arr, y_arr).statistic)
    generator = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(n_permutations):
        permuted = generator.permutation(y_arr)
        rho = float(stats.spearmanr(x_arr, permuted).statistic)
        exceedances += int(abs(rho) >= abs(observed))
    p_value = (exceedances + 1.0) / (n_permutations + 1.0)
    return observed, float(p_value)


def patch_residual_associations(
    block_field: pd.DataFrame,
    n_permutations: int = 999,
    seed: int = 20260822,
) -> pd.DataFrame:
    records: list[dict[str, float | int | str]] = []
    for feature_index, feature in enumerate(
        PATCH_STATIC_FEATURES + PATCH_DYNAMIC_FEATURES
    ):
        rho, p_value = permutation_spearman(
            block_field[feature],
            block_field["log_hazard_residual"],
            n_permutations=n_permutations,
            seed=seed + feature_index,
        )
        records.append(
            {
                "site": str(block_field["site"].iloc[0]),
                "block_width_m": float(block_field["block_width_m"].iloc[0]),
                "feature": feature,
                "feature_group": (
                    "static" if feature in PATCH_STATIC_FEATURES else "dynamic"
                ),
                "n_blocks": int(len(block_field)),
                "spearman_rho": rho,
                "permutation_p": p_value,
            }
        )
    return pd.DataFrame.from_records(records)


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    output = np.full(values.shape, np.nan, dtype=float)
    finite_indices = np.flatnonzero(np.isfinite(values))
    if finite_indices.size == 0:
        return output
    ordered_indices = finite_indices[np.argsort(values[finite_indices])]
    ordered = values[ordered_indices]
    adjusted = ordered * len(ordered) / np.arange(1, len(ordered) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output[ordered_indices] = np.clip(adjusted, 0.0, 1.0)
    return output


def multiscale_field_consistency(
    fields: dict[float, pd.DataFrame],
    pixel_size_m: float = 0.25,
) -> pd.DataFrame:
    records: list[dict[str, float | str]] = []
    scales = sorted(fields)
    for fine_index, fine_scale in enumerate(scales):
        for coarse_scale in scales[fine_index + 1 :]:
            fine = fields[fine_scale].copy()
            coarse = fields[coarse_scale].copy()
            coarse_pixels = int(round(coarse_scale / pixel_size_m))
            fine["coarse_block_row"] = (
                (fine["center_y_m"] / pixel_size_m).astype(int) // coarse_pixels
            )
            fine["coarse_block_col"] = (
                (fine["center_x_m"] / pixel_size_m).astype(int) // coarse_pixels
            )
            joined = fine.merge(
                coarse[
                    ["block_row", "block_col", "log_hazard_residual"]
                ].rename(
                    columns={
                        "block_row": "coarse_block_row",
                        "block_col": "coarse_block_col",
                        "log_hazard_residual": "coarse_log_hazard_residual",
                    }
                ),
                on=["coarse_block_row", "coarse_block_col"],
                how="inner",
            )
            rank = stats.spearmanr(
                joined["log_hazard_residual"],
                joined["coarse_log_hazard_residual"],
            )
            records.append(
                {
                    "site": str(fine["site"].iloc[0]),
                    "fine_scale_m": fine_scale,
                    "coarse_scale_m": coarse_scale,
                    "n_fine_blocks": int(len(joined)),
                    "spearman_rho": float(rank.statistic),
                    "p_value": float(rank.pvalue),
                }
            )
    return pd.DataFrame.from_records(records)

