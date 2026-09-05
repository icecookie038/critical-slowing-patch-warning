"""Reproducible benchmarks from published ecological CSD experiments.

The module deliberately separates three questions:

* temporal warning performance in a high-frequency predator--prey experiment;
* replicate precision and driver dependence in a yeast experiment; and
* spatial warning and descriptive patch structure in a macroalgal experiment.

Patch descriptors are never used to define whether CSD is present.  They are
secondary descriptions of the spatial organization in the field experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import md5
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import label


CLEMENTS_TRANSITION_TUBES = {
    14,
    16,
    21,
    23,
    26,
    28,
    29,
    30,
    32,
    35,
    38,
    44,
    46,
    48,
    49,
    53,
    57,
}

CLEMENTS_COMPOSITES: dict[str, tuple[str, ...]] = {
    "cv": ("cv",),
    "acf_cv": ("acf", "cv"),
    # Best overall 2-sigma composite reported by Clements & Ozgul (2016).
    "published_trait": ("cv", "size", "size_sd"),
}

RINDI_CLASSICAL_METRICS = (
    "spatial_sd",
    "spatial_cv",
    "spatial_skewness",
    "moran_lag1",
    "low_frequency_power",
)

RINDI_PATCH_METRICS = (
    "turf_fraction",
    "component_density",
    "largest_turf_patch_fraction",
    "edge_density",
)


def file_md5(path: Path) -> str:
    digest = md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_sd(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if len(array) < 2:
        return float("nan")
    return float(np.std(array, ddof=1))


def _expanding_zscore(value: float, history: Sequence[float]) -> float:
    history_array = np.asarray(history, dtype=float)
    finite = history_array[np.isfinite(history_array)]
    scale = _sample_sd(finite)
    if not np.isfinite(value) or not np.isfinite(scale) or scale <= 0:
        return float("nan")
    return float((value - np.mean(finite)) / scale)


def _linear_interpolate(
    x: np.ndarray,
    y: np.ndarray,
    *,
    zero_is_missing: bool = False,
) -> np.ndarray:
    values = np.asarray(y, dtype=float).copy()
    if zero_is_missing:
        values[values == 0] = np.nan
    valid = np.isfinite(values)
    result = np.full(values.shape, np.nan, dtype=float)
    if not valid.any():
        return result
    inside = (x >= x[valid].min()) & (x <= x[valid].max())
    result[inside] = np.interp(x[inside], x[valid], values[valid])
    return result


def r_acf_lag1(values: Sequence[float]) -> float:
    """Lag-1 ACF using the normalization used by R's ``acf``."""
    array = np.asarray(values, dtype=float)
    if len(array) < 2 or not np.isfinite(array).all():
        return float("nan")
    centered = array - array.mean()
    denominator = float(np.dot(centered, centered))
    if denominator <= 0:
        return float("nan")
    return float(np.dot(centered[:-1], centered[1:]) / denominator)


def load_clements_workbook(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path, engine="xlrd")
    unnamed = [column for column in frame.columns if str(column).startswith("Unnamed")]
    if unnamed:
        frame = frame.drop(columns=unnamed)
    required = {"Day", "Tube", "Treatment", "Count", "mean.size", "sd.size"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Clements workbook is missing columns: {sorted(missing)}")
    return frame.sort_values(["Tube", "Day"]).reset_index(drop=True)


def estimate_clements_tipping_points(frame: pd.DataFrame) -> pd.DataFrame:
    """Port the published R/LOESS realized-growth transition calculation."""
    try:
        from skmisc.loess import loess
    except ImportError as exc:  # pragma: no cover - dependency message
        raise ImportError(
            "scikit-misc is required for the published LOESS transition port"
        ) from exc

    records: list[dict[str, float | int | str | bool]] = []
    for tube, group in frame.groupby("Tube", sort=True):
        group = group.sort_values("Day")
        days = group["Day"].to_numpy(dtype=float)
        counts = _linear_interpolate(
            days,
            group["Count"].to_numpy(dtype=float),
            zero_is_missing=True,
        )
        ratio = counts[1:] / counts[:-1]
        ratio_days = days[:-1]
        valid = np.isfinite(ratio)
        model = loess(ratio_days[valid], ratio[valid], span=0.75, degree=2)
        model.fit()
        prediction_days = np.arange(
            max(12, int(np.ceil(ratio_days[valid].min()))),
            min(45, int(np.floor(ratio_days[valid].max()))) + 1,
        )
        fitted = np.asarray(model.predict(prediction_days).values, dtype=float)
        tipping_day = float("nan")
        if len(fitted) and fitted[-1] < 1.0:
            nondeclining = np.flatnonzero(fitted >= 1.0)
            if len(nondeclining):
                tipping_day = float(prediction_days[nondeclining[-1]])
        records.append(
            {
                "tube": int(tube),
                "treatment": str(group["Treatment"].iloc[0]),
                "tipping_day": tipping_day,
                "last_fitted_growth": float(fitted[-1]),
                "n_fitted_days": int(len(fitted)),
                "transition_inferred": bool(np.isfinite(tipping_day)),
            }
        )
    return pd.DataFrame.from_records(records)


def _clements_window(group: pd.DataFrame, tipping_day: float) -> pd.DataFrame:
    data = group.sort_values("Day").copy()
    days = data["Day"].to_numpy(dtype=float)
    data["mean.size"] = _linear_interpolate(
        days, data["mean.size"].to_numpy(dtype=float)
    )
    data["sd.size"] = _linear_interpolate(
        days, data["sd.size"].to_numpy(dtype=float)
    )
    if np.isfinite(tipping_day):
        data = data.loc[data["Day"] < tipping_day].copy()
    if str(data["Treatment"].iloc[0]) == "Constant":
        return data.head(20).reset_index(drop=True)

    zero_positions = np.flatnonzero(data["Count"].to_numpy(dtype=float) == 0)
    if len(zero_positions):
        counts = data["Count"].to_numpy(dtype=float)
        for index in range(1, len(counts)):
            if counts[index] == 0 and counts[index - 1] == 0:
                cutoff_day = float(data["Day"].iloc[index - 1])
                data = data.loc[data["Day"] < cutoff_day].copy()
                break
    return data.tail(20).reset_index(drop=True)


def clements_expanding_indicators(window: pd.DataFrame) -> pd.DataFrame:
    histories: dict[str, list[float]] = {
        "cv": [],
        "acf": [],
        "size": [],
        "size_sd": [],
    }
    records: list[dict[str, float]] = []
    for length in range(2, len(window) + 1):
        subset = window.iloc[:length]
        count = subset["Count"].to_numpy(dtype=float)
        mean_count = float(np.mean(count))
        values = {
            "cv": _sample_sd(count) / mean_count if mean_count != 0 else np.nan,
            "acf": r_acf_lag1(count),
            "size": float(subset["mean.size"].mean()),
            "size_sd": float(subset["sd.size"].mean()),
        }
        normalized: dict[str, float] = {}
        for name, value in values.items():
            histories[name].append(float(value))
            normalized[name] = _expanding_zscore(value, histories[name])
        # Declining mean body size is the expected warning direction.
        normalized["size"] *= -1.0
        records.append(
            {
                "day": float(subset["Day"].iloc[-1]),
                **normalized,
            }
        )
    return pd.DataFrame.from_records(records)


def clements_threshold_signal(
    indicators: pd.DataFrame,
    components: Sequence[str],
    sigma: float = 2.0,
) -> tuple[bool, float]:
    if indicators.empty:
        return False, float("nan")
    composite = indicators[list(components)].sum(
        axis=1, min_count=len(components)
    ).to_numpy(dtype=float)
    running_mean = np.full(len(composite), np.nan)
    running_sd = np.full(len(composite), np.nan)
    for index in range(len(composite)):
        prefix = composite[: index + 1]
        finite = prefix[np.isfinite(prefix)]
        if len(finite):
            running_mean[index] = finite.mean()
        running_sd[index] = _sample_sd(finite)
    hit = (
        np.isfinite(composite)
        & np.isfinite(running_mean)
        & np.isfinite(running_sd)
        & (composite > running_mean + sigma * running_sd)
    )
    if not hit.any():
        return False, float("nan")
    first = int(np.flatnonzero(hit)[0])
    return True, float(indicators["day"].iloc[first])


def clements_replication(
    frame: pd.DataFrame,
    tipping_points: pd.DataFrame,
    *,
    sigma: float = 2.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tip_map = tipping_points.set_index("tube")["tipping_day"].to_dict()
    records: list[dict[str, float | int | str | bool]] = []
    for tube, group in frame.groupby("Tube", sort=True):
        treatment = str(group["Treatment"].iloc[0])
        tipping_day = float(tip_map[int(tube)])
        if treatment == "Constant" and int(tube) == 14:
            continue
        if treatment != "Constant" and not np.isfinite(tipping_day):
            continue
        window = _clements_window(group, tipping_day)
        indicators = clements_expanding_indicators(window)
        for metric, components in CLEMENTS_COMPOSITES.items():
            signal, signal_day = clements_threshold_signal(
                indicators, components, sigma=sigma
            )
            records.append(
                {
                    "tube": int(tube),
                    "treatment": treatment,
                    "class": "constant" if treatment == "Constant" else "deteriorating",
                    "tipping_day": tipping_day,
                    "n_observations": int(len(window)),
                    "sampling_interval_days": 1,
                    "phase": 0,
                    "metric": metric,
                    "signal": signal,
                    "signal_day": signal_day,
                    "eligible": len(window) >= 6,
                }
            )
    detail = pd.DataFrame.from_records(records)
    summary = summarize_clements_signals(detail)
    return detail, summary


def summarize_clements_signals(detail: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        detail.groupby(["sampling_interval_days", "metric", "class"], as_index=False)
        .agg(
            n_trials=("signal", "size"),
            n_tubes=("tube", "nunique"),
            eligible_fraction=("eligible", "mean"),
            signal_rate_all=("signal", "mean"),
        )
    )
    eligible = (
        detail.loc[detail["eligible"]]
        .groupby(["sampling_interval_days", "metric", "class"])["signal"]
        .mean()
        .rename("signal_rate_eligible")
        .reset_index()
    )
    grouped = grouped.merge(
        eligible,
        on=["sampling_interval_days", "metric", "class"],
        how="left",
    )
    wide = grouped.pivot(
        index=["sampling_interval_days", "metric"],
        columns="class",
        values="signal_rate_all",
    )
    grouped = grouped.merge(
        (
            wide.get("deteriorating", np.nan) - wide.get("constant", np.nan)
        ).rename("normalized_metric_score"),
        on=["sampling_interval_days", "metric"],
        how="left",
    )
    return grouped.sort_values(
        ["sampling_interval_days", "metric", "class"]
    ).reset_index(drop=True)


def clements_temporal_downsampling(
    frame: pd.DataFrame,
    tipping_points: pd.DataFrame,
    intervals: Sequence[int] = (1, 2, 3, 4),
    *,
    sigma: float = 2.0,
    min_observations: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tip_map = tipping_points.set_index("tube")["tipping_day"].to_dict()
    records: list[dict[str, float | int | str | bool]] = []
    for tube, group in frame.groupby("Tube", sort=True):
        treatment = str(group["Treatment"].iloc[0])
        tipping_day = float(tip_map[int(tube)])
        if treatment == "Constant" and int(tube) == 14:
            continue
        if treatment != "Constant" and not np.isfinite(tipping_day):
            continue
        full_window = _clements_window(group, tipping_day)
        for interval in intervals:
            if interval < 1:
                raise ValueError("Sampling intervals must be positive integers")
            for phase in range(interval):
                window = full_window.iloc[phase::interval].reset_index(drop=True)
                eligible = len(window) >= min_observations
                indicators = (
                    clements_expanding_indicators(window)
                    if eligible
                    else pd.DataFrame()
                )
                for metric, components in CLEMENTS_COMPOSITES.items():
                    signal, signal_day = clements_threshold_signal(
                        indicators, components, sigma=sigma
                    )
                    records.append(
                        {
                            "tube": int(tube),
                            "treatment": treatment,
                            "class": (
                                "constant"
                                if treatment == "Constant"
                                else "deteriorating"
                            ),
                            "tipping_day": tipping_day,
                            "n_observations": int(len(window)),
                            "sampling_interval_days": int(interval),
                            "phase": int(phase),
                            "metric": metric,
                            "signal": bool(signal),
                            "signal_day": signal_day,
                            "eligible": bool(eligible),
                        }
                    )
    detail = pd.DataFrame.from_records(records)
    return detail, summarize_clements_signals(detail)


def load_dai_deterioration(root: Path) -> dict[str, np.ndarray]:
    files = {
        "dilution_factor": "all_DF.txt",
        "sucrose": "all_SUC.txt",
        "control": "all_DF_control.txt",
    }
    arrays = {name: np.loadtxt(root / filename) for name, filename in files.items()}
    if any(array.ndim != 2 for array in arrays.values()):
        raise ValueError("Dai input arrays must be replicate-by-time matrices")
    return arrays


def estimate_collapse_index(populations: np.ndarray, ratio_threshold: float = 0.5) -> int:
    means = np.mean(populations, axis=0)
    ratios = means[1:] / means[:-1]
    crossings = np.flatnonzero(ratios < ratio_threshold)
    if not len(crossings):
        raise ValueError("No collapse crossing was found")
    return int(crossings[0] + 1)


def dai_indicators(populations: np.ndarray) -> dict[str, np.ndarray]:
    populations = np.asarray(populations, dtype=float)
    means = populations.mean(axis=0)
    cv = populations.std(axis=0, ddof=1) / means
    ar1 = np.full(populations.shape[1] - 1, np.nan)
    for time in range(populations.shape[1] - 1):
        current = populations[:, time] - populations[:, time].mean()
        following = populations[:, time + 1] - populations[:, time + 1].mean()
        denominator = float(np.dot(current, current))
        if denominator > 0:
            ar1[time] = float(np.dot(current, following) / denominator)
    autocorrelation_time = np.full_like(ar1, np.nan)
    valid = (ar1 > 0) & (ar1 < 1)
    autocorrelation_time[valid] = -1.0 / np.log(ar1[valid])
    return {
        "cv": cv,
        "ar1_slope": ar1,
        "autocorrelation_time": autocorrelation_time,
    }


def _indicator_change(
    curve: np.ndarray,
    tipping_index: int,
    *,
    window: int = 10,
    baseline_points: int = 5,
    late_points: int = 3,
) -> dict[str, float]:
    start = tipping_index - window
    if start < 0 or tipping_index > len(curve):
        raise ValueError("Indicator curve does not contain the pre-collapse window")
    segment = np.asarray(curve[start:tipping_index], dtype=float)
    early = segment[:baseline_points]
    late = segment[-late_points:]
    tau = stats.kendalltau(
        np.arange(len(segment)), segment, nan_policy="omit"
    )
    return {
        "early_mean": float(np.nanmean(early)),
        "late_mean": float(np.nanmean(late)),
        "late_minus_early": float(np.nanmean(late) - np.nanmean(early)),
        "kendall_tau": float(tau.statistic),
        "kendall_p": float(tau.pvalue),
    }


def dai_replication(
    arrays: Mapping[str, np.ndarray],
    *,
    pre_collapse_window: int = 10,
) -> tuple[pd.DataFrame, dict[str, int]]:
    tipping = {
        "dilution_factor": estimate_collapse_index(arrays["dilution_factor"]),
        "sucrose": estimate_collapse_index(arrays["sucrose"]),
    }
    records: list[dict[str, float | int | str]] = []
    for driver, tipping_index in tipping.items():
        for environment in (driver, "control"):
            indicator_curves = dai_indicators(arrays[environment])
            for indicator in ("cv", "ar1_slope", "autocorrelation_time"):
                change = _indicator_change(
                    indicator_curves[indicator],
                    tipping_index,
                    window=pre_collapse_window,
                )
                records.append(
                    {
                        "driver": driver,
                        "environment": environment,
                        "indicator": indicator,
                        "n_replicates": int(arrays[environment].shape[0]),
                        "n_time_points": int(arrays[environment].shape[1]),
                        "tipping_index": int(tipping_index),
                        **change,
                    }
                )
    return pd.DataFrame.from_records(records), tipping


def _rank_similarity(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3:
        return float("nan")
    return float(stats.spearmanr(left[valid], right[valid]).statistic)


def dai_replicate_precision(
    arrays: Mapping[str, np.ndarray],
    tipping: Mapping[str, int],
    sample_sizes: Sequence[int] = (48, 24, 12, 6),
    *,
    n_iterations: int = 500,
    seed: int = 2026,
    pre_collapse_window: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    detail_records: list[dict[str, float | int | str | bool]] = []
    for driver in ("dilution_factor", "sucrose"):
        tipping_index = int(tipping[driver])
        full_curves = dai_indicators(arrays[driver])
        full_control_curves = dai_indicators(arrays["control"])
        for sample_size in sample_sizes:
            if sample_size > arrays[driver].shape[0]:
                continue
            iterations = 1 if sample_size == arrays[driver].shape[0] else n_iterations
            for iteration in range(iterations):
                if sample_size == arrays[driver].shape[0]:
                    driver_index = np.arange(sample_size)
                    control_index = np.arange(sample_size)
                else:
                    driver_index = rng.choice(
                        arrays[driver].shape[0], sample_size, replace=False
                    )
                    control_index = rng.choice(
                        arrays["control"].shape[0], sample_size, replace=False
                    )
                sampled = dai_indicators(arrays[driver][driver_index])
                sampled_control = dai_indicators(arrays["control"][control_index])
                for indicator in ("cv", "ar1_slope"):
                    driver_change = _indicator_change(
                        sampled[indicator],
                        tipping_index,
                        window=pre_collapse_window,
                    )
                    control_change = _indicator_change(
                        sampled_control[indicator],
                        tipping_index,
                        window=pre_collapse_window,
                    )
                    start = tipping_index - pre_collapse_window
                    sampled_segment = sampled[indicator][start:tipping_index]
                    full_segment = full_curves[indicator][start:tipping_index]
                    full_control_segment = full_control_curves[indicator][
                        start:tipping_index
                    ]
                    full_range = float(np.nanmax(full_segment) - np.nanmin(full_segment))
                    normalized_rmse = (
                        float(
                            np.sqrt(
                                np.nanmean((sampled_segment - full_segment) ** 2)
                            )
                            / full_range
                        )
                        if full_range > 0
                        else float("nan")
                    )
                    detail_records.append(
                        {
                            "driver": driver,
                            "indicator": indicator,
                            "sample_size": int(sample_size),
                            "iteration": int(iteration),
                            "driver_delta": driver_change["late_minus_early"],
                            "control_delta": control_change["late_minus_early"],
                            "positive_direction": (
                                driver_change["late_minus_early"] > 0
                            ),
                            "greater_than_control": (
                                driver_change["late_minus_early"]
                                > control_change["late_minus_early"]
                            ),
                            "rank_similarity_to_full": _rank_similarity(
                                sampled_segment, full_segment
                            ),
                            "normalized_rmse_to_full": normalized_rmse,
                            "full_control_rank_similarity": _rank_similarity(
                                full_control_segment, full_control_segment
                            ),
                        }
                    )
    detail = pd.DataFrame.from_records(detail_records)
    summary = (
        detail.groupby(["driver", "indicator", "sample_size"], as_index=False)
        .agg(
            n_iterations=("iteration", "size"),
            median_driver_delta=("driver_delta", "median"),
            median_control_delta=("control_delta", "median"),
            positive_direction_rate=("positive_direction", "mean"),
            greater_than_control_rate=("greater_than_control", "mean"),
            median_rank_similarity=("rank_similarity_to_full", "median"),
            median_normalized_rmse=("normalized_rmse_to_full", "median"),
        )
        .sort_values(["driver", "indicator", "sample_size"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    return detail, summary


@dataclass(frozen=True)
class RindiGrid:
    year: int
    plot: int
    removal_percent: float
    turf_score: np.ndarray


def load_rindi_grids(root: Path) -> list[RindiGrid]:
    grids: list[RindiGrid] = []
    for year, filename in ((2014, "Exp_data_2014.txt"), (2015, "Exp_data_2015.txt")):
        frame = pd.read_csv(root / filename, sep="\t")
        for plot, group in frame.groupby("Number", sort=True):
            scores = group.iloc[:, 2:].to_numpy(dtype=float)
            if scores.shape != (5, 30):
                raise ValueError(
                    f"Expected a 5x30 grid for plot {plot}, {year}; got {scores.shape}"
                )
            grids.append(
                RindiGrid(
                    year=year,
                    plot=int(plot),
                    removal_percent=float(group["%_cys_removed"].iloc[0]),
                    turf_score=scores,
                )
            )
    return grids


def _spatial_plane_residuals(
    understory_cover: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    rows, cols = np.indices(understory_cover.shape)
    design = np.column_stack(
        [np.ones(mask.sum()), rows[mask], cols[mask]]
    )
    coefficients = np.linalg.lstsq(
        design, understory_cover[mask], rcond=None
    )[0]
    plane = coefficients[0] + coefficients[1] * rows + coefficients[2] * cols
    return np.where(mask, understory_cover - plane, np.nan)


def _moran_lag1(residuals: np.ndarray) -> float:
    valid = np.isfinite(residuals)
    centered = residuals - np.nanmean(residuals)
    denominator = float(np.nansum(centered**2))
    horizontal = valid[:, :-1] & valid[:, 1:]
    vertical = valid[:-1, :] & valid[1:, :]
    n_edges = int(horizontal.sum() + vertical.sum())
    if denominator <= 0 or n_edges == 0:
        return float("nan")
    edge_product = float(
        np.sum(centered[:, :-1][horizontal] * centered[:, 1:][horizontal])
        + np.sum(centered[:-1, :][vertical] * centered[1:, :][vertical])
    )
    # The usual directed-neighbour definition reduces to this expression when
    # every undirected edge is included in both directions.
    return float(valid.sum() / n_edges * edge_product / denominator)


def _low_frequency_power(residuals: np.ndarray, cutoff: float = 0.10) -> float:
    valid = np.isfinite(residuals)
    filled = np.where(valid, residuals, 0.0)
    power = np.abs(np.fft.fft2(filled)) ** 2
    row_frequency = np.abs(np.fft.fftfreq(filled.shape[0]))
    col_frequency = np.abs(np.fft.fftfreq(filled.shape[1]))
    low = (
        (row_frequency[:, None] <= cutoff * row_frequency.max() + 1e-12)
        & (col_frequency[None, :] <= cutoff * col_frequency.max() + 1e-12)
    )
    low[0, 0] = False
    if not low.any() or valid.sum() == 0:
        return float("nan")
    return float(np.mean(power[low]) / valid.sum())


def rindi_grid_metrics(
    turf_score: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    turf_threshold: int = 2,
) -> dict[str, float]:
    score = np.asarray(turf_score, dtype=float)
    if mask is None:
        mask = np.isfinite(score)
    else:
        mask = np.asarray(mask, dtype=bool) & np.isfinite(score)
    if mask.sum() < 6:
        raise ValueError("At least six spatial observations are required")
    understory = 100.0 - 25.0 * score
    residuals = _spatial_plane_residuals(understory, mask)
    residual_values = residuals[mask]
    raw_values = understory[mask]
    spatial_sd = float(np.std(residual_values, ddof=1))
    mean_understory = float(np.mean(raw_values))

    turf = (score >= turf_threshold) & mask
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    components, n_components = label(turf, structure=structure)
    component_sizes = (
        np.bincount(components[components > 0].ravel())
        if n_components
        else np.array([], dtype=int)
    )
    horizontal_valid = mask[:, :-1] & mask[:, 1:]
    vertical_valid = mask[:-1, :] & mask[1:, :]
    n_edges = int(horizontal_valid.sum() + vertical_valid.sum())
    different_edges = int(
        ((turf[:, :-1] != turf[:, 1:]) & horizontal_valid).sum()
        + ((turf[:-1, :] != turf[1:, :]) & vertical_valid).sum()
    )
    n_observed = int(mask.sum())
    return {
        "n_observed_cells": float(n_observed),
        "mean_understory_cover": mean_understory,
        "spatial_sd": spatial_sd,
        "spatial_cv": (
            spatial_sd / mean_understory if mean_understory != 0 else np.nan
        ),
        "spatial_skewness": float(stats.skew(residual_values, bias=True)),
        "moran_lag1": _moran_lag1(residuals),
        "low_frequency_power": _low_frequency_power(residuals),
        "turf_fraction": float(turf.sum() / n_observed),
        "component_density": float(n_components / n_observed),
        "largest_turf_patch_fraction": (
            float(component_sizes.max() / n_observed)
            if len(component_sizes)
            else 0.0
        ),
        "edge_density": (
            float(different_edges / n_edges) if n_edges else float("nan")
        ),
    }


def compute_rindi_metrics(
    grids: Sequence[RindiGrid],
    *,
    masks: Mapping[tuple[int, int], np.ndarray] | None = None,
    turf_threshold: int = 2,
) -> pd.DataFrame:
    records: list[dict[str, float | int]] = []
    for grid in grids:
        mask = None if masks is None else masks[(grid.year, grid.plot)]
        records.append(
            {
                "year": grid.year,
                "plot": grid.plot,
                "removal_percent": grid.removal_percent,
                "turf_threshold": int(turf_threshold),
                **rindi_grid_metrics(
                    grid.turf_score,
                    mask=mask,
                    turf_threshold=turf_threshold,
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def _year_adjusted_slope(frame: pd.DataFrame, metric: str) -> float:
    subset = frame[["year", "removal_percent", metric]].dropna()
    # Pandas 3 may expose read-only arrays; centering needs owned buffers.
    x = subset["removal_percent"].to_numpy(dtype=float, copy=True)
    y = subset[metric].to_numpy(dtype=float, copy=True)
    years = subset["year"].to_numpy()
    for year in np.unique(years):
        selection = years == year
        x[selection] -= x[selection].mean()
        y[selection] -= y[selection].mean()
    denominator = float(np.dot(x, x))
    return float(np.dot(x, y) / denominator) if denominator > 0 else np.nan


def blocked_permutation_slope(
    frame: pd.DataFrame,
    metric: str,
    *,
    n_permutations: int = 999,
    seed: int = 2026,
) -> tuple[float, float]:
    observed = _year_adjusted_slope(frame, metric)
    plot_removal = (
        frame[["plot", "removal_percent"]]
        .drop_duplicates("plot")
        .sort_values("plot")
    )
    plots = plot_removal["plot"].to_numpy()
    removals = plot_removal["removal_percent"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    permuted = np.empty(n_permutations, dtype=float)
    for iteration in range(n_permutations):
        mapping = dict(zip(plots, rng.permutation(removals), strict=True))
        shuffled = frame.copy()
        shuffled["removal_percent"] = shuffled["plot"].map(mapping)
        permuted[iteration] = _year_adjusted_slope(shuffled, metric)
    p_value = float((1 + np.sum(permuted >= observed)) / (n_permutations + 1))
    return observed, p_value


def rindi_replication(
    metrics: pd.DataFrame,
    *,
    n_permutations: int = 999,
    seed: int = 2026,
) -> pd.DataFrame:
    records: list[dict[str, float | str | bool]] = []
    for index, metric in enumerate((*RINDI_CLASSICAL_METRICS, *RINDI_PATCH_METRICS)):
        slope, p_value = blocked_permutation_slope(
            metrics,
            metric,
            n_permutations=n_permutations,
            seed=seed + index,
        )
        rank = stats.spearmanr(
            metrics["removal_percent"], metrics[metric], nan_policy="omit"
        )
        records.append(
            {
                "metric": metric,
                "metric_role": (
                    "published_spatial_ews"
                    if metric in RINDI_CLASSICAL_METRICS
                    else "patch_explanation"
                ),
                "year_adjusted_slope": slope,
                "one_sided_block_permutation_p": p_value,
                "spearman_rho": float(rank.statistic),
                "spearman_p": float(rank.pvalue),
                "positive_direction": bool(slope > 0),
            }
        )
    return pd.DataFrame.from_records(records)


def rindi_spatial_subsampling(
    grids: Sequence[RindiGrid],
    full_metrics: pd.DataFrame,
    fractions: Sequence[float] = (1.0, 0.75, 0.50, 0.25),
    *,
    n_iterations: int = 200,
    seed: int = 2026,
    turf_threshold: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    metric_names = (*RINDI_CLASSICAL_METRICS, *RINDI_PATCH_METRICS)
    full_slopes = {
        metric: _year_adjusted_slope(full_metrics, metric)
        for metric in metric_names
    }
    detail_records: list[dict[str, float | int | str | bool]] = []
    for fraction in fractions:
        if not 0 < fraction <= 1:
            raise ValueError("Spatial sampling fractions must be in (0, 1]")
        iterations = 1 if fraction == 1.0 else n_iterations
        for iteration in range(iterations):
            masks: dict[tuple[int, int], np.ndarray] = {}
            for grid in grids:
                n_cells = grid.turf_score.size
                n_keep = max(6, int(round(fraction * n_cells)))
                flat_mask = np.zeros(n_cells, dtype=bool)
                if n_keep == n_cells:
                    flat_mask[:] = True
                else:
                    flat_mask[rng.choice(n_cells, n_keep, replace=False)] = True
                masks[(grid.year, grid.plot)] = flat_mask.reshape(
                    grid.turf_score.shape
                )
            sampled = compute_rindi_metrics(
                grids, masks=masks, turf_threshold=turf_threshold
            )
            for metric in metric_names:
                slope = _year_adjusted_slope(sampled, metric)
                merged = full_metrics[["year", "plot", metric]].merge(
                    sampled[["year", "plot", metric]],
                    on=["year", "plot"],
                    suffixes=("_full", "_sampled"),
                    validate="one_to_one",
                )
                detail_records.append(
                    {
                        "sampling_fraction": float(fraction),
                        "iteration": int(iteration),
                        "metric": metric,
                        "metric_role": (
                            "published_spatial_ews"
                            if metric in RINDI_CLASSICAL_METRICS
                            else "patch_explanation"
                        ),
                        "year_adjusted_slope": slope,
                        "full_slope": full_slopes[metric],
                        "direction_retained": bool(
                            np.sign(slope) == np.sign(full_slopes[metric])
                        ),
                        "rank_similarity_to_full": _rank_similarity(
                            merged[f"{metric}_sampled"].to_numpy(dtype=float),
                            merged[f"{metric}_full"].to_numpy(dtype=float),
                        ),
                    }
                )
    detail = pd.DataFrame.from_records(detail_records)
    summary = (
        detail.groupby(
            ["sampling_fraction", "metric", "metric_role"], as_index=False
        )
        .agg(
            n_iterations=("iteration", "size"),
            full_slope=("full_slope", "first"),
            median_slope=("year_adjusted_slope", "median"),
            slope_q05=("year_adjusted_slope", lambda x: x.quantile(0.05)),
            slope_q95=("year_adjusted_slope", lambda x: x.quantile(0.95)),
            direction_retention_rate=("direction_retained", "mean"),
            median_rank_similarity=("rank_similarity_to_full", "median"),
        )
        .sort_values(["metric", "sampling_fraction"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return detail, summary


def rindi_patch_threshold_sensitivity(
    grids: Sequence[RindiGrid],
    thresholds: Sequence[int] = (1, 2, 3),
    *,
    n_permutations: int = 999,
    seed: int = 2026,
) -> pd.DataFrame:
    records: list[dict[str, float | int | str]] = []
    for threshold in thresholds:
        metrics = compute_rindi_metrics(grids, turf_threshold=threshold)
        for offset, metric in enumerate(RINDI_PATCH_METRICS):
            slope, p_value = blocked_permutation_slope(
                metrics,
                metric,
                n_permutations=n_permutations,
                seed=seed + threshold * 20 + offset,
            )
            records.append(
                {
                    "turf_threshold_score": int(threshold),
                    "minimum_turf_cover_percent": int(threshold * 25),
                    "metric": metric,
                    "year_adjusted_slope": slope,
                    "one_sided_block_permutation_p": p_value,
                }
            )
    return pd.DataFrame.from_records(records)
