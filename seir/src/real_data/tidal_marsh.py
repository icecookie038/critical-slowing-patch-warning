"""Tidal-marsh recovery events and spatial patch context.

This module ports the right-censored recovery estimator used by van Belzen
et al. (2017) and adds scale-aware aggregation into spatial blocks.  Recovery
hazards are the primary resilience measurement.  Patch descriptors are kept
separate and are intended only as explanatory spatial covariates.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage, stats


@dataclass(frozen=True)
class TidalMarshSite:
    name: str
    years: tuple[int, ...]
    bin_min: float
    bin_max: float
    bin_step: float = 0.01
    pixel_size_m: float = 0.25


SITES: dict[str, TidalMarshSite] = {
    "Hellegat": TidalMarshSite(
        name="Hellegat",
        years=(1976, 1982, 1993, 1996, 1998, 2001, 2004, 2008, 2010, 2011, 2012),
        bin_min=0.10,
        bin_max=0.46,
    ),
    "Paulina": TidalMarshSite(
        name="Paulina",
        years=(1982, 1996, 1998, 2001, 2008, 2010, 2011, 2012),
        bin_min=0.30,
        bin_max=0.43,
    ),
}


@dataclass
class TidalMarshData:
    site: TidalMarshSite
    vegetation: np.ndarray
    inundation: np.ndarray
    bundle_root: Path


@dataclass
class RecoveryEventBatch:
    disturbance_index: int
    rows: np.ndarray
    cols: np.ndarray
    duration_years: np.ndarray
    recovered: np.ndarray


def resolve_bundle_root(data_root: Path | str) -> Path:
    """Resolve either the extraction root or the bundle directory itself."""
    root = Path(data_root)
    candidates = [
        root,
        root / "Data Bundle NCOMMS",
        root / "Data Bundle NCOMMS" / "Data",
    ]
    for candidate in candidates:
        if (candidate / "Data" / "data" / "Hellegat").is_dir():
            return candidate
        if (candidate / "data" / "Hellegat").is_dir():
            return candidate.parent
    raise FileNotFoundError(
        "Could not locate 'Data Bundle NCOMMS/Data/data'. Run "
        "scripts/download_tidal_marsh_data.py first."
    )


def analysis_bin_edges(site: TidalMarshSite) -> np.ndarray:
    """Return the exact bin edges used by the authors' MATLAB loop.

    The published code creates an inclusive vector from ``bin_min`` through
    ``bin_max`` but iterates only to ``n - 2``.  Dropping the final edge here
    reproduces its 35 Hellegat and 12 Paulina intervals.
    """
    inclusive = np.arange(
        site.bin_min,
        site.bin_max + site.bin_step / 2.0,
        site.bin_step,
        dtype=float,
    )
    return inclusive[:-1]


def _data_directory(bundle_root: Path, site_name: str) -> Path:
    return bundle_root / "Data" / "data" / site_name


def load_tidal_marsh_site(
    data_root: Path | str,
    site_name: str,
    inundation_dtype: np.dtype = np.float32,
) -> TidalMarshData:
    """Load binary vegetation snapshots and the inundation field."""
    if site_name not in SITES:
        raise KeyError(f"Unknown site {site_name!r}; choose from {sorted(SITES)}")
    site = SITES[site_name]
    bundle_root = resolve_bundle_root(data_root)
    site_dir = _data_directory(bundle_root, site_name)

    snapshots: list[np.ndarray] = []
    for year in site.years:
        path = site_dir / f"{site_name}{year}bin.tif"
        if not path.exists():
            raise FileNotFoundError(path)
        snapshots.append(np.asarray(Image.open(path), dtype=bool))

    vegetation = np.stack(snapshots, axis=0)
    inundation = np.loadtxt(
        site_dir / f"{site_name}_IFmap.txt",
        delimiter=",",
        dtype=inundation_dtype,
    )
    if vegetation.shape[1:] != inundation.shape:
        raise ValueError(
            f"Shape mismatch for {site_name}: vegetation={vegetation.shape}, "
            f"inundation={inundation.shape}"
        )
    return TidalMarshData(site, vegetation, inundation, bundle_root)


def iter_recovery_events(
    vegetation: np.ndarray,
    years: Sequence[int],
) -> Iterator[RecoveryEventBatch]:
    """Yield loss events with first-recovery times and right censoring.

    A disturbance occurs when a vegetated pixel is absent in the next image.
    Time starts at that next image.  The first later vegetated observation is
    the recovery event; otherwise observation ends right-censored at the last
    image.  Losses first seen in the final image have zero follow-up and are
    excluded, matching the authors' ``data2bin > 0`` filter.
    """
    seq = np.asarray(vegetation, dtype=bool)
    years_arr = np.asarray(years, dtype=float)
    if seq.ndim != 3:
        raise ValueError(f"vegetation must have shape (T,H,W), got {seq.shape}")
    if len(years_arr) != seq.shape[0]:
        raise ValueError("years must match the number of snapshots")
    if np.any(np.diff(years_arr) <= 0):
        raise ValueError("years must be unique and strictly increasing")

    for start in range(1, seq.shape[0] - 1):
        disturbed = seq[start - 1] & ~seq[start]
        rows, cols = np.nonzero(disturbed)
        if rows.size == 0:
            continue

        duration = np.full(rows.size, years_arr[-1] - years_arr[start], dtype=float)
        recovered = np.zeros(rows.size, dtype=bool)
        active = np.ones(rows.size, dtype=bool)

        for future in range(start + 1, seq.shape[0]):
            active_positions = np.flatnonzero(active)
            if active_positions.size == 0:
                break
            now_recovered = seq[
                future,
                rows[active_positions],
                cols[active_positions],
            ]
            recovered_positions = active_positions[now_recovered]
            if recovered_positions.size:
                recovered[recovered_positions] = True
                duration[recovered_positions] = years_arr[future] - years_arr[start]
                active[recovered_positions] = False

        yield RecoveryEventBatch(start, rows, cols, duration, recovered)


def _inundation_bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(edges, values, side="right") - 1
    n_bins = len(edges) - 1
    valid = (indices >= 0) & (indices < n_bins) & (values < edges[-1])
    return np.where(valid, indices, -1)


def _poisson_rate_interval(
    events: np.ndarray,
    exposure_days: np.ndarray,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    events = np.asarray(events, dtype=float)
    exposure_days = np.asarray(exposure_days, dtype=float)
    lower = np.zeros_like(events, dtype=float)
    positive = events > 0
    lower[positive] = 0.5 * stats.chi2.ppf(alpha / 2.0, 2.0 * events[positive])
    upper = 0.5 * stats.chi2.ppf(1.0 - alpha / 2.0, 2.0 * (events + 1.0))
    lower_rate = np.divide(
        lower,
        exposure_days,
        out=np.full_like(lower, np.nan),
        where=exposure_days > 0,
    )
    upper_rate = np.divide(
        upper,
        exposure_days,
        out=np.full_like(upper, np.nan),
        where=exposure_days > 0,
    )
    return lower_rate, upper_rate


def aggregate_recovery_units(
    data: TidalMarshData,
    block_widths_m: Sequence[float] = (32.0, 64.0, 128.0),
) -> tuple[dict[float, pd.DataFrame], pd.DataFrame, dict[str, int]]:
    """Aggregate censored recovery events by block and inundation interval."""
    site = data.site
    edges = analysis_bin_edges(site)
    n_bins = len(edges) - 1
    height, width = data.inundation.shape

    block_specs: dict[float, dict[str, object]] = {}
    for width_m in block_widths_m:
        block_pixels_float = width_m / site.pixel_size_m
        block_pixels = int(round(block_pixels_float))
        if not np.isclose(block_pixels, block_pixels_float):
            raise ValueError(f"Block width {width_m} m is not pixel aligned")
        n_block_rows = ceil(height / block_pixels)
        n_block_cols = ceil(width / block_pixels)
        size = n_block_rows * n_block_cols * n_bins
        block_specs[float(width_m)] = {
            "block_pixels": block_pixels,
            "n_block_rows": n_block_rows,
            "n_block_cols": n_block_cols,
            "losses": np.zeros(size, dtype=np.int64),
            "recoveries": np.zeros(size, dtype=np.int64),
            "exposure_years": np.zeros(size, dtype=np.float64),
        }

    direct_losses = np.zeros(n_bins, dtype=np.int64)
    direct_recoveries = np.zeros(n_bins, dtype=np.int64)
    direct_exposure_years = np.zeros(n_bins, dtype=np.float64)
    total_events = 0
    total_recoveries = 0

    for batch in iter_recovery_events(data.vegetation, site.years):
        inundation = data.inundation[batch.rows, batch.cols]
        bin_index = _inundation_bins(inundation, edges)
        valid = bin_index >= 0
        if not valid.any():
            continue
        rows = batch.rows[valid]
        cols = batch.cols[valid]
        bin_valid = bin_index[valid]
        duration = batch.duration_years[valid]
        recovered = batch.recovered[valid].astype(np.int64)
        total_events += int(valid.sum())
        total_recoveries += int(recovered.sum())

        direct_losses += np.bincount(bin_valid, minlength=n_bins)
        direct_recoveries += np.bincount(
            bin_valid, weights=recovered, minlength=n_bins
        ).astype(np.int64)
        direct_exposure_years += np.bincount(
            bin_valid, weights=duration, minlength=n_bins
        )

        for spec in block_specs.values():
            block_pixels = int(spec["block_pixels"])
            n_block_cols = int(spec["n_block_cols"])
            block_id = (rows // block_pixels) * n_block_cols + (cols // block_pixels)
            key = block_id * n_bins + bin_valid
            spec["losses"] += np.bincount(
                key, minlength=len(spec["losses"])
            ).astype(np.int64)
            spec["recoveries"] += np.bincount(
                key, weights=recovered, minlength=len(spec["recoveries"])
            ).astype(np.int64)
            spec["exposure_years"] += np.bincount(
                key, weights=duration, minlength=len(spec["exposure_years"])
            )

    direct_exposure_days = direct_exposure_years * 365.0
    direct_rate = np.divide(
        direct_recoveries,
        direct_exposure_days,
        out=np.zeros_like(direct_exposure_days),
        where=direct_exposure_days > 0,
    )
    direct_lower, direct_upper = _poisson_rate_interval(
        direct_recoveries, direct_exposure_days
    )
    direct = pd.DataFrame(
        {
            "site": site.name,
            "inundation_fraction": (edges[:-1] + edges[1:]) / 2.0,
            "inundation_percent": (edges[:-1] + edges[1:]) * 50.0,
            "n_losses": direct_losses,
            "n_recovered": direct_recoveries,
            "n_censored": direct_losses - direct_recoveries,
            "exposure_years": direct_exposure_years,
            "hazard_per_day": direct_rate,
            "hazard_ci_low_per_day": direct_lower,
            "hazard_ci_high_per_day": direct_upper,
        }
    )

    outputs: dict[float, pd.DataFrame] = {}
    for width_m, spec in block_specs.items():
        n_block_cols = int(spec["n_block_cols"])
        block_pixels = int(spec["block_pixels"])
        losses = np.asarray(spec["losses"])
        recoveries = np.asarray(spec["recoveries"])
        exposure_years = np.asarray(spec["exposure_years"])
        occupied = np.flatnonzero(losses > 0)
        block_id = occupied // n_bins
        bin_index = occupied % n_bins
        block_row = block_id // n_block_cols
        block_col = block_id % n_block_cols
        exposure_days = exposure_years[occupied] * 365.0
        rate = recoveries[occupied] / exposure_days
        ci_low, ci_high = _poisson_rate_interval(
            recoveries[occupied], exposure_days
        )
        x0 = block_col * block_pixels
        y0 = block_row * block_pixels
        x1 = np.minimum(x0 + block_pixels, width)
        y1 = np.minimum(y0 + block_pixels, height)

        outputs[width_m] = pd.DataFrame(
            {
                "site": site.name,
                "block_width_m": width_m,
                "block_pixels": block_pixels,
                "block_id": block_id,
                "block_row": block_row,
                "block_col": block_col,
                "row_start": y0,
                "row_stop": y1,
                "col_start": x0,
                "col_stop": x1,
                "center_y_m": (y0 + y1) * site.pixel_size_m / 2.0,
                "center_x_m": (x0 + x1) * site.pixel_size_m / 2.0,
                "inundation_bin": bin_index,
                "inundation_fraction": (edges[bin_index] + edges[bin_index + 1]) / 2.0,
                "inundation_percent": (edges[bin_index] + edges[bin_index + 1]) * 50.0,
                "n_losses": losses[occupied],
                "n_recovered": recoveries[occupied],
                "n_censored": losses[occupied] - recoveries[occupied],
                "censor_fraction": 1.0 - recoveries[occupied] / losses[occupied],
                "exposure_years": exposure_years[occupied],
                "hazard_per_year": recoveries[occupied] / exposure_years[occupied],
                "hazard_per_day": rate,
                "hazard_ci_low_per_day": ci_low,
                "hazard_ci_high_per_day": ci_high,
            }
        )

    counts = {
        "events_in_analysis_gradient": total_events,
        "recoveries_in_analysis_gradient": total_recoveries,
    }
    return outputs, direct, counts


def published_recovery_rates(data: TidalMarshData) -> pd.DataFrame:
    path = (
        data.bundle_root
        / "Figures"
        / "Figure 2"
        / f"{data.site.name}_RecovRate.csv"
    )
    frame = pd.read_csv(
        path,
        header=None,
        names=[
            "inundation_percent",
            "published_hazard_per_day",
            "published_ci_upper_delta",
            "published_ci_lower_delta",
        ],
    )
    return frame


def compare_published_recovery_rates(
    direct: pd.DataFrame,
    published: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    direct_keyed = direct.assign(
        _bin_key=direct["inundation_percent"].round(6)
    )
    published_keyed = published.assign(
        _bin_key=published["inundation_percent"].round(6)
    ).drop(columns="inundation_percent")
    merged = direct_keyed.merge(published_keyed, on="_bin_key", how="inner")
    merged = merged.drop(columns="_bin_key")
    merged["relative_error"] = np.abs(
        merged["hazard_per_day"] - merged["published_hazard_per_day"]
    ) / merged["published_hazard_per_day"]
    regression = stats.linregress(
        merged["inundation_percent"], merged["hazard_per_day"]
    )
    summary = {
        "n_bins": int(len(merged)),
        "slope_per_percent": float(regression.slope),
        "intercept": float(regression.intercept),
        "r_squared": float(regression.rvalue**2),
        "p_value": float(regression.pvalue),
        "max_relative_replication_error": float(merged["relative_error"].max()),
    }
    return merged, summary


def _coarsen_mean(array: np.ndarray, factor: int) -> np.ndarray:
    array = np.asarray(array, dtype=float)
    height = (array.shape[0] // factor) * factor
    width = (array.shape[1] // factor) * factor
    if height == 0 or width == 0:
        return np.empty((0, 0), dtype=float)
    trimmed = array[:height, :width]
    return trimmed.reshape(
        height // factor,
        factor,
        width // factor,
        factor,
    ).mean(axis=(1, 3))


def _snapshot_patch_metrics(mask: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    mask = np.asarray(mask, dtype=bool) & valid
    valid_count = int(valid.sum())
    if valid_count == 0:
        return {
            "coverage": np.nan,
            "component_density": np.nan,
            "largest_component_fraction": np.nan,
            "edge_density": np.nan,
        }

    vegetated = int(mask.sum())
    labels, n_components = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    if n_components:
        sizes = np.bincount(labels.ravel())[1:]
        largest_fraction = float(sizes.max() / vegetated)
    else:
        largest_fraction = 0.0

    horizontal_valid = valid[:, :-1] & valid[:, 1:]
    vertical_valid = valid[:-1, :] & valid[1:, :]
    transitions = int(((mask[:, :-1] != mask[:, 1:]) & horizontal_valid).sum())
    transitions += int(((mask[:-1, :] != mask[1:, :]) & vertical_valid).sum())
    neighbor_pairs = int(horizontal_valid.sum() + vertical_valid.sum())

    return {
        "coverage": vegetated / valid_count,
        "component_density": n_components / valid_count * 10_000.0,
        "largest_component_fraction": largest_fraction,
        "edge_density": transitions / neighbor_pairs if neighbor_pairs else 0.0,
    }


PATCH_STATIC_FEATURES = [
    "coverage_mean",
    "component_density_mean",
    "largest_component_fraction_mean",
    "edge_density_mean",
]

PATCH_DYNAMIC_FEATURES = [
    "turnover_rate",
    "coverage_change_rate",
    "fragmentation_change_rate",
    "edge_change_rate",
    "largest_component_change_rate",
]


def snapshot_patch_context(
    data: TidalMarshData,
    block_width_m: float,
    patch_resolution_m: float = 1.0,
    min_valid_cells: int = 16,
) -> pd.DataFrame:
    """Compute time-indexed patch covariates without future information.

    Each record describes one spatial block in one vegetation snapshot.  The
    static variables use only that snapshot.  Dynamic variables use only the
    transition from the immediately preceding snapshot to the current one and
    are annualized by the actual observation interval.  This table is suitable
    for time-dependent survival covariates; unlike :func:`summarize_patch_context`,
    it never averages over later images.
    """
    site = data.site
    block_pixels_float = block_width_m / site.pixel_size_m
    block_pixels = int(round(block_pixels_float))
    coarse_factor_float = patch_resolution_m / site.pixel_size_m
    coarse_factor = int(round(coarse_factor_float))
    if not np.isclose(block_pixels_float, block_pixels):
        raise ValueError("block_width_m must align with source pixels")
    if not np.isclose(coarse_factor_float, coarse_factor) or coarse_factor < 1:
        raise ValueError("patch_resolution_m must be an integer source-pixel multiple")

    edges = analysis_bin_edges(site)
    analysis_valid = (data.inundation >= edges[0]) & (data.inundation < edges[-1])
    height, width = analysis_valid.shape
    n_block_rows = ceil(height / block_pixels)
    n_block_cols = ceil(width / block_pixels)
    years = np.asarray(site.years, dtype=float)
    records: list[dict[str, float | int | str]] = []

    for block_row in range(n_block_rows):
        y0 = block_row * block_pixels
        y1 = min(y0 + block_pixels, height)
        for block_col in range(n_block_cols):
            x0 = block_col * block_pixels
            x1 = min(x0 + block_pixels, width)
            valid_fraction = _coarsen_mean(
                analysis_valid[y0:y1, x0:x1], coarse_factor
            )
            if valid_fraction.size == 0:
                continue
            valid = valid_fraction >= 0.5
            valid_count = int(valid.sum())
            if valid_count < min_valid_cells:
                continue

            masks: list[np.ndarray] = []
            metrics: list[dict[str, float]] = []
            for vegetation in data.vegetation:
                vegetation_fraction = _coarsen_mean(
                    vegetation[y0:y1, x0:x1], coarse_factor
                )
                mask = vegetation_fraction >= 0.5
                masks.append(mask)
                metrics.append(_snapshot_patch_metrics(mask, valid))

            for snapshot_index, current in enumerate(metrics):
                record: dict[str, float | int | str] = {
                    "site": site.name,
                    "block_width_m": float(block_width_m),
                    "block_id": block_row * n_block_cols + block_col,
                    "block_row": block_row,
                    "block_col": block_col,
                    "snapshot_index": snapshot_index,
                    "snapshot_year": float(years[snapshot_index]),
                    "patch_resolution_m": float(patch_resolution_m),
                    "valid_patch_cells": valid_count,
                    "coverage": float(current["coverage"]),
                    "component_density": float(current["component_density"]),
                    "largest_component_fraction": float(
                        current["largest_component_fraction"]
                    ),
                    "edge_density": float(current["edge_density"]),
                }
                dynamic_names = {
                    "turnover_rate": np.nan,
                    "coverage_change_rate": np.nan,
                    "fragmentation_change_rate": np.nan,
                    "edge_change_rate": np.nan,
                    "largest_component_change_rate": np.nan,
                }
                if snapshot_index > 0:
                    previous = metrics[snapshot_index - 1]
                    dt = years[snapshot_index] - years[snapshot_index - 1]
                    dynamic_names = {
                        "turnover_rate": float(
                            ((masks[snapshot_index] != masks[snapshot_index - 1]) & valid).sum()
                        )
                        / valid_count
                        / dt,
                        "coverage_change_rate": abs(
                            current["coverage"] - previous["coverage"]
                        )
                        / dt,
                        "fragmentation_change_rate": abs(
                            current["component_density"]
                            - previous["component_density"]
                        )
                        / dt,
                        "edge_change_rate": abs(
                            current["edge_density"] - previous["edge_density"]
                        )
                        / dt,
                        "largest_component_change_rate": abs(
                            current["largest_component_fraction"]
                            - previous["largest_component_fraction"]
                        )
                        / dt,
                    }
                record.update(dynamic_names)
                records.append(record)

    return pd.DataFrame.from_records(records)


def summarize_patch_context(
    data: TidalMarshData,
    block_width_m: float,
    patch_resolution_m: float = 1.0,
    min_valid_cells: int = 16,
) -> pd.DataFrame:
    """Describe vegetation-patch organization in each recovery block.

    Metrics are computed at a fixed patch resolution and summarized over the
    complete image sequence.  They are explanatory covariates, not recovery
    labels and not early-warning alarms.
    """
    site = data.site
    block_pixels_float = block_width_m / site.pixel_size_m
    block_pixels = int(round(block_pixels_float))
    coarse_factor_float = patch_resolution_m / site.pixel_size_m
    coarse_factor = int(round(coarse_factor_float))
    if not np.isclose(block_pixels_float, block_pixels):
        raise ValueError("block_width_m must align with source pixels")
    if not np.isclose(coarse_factor_float, coarse_factor) or coarse_factor < 1:
        raise ValueError("patch_resolution_m must be an integer source-pixel multiple")

    edges = analysis_bin_edges(site)
    analysis_valid = (data.inundation >= edges[0]) & (data.inundation < edges[-1])
    height, width = analysis_valid.shape
    n_block_rows = ceil(height / block_pixels)
    n_block_cols = ceil(width / block_pixels)
    years = np.asarray(site.years, dtype=float)
    records: list[dict[str, float | int | str]] = []

    for block_row in range(n_block_rows):
        y0 = block_row * block_pixels
        y1 = min(y0 + block_pixels, height)
        for block_col in range(n_block_cols):
            x0 = block_col * block_pixels
            x1 = min(x0 + block_pixels, width)
            valid_fraction = _coarsen_mean(
                analysis_valid[y0:y1, x0:x1], coarse_factor
            )
            if valid_fraction.size == 0:
                continue
            valid = valid_fraction >= 0.5
            if int(valid.sum()) < min_valid_cells:
                continue

            masks: list[np.ndarray] = []
            snapshot_metrics: list[dict[str, float]] = []
            for snapshot in data.vegetation:
                vegetation_fraction = _coarsen_mean(
                    snapshot[y0:y1, x0:x1], coarse_factor
                )
                mask = vegetation_fraction >= 0.5
                masks.append(mask)
                snapshot_metrics.append(_snapshot_patch_metrics(mask, valid))

            snapshot = pd.DataFrame(snapshot_metrics)
            dynamic = {
                "turnover_rate": [],
                "coverage_change_rate": [],
                "fragmentation_change_rate": [],
                "edge_change_rate": [],
                "largest_component_change_rate": [],
            }
            for index in range(1, len(masks)):
                dt = years[index] - years[index - 1]
                dynamic["turnover_rate"].append(
                    float(((masks[index] != masks[index - 1]) & valid).sum())
                    / float(valid.sum())
                    / dt
                )
                dynamic["coverage_change_rate"].append(
                    abs(snapshot.loc[index, "coverage"] - snapshot.loc[index - 1, "coverage"])
                    / dt
                )
                dynamic["fragmentation_change_rate"].append(
                    abs(
                        snapshot.loc[index, "component_density"]
                        - snapshot.loc[index - 1, "component_density"]
                    )
                    / dt
                )
                dynamic["edge_change_rate"].append(
                    abs(snapshot.loc[index, "edge_density"] - snapshot.loc[index - 1, "edge_density"])
                    / dt
                )
                dynamic["largest_component_change_rate"].append(
                    abs(
                        snapshot.loc[index, "largest_component_fraction"]
                        - snapshot.loc[index - 1, "largest_component_fraction"]
                    )
                    / dt
                )

            record: dict[str, float | int | str] = {
                "site": site.name,
                "block_width_m": float(block_width_m),
                "block_pixels": block_pixels,
                "block_id": block_row * n_block_cols + block_col,
                "block_row": block_row,
                "block_col": block_col,
                "row_start": y0,
                "row_stop": y1,
                "col_start": x0,
                "col_stop": x1,
                "center_y_m": (y0 + y1) * site.pixel_size_m / 2.0,
                "center_x_m": (x0 + x1) * site.pixel_size_m / 2.0,
                "patch_resolution_m": patch_resolution_m,
                "valid_patch_cells": int(valid.sum()),
                "coverage_mean": float(snapshot["coverage"].mean()),
                "component_density_mean": float(snapshot["component_density"].mean()),
                "largest_component_fraction_mean": float(
                    snapshot["largest_component_fraction"].mean()
                ),
                "edge_density_mean": float(snapshot["edge_density"].mean()),
            }
            for name, values in dynamic.items():
                record[name] = float(np.mean(values)) if values else np.nan
            records.append(record)

    return pd.DataFrame.from_records(records)


def filter_supported_units(
    units: pd.DataFrame,
    min_losses: int = 50,
    min_recoveries: int = 10,
) -> pd.DataFrame:
    supported = units.loc[
        (units["n_losses"] >= min_losses)
        & (units["n_recovered"] >= min_recoveries)
        & np.isfinite(units["hazard_per_year"])
        & (units["hazard_per_year"] > 0)
    ].copy()
    return supported.reset_index(drop=True)
