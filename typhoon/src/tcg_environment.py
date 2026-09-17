from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.tcg_case_control import (
    _exact_pair_permutation_p,
    _linear_slope,
    _pair_bootstrap_auc_ci,
    lopo_predict,
)
from src.tcg_pilot import ROOT, _haversine_km


DEFAULT_CONFIG = ROOT / "configs" / "pilot_v0_3_environment.json"

PATH_KEYS = (
    "case_control_manifest_path",
    "cloud_frame_features_path",
    "era5_request_dir",
    "era5_request_index_path",
    "era5_raw_dir",
    "era5_download_log_path",
    "environment_frame_features_path",
    "merged_frame_features_path",
    "environment_case_features_path",
    "environment_metrics_path",
    "environment_predictions_path",
    "environment_report_path",
    "environment_status_report_path",
)

VARIABLE_ALIASES = {
    "u": ("u", "u_component_of_wind"),
    "v": ("v", "v_component_of_wind"),
    "r": ("r", "relative_humidity"),
    "vo": ("vo", "vorticity", "relative_vorticity"),
}

CLOUD_ORGANIZATION_FRAME_FEATURES = (
    "cold_cloud_fraction",
    "very_cold_fraction",
    "radial_concentration",
    "azimuthal_symmetry",
    "largest_patch_fraction",
    "cold_centroid_offset_km",
)


def load_environment_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in PATH_KEYS:
        cfg[key] = str((ROOT / cfg[key]).resolve())
    return cfg


def _relative_to_root(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _month_request_records(manifest: pd.DataFrame, cfg: dict) -> list[dict]:
    """Build auditable CDS requests, one case-month at a time.

    Month grouping avoids requesting invalid or extraneous date cross-products
    when a 72-hour case window crosses a calendar-month boundary.
    """
    required = {
        "sid",
        "name",
        "outcome",
        "observation_time",
        "center_lat",
        "center_lon",
    }
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"manifest missing columns: {sorted(missing)}")
    work = manifest.copy()
    work["observation_time"] = pd.to_datetime(
        work["observation_time"], utc=True, errors="raise"
    )
    records: list[dict] = []
    margin = float(cfg["area_margin_deg"])
    raw_dir = Path(cfg["era5_raw_dir"])
    request_dir = Path(cfg["era5_request_dir"])
    for sid, case in work.groupby("sid", sort=True):
        lons = case["center_lon"].to_numpy(dtype=float)
        if np.ptp(lons) > 180.0:
            raise ValueError(
                f"case {sid} crosses the dateline; split/unwrap longitudes first"
            )
        north = min(
            90.0,
            math.ceil((case["center_lat"].max() + margin) * 4) / 4,
        )
        south = max(
            -90.0,
            math.floor((case["center_lat"].min() - margin) * 4) / 4,
        )
        west = max(
            -180.0,
            math.floor((case["center_lon"].min() - margin) * 4) / 4,
        )
        east = min(
            180.0,
            math.ceil((case["center_lon"].max() + margin) * 4) / 4,
        )
        groups = case.groupby(
            [
                case["observation_time"].dt.year,
                case["observation_time"].dt.month,
            ],
            sort=True,
        )
        for (year, month), block in groups:
            stamp = f"{int(year):04d}{int(month):02d}"
            request_id = f"{sid}_{stamp}"
            target = raw_dir / f"{request_id}.nc"
            request_path = request_dir / f"{request_id}.json"
            request = {
                "product_type": ["reanalysis"],
                "variable": list(cfg["variables"]),
                "pressure_level": [
                    str(x) for x in cfg["pressure_levels_hpa"]
                ],
                "year": [f"{int(year):04d}"],
                "month": [f"{int(month):02d}"],
                "day": sorted(
                    block["observation_time"].dt.strftime("%d").unique()
                ),
                "time": sorted(
                    block["observation_time"].dt.strftime("%H:%M").unique()
                ),
                "data_format": "netcdf",
                "download_format": "unarchived",
                "area": [north, west, south, east],
            }
            payload = {
                "dataset": cfg["cds_dataset"],
                "request": request,
                "target": _relative_to_root(target),
                "case_observation_times": block["observation_time"]
                .dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                .tolist(),
            }
            records.append(
                {
                    "request_id": request_id,
                    "sid": str(sid),
                    "name": str(case["name"].iloc[0]),
                    "outcome": int(case["outcome"].iloc[0]),
                    "year": int(year),
                    "month": int(month),
                    "n_case_times": int(len(block)),
                    "north": north,
                    "west": west,
                    "south": south,
                    "east": east,
                    "request_json_path": _relative_to_root(request_path),
                    "target_path": _relative_to_root(target),
                    "payload": payload,
                }
            )
    return records


def _credentials_status() -> str:
    if (Path.home() / ".cdsapirc").exists():
        return "configured_file"
    if os.environ.get("CDSAPI_KEY") or os.environ.get("CDSAPI_RC"):
        return "configured_environment"
    return "missing"


def plan_environment_requests(cfg: dict) -> pd.DataFrame:
    manifest_path = Path(cfg["case_control_manifest_path"])
    if not manifest_path.exists():
        raise FileNotFoundError(f"case-control manifest not found: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    records = _month_request_records(manifest, cfg)
    request_dir = Path(cfg["era5_request_dir"])
    request_dir.mkdir(parents=True, exist_ok=True)
    index_rows = []
    for record in records:
        request_path = ROOT / record["request_json_path"]
        request_path.parent.mkdir(parents=True, exist_ok=True)
        request_path.write_text(
            json.dumps(record["payload"], indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        index_rows.append(
            {key: value for key, value in record.items() if key != "payload"}
        )
    index = pd.DataFrame(index_rows).sort_values(["sid", "year", "month"])
    index_path = Path(cfg["era5_request_index_path"])
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(index_path, index=False)
    status_path = Path(cfg["environment_status_report_path"])
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(
        "# v0.3 ERA5 environment-baseline status\n\n"
        f"- Case trajectories: {manifest['sid'].nunique()}\n"
        f"- Target observation times: {len(manifest)}\n"
        f"- Case-month CDS requests: {len(index)}\n"
        f"- Variables: {', '.join(cfg['variables'])}\n"
        f"- Pressure levels: "
        f"{', '.join(map(str, cfg['pressure_levels_hpa']))} hPa\n"
        f"- Analysis radii: "
        f"{', '.join(map(str, cfg['analysis_radii_km']))} km\n"
        f"- CDS credential status: {_credentials_status()}\n\n"
        "The request plan is complete. Downloading requires a personal CDS API "
        "token and prior acceptance of the ERA5 pressure-level dataset terms. "
        "No synthetic environment values are substituted when credentials are "
        "unavailable.\n",
        encoding="utf-8",
    )
    print(
        f"ERA5 plan: {len(index)} requests for "
        f"{manifest['sid'].nunique()} cases / {len(manifest)} target times "
        f"-> {index_path}"
    )
    print(f"CDS credentials: {_credentials_status()}")
    return index


def _sha256(path: Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def download_environment_data(
    cfg: dict, limit: int | None = None
) -> pd.DataFrame:
    try:
        import cdsapi
    except ImportError as exc:
        raise RuntimeError(
            "cdsapi is not installed; run: "
            "python -m pip install -r requirements.txt"
        ) from exc
    if _credentials_status() == "missing":
        raise RuntimeError(
            "CDS credentials are missing. Accept the ERA5 pressure-level terms "
            "and configure ~/.cdsapirc as documented by Copernicus."
        )
    index_path = Path(cfg["era5_request_index_path"])
    if not index_path.exists():
        plan_environment_requests(cfg)
    index = pd.read_csv(index_path)
    if limit is not None:
        index = index.head(limit)
    client = cdsapi.Client()
    log_rows = []
    for position, row in enumerate(index.itertuples(index=False), start=1):
        request_path = ROOT / row.request_json_path
        payload = json.loads(request_path.read_text(encoding="utf-8"))
        target = ROOT / payload["target"]
        target.parent.mkdir(parents=True, exist_ok=True)
        status = (
            "cached"
            if target.exists() and target.stat().st_size > 0
            else "downloaded"
        )
        error = ""
        try:
            if status == "downloaded":
                client.retrieve(
                    payload["dataset"], payload["request"], str(target)
                )
            if not target.exists() or target.stat().st_size == 0:
                raise RuntimeError("CDS request returned no usable file")
            size_bytes = target.stat().st_size
            checksum = _sha256(target)
        except Exception as exc:  # retain an auditable partial log
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
            size_bytes = target.stat().st_size if target.exists() else 0
            checksum = ""
        log_rows.append(
            {
                "request_id": row.request_id,
                "sid": row.sid,
                "status": status,
                "bytes": size_bytes,
                "sha256": checksum,
                "target_path": payload["target"],
                "error": error,
            }
        )
        print(
            f"ERA5 {position}/{len(index)} {row.request_id}: {status}",
            flush=True,
        )
    log = pd.DataFrame(log_rows)
    log_path = Path(cfg["era5_download_log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(log_path, index=False)
    failures = int(log["status"].eq("failed").sum())
    if failures:
        raise RuntimeError(
            f"{failures}/{len(log)} ERA5 requests failed; see {log_path}"
        )
    return log


def _weighted_radius_mean(
    values: np.ndarray,
    lons: np.ndarray,
    lats: np.ndarray,
    center_lon: float,
    center_lat: float,
    radius_km: float,
) -> float:
    values = np.asarray(values, dtype=float)
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    distance = _haversine_km(lons, lats, center_lon, center_lat)
    valid = (
        np.isfinite(values)
        & np.isfinite(distance)
        & (distance <= radius_km)
    )
    if valid.sum() < 4:
        raise ValueError(
            f"fewer than four ERA5 grid cells inside {radius_km:g} km"
        )
    weights = np.clip(np.cos(np.deg2rad(lats)), 0.05, None)
    return float(np.average(values[valid], weights=weights[valid]))


def compute_environment_features(
    fields: dict[str, np.ndarray],
    pressure_levels_hpa: Iterable[int | float],
    lons: np.ndarray,
    lats: np.ndarray,
    center_lon: float,
    center_lat: float,
    radii_km: Iterable[int | float],
) -> dict[str, float]:
    """Compute the three prespecified environmental state variables.

    Arrays in ``fields`` have shape ``(pressure, latitude, longitude)``.
    Relative humidity is expected in percent and vorticity in s^-1.
    """
    levels = np.asarray(list(pressure_levels_hpa), dtype=float)
    required_levels = {200.0, 600.0, 700.0, 850.0}
    if not required_levels.issubset(set(levels.tolist())):
        raise ValueError(
            f"required pressure levels missing: {sorted(required_levels)}"
        )
    for name in ("u", "v", "r", "vo"):
        if name not in fields:
            raise ValueError(f"ERA5 field missing: {name}")
        if np.asarray(fields[name]).shape[0] != len(levels):
            raise ValueError(f"{name} pressure dimension does not match levels")

    def level(field: str, pressure: float) -> np.ndarray:
        index = int(np.flatnonzero(np.isclose(levels, pressure))[0])
        return np.asarray(fields[field][index], dtype=float)

    result: dict[str, float] = {}
    for radius in radii_km:
        radius = float(radius)
        suffix = f"{int(radius)}km"
        vo850 = _weighted_radius_mean(
            level("vo", 850),
            lons,
            lats,
            center_lon,
            center_lat,
            radius,
        )
        rh_means = [
            _weighted_radius_mean(
                level("r", pressure),
                lons,
                lats,
                center_lon,
                center_lat,
                radius,
            )
            for pressure in (600, 700, 850)
        ]
        u200 = _weighted_radius_mean(
            level("u", 200), lons, lats, center_lon, center_lat, radius
        )
        u850 = _weighted_radius_mean(
            level("u", 850), lons, lats, center_lon, center_lat, radius
        )
        v200 = _weighted_radius_mean(
            level("v", 200), lons, lats, center_lon, center_lat, radius
        )
        v850 = _weighted_radius_mean(
            level("v", 850), lons, lats, center_lon, center_lat, radius
        )
        result[f"vorticity_850_mean_s1__{suffix}"] = vo850
        result[f"rh_600_850_mean_pct__{suffix}"] = float(np.mean(rh_means))
        result[f"shear_200_850_ms__{suffix}"] = float(
            math.hypot(u200 - u850, v200 - v850)
        )
    return result


def _find_name(
    candidates: Iterable[str], available: Iterable[str], kind: str
) -> str:
    available_set = set(available)
    for name in candidates:
        if name in available_set:
            return name
    raise ValueError(
        f"could not find {kind}; available={sorted(available_set)}"
    )


def _canonical_slice(dataset, observation_time: pd.Timestamp, cfg: dict):
    time_name = _find_name(
        ("valid_time", "time"), dataset.coords, "time coordinate"
    )
    level_name = _find_name(
        ("pressure_level", "isobaricInhPa", "level"),
        dataset.coords,
        "pressure coordinate",
    )
    lat_name = _find_name(
        ("latitude", "lat"), dataset.coords, "latitude coordinate"
    )
    lon_name = _find_name(
        ("longitude", "lon"), dataset.coords, "longitude coordinate"
    )
    time_dim = dataset[time_name].dims[0]
    level_dim = dataset[level_name].dims[0]
    lat_dim = dataset[lat_name].dims[0]
    lon_dim = dataset[lon_name].dims[0]
    times = pd.to_datetime(dataset[time_name].values, utc=True)
    offsets = np.abs((times - observation_time).total_seconds()) / 60.0
    time_index = int(np.argmin(offsets))
    if float(offsets[time_index]) > float(cfg["max_time_offset_minutes"]):
        raise ValueError(
            f"nearest ERA5 time is {offsets[time_index]:.1f} minutes from target"
        )
    selected = dataset.isel({time_dim: time_index})
    lat_values = np.asarray(selected[lat_name].values, dtype=float)
    lon_values = np.asarray(selected[lon_name].values, dtype=float)
    lons, lats = np.meshgrid(lon_values, lat_values)
    levels = np.asarray(selected[level_name].values, dtype=float)
    order = [
        int(np.argmin(np.abs(levels - target)))
        for target in cfg["pressure_levels_hpa"]
    ]
    for index, target in zip(
        order, cfg["pressure_levels_hpa"], strict=True
    ):
        if abs(levels[index] - target) > 1e-6:
            raise ValueError(
                f"ERA5 pressure levels do not match request: {levels.tolist()}"
            )
    fields = {}
    for canonical, aliases in VARIABLE_ALIASES.items():
        variable_name = _find_name(
            aliases, selected.data_vars, f"{canonical} variable"
        )
        data = selected[variable_name].isel({level_dim: order})
        extra_dims = [
            dim
            for dim in data.dims
            if dim not in (level_dim, lat_dim, lon_dim)
        ]
        for dim in extra_dims:
            data = data.isel({dim: 0})
        fields[canonical] = np.asarray(
            data.transpose(level_dim, lat_dim, lon_dim).values,
            dtype=float,
        )
    return (
        fields,
        np.asarray(cfg["pressure_levels_hpa"], dtype=float),
        lons,
        lats,
        times[time_index],
    )


def _open_case_dataset(paths: list[Path]):
    import xarray as xr

    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"ERA5 case files missing: {missing}")
    datasets = [xr.open_dataset(path) for path in paths]
    if len(datasets) == 1:
        return datasets[0], datasets
    return xr.combine_by_coords(
        datasets, combine_attrs="override"
    ), datasets


def process_environment_data(cfg: dict) -> pd.DataFrame:
    manifest = pd.read_csv(
        cfg["case_control_manifest_path"],
        parse_dates=["observation_time", "endpoint_time"],
    )
    index = pd.read_csv(cfg["era5_request_index_path"])
    rows = []
    for sid, case in manifest.groupby("sid", sort=True):
        targets = [
            ROOT / path
            for path in index[
                index["sid"].astype(str).eq(str(sid))
            ]["target_path"]
        ]
        dataset, handles = _open_case_dataset(targets)
        try:
            for row in case.sort_values("observation_time").itertuples(
                index=False
            ):
                fields, levels, lons, lats, matched_time = _canonical_slice(
                    dataset, row.observation_time, cfg
                )
                features = compute_environment_features(
                    fields,
                    levels,
                    lons,
                    lats,
                    float(row.center_lon),
                    float(row.center_lat),
                    cfg["analysis_radii_km"],
                )
                rows.append(
                    {
                        "sid": str(row.sid),
                        "name": row.name,
                        "outcome": int(row.outcome),
                        "pair_id": int(row.pair_id),
                        "endpoint_time": row.endpoint_time,
                        "observation_time": row.observation_time,
                        "hours_to_endpoint": int(row.hours_to_endpoint),
                        "center_lat": float(row.center_lat),
                        "center_lon": float(row.center_lon),
                        "era5_matched_time": matched_time,
                        **features,
                    }
                )
        finally:
            dataset.close()
            for handle in handles:
                handle.close()
        print(f"ERA5 processed {sid}: {len(case)} frames", flush=True)
    environment = pd.DataFrame(rows).sort_values(
        ["sid", "observation_time"]
    )
    expected = len(manifest)
    if len(environment) != expected:
        raise ValueError(
            f"incomplete ERA5 processing: {len(environment)}/{expected}"
        )
    feature_columns = [
        col
        for col in environment.columns
        if col.startswith(("vorticity_", "rh_", "shear_"))
    ]
    if environment[feature_columns].isna().any().any():
        raise ValueError("environment feature table contains missing values")
    path = Path(cfg["environment_frame_features_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    environment.to_csv(path, index=False)

    cloud = pd.read_csv(cfg["cloud_frame_features_path"])
    merged = cloud.merge(
        environment[["sid", "observation_time", *feature_columns]],
        on=["sid", "observation_time"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(cloud):
        raise ValueError(
            f"cloud/environment merge is incomplete: "
            f"{len(merged)}/{len(cloud)}"
        )
    merged.to_csv(cfg["merged_frame_features_path"], index=False)
    return environment


def _environment_feature_names(cfg: dict) -> list[str]:
    names = []
    for radius in cfg["analysis_radii_km"]:
        suffix = f"{int(radius)}km"
        names.extend(
            [
                f"vorticity_850_mean_s1__{suffix}",
                f"rh_600_850_mean_pct__{suffix}",
                f"shear_200_850_ms__{suffix}",
            ]
        )
    return names


def aggregate_environment_cases(
    frames: pd.DataFrame, cfg: dict
) -> pd.DataFrame:
    env_features = _environment_feature_names(cfg)
    records = []
    for sid, case in frames.groupby("sid", sort=True):
        case = case.sort_values("hours_to_endpoint")
        early = case[case["hours_to_endpoint"].between(-72, -48)]
        late = case[case["hours_to_endpoint"].between(-24, -3)]
        endpoint = pd.to_datetime(
            case["endpoint_time"].iloc[0], utc=True
        )
        day = float(endpoint.dayofyear)
        record = {
            "sid": str(sid),
            "name": case["name"].iloc[0],
            "outcome": int(case["outcome"].iloc[0]),
            "pair_id": int(case["pair_id"].iloc[0]),
            "endpoint_time": endpoint,
            "endpoint_type": (
                case["endpoint_type"].iloc[0]
                if "endpoint_type" in case.columns
                else "unspecified"
            ),
            "median_lat": float(case["center_lat"].median()),
            "median_lon": float(case["center_lon"].median()),
            "endpoint_day_sin": math.sin(2 * math.pi * day / 365.25),
            "endpoint_day_cos": math.cos(2 * math.pi * day / 365.25),
            "n_frames": int(len(case)),
        }
        hours = case["hours_to_endpoint"].to_numpy(dtype=float)
        features = [
            *env_features,
            *CLOUD_ORGANIZATION_FRAME_FEATURES,
        ]
        for feature in features:
            values = case[feature].to_numpy(dtype=float)
            early_median = float(early[feature].median())
            late_median = float(late[feature].median())
            record[f"{feature}__early_median"] = early_median
            record[f"{feature}__late_median"] = late_median
            record[f"{feature}__late_minus_early"] = (
                late_median - early_median
            )
            record[f"{feature}__slope_per_day"] = _linear_slope(
                hours, values
            )
        records.append(record)
    return pd.DataFrame(records).sort_values(
        ["pair_id", "outcome"], ascending=[True, False]
    )


def environment_model_features(cfg: dict) -> dict[str, list[str]]:
    suffix = f"{int(cfg['primary_radius_km'])}km"
    environment = [
        f"vorticity_850_mean_s1__{suffix}__late_median",
        f"rh_600_850_mean_pct__{suffix}__late_median",
        f"shear_200_850_ms__{suffix}__late_median",
    ]
    cloud = [
        f"{name}__late_median"
        for name in ("cold_cloud_fraction", "very_cold_fraction")
    ]
    organization = [
        f"{name}__late_median"
        for name in (
            "radial_concentration",
            "azimuthal_symmetry",
            "largest_patch_fraction",
            "cold_centroid_offset_km",
        )
    ]
    return {
        "metadata_only": [
            "median_lat",
            "median_lon",
            "endpoint_day_sin",
            "endpoint_day_cos",
        ],
        "environment_only": environment,
        "cloud_area_only": cloud,
        "environment_plus_cloud": [*environment, *cloud],
        "environment_plus_cloud_plus_organization": [
            *environment,
            *cloud,
            *organization,
        ],
    }


def evaluate_environment_models(cases: pd.DataFrame, cfg: dict):
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        roc_auc_score,
    )

    metric_rows = []
    prediction_frames = []
    for name, features in environment_model_features(cfg).items():
        prediction = lopo_predict(cases, features, name, cfg)
        auc = float(
            roc_auc_score(
                prediction["outcome"], prediction["probability"]
            )
        )
        ci_low, ci_high = _pair_bootstrap_auc_ci(prediction, cfg)
        p_value, n_permutations = _exact_pair_permutation_p(
            cases, features, auc, cfg
        )
        metric_rows.append(
            {
                "model": name,
                "n_cases": len(cases),
                "n_pairs": cases["pair_id"].nunique(),
                "n_features": len(features),
                "roc_auc": auc,
                "pair_bootstrap_ci_low": ci_low,
                "pair_bootstrap_ci_high": ci_high,
                "pr_auc": float(
                    average_precision_score(
                        prediction["outcome"], prediction["probability"]
                    )
                ),
                "brier_score": float(
                    brier_score_loss(
                        prediction["outcome"], prediction["probability"]
                    )
                ),
                "exact_within_pair_permutation_p_one_sided": p_value,
                "n_exact_permutations": n_permutations,
            }
        )
        prediction_frames.append(prediction)
    return (
        pd.DataFrame(metric_rows),
        pd.concat(prediction_frames, ignore_index=True),
    )


def write_environment_report(
    cases: pd.DataFrame, metrics: pd.DataFrame, cfg: dict
) -> Path:
    by_name = metrics.set_index("model")
    env_auc = by_name.loc["environment_only", "roc_auc"]
    cloud_auc = by_name.loc["environment_plus_cloud", "roc_auc"]
    full_auc = by_name.loc[
        "environment_plus_cloud_plus_organization", "roc_auc"
    ]
    text = f"""# v0.3 ERA5 environment-baseline report

## Scope

This analysis adds three prespecified ERA5 environmental variables at a
{cfg['primary_radius_km']}-km radius: 850-hPa relative vorticity, mean
600/700/850-hPa relative humidity, and 200--850-hPa vertical wind shear.
Validation remains leave-one-matched-pair-out over only {len(cases)} tracks.

## Results

{metrics.to_markdown(index=False)}

The environment-only ROC-AUC is {env_auc:.3f}. Adding cold-cloud area changes it
to {cloud_auc:.3f}; adding the four organization descriptors changes it to
{full_auc:.3f}. These differences are diagnostic only: five matched pairs cannot
establish stable out-of-sample improvement, and the exact one-sided paired
permutation p-value cannot be smaller than 0.0625.

## Decision rule

Do not advance to DynCL, SINDy, Jacobian eigenvalues, or a critical-slowing claim
from this pilot alone. First improve environmental/geographic matching and expand
the objectively tracked nondeveloping cloud-cluster sample. Representation
learning is justified only if organization information repeatedly improves the
environment-only baseline under grouped year/case validation.
"""
    path = Path(cfg["environment_report_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def analyze_environment(cfg: dict):
    frames = pd.read_csv(
        cfg["merged_frame_features_path"],
        parse_dates=["observation_time", "endpoint_time"],
    )
    cases = aggregate_environment_cases(frames, cfg)
    metrics, predictions = evaluate_environment_models(cases, cfg)
    cases.to_csv(cfg["environment_case_features_path"], index=False)
    metrics.to_csv(cfg["environment_metrics_path"], index=False)
    predictions.to_csv(cfg["environment_predictions_path"], index=False)
    report = write_environment_report(cases, metrics, cfg)
    print(metrics.to_string(index=False))
    print(f"report -> {report}")
    return metrics, predictions


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="TC-genesis ERA5 environment pilot v0.3"
    )
    parser.add_argument(
        "command",
        choices=["plan", "download", "process", "analyze", "run-all"],
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    cfg = load_environment_config(args.config)
    if args.command == "plan":
        plan_environment_requests(cfg)
    elif args.command == "download":
        download_environment_data(cfg, limit=args.limit)
    elif args.command == "process":
        process_environment_data(cfg)
    elif args.command == "analyze":
        analyze_environment(cfg)
    else:
        plan_environment_requests(cfg)
        download_environment_data(cfg, limit=args.limit)
        process_environment_data(cfg)
        analyze_environment(cfg)


if __name__ == "__main__":
    main()
