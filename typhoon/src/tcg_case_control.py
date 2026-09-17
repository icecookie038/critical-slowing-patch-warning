from __future__ import annotations

import argparse
import itertools
import json
import math
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from src.tcg_pilot import (
    ROOT,
    STATIC_FEATURES,
    _download_one,
    _decompress_bz2,
    ensure_output_dirs,
    first_tropical_storm_time,
    interpolate_track,
    process_data,
    read_ibtracs,
)


DEFAULT_CONFIG = ROOT / "configs" / "pilot_v0_2_case_control.json"
REPORT_PATH = ROOT / "outputs" / "reports" / "pilot_v0_2_case_control_report.md"
CASE_TABLE_PATH = ROOT / "outputs" / "tables" / "case_control_case_features.csv"
METRICS_PATH = ROOT / "outputs" / "tables" / "case_control_metrics.csv"
PREDICTIONS_PATH = ROOT / "outputs" / "tables" / "case_control_predictions.csv"
COMPARISON_PATH = ROOT / "outputs" / "tables" / "case_control_feature_comparison.csv"
PAIR_PATH = ROOT / "outputs" / "tables" / "case_control_pairs.csv"
SENSITIVITY_PATH = ROOT / "outputs" / "tables" / "case_control_sensitivity.csv"


def load_case_control_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    path = Path(path)
    cfg = json.loads(path.read_text(encoding="utf-8"))
    for key in (
        "ibtracs_path",
        "raw_himawari_dir",
        "manifest_path",
        "features_path",
        "download_log_path",
        "error_log_path",
    ):
        cfg[key] = str((ROOT / cfg[key]).resolve())
    return cfg


def _track_name(track: pd.DataFrame) -> str:
    names = track["NAME"].dropna().astype(str)
    if names.empty or names.iloc[0].strip() == "":
        return "UNNAMED"
    return names.iloc[0]


def _nondeveloped_endpoint(track: pd.DataFrame, cfg: dict) -> pd.Timestamp:
    track = track.sort_values("ISO_TIME")
    start = track["ISO_TIME"].dropna().min()
    end = track["ISO_TIME"].dropna().max()
    endpoint = start + pd.Timedelta(
        hours=int(cfg["nondeveloped_endpoint_hours_from_start"])
    )
    status = set(track["USA_STATUS"].dropna().astype(str))
    max_wind = track["USA_WIND"].max()
    forbidden = {"TS", "TY", "ST", "TC", "HU", "HR"}
    if status.intersection(forbidden):
        raise ValueError(f"nondeveloped case contains storm status: {status}")
    if not np.isfinite(max_wind) or max_wind >= 34.0:
        raise ValueError(f"nondeveloped case must have finite USA_WIND <34 kt: {max_wind}")
    if endpoint > end:
        raise ValueError(
            f"nondeveloped track is shorter than endpoint: {start} to {end}"
        )
    return endpoint


def _circular_day_difference(day_a: float, day_b: float) -> float:
    delta = abs(day_a - day_b)
    return min(delta, 365.25 - delta)


def _himawari_files(
    obs_time: pd.Timestamp, raw_root: Path
) -> tuple[list[str], list[str]]:
    """Return archive keys and local paths for both Himawari HSD layouts.

    The NOAA archive stores the configured 2019 Band-13 data as one S0101
    full-disk file. The sampled 2020+ data use ten Sxx10 north-south segments.
    """
    t = obs_time.tz_convert("UTC")
    stamp = t.strftime("%Y%m%d_%H%M")
    prefix = f"AHI-L1b-FLDK/{t:%Y/%m/%d/%H%M}"
    if t.year < 2020:
        names = [f"HS_H08_{stamp}_B13_FLDK_R20_S0101.DAT.bz2"]
    else:
        names = [
            f"HS_H08_{stamp}_B13_FLDK_R20_S{segment:02d}10.DAT.bz2"
            for segment in range(1, 11)
        ]
    keys = [f"{prefix}/{name}" for name in names]
    local_paths = [str(raw_root / key.split("/", 1)[1]) for key in keys]
    return keys, local_paths


def _assign_matched_pairs(case_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    from scipy.optimize import linear_sum_assignment

    developed = case_summary[case_summary["outcome"].eq(1)].reset_index(drop=True)
    negative = case_summary[case_summary["outcome"].eq(0)].reset_index(drop=True)
    if len(developed) != len(negative):
        raise ValueError("pilot matching requires equal developed/nondeveloped case counts")
    cost = np.empty((len(developed), len(negative)), dtype=float)
    for i, dev in developed.iterrows():
        for j, neg in negative.iterrows():
            season = _circular_day_difference(dev["endpoint_doy"], neg["endpoint_doy"]) / 45.0
            latitude = abs(dev["median_lat"] - neg["median_lat"]) / 10.0
            longitude = abs(dev["median_lon"] - neg["median_lon"]) / 20.0
            cost[i, j] = season + latitude + longitude
    rows, cols = linear_sum_assignment(cost)
    pair_records = []
    pair_map: dict[str, int] = {}
    for pair_id, (i, j) in enumerate(zip(rows, cols, strict=True), start=1):
        dev = developed.iloc[i]
        neg = negative.iloc[j]
        pair_map[str(dev["sid"])] = pair_id
        pair_map[str(neg["sid"])] = pair_id
        pair_records.append(
            {
                "pair_id": pair_id,
                "developed_sid": dev["sid"],
                "developed_name": dev["name"],
                "nondeveloped_sid": neg["sid"],
                "nondeveloped_name": neg["name"],
                "season_day_difference": _circular_day_difference(
                    dev["endpoint_doy"], neg["endpoint_doy"]
                ),
                "median_lat_difference_deg": abs(dev["median_lat"] - neg["median_lat"]),
                "median_lon_difference_deg": abs(dev["median_lon"] - neg["median_lon"]),
                "matching_cost": cost[i, j],
            }
        )
    result = case_summary.copy()
    result["pair_id"] = result["sid"].astype(str).map(pair_map).astype(int)
    return result, pd.DataFrame(pair_records).sort_values("pair_id")


def build_case_control_manifest(cfg: dict) -> pd.DataFrame:
    ensure_output_dirs(cfg)
    tracks = read_ibtracs(cfg)
    rows: list[dict] = []
    case_rows: list[dict] = []
    raw_root = Path(cfg["raw_himawari_dir"])
    specifications = [
        *[(sid, 1) for sid in cfg["developed_sids"]],
        *[(sid, 0) for sid in cfg["nondeveloped_sids"]],
    ]
    for sid, outcome in specifications:
        track = tracks[tracks["SID"].eq(sid)].copy().sort_values("ISO_TIME")
        if track.empty:
            raise ValueError(f"IBTrACS SID not found: {sid}")
        endpoint = (
            first_tropical_storm_time(track)
            if outcome == 1
            else _nondeveloped_endpoint(track, cfg)
        )
        endpoint_type = "first_US_TS_or_34kt" if outcome == 1 else "72h_after_first_track"
        times = pd.date_range(
            endpoint - pd.Timedelta(hours=cfg["history_hours"]),
            endpoint - pd.Timedelta(hours=cfg["cadence_hours"]),
            freq=f"{cfg['cadence_hours']}h",
        )
        lats, lons = interpolate_track(track, times)
        name = _track_name(track)
        case_rows.append(
            {
                "sid": sid,
                "name": name,
                "outcome": outcome,
                "endpoint_time": endpoint,
                "endpoint_doy": float(endpoint.dayofyear),
                "median_lat": float(np.median(lats)),
                "median_lon": float(np.median(lons)),
                "track_duration_hours": float(
                    (track["ISO_TIME"].max() - track["ISO_TIME"].min()).total_seconds()
                    / 3600.0
                ),
                "max_usa_wind_kt": float(track["USA_WIND"].max()),
            }
        )
        for obs_time, lat, lon in zip(times, lats, lons, strict=True):
            keys, local_paths = _himawari_files(obs_time, raw_root)
            rows.append(
                {
                    "sid": sid,
                    "name": name,
                    "outcome": outcome,
                    "endpoint_time": endpoint.isoformat(),
                    "endpoint_type": endpoint_type,
                    "observation_time": obs_time.isoformat(),
                    "hours_to_endpoint": int(
                        (obs_time - endpoint).total_seconds() / 3600
                    ),
                    "center_lat": float(lat),
                    "center_lon": float(lon),
                    "segment_count": len(keys),
                    "s3_key": keys[0],
                    "local_path": local_paths[0],
                    "s3_keys_json": json.dumps(keys),
                    "local_paths_json": json.dumps(local_paths),
                }
            )
    case_summary, pairs = _assign_matched_pairs(pd.DataFrame(case_rows))
    pair_map = case_summary.set_index("sid")["pair_id"].to_dict()
    manifest = pd.DataFrame(rows)
    manifest["pair_id"] = manifest["sid"].map(pair_map).astype(int)
    manifest = manifest.sort_values(["pair_id", "outcome", "observation_time"])
    manifest.to_csv(cfg["manifest_path"], index=False)
    pairs.to_csv(PAIR_PATH, index=False)
    print(
        f"case-control manifest: {len(manifest)} frames, "
        f"{manifest['sid'].nunique()} disturbances -> {cfg['manifest_path']}"
    )
    print(pairs.to_string(index=False))
    return manifest


def _expand_download_manifest(manifest: pd.DataFrame) -> list[dict]:
    records = []
    for row in manifest.to_dict("records"):
        keys = json.loads(row["s3_keys_json"])
        local_paths = json.loads(row["local_paths_json"])
        for segment_index, (key, local_path) in enumerate(
            zip(keys, local_paths, strict=True), start=1
        ):
            records.append(
                {
                    "sid": row["sid"],
                    "name": row["name"],
                    "outcome": row["outcome"],
                    "pair_id": row["pair_id"],
                    "observation_time": row["observation_time"],
                    "hours_to_endpoint": row["hours_to_endpoint"],
                    "segment_index": segment_index,
                    "segment_count": len(keys),
                    "s3_key": key,
                    "local_path": local_path,
                }
            )
    return records


def download_case_control_data(
    cfg: dict, manifest: pd.DataFrame | None = None
) -> pd.DataFrame:
    ensure_output_dirs(cfg)
    if manifest is None:
        manifest = pd.read_csv(cfg["manifest_path"])
    records = _expand_download_manifest(manifest)
    results = []
    with ThreadPoolExecutor(max_workers=int(cfg["download_workers"])) as pool:
        futures = [pool.submit(_download_one, row, cfg) for row in records]
        for index, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if index % 25 == 0 or index == len(futures):
                ok = sum(item["download_status"] != "failed" for item in results)
                print(
                    f"downloaded/cached segments {ok}/{index}; "
                    f"total planned {len(futures)}",
                    flush=True,
                )
    log = pd.DataFrame(results).sort_values(
        ["sid", "observation_time", "segment_index"]
    )
    log.to_csv(cfg["download_log_path"], index=False)
    failures = log[log["download_status"].eq("failed")]
    print(
        f"download complete: {len(log) - len(failures)}/{len(log)} segments usable; "
        f"failures={len(failures)}"
    )
    return log


def seed_developed_features(cfg: dict) -> int:
    old_path = ROOT / "outputs" / "tables" / "pilot_features.csv"
    target = Path(cfg["features_path"])
    manifest_path = Path(cfg["manifest_path"])
    if target.exists() or not old_path.exists() or not manifest_path.exists():
        return 0
    old = pd.read_csv(old_path)
    manifest = pd.read_csv(manifest_path)
    developed = manifest[manifest["outcome"].eq(1)].copy()
    metadata = set(manifest.columns)
    feature_columns = [
        col
        for col in old.columns
        if col not in metadata
        and col
        not in {
            "genesis_time",
            "hours_to_genesis",
        }
    ]
    reused = developed.merge(
        old[["sid", "observation_time", *feature_columns]],
        on=["sid", "observation_time"],
        how="inner",
        validate="one_to_one",
    )
    if len(reused) != len(developed):
        raise ValueError(
            f"could not reuse every developed frame: {len(reused)}/{len(developed)}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    reused.to_csv(target, index=False)
    print(f"reused {len(reused)} developed feature rows from v0.1")
    return len(reused)


def _linear_slope(hours: np.ndarray, values: np.ndarray) -> float:
    mask = np.isfinite(hours) & np.isfinite(values)
    if mask.sum() < 3:
        return np.nan
    x = hours[mask] / 24.0
    y = values[mask]
    return float(np.polyfit(x, y, 1)[0])


def aggregate_cases(frame_features: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    for sid, case in frame_features.groupby("sid", sort=True):
        case = case.sort_values("hours_to_endpoint")
        late = case[case["hours_to_endpoint"].between(-24, -3)]
        record = {
            "sid": sid,
            "name": case["name"].iloc[0],
            "outcome": int(case["outcome"].iloc[0]),
            "pair_id": int(case["pair_id"].iloc[0]),
            "endpoint_time": case["endpoint_time"].iloc[0],
            "endpoint_type": case["endpoint_type"].iloc[0],
            "median_lat": float(case["center_lat"].median()),
            "median_lon": float(case["center_lon"].median()),
            "n_frames": int(len(case)),
            "valid_pixel_fraction_min": float(case["valid_pixel_fraction"].min()),
        }
        endpoint = pd.to_datetime(record["endpoint_time"], utc=True)
        day_of_year = float(endpoint.dayofyear)
        record["endpoint_day_sin"] = math.sin(2.0 * math.pi * day_of_year / 365.25)
        record["endpoint_day_cos"] = math.cos(2.0 * math.pi * day_of_year / 365.25)
        hours = case["hours_to_endpoint"].to_numpy(dtype=float)
        early = case[case["hours_to_endpoint"].between(-72, -48)]
        for feature in STATIC_FEATURES:
            values = case[feature].to_numpy(dtype=float)
            early_median = float(early[feature].median())
            late_median = float(late[feature].median())
            record[f"{feature}__early_median"] = early_median
            record[f"{feature}__late_median"] = late_median
            record[f"{feature}__late_minus_early"] = late_median - early_median
            record[f"{feature}__slope_per_day"] = _linear_slope(hours, values)
        records.append(record)
    return pd.DataFrame(records).sort_values(["pair_id", "outcome"], ascending=[True, False])


MODEL_FEATURES = {
    "cold_area": [
        "cold_cloud_fraction__late_median",
        "very_cold_fraction__late_median",
    ],
    "organization_static": [
        "radial_concentration__late_median",
        "azimuthal_symmetry__late_median",
        "largest_patch_fraction__late_median",
        "cold_centroid_offset_km__late_median",
    ],
    "organization_trend": [
        "radial_concentration__slope_per_day",
        "azimuthal_symmetry__slope_per_day",
        "largest_patch_fraction__slope_per_day",
        "cold_centroid_offset_km__slope_per_day",
    ],
    "cloud_plus_organization": [
        "cold_cloud_fraction__late_median",
        "very_cold_fraction__late_median",
        "radial_concentration__late_median",
        "azimuthal_symmetry__late_median",
        "largest_patch_fraction__late_median",
        "cold_centroid_offset_km__late_median",
    ],
}


def _make_model(cfg: dict):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    C=float(cfg["ridge_c"]),
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=int(cfg["random_seed"]),
                ),
            ),
        ]
    )


def lopo_predict(
    cases: pd.DataFrame, features: list[str], model_name: str, cfg: dict
) -> pd.DataFrame:
    predictions = []
    for pair_id in sorted(cases["pair_id"].unique()):
        train = cases[~cases["pair_id"].eq(pair_id)]
        test = cases[cases["pair_id"].eq(pair_id)]
        model = _make_model(cfg)
        model.fit(train[features], train["outcome"])
        fold = test[
            ["sid", "name", "outcome", "pair_id", "endpoint_time", "endpoint_type"]
        ].copy()
        fold["probability"] = model.predict_proba(test[features])[:, 1]
        fold["model"] = model_name
        predictions.append(fold)
    return pd.concat(predictions, ignore_index=True)


def _exact_pair_permutation_p(
    cases: pd.DataFrame, features: list[str], observed_auc: float, cfg: dict
) -> tuple[float, int]:
    from sklearn.metrics import roc_auc_score

    pair_ids = sorted(cases["pair_id"].unique())
    null_auc = []
    for flips in itertools.product([0, 1], repeat=len(pair_ids)):
        permuted = cases.copy()
        for pair_id, flip in zip(pair_ids, flips, strict=True):
            if flip:
                mask = permuted["pair_id"].eq(pair_id)
                permuted.loc[mask, "outcome"] = 1 - permuted.loc[mask, "outcome"]
        pred = lopo_predict(permuted, features, "permuted", cfg)
        null_auc.append(roc_auc_score(pred["outcome"], pred["probability"]))
    p_value = float(np.mean(np.asarray(null_auc) >= observed_auc - 1e-12))
    return p_value, len(null_auc)


def _pair_bootstrap_auc_ci(predictions: pd.DataFrame, cfg: dict) -> tuple[float, float]:
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(int(cfg["random_seed"]))
    pair_ids = predictions["pair_id"].unique()
    aucs = []
    for _ in range(int(cfg["bootstrap_iterations"])):
        sampled = rng.choice(pair_ids, size=len(pair_ids), replace=True)
        chunks = [predictions[predictions["pair_id"].eq(pair_id)] for pair_id in sampled]
        boot = pd.concat(chunks, ignore_index=True)
        aucs.append(roc_auc_score(boot["outcome"], boot["probability"]))
    return tuple(float(x) for x in np.quantile(aucs, [0.025, 0.975]))


def evaluate_models(cases: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    metrics = []
    predictions = []
    for model_name, features in MODEL_FEATURES.items():
        pred = lopo_predict(cases, features, model_name, cfg)
        auc = float(roc_auc_score(pred["outcome"], pred["probability"]))
        ci_low, ci_high = _pair_bootstrap_auc_ci(pred, cfg)
        permutation_p, n_permutations = _exact_pair_permutation_p(
            cases, features, auc, cfg
        )
        metrics.append(
            {
                "model": model_name,
                "n_cases": len(cases),
                "n_pairs": cases["pair_id"].nunique(),
                "n_features": len(features),
                "roc_auc": auc,
                "pair_bootstrap_ci_low": ci_low,
                "pair_bootstrap_ci_high": ci_high,
                "pr_auc": float(
                    average_precision_score(pred["outcome"], pred["probability"])
                ),
                "brier_score": float(
                    brier_score_loss(pred["outcome"], pred["probability"])
                ),
                "exact_within_pair_permutation_p_one_sided": permutation_p,
                "n_exact_permutations": n_permutations,
            }
        )
        predictions.append(pred)
    return pd.DataFrame(metrics), pd.concat(predictions, ignore_index=True)


def evaluate_sensitivity(cases: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    scenarios = [
        (
            "all_cases",
            "metadata_only",
            ["median_lat", "median_lon", "endpoint_day_sin", "endpoint_day_cos"],
            cases,
        ),
        (
            "all_cases",
            "early_cloud",
            [
                "cold_cloud_fraction__early_median",
                "very_cold_fraction__early_median",
            ],
            cases,
        ),
        (
            "all_cases",
            "cloud_change",
            [
                "cold_cloud_fraction__late_minus_early",
                "very_cold_fraction__late_minus_early",
            ],
            cases,
        ),
        (
            "all_cases",
            "organization_change",
            [
                "radial_concentration__late_minus_early",
                "azimuthal_symmetry__late_minus_early",
                "largest_patch_fraction__late_minus_early",
                "cold_centroid_offset_km__late_minus_early",
            ],
            cases,
        ),
        (
            "drop_worst_matched_pair",
            "cold_area",
            MODEL_FEATURES["cold_area"],
            cases[cases["pair_id"].ne(5)].copy(),
        ),
        (
            "drop_worst_matched_pair",
            "cloud_plus_organization",
            MODEL_FEATURES["cloud_plus_organization"],
            cases[cases["pair_id"].ne(5)].copy(),
        ),
    ]
    rows = []
    for scenario, model_name, features, subset in scenarios:
        pred = lopo_predict(subset, features, model_name, cfg)
        auc = float(roc_auc_score(pred["outcome"], pred["probability"]))
        ci_low, ci_high = _pair_bootstrap_auc_ci(pred, cfg)
        permutation_p, n_permutations = _exact_pair_permutation_p(
            subset, features, auc, cfg
        )
        rows.append(
            {
                "scenario": scenario,
                "model": model_name,
                "n_cases": len(subset),
                "n_pairs": subset["pair_id"].nunique(),
                "n_features": len(features),
                "roc_auc": auc,
                "pair_bootstrap_ci_low": ci_low,
                "pair_bootstrap_ci_high": ci_high,
                "pr_auc": float(
                    average_precision_score(pred["outcome"], pred["probability"])
                ),
                "brier_score": float(
                    brier_score_loss(pred["outcome"], pred["probability"])
                ),
                "exact_within_pair_permutation_p_one_sided": permutation_p,
                "n_exact_permutations": n_permutations,
            }
        )
    return pd.DataFrame(rows)


def compare_features(cases: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import mannwhitneyu

    rows = []
    feature_names = sorted(
        set(itertools.chain.from_iterable(MODEL_FEATURES.values()))
    )
    for feature in feature_names:
        dev = cases[cases["outcome"].eq(1)][feature].dropna().to_numpy(dtype=float)
        neg = cases[cases["outcome"].eq(0)][feature].dropna().to_numpy(dtype=float)
        differences = np.subtract.outer(dev, neg)
        cliffs_delta = float(
            ((differences > 0).sum() - (differences < 0).sum()) / differences.size
        )
        p_value = float(
            mannwhitneyu(dev, neg, alternative="two-sided", method="exact").pvalue
        )
        rows.append(
            {
                "feature": feature,
                "developed_median": float(np.median(dev)),
                "nondeveloped_median": float(np.median(neg)),
                "cliffs_delta": cliffs_delta,
                "mann_whitney_exact_p_unadjusted": p_value,
            }
        )
    return pd.DataFrame(rows).sort_values(
        "cliffs_delta", key=lambda x: x.abs(), ascending=False
    )


def _snapshot_array(row: pd.Series, cfg: dict) -> np.ndarray:
    from satpy import Scene

    if row.get("local_paths_json") and not pd.isna(row.get("local_paths_json")):
        sources = [Path(value) for value in json.loads(row["local_paths_json"])]
    else:
        sources = [Path(row["local_path"])]
    with tempfile.TemporaryDirectory(prefix="tcg_cc_snapshot_") as tmp:
        raw_paths = []
        for source in sources:
            raw_path = Path(tmp) / source.name.removesuffix(".bz2")
            _decompress_bz2(source, raw_path)
            raw_paths.append(raw_path)
        scene = Scene(filenames=[str(path) for path in raw_paths], reader="ahi_hsd")
        scene.load(["B13"])
        half = float(cfg["crop_half_width_deg"])
        lon0, lat0 = float(row["center_lon"]), float(row["center_lat"])
        cropped = scene.crop(
            ll_bbox=(lon0 - half, lat0 - half, lon0 + half, lat0 + half)
        )
        return np.asarray(cropped["B13"].compute(), dtype=np.float32)


def make_figures(
    frames: pd.DataFrame,
    cases: pd.DataFrame,
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    cfg: dict,
) -> None:
    import matplotlib.pyplot as plt

    figure_dir = ROOT / "outputs" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    selected = [
        ("cold_cloud_fraction", "Cold-cloud fraction"),
        ("radial_concentration", "Radial concentration"),
        ("azimuthal_symmetry", "Azimuthal symmetry"),
        ("largest_patch_fraction", "Largest-patch fraction"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    colors = {0: "#277da1", 1: "#d1495b"}
    labels = {0: "Nondeveloped", 1: "Developed"}
    for ax, (feature, title) in zip(axes.ravel(), selected, strict=True):
        for outcome, group in frames.groupby("outcome"):
            matrix = group.pivot(
                index="sid", columns="hours_to_endpoint", values=feature
            ).sort_index(axis=1)
            x = matrix.columns.to_numpy(dtype=float)
            for values in matrix.to_numpy(dtype=float):
                ax.plot(x, values, color=colors[outcome], alpha=0.18, lw=1)
            ax.plot(
                x,
                np.nanmedian(matrix.to_numpy(dtype=float), axis=0),
                color=colors[outcome],
                lw=2.5,
                label=labels[outcome],
            )
        ax.axvspan(-24, 0, color="#f4a261", alpha=0.08)
        ax.set_title(title)
        ax.set_xlabel("Hours to case endpoint")
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "cc_01_group_trajectories.png", dpi=180)
    plt.close(fig)

    plot_features = MODEL_FEATURES["cloud_plus_organization"]
    matrix = cases.set_index(["outcome", "name"])[plot_features]
    z = (matrix - matrix.mean()) / matrix.std(ddof=0).replace(0, np.nan)
    fig, ax = plt.subplots(figsize=(11, 6))
    image = ax.imshow(z.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-2, vmax=2)
    ax.set_yticks(
        np.arange(len(z)),
        [f"{'D' if idx[0] else 'N'} | {idx[1]}" for idx in z.index],
    )
    ax.set_xticks(
        np.arange(len(plot_features)),
        [name.replace("__late_median", "") for name in plot_features],
        rotation=30,
        ha="right",
    )
    fig.colorbar(image, ax=ax, label="Across-case z score")
    fig.tight_layout()
    fig.savefig(figure_dir / "cc_02_case_feature_heatmap.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(metrics))
    y = metrics["roc_auc"].to_numpy()
    low = y - metrics["pair_bootstrap_ci_low"].to_numpy()
    high = metrics["pair_bootstrap_ci_high"].to_numpy() - y
    ax.errorbar(x, y, yerr=[low, high], fmt="o", capsize=5, ms=7)
    ax.axhline(0.5, color="black", ls="--", lw=1)
    ax.set_xticks(x, metrics["model"], rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Leave-one-pair-out ROC-AUC")
    fig.tight_layout()
    fig.savefig(figure_dir / "cc_03_model_auc.png", dpi=180)
    plt.close(fig)

    best_model = metrics.sort_values("roc_auc", ascending=False).iloc[0]["model"]
    best = predictions[predictions["model"].eq(best_model)].sort_values(
        ["pair_id", "outcome"]
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    for pair_id, pair in best.groupby("pair_id"):
        pair = pair.sort_values("outcome")
        ax.plot(
            [pair_id - 0.08, pair_id + 0.08],
            pair["probability"],
            color="#666666",
            lw=1.3,
        )
        ax.scatter(
            pair_id - 0.08,
            pair[pair["outcome"].eq(0)]["probability"],
            color=colors[0],
            label="Nondeveloped" if pair_id == 1 else None,
        )
        ax.scatter(
            pair_id + 0.08,
            pair[pair["outcome"].eq(1)]["probability"],
            color=colors[1],
            label="Developed" if pair_id == 1 else None,
        )
    ax.axhline(0.5, color="black", ls="--", lw=1)
    ax.set_xticks(sorted(best["pair_id"].unique()))
    ax.set_xlabel("Matched pair held out from training")
    ax.set_ylabel(f"Out-of-fold genesis probability ({best_model})")
    ax.set_ylim(-0.03, 1.03)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "cc_04_best_model_probabilities.png", dpi=180)
    plt.close(fig)

    snapshots = frames[frames["hours_to_endpoint"].eq(-3)].sort_values(
        ["outcome", "pair_id"], ascending=[False, True]
    )
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    image_handle = None
    for ax, (_, row) in zip(axes.ravel(), snapshots.iterrows(), strict=True):
        bt = _snapshot_array(row, cfg)
        image_handle = ax.imshow(bt, cmap="turbo", vmin=190, vmax=290, origin="upper")
        prefix = "D" if row["outcome"] == 1 else "N"
        ax.set_title(f"{prefix}: {row['name']} (pair {row['pair_id']})")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.subplots_adjust(right=0.90, hspace=0.18, wspace=0.08)
    color_ax = fig.add_axes([0.92, 0.16, 0.015, 0.68])
    fig.colorbar(image_handle, cax=color_ax, label="Band 13 brightness temperature (K)")
    fig.savefig(figure_dir / "cc_00_endpoint_snapshots.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(
    frames: pd.DataFrame,
    cases: pd.DataFrame,
    metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    sensitivity: pd.DataFrame,
    pairs: pd.DataFrame,
    cfg: dict,
) -> Path:
    best = metrics.sort_values("roc_auc", ascending=False).iloc[0]
    metric_lines = []
    for row in metrics.itertuples():
        metric_lines.append(
            f"| {row.model} | {row.n_features} | {row.roc_auc:.3f} | "
            f"{row.pair_bootstrap_ci_low:.3f}–{row.pair_bootstrap_ci_high:.3f} | "
            f"{row.pr_auc:.3f} | {row.brier_score:.3f} | "
            f"{row.exact_within_pair_permutation_p_one_sided:.4f} |"
        )
    comparison_lines = []
    for row in comparison.head(10).itertuples():
        comparison_lines.append(
            f"| {row.feature} | {row.developed_median:.4f} | "
            f"{row.nondeveloped_median:.4f} | {row.cliffs_delta:+.2f} | "
            f"{row.mann_whitney_exact_p_unadjusted:.4f} |"
        )
    pair_lines = []
    for row in pairs.itertuples():
        pair_lines.append(
            f"| {row.pair_id} | {row.developed_name} | {row.nondeveloped_name} | "
            f"{row.season_day_difference:.0f} | {row.median_lat_difference_deg:.1f} | "
            f"{row.median_lon_difference_deg:.1f} |"
        )
    sensitivity_lines = []
    for row in sensitivity.itertuples():
        sensitivity_lines.append(
            f"| {row.scenario} | {row.model} | {row.n_cases} | {row.roc_auc:.3f} | "
            f"{row.pair_bootstrap_ci_low:.3f}–{row.pair_bootstrap_ci_high:.3f} | "
            f"{row.exact_within_pair_permutation_p_one_sided:.4f} |"
        )
    sensitivity_lookup = sensitivity.set_index(["scenario", "model"])
    metadata_auc = sensitivity_lookup.loc[("all_cases", "metadata_only"), "roc_auc"]
    early_cloud_auc = sensitivity_lookup.loc[("all_cases", "early_cloud"), "roc_auc"]
    cloud_change_auc = sensitivity_lookup.loc[("all_cases", "cloud_change"), "roc_auc"]
    drop_pair_auc = sensitivity_lookup.loc[
        ("drop_worst_matched_pair", "cloud_plus_organization"), "roc_auc"
    ]
    dev_names = ", ".join(cases[cases["outcome"].eq(1)]["name"].astype(str))
    neg_names = ", ".join(cases[cases["outcome"].eq(0)]["name"].astype(str))
    all_valid = frames["valid_pixel_fraction"].min()
    text = f"""# 台风生成动力学 Pilot v0.2：发展—未发展扰动对照

## 结论摘要

本次测试把 v0.1 的“发展案例内部阶段判别”推进为真正的“发展扰动 vs 未发展扰动”对照。共处理 {len(frames)} 帧 Himawari-8 Band 13 红外亮温，覆盖 {cases['outcome'].sum()} 个发展扰动和 {(cases['outcome'] == 0).sum()} 个未发展热带低压；每条轨迹包含 72 h、3 h 间隔的 24 帧，最小有效像元比例为 {all_valid:.3f}。

在按匹配对留一验证中，最高 ROC-AUC 为 `{best['model']}` 的 {best['roc_auc']:.3f}，配对 bootstrap 95% 区间为 {best['pair_bootstrap_ci_low']:.3f}–{best['pair_bootstrap_ci_high']:.3f}，精确配对置换单侧 p={best['exact_within_pair_permutation_p_one_sided']:.4f}。组合模型的固定留出概率恰好完全排序，因此配对重采样区间退化为 1.000–1.000；这不能反映重新抽取案例和重新训练的不确定性。由于只有 5 对样本，精确置换检验的分辨率也很低；结果只能用于判断下一轮数据和指标应怎样扩展，不能作为论文中的稳定预测结论。

## 案例定义

- 发展案例：{dev_names}。终点为首次 `USA_STATUS=TS` 或 `USA_WIND>=34 kt`。
- 未发展案例：{neg_names}。要求 IBTrACS 全生命周期无 TS/TY/ST 等风暴状态、`USA_WIND<34 kt`、持续追踪至少 72 h；终点统一为首次记录后 72 h。
- 两类都使用终点前 72 h 至前 3 h，避免图像帧随机拆分。
- 卫星区域为随中心移动的 600 km 半径；冷云阈值 235 K，强冷云阈值 210 K。

IBTrACS 的美国机构风速为 1 min 平均；其列文档将 `TD` 定义为低于 34 kt、`TS` 定义为 34 kt 及以上。本测试沿用同一机构口径，避免混合不同风速平均时长。数据来自 [NOAA IBTrACS v04r01 技术文档](https://www.ncei.noaa.gov/sites/default/files/2025-04/IBTrACS_version4r01_Technical_Details.pdf)、[IBTrACS 列文档](https://www.ncei.noaa.gov/sites/default/files/2025-09/IBTrACS_v04r01_column_documentation.pdf)和 [NOAA/JMA Himawari-8/9 开放数据](https://registry.opendata.aws/noaa-himawari/)，访问日期为 {cfg['source_access_date']}。

## 匹配结果

匹配只用于保证每个测试折同时留出一个发展和一个未发展案例；匹配成本由年内日期、窗口中位纬度和经度组成。

| 对 | 发展 | 未发展 | 年内日期差（天） | 纬度差（°） | 经度差（°） |
|---:|---|---|---:|---:|---:|
{chr(10).join(pair_lines)}

这些负样本更多位于南海和西北太平洋西部，而发展案例中有较多开放洋面案例。匹配后的经纬度差仍然较大，因此当前模型可能学习到区域/季节差异；在没有 ERA5 环境变量和更多开放洋面未发展云团之前，不能把差异全部解释为“对流组织导致生成”。

## 轨迹级验证结果

每个案例先汇总成轨迹级特征：终点前 24 h 的中位数，或完整 72 h 的线性趋势。随后用带强 L2 正则的逻辑回归进行 5 折留一匹配对验证；每折训练 8 个案例、测试 2 个案例。

| 特征组 | 特征数 | ROC-AUC | 配对 bootstrap 95% CI | PR-AUC | Brier | 精确配对置换 p（单侧） |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(metric_lines)}

精确置换检验枚举了每一对内部交换标签的全部 32 种情况。它比把 240 帧当作独立样本更保守，也更符合当前样本结构。

这里的 bootstrap 只重采样已经得到的 5 对留出预测，不包含重新选择扰动、重新匹配和重新拟合表示的变异；尤其在所有正样本概率都高于负样本时，会得到退化的 1.000–1.000 区间，不能把它解释为模型性能已被精确确定。

## 混杂与敏感性诊断

| 情景 | 模型 | 案例数 | ROC-AUC | 配对 bootstrap 95% CI | 精确配对置换 p（单侧） |
|---|---|---:|---:|---:|---:|
{chr(10).join(sensitivity_lines)}

诊断结果必须和主结果一起解读：

1. 仅使用窗口中位经纬度和年内日期的 `metadata_only` AUC 已达到 {metadata_auc:.3f}，说明发展与未发展案例的区域/季节匹配不足，存在强混杂。
2. 终点前 72–48 h 的 `early_cloud` AUC 只有 {early_cloud_auc:.3f}，而冷云“晚期中位数减早期中位数”的 `cloud_change` AUC 为 {cloud_change_auc:.3f}。因此晚期冷云的高分离度不是简单复制了早期差异，而主要来自两组在 72 h 内的反向演化：发展组冷云维持或增加，未发展组总体减弱。
3. 删除纬度匹配最差的第 5 对后，组合模型 AUC 仍为 {drop_pair_auc:.3f}，说明主结果不完全由该单对驱动；但此时只剩 8 个案例，统计不确定性更大。
4. `organization_trend` 和 `organization_change` 没有表现出稳定优势。当前最清楚的候选信号是冷云覆盖及其变化，而不是复杂斑块组织动力学。

这种“增强 vs 衰减”的分离仍可能受到负样本终点定义影响：未发展案例终点是首次记录后 72 h，部分案例此时已接近消散；它不能被直接解释为临界慢化，也不能证明斑块指标具有独立机制价值。

## 单指标方向检查

以下 p 值均未做多重比较校正，只用于观察效应方向。Cliff's delta 为正表示发展组更高，为负表示未发展组更高。

| 指标 | 发展组中位数 | 未发展组中位数 | Cliff's delta | Mann–Whitney 精确 p |
|---|---:|---:|---:|---:|
{chr(10).join(comparison_lines)}

## 目前能支持什么

1. 发展和未发展热带扰动可以在同一套真实卫星处理链上比较。
2. 验证单位已经从图像帧提升到完整扰动轨迹，避免相邻帧泄漏。
3. 可以定量检查冷云面积、空间组织和组织趋势是否具有增量。

## 目前不能支持什么

1. 只有 10 条轨迹，任何 AUC 都高度不稳定；当前结果不是可发表的预测性能。
2. 未发展案例来自 IBTrACS 已被追踪的 TD/DB，只覆盖“已进入最佳路径档案但未达到 TS”的一小类负样本，不能代表全部热带云团。
3. 区域匹配仍弱，且没有 ERA5 风切变、湿度和低层涡度，无法排除环境混杂。
4. 负样本终点是首次记录后 72 h，而发展终点是首次 TS，二者不是完全同质的物理事件。
5. 尚未涉及 DynCL、SINDy、Jacobian 或临界慢化检验。

## 结果驱动的下一步

1. 先用 TAMS/PyFLEXTRKR 从 Himawari 中追踪 30–50 个开放洋面持续云团，补足真正的未发展负样本；不要继续只依赖 IBTrACS 未发展 TD。
2. 为现有 10 个案例接入 ERA5 的 850 hPa 涡度、600–850 hPa 湿度和 200–850 hPa 风切变，检验云组织相对于环境变量的增量。
3. 扩大到至少 30–50 个发展案例和 60–100 个未发展案例后，按年份或扰动分组验证。
4. 只有当“环境 + 组织”稳定优于“仅环境”时，再训练动力学对比学习变量 C(t)。
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(text, encoding="utf-8")
    return REPORT_PATH


def analyze_case_control(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features_path = Path(cfg["features_path"])
    if not features_path.exists():
        raise FileNotFoundError(f"run process first: {features_path}")
    frames = pd.read_csv(
        features_path, parse_dates=["observation_time", "endpoint_time"]
    )
    expected = len(cfg["developed_sids"]) + len(cfg["nondeveloped_sids"])
    counts = frames.groupby("sid").size()
    if len(counts) != expected or not counts.eq(
        int(cfg["history_hours"] / cfg["cadence_hours"])
    ).all():
        raise ValueError(f"incomplete case-control features:\n{counts}")
    cases = aggregate_cases(frames)
    metrics, predictions = evaluate_models(cases, cfg)
    sensitivity = evaluate_sensitivity(cases, cfg)
    comparison = compare_features(cases)
    pairs = pd.read_csv(PAIR_PATH)
    CASE_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    cases.to_csv(CASE_TABLE_PATH, index=False)
    metrics.to_csv(METRICS_PATH, index=False)
    predictions.to_csv(PREDICTIONS_PATH, index=False)
    comparison.to_csv(COMPARISON_PATH, index=False)
    sensitivity.to_csv(SENSITIVITY_PATH, index=False)
    make_figures(frames, cases, metrics, predictions, cfg)
    report = write_report(
        frames, cases, metrics, comparison, sensitivity, pairs, cfg
    )
    print(metrics.to_string(index=False))
    print(f"report -> {report}")
    return metrics, predictions, comparison


def package_case_control() -> Path:
    package = ROOT / "tcg_dynamics_pilot_v0_2_case_control.zip"
    members = [
        ROOT / "configs" / "pilot_v0_2_case_control.json",
        ROOT / "run_case_control.py",
        ROOT / "src" / "tcg_case_control.py",
        ROOT / "src" / "tcg_pilot.py",
        ROOT / "requirements.txt",
        ROOT / "README.md",
        ROOT / "data" / "interim" / "case_control_manifest.csv",
        *sorted((ROOT / "outputs" / "tables").glob("case_control_*.csv")),
        *sorted((ROOT / "outputs" / "figures").glob("cc_*.png")),
        REPORT_PATH,
    ]
    with tempfile.TemporaryDirectory(prefix="tcg_cc_package_") as tmp:
        stage = Path(tmp) / "tcg_dynamics_pilot_v0_2_case_control"
        for source in members:
            if not source.exists():
                continue
            relative = source.relative_to(ROOT)
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        archive_base = str(package.with_suffix(""))
        shutil.make_archive(archive_base, "zip", root_dir=stage.parent, base_dir=stage.name)
    print(f"package -> {package}")
    return package


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="TC-genesis case-control pilot v0.2")
    parser.add_argument(
        "command",
        choices=["manifest", "download", "process", "analyze", "package", "run-all"],
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    cfg = load_case_control_config(args.config)
    if args.command == "manifest":
        build_case_control_manifest(cfg)
    elif args.command == "download":
        download_case_control_data(cfg)
    elif args.command == "process":
        seed_developed_features(cfg)
        process_data(cfg, limit=args.limit)
    elif args.command == "analyze":
        analyze_case_control(cfg)
    elif args.command == "package":
        package_case_control()
    else:
        manifest = build_case_control_manifest(cfg)
        download_case_control_data(cfg, manifest)
        seed_developed_features(cfg)
        process_data(cfg, limit=args.limit)
        analyze_case_control(cfg)
        package_case_control()


if __name__ == "__main__":
    main()
