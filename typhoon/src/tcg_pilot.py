from __future__ import annotations

import argparse
import bz2
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "pilot_v0_1.json"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
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


def ensure_output_dirs(cfg: dict) -> None:
    paths = [
        cfg["raw_himawari_dir"],
        cfg["manifest_path"],
        cfg["features_path"],
        cfg["download_log_path"],
        cfg["error_log_path"],
        ROOT / "outputs" / "figures",
        ROOT / "outputs" / "reports",
    ]
    for item in paths:
        p = Path(item)
        (p if p.suffix == "" else p.parent).mkdir(parents=True, exist_ok=True)


def read_ibtracs(cfg: dict) -> pd.DataFrame:
    columns = [
        "SID",
        "SEASON",
        "NAME",
        "ISO_TIME",
        "LAT",
        "LON",
        "USA_STATUS",
        "USA_WIND",
        "WMO_WIND",
        "TRACK_TYPE",
    ]
    df = pd.read_csv(
        cfg["ibtracs_path"], skiprows=[1], usecols=columns, low_memory=False
    )
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce", utc=True)
    for col in ["LAT", "LON", "USA_WIND", "WMO_WIND"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[df["TRACK_TYPE"].fillna("main").eq("main")].copy()


def first_tropical_storm_time(track: pd.DataFrame) -> pd.Timestamp:
    """Use the first JTWC tropical-storm classification / >=34 kt report.

    This operational threshold is explicit and reproducible. It is a pilot event
    definition, not a claim that all physical genesis occurs at exactly 34 kt.
    """
    track = track.sort_values("ISO_TIME")
    candidates = track[
        track["USA_STATUS"].eq("TS") | track["USA_WIND"].ge(34.0)
    ]
    if candidates.empty:
        raise ValueError("no USA tropical-storm event (TS or >=34 kt) found")
    return candidates.iloc[0]["ISO_TIME"]


def interpolate_track(track: pd.DataFrame, times: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    valid = track.dropna(subset=["ISO_TIME", "LAT", "LON"]).sort_values("ISO_TIME")
    if valid.empty:
        raise ValueError("track has no valid time/position records")
    xp = valid["ISO_TIME"].astype("int64").to_numpy(dtype=np.float64)
    x = times.astype("int64").to_numpy(dtype=np.float64)
    lats = np.interp(x, xp, valid["LAT"].to_numpy(dtype=float))
    lons = np.interp(x, xp, valid["LON"].to_numpy(dtype=float))
    return lats, lons


def himawari_key(obs_time: pd.Timestamp) -> str:
    t = obs_time.tz_convert("UTC")
    stamp = t.strftime("%Y%m%d_%H%M")
    return (
        f"AHI-L1b-FLDK/{t:%Y/%m/%d/%H%M}/"
        f"HS_H08_{stamp}_B13_FLDK_R20_S0101.DAT.bz2"
    )


def build_manifest(cfg: dict) -> pd.DataFrame:
    ensure_output_dirs(cfg)
    tracks = read_ibtracs(cfg)
    rows: list[dict] = []
    raw_root = Path(cfg["raw_himawari_dir"])
    for sid in cfg["case_sids"]:
        track = tracks[tracks["SID"].eq(sid)].copy()
        if track.empty:
            raise ValueError(f"IBTrACS SID not found: {sid}")
        genesis = first_tropical_storm_time(track)
        times = pd.date_range(
            genesis - pd.Timedelta(hours=cfg["history_hours"]),
            genesis - pd.Timedelta(hours=cfg["cadence_hours"]),
            freq=f"{cfg['cadence_hours']}h",
        )
        lats, lons = interpolate_track(track, times)
        name = str(track["NAME"].dropna().iloc[0])
        for obs_time, lat, lon in zip(times, lats, lons, strict=True):
            key = himawari_key(obs_time)
            local = raw_root / key.split("/", 1)[1]
            rows.append(
                {
                    "sid": sid,
                    "name": name,
                    "genesis_time": genesis.isoformat(),
                    "observation_time": obs_time.isoformat(),
                    "hours_to_genesis": int((obs_time - genesis).total_seconds() / 3600),
                    "center_lat": float(lat),
                    "center_lon": float(lon),
                    "s3_key": key,
                    "local_path": str(local),
                }
            )
    manifest = pd.DataFrame(rows).sort_values(["sid", "observation_time"])
    manifest.to_csv(cfg["manifest_path"], index=False)
    print(
        f"manifest: {len(manifest)} frames, {manifest['sid'].nunique()} storms -> "
        f"{cfg['manifest_path']}"
    )
    return manifest


def _sha256(path: Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def _download_one(row: dict, cfg: dict, retries: int = 3) -> dict:
    destination = Path(row["local_path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 1_000_000:
        return {
            **row,
            "download_status": "cached",
            "bytes": destination.stat().st_size,
            "sha256": _sha256(destination),
            "download_error": "",
        }
    url = f"{cfg['s3_base_url'].rstrip('/')}/{row['s3_key']}"
    partial = destination.with_suffix(destination.suffix + ".part")
    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "tcg-pilot/0.1"})
            with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as out:
                shutil.copyfileobj(response, out, length=1 << 20)
            if partial.stat().st_size < 1_000_000:
                raise IOError(f"downloaded file is unexpectedly small: {partial.stat().st_size}")
            os.replace(partial, destination)
            return {
                **row,
                "download_status": "downloaded",
                "bytes": destination.stat().st_size,
                "sha256": _sha256(destination),
                "download_error": "",
            }
        except Exception as exc:  # network failures need to be recorded, not hidden
            last_error = f"{type(exc).__name__}: {exc}"
            partial.unlink(missing_ok=True)
            if isinstance(exc, urllib.error.HTTPError) and exc.code == 404:
                break
            time.sleep(attempt * 2)
    return {
        **row,
        "download_status": "failed",
        "bytes": 0,
        "sha256": "",
        "download_error": last_error,
    }


def download_data(cfg: dict, manifest: pd.DataFrame | None = None) -> pd.DataFrame:
    ensure_output_dirs(cfg)
    if manifest is None:
        if Path(cfg["manifest_path"]).exists():
            manifest = pd.read_csv(cfg["manifest_path"])
        else:
            manifest = build_manifest(cfg)
    records = manifest.to_dict("records")
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=int(cfg["download_workers"])) as pool:
        futures = [pool.submit(_download_one, row, cfg) for row in records]
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if index % 10 == 0 or index == len(futures):
                ok = sum(r["download_status"] != "failed" for r in results)
                print(f"downloaded/cached {ok}/{index}; total planned {len(futures)}", flush=True)
    log = pd.DataFrame(results).sort_values(["sid", "observation_time"])
    log.to_csv(cfg["download_log_path"], index=False)
    failures = log[log["download_status"].eq("failed")]
    print(
        f"download complete: {len(log) - len(failures)}/{len(log)} usable; "
        f"failures={len(failures)}"
    )
    return log


def _haversine_km(lons: np.ndarray, lats: np.ndarray, lon0: float, lat0: float) -> np.ndarray:
    radius = 6371.0088
    phi1 = np.deg2rad(lats)
    phi0 = math.radians(lat0)
    dphi = phi1 - phi0
    dlambda = np.deg2rad(((lons - lon0 + 180.0) % 360.0) - 180.0)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * math.cos(phi0) * np.sin(dlambda / 2.0) ** 2
    return 2.0 * radius * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def compute_cloud_features(
    bt: np.ndarray,
    lons: np.ndarray,
    lats: np.ndarray,
    center_lon: float,
    center_lat: float,
    cfg: dict,
) -> dict:
    from scipy import ndimage

    bt = np.asarray(bt, dtype=np.float32)
    lons = np.asarray(lons, dtype=np.float32)
    lats = np.asarray(lats, dtype=np.float32)
    distance = _haversine_km(lons, lats, center_lon, center_lat)
    geometric_domain = np.isfinite(distance) & (distance <= float(cfg["outer_radius_km"]))
    finite = np.isfinite(bt) & np.isfinite(distance)
    domain = finite & geometric_domain
    if domain.sum() < 100:
        raise ValueError("fewer than 100 valid pixels in the analysis radius")

    area_weights = np.clip(np.cos(np.deg2rad(lats)), 0.05, None)
    domain_weight = float(area_weights[domain].sum())
    cold = domain & (bt < float(cfg["cold_threshold_k"]))
    very_cold = domain & (bt < float(cfg["very_cold_threshold_k"]))
    cold_weight = float(area_weights[cold].sum())
    inner_cold = cold & (distance <= float(cfg["inner_radius_km"]))

    # Keep only non-trivial connected cold-cloud objects.
    labels, object_count = ndimage.label(cold, structure=np.ones((3, 3), dtype=np.uint8))
    counts = np.bincount(labels.ravel()) if object_count else np.array([0])
    valid_ids = np.flatnonzero(counts >= int(cfg["min_patch_pixels"]))
    valid_ids = valid_ids[valid_ids != 0]
    filtered = np.isin(labels, valid_ids) if len(valid_ids) else np.zeros_like(cold)
    filtered_labels, patch_count = ndimage.label(filtered, structure=np.ones((3, 3), dtype=np.uint8))
    component_areas = np.bincount(filtered_labels.ravel(), weights=area_weights.ravel())
    component_areas = component_areas[1:] if len(component_areas) > 1 else np.array([])
    largest_patch_fraction = (
        float(component_areas.max() / cold_weight)
        if len(component_areas) and cold_weight > 0
        else 0.0
    )

    dx = ((lons - center_lon + 180.0) % 360.0 - 180.0) * 111.32 * math.cos(math.radians(center_lat))
    dy = (lats - center_lat) * 110.57
    angle = (np.arctan2(dy, dx) + 2.0 * np.pi) % (2.0 * np.pi)
    sector_cold_fractions = []
    for sector in range(8):
        in_sector = domain & (angle >= sector * np.pi / 4.0) & (angle < (sector + 1) * np.pi / 4.0)
        denom = float(area_weights[in_sector].sum())
        numer = float(area_weights[in_sector & cold].sum())
        sector_cold_fractions.append(numer / denom if denom > 0 else np.nan)
    sector_values = np.asarray(sector_cold_fractions, dtype=float)
    sector_mean = float(np.nanmean(sector_values))
    symmetry = (
        float(np.clip(1.0 - np.nanstd(sector_values) / (sector_mean + 1e-8), 0.0, 1.0))
        if sector_mean > 0
        else 0.0
    )

    if cold_weight > 0:
        cold_x = float(np.average(dx[cold], weights=area_weights[cold]))
        cold_y = float(np.average(dy[cold], weights=area_weights[cold]))
        centroid_offset = float(math.hypot(cold_x, cold_y))
        mean_cold_radius = float(np.average(distance[cold], weights=area_weights[cold]))
    else:
        centroid_offset = np.nan
        mean_cold_radius = np.nan

    return {
        "valid_pixel_fraction": float(domain.sum() / geometric_domain.sum()),
        "mean_bt_k": float(np.average(bt[domain], weights=area_weights[domain])),
        "min_bt_k": float(np.nanmin(bt[domain])),
        "cold_cloud_fraction": cold_weight / domain_weight,
        "very_cold_fraction": float(area_weights[very_cold].sum() / domain_weight),
        "radial_concentration": float(area_weights[inner_cold].sum() / (cold_weight + 1e-12)),
        "azimuthal_symmetry": symmetry,
        "largest_patch_fraction": largest_patch_fraction,
        "patch_count": int(patch_count),
        "patch_density_per_10k_px": float(patch_count / domain.sum() * 10_000.0),
        "cold_centroid_offset_km": centroid_offset,
        "mean_cold_radius_km": mean_cold_radius,
    }


def _decompress_bz2(source: Path, destination: Path) -> None:
    with bz2.open(source, "rb") as compressed, destination.open("wb") as raw:
        shutil.copyfileobj(compressed, raw, length=1 << 20)


def process_one_frame(row: dict, cfg: dict, temp_dir: Path) -> dict:
    from satpy import Scene

    if row.get("local_paths_json") and not pd.isna(row.get("local_paths_json")):
        sources = [Path(value) for value in json.loads(row["local_paths_json"])]
    else:
        sources = [Path(row["local_path"])]
    missing = [source for source in sources if not source.exists()]
    if missing:
        raise FileNotFoundError(", ".join(str(path) for path in missing))
    raw_paths = []
    for source in sources:
        raw_path = temp_dir / source.name.removesuffix(".bz2")
        _decompress_bz2(source, raw_path)
        raw_paths.append(raw_path)
    try:
        scene = Scene(filenames=[str(path) for path in raw_paths], reader="ahi_hsd")
        scene.load(["B13"])
        half = float(cfg["crop_half_width_deg"])
        lon0, lat0 = float(row["center_lon"]), float(row["center_lat"])
        cropped = scene.crop(ll_bbox=(lon0 - half, lat0 - half, lon0 + half, lat0 + half))
        data = cropped["B13"]
        bt = np.asarray(data.compute(), dtype=np.float32)
        lons, lats = data.attrs["area"].get_lonlats()
        features = compute_cloud_features(bt, lons, lats, lon0, lat0, cfg)
        return {**row, **features}
    finally:
        for raw_path in raw_paths:
            raw_path.unlink(missing_ok=True)


def process_data(cfg: dict, limit: int | None = None) -> pd.DataFrame:
    ensure_output_dirs(cfg)
    manifest = pd.read_csv(cfg["manifest_path"])
    if limit is not None:
        manifest = manifest.head(limit)
    existing_path = Path(cfg["features_path"])
    if existing_path.exists():
        existing = pd.read_csv(existing_path)
        done = set(zip(existing["sid"].astype(str), existing["observation_time"].astype(str)))
    else:
        existing = pd.DataFrame()
        done = set()
    rows: list[dict] = []
    errors: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="tcg_h8_") as tmp:
        temp_dir = Path(tmp)
        for index, row in enumerate(manifest.to_dict("records"), start=1):
            key = (str(row["sid"]), str(row["observation_time"]))
            if key in done:
                continue
            try:
                rows.append(process_one_frame(row, cfg, temp_dir))
                relative_hour = row.get(
                    "hours_to_endpoint", row.get("hours_to_genesis", "")
                )
                print(
                    f"processed {index}/{len(manifest)} {row['sid']} {relative_hour}h",
                    flush=True,
                )
            except Exception as exc:
                errors.append(
                    {
                        "sid": row["sid"],
                        "observation_time": row["observation_time"],
                        "local_path": row["local_path"],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                print(f"processing failed {index}/{len(manifest)}: {errors[-1]['error']}", flush=True)
    combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    if not combined.empty:
        combined = combined.drop_duplicates(["sid", "observation_time"]).sort_values(
            ["sid", "observation_time"]
        )
        combined.to_csv(cfg["features_path"], index=False)
    pd.DataFrame(
        errors, columns=["sid", "observation_time", "local_path", "error"]
    ).to_csv(cfg["error_log_path"], index=False)
    print(f"features: {len(combined)} rows; new errors={len(errors)}")
    return combined


STATIC_FEATURES = [
    "cold_cloud_fraction",
    "very_cold_fraction",
    "radial_concentration",
    "azimuthal_symmetry",
    "largest_patch_fraction",
    "patch_density_per_10k_px",
    "cold_centroid_offset_km",
    "mean_cold_radius_km",
    "mean_bt_k",
    "min_bt_k",
]


def add_dynamic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["sid", "observation_time"]).copy()
    for col in [
        "cold_cloud_fraction",
        "radial_concentration",
        "azimuthal_symmetry",
        "largest_patch_fraction",
        "cold_centroid_offset_km",
    ]:
        df[f"delta_{col}"] = df.groupby("sid")[col].diff()
    return df


def _cluster_bootstrap_ci(
    predictions: pd.DataFrame, metric, rng: np.random.Generator, n_boot: int = 2000
) -> tuple[float, float]:
    storms = predictions["sid"].unique()
    values = []
    for _ in range(n_boot):
        sampled = rng.choice(storms, size=len(storms), replace=True)
        chunks = []
        for draw, sid in enumerate(sampled):
            part = predictions[predictions["sid"].eq(sid)].copy()
            part["bootstrap_group"] = draw
            chunks.append(part)
        boot = pd.concat(chunks, ignore_index=True)
        if boot["label"].nunique() < 2:
            continue
        values.append(metric(boot["label"], boot["probability"]))
    if not values:
        return np.nan, np.nan
    return tuple(np.quantile(values, [0.025, 0.975]))


def evaluate_loso(
    data: pd.DataFrame, features: list[str], model_name: str, seed: int
) -> tuple[dict, pd.DataFrame]:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        roc_auc_score,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    work = data.dropna(subset=["label"]).copy()
    predictions = []
    per_storm_auc = []
    for sid in sorted(work["sid"].unique()):
        train = work[~work["sid"].eq(sid)]
        test = work[work["sid"].eq(sid)]
        model = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
                (
                    "logistic",
                    LogisticRegression(
                        C=1.0,
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=seed,
                    ),
                ),
            ]
        )
        model.fit(train[features], train["label"].astype(int))
        probability = model.predict_proba(test[features])[:, 1]
        fold = test[["sid", "name", "observation_time", "hours_to_genesis", "label"]].copy()
        fold["probability"] = probability
        fold["model"] = model_name
        predictions.append(fold)
        per_storm_auc.append(roc_auc_score(test["label"], probability))
    pred = pd.concat(predictions, ignore_index=True)
    y = pred["label"].astype(int)
    p = pred["probability"]
    rng = np.random.default_rng(seed)
    auc_low, auc_high = _cluster_bootstrap_ci(pred, roc_auc_score, rng)
    metrics = {
        "model": model_name,
        "n_frames": len(pred),
        "n_storms": pred["sid"].nunique(),
        "roc_auc_pooled": roc_auc_score(y, p),
        "roc_auc_storm_mean": float(np.mean(per_storm_auc)),
        "roc_auc_cluster_ci_low": auc_low,
        "roc_auc_cluster_ci_high": auc_high,
        "pr_auc": average_precision_score(y, p),
        "balanced_accuracy_at_0p5": balanced_accuracy_score(y, p >= 0.5),
        "brier_score": brier_score_loss(y, p),
    }
    return metrics, pred


def compare_stages(df: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
    from scipy.stats import wilcoxon

    records = []
    for feature in features:
        pairs = []
        for sid, storm in df.groupby("sid"):
            early = storm[storm["hours_to_genesis"].between(-72, -48)][feature].median()
            late = storm[storm["hours_to_genesis"].between(-24, -3)][feature].median()
            if np.isfinite(early) and np.isfinite(late):
                pairs.append((sid, early, late))
        differences = np.asarray([late - early for _, early, late in pairs])
        try:
            p_value = float(wilcoxon(differences, alternative="two-sided").pvalue)
        except ValueError:
            p_value = 1.0
        records.append(
            {
                "feature": feature,
                "n_storms": len(pairs),
                "early_median_across_storms": float(np.median([p[1] for p in pairs])),
                "late_median_across_storms": float(np.median([p[2] for p in pairs])),
                "median_paired_change": float(np.median(differences)),
                "storms_increasing": int((differences > 0).sum()),
                "wilcoxon_p_unadjusted": p_value,
            }
        )
    return pd.DataFrame(records)


def make_snapshot_figure(df: pd.DataFrame, cfg: dict) -> None:
    import matplotlib.pyplot as plt
    from satpy import Scene

    figures = ROOT / "outputs" / "figures"
    selected = df[df["hours_to_genesis"].isin([-72, -3])].sort_values(
        ["name", "hours_to_genesis"]
    )
    fig, axes = plt.subplots(selected["name"].nunique(), 2, figsize=(8, 15))
    image_handle = None
    with tempfile.TemporaryDirectory(prefix="tcg_snapshot_") as tmp:
        temp_dir = Path(tmp)
        for row_index, (name, storm) in enumerate(selected.groupby("name", sort=True)):
            for col_index, (_, row) in enumerate(storm.sort_values("hours_to_genesis").iterrows()):
                source = Path(row["local_path"])
                raw_path = temp_dir / source.name.removesuffix(".bz2")
                _decompress_bz2(source, raw_path)
                try:
                    scene = Scene(filenames=[str(raw_path)], reader="ahi_hsd")
                    scene.load(["B13"])
                    half = float(cfg["crop_half_width_deg"])
                    lon0, lat0 = float(row["center_lon"]), float(row["center_lat"])
                    cropped = scene.crop(
                        ll_bbox=(lon0 - half, lat0 - half, lon0 + half, lat0 + half)
                    )
                    bt = np.asarray(cropped["B13"].compute(), dtype=np.float32)
                finally:
                    raw_path.unlink(missing_ok=True)
                ax = axes[row_index, col_index]
                image_handle = ax.imshow(bt, cmap="turbo", vmin=190, vmax=290, origin="upper")
                ax.set_title(f"{name}: {int(row['hours_to_genesis'])} h")
                ax.set_xticks([])
                ax.set_yticks([])
    fig.subplots_adjust(right=0.86, hspace=0.18, wspace=0.08)
    color_ax = fig.add_axes([0.89, 0.15, 0.025, 0.7])
    fig.colorbar(image_handle, cax=color_ax, label="Band 13 brightness temperature (K)")
    fig.savefig(figures / "00_ir_early_late_snapshots.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_figures(
    df: pd.DataFrame, predictions: pd.DataFrame, comparison: pd.DataFrame, cfg: dict
) -> None:
    import matplotlib.pyplot as plt

    figures = ROOT / "outputs" / "figures"
    plt.style.use("seaborn-v0_8-whitegrid")
    selected = [
        ("cold_cloud_fraction", "Cold-cloud fraction"),
        ("radial_concentration", "Radial concentration"),
        ("azimuthal_symmetry", "Azimuthal symmetry"),
        ("largest_patch_fraction", "Largest-patch fraction"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for ax, (feature, label) in zip(axes.ravel(), selected, strict=True):
        for name, storm in df.groupby("name"):
            ax.plot(storm["hours_to_genesis"], storm[feature], marker="o", ms=3, lw=1.2, alpha=0.8, label=name)
        ax.axvspan(-24, 0, color="#e76f51", alpha=0.08)
        ax.set_title(label)
        ax.set_xlabel("Hours to first tropical-storm classification")
    axes[0, 0].legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "01_feature_trajectories.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    storm_names = sorted(predictions["name"].unique())
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    storm_colors = {
        name: color_cycle[index % len(color_cycle)] for index, name in enumerate(storm_names)
    }
    for (model, name), storm in predictions.groupby(["model", "name"]):
        if model != "cold_area_only":
            continue
        for _, segment in storm.assign(
            period=np.where(storm["hours_to_genesis"] <= -48, "early", "late")
        ).groupby("period"):
            ax.plot(
                segment["hours_to_genesis"],
                segment["probability"],
                marker="o",
                lw=1.4,
                color=storm_colors[name],
                label=name if segment["period"].iloc[0] == "early" else None,
            )
    ax.axhline(0.5, color="black", ls="--", lw=1)
    ax.set_xlabel("Hours to first tropical-storm classification")
    ax.set_ylabel("LOSO probability of late pre-genesis stage")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "02_loso_probabilities.png", dpi=180)
    plt.close(fig)

    plot = comparison.set_index("feature").loc[[x[0] for x in selected]].reset_index()
    labels = [x[1] for x in selected]
    x = np.arange(len(plot))
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.36
    ax.bar(x - width / 2, plot["early_median_across_storms"], width, label="Early (-72 to -48 h)")
    ax.bar(x + width / 2, plot["late_median_across_storms"], width, label="Late (-24 to -3 h)")
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.set_ylabel("Across-storm median")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "03_early_late_comparison.png", dpi=180)
    plt.close(fig)
    make_snapshot_figure(df, cfg)


def write_report(
    df: pd.DataFrame,
    metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    per_storm_auc: pd.DataFrame,
) -> Path:
    report_path = ROOT / "outputs" / "reports" / "pilot_v0_1_report.md"
    best = metrics.sort_values("roc_auc_storm_mean", ascending=False).iloc[0]
    comparison_lines = []
    for row in comparison.sort_values("wilcoxon_p_unadjusted").head(6).itertuples():
        comparison_lines.append(
            f"| {row.feature} | {row.early_median_across_storms:.4f} | "
            f"{row.late_median_across_storms:.4f} | {row.median_paired_change:+.4f} | "
            f"{row.storms_increasing}/{row.n_storms} | {row.wilcoxon_p_unadjusted:.4f} |"
        )
    metric_lines = []
    for row in metrics.itertuples():
        metric_lines.append(
            f"| {row.model} | {row.roc_auc_pooled:.3f} | {row.roc_auc_storm_mean:.3f} | "
            f"{row.roc_auc_cluster_ci_low:.3f}–{row.roc_auc_cluster_ci_high:.3f} | "
            f"{row.pr_auc:.3f} | {row.brier_score:.3f} |"
        )
    storm_lines = []
    for row in per_storm_auc.itertuples():
        storm_lines.append(
            f"| {row.model} | {row.BAILU:.3f} | {row.DANAS:.3f} | "
            f"{row.LINGLING:.3f} | {row.MITAG:.3f} | {row.PODUL:.3f} |"
        )
    text = f"""# 台风生成动力学真实资料 Pilot v0.1

## 结论摘要

本次测试成功处理 {len(df)} 帧 Himawari-8 Band 13 红外亮温，覆盖 {df['sid'].nunique()} 个西北太平洋发展案例。事件定义为 IBTrACS 中首次 JTWC `TS` 或 `USA_WIND >= 34 kt` 的时刻。

在完整风暴留一交叉验证中，表现最好的阶段判别模型为 `{best['model']}`：逐风暴 ROC-AUC 均值为 {best['roc_auc_storm_mean']:.3f}，池化 ROC-AUC 为 {best['roc_auc_pooled']:.3f}，风暴级聚类 bootstrap 95% 区间为 {best['roc_auc_cluster_ci_low']:.3f}–{best['roc_auc_cluster_ci_high']:.3f}。该区间跨过 0.5，且逐案例结果差异很大，因此只能称为弱而不稳健的试验信号。

这说明当前下载、中心跟踪、裁剪、亮温转换、云团指标和分组验证链路已经跑通。它不能证明模型能够区分“发展扰动”和“未发展扰动”，也不能单独证明存在临界慢化；后两项需要加入匹配的未发展云团以及 ERA5/IMERG 后再检验。

## 数据和方法

- 案例：Danas、Bailu、Podul、Lingling、Mitag（2019）。
- 卫星：Himawari-8 AHI Band 13（10.4 μm，2 km），NOAA 开放对象存储。
- 时间窗：首次热带风暴分类前 72 h 至前 3 h，间隔 3 h。
- 空间窗：随 IBTrACS 中心移动，600 km 半径。
- 冷云阈值：235 K；强冷云阈值：210 K。
- 判别任务：早期（−72～−48 h）与临近生成（−24～−3 h）。
- 验证：按 SID 留一整场风暴，禁止相邻帧跨训练/测试集。

数据来自 [NOAA IBTrACS v04r01](https://www.ncei.noaa.gov/products/international-best-track-archive) 与 [NOAA/JMA Himawari-8/9 开放数据](https://registry.opendata.aws/noaa-himawari/)，访问日期为 2026-08-02。

## 阶段判别结果

| 模型 | 池化 ROC-AUC | 逐风暴 AUC 均值 | 聚类 bootstrap 95% CI | PR-AUC | Brier |
|---|---:|---:|---:|---:|---:|
{chr(10).join(metric_lines)}

### 各留出风暴 ROC-AUC

| 模型 | Bailu | Danas | Lingling | Mitag | Podul |
|---|---:|---:|---:|---:|---:|
{chr(10).join(storm_lines)}

单冷云面积模型的均值明显受到 Bailu（AUC=1.000）拉高，其余案例大多仅略高于或低于随机水平；这也是当前不能把 0.622 当作稳定预测能力的直接证据。

## 早期—临近生成配对比较

以下 Wilcoxon p 值未做多重比较校正，且只有 5 个案例，应视为效应方向检查，而不是确认性显著性检验。

| 指标 | 早期中位数 | 临近中位数 | 配对变化中位数 | 上升案例 | p（未校正） |
|---|---:|---:|---:|---:|---:|
{chr(10).join(comparison_lines)}

## 第一版能支持什么

1. 真实卫星数据链和中心跟随裁剪可复现。
2. 斑块/组织指标可以从亮温场稳定计算。
3. 可用整场风暴留一法检查生成前时间阶段信号。

## 第一版不能支持什么

1. 没有未发展扰动，因此不能报告真实的台风生成预测能力。
2. 没有 ERA5 环境变量，无法排除风切变、水汽和背景涡度的混杂。
3. 只有 5 个案例，置信区间主要用于暴露不确定性，不足以形成论文结论。
4. 尚未拟合动力学方程，不能声称 SINDy 已发现控制方程。
5. 尚未做恢复率、Jacobian 特征值和空模型检验，不能声称发现临界慢化。

## 下一步固定顺序

1. 自动追踪并人工复核至少 5 个匹配未发展云团。
2. 接入 ERA5 的 850 hPa 涡度、600–850 hPa 湿度和 200–850 hPa 风切变。
3. 增加 IMERG 降水持续性，再建立发展/未发展逻辑回归基线。
4. 扩展到至少 30–50 个发展案例和 60–100 个未发展案例。
5. 基线确认组织信息有增量后，再训练动力学对比学习表示。
"""
    report_path.write_text(text, encoding="utf-8")
    return report_path


def analyze(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features_path = Path(cfg["features_path"])
    if not features_path.exists():
        raise FileNotFoundError(f"run process first: {features_path}")
    df = pd.read_csv(features_path, parse_dates=["observation_time", "genesis_time"])
    df = add_dynamic_features(df)
    # Middle interval is deliberately excluded to create a clean diagnostic contrast.
    df["label"] = np.nan
    df.loc[df["hours_to_genesis"].between(-72, -48), "label"] = 0
    df.loc[df["hours_to_genesis"].between(-24, -3), "label"] = 1
    dynamic = [col for col in df.columns if col.startswith("delta_")]
    models = {
        "cold_area_only": ["cold_cloud_fraction"],
        "organization_static": STATIC_FEATURES,
        "organization_dynamic": STATIC_FEATURES + dynamic,
    }
    metrics, predictions = [], []
    for model_name, cols in models.items():
        result, pred = evaluate_loso(df, cols, model_name, int(cfg["random_seed"]))
        metrics.append(result)
        predictions.append(pred)
    metrics_df = pd.DataFrame(metrics)
    predictions_df = pd.concat(predictions, ignore_index=True)
    from sklearn.metrics import roc_auc_score

    per_storm_auc = (
        predictions_df.groupby(["model", "name"])
        .apply(
            lambda group: roc_auc_score(group["label"], group["probability"]),
            include_groups=False,
        )
        .unstack()
        .reset_index()
    )
    comparison = compare_stages(df, STATIC_FEATURES)
    out = ROOT / "outputs" / "tables"
    metrics_df.to_csv(out / "pilot_metrics.csv", index=False)
    predictions_df.to_csv(out / "pilot_predictions.csv", index=False)
    per_storm_auc.to_csv(out / "pilot_per_storm_auc.csv", index=False)
    comparison.to_csv(out / "pilot_stage_comparison.csv", index=False)
    make_figures(df, predictions_df, comparison, cfg)
    report = write_report(df, metrics_df, comparison, per_storm_auc)
    print(metrics_df.to_string(index=False))
    print(f"report -> {report}")
    return metrics_df, predictions_df, comparison


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="TC-genesis Himawari pilot v0.1")
    parser.add_argument(
        "command", choices=["manifest", "download", "process", "analyze", "run-all"]
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--limit", type=int, default=None, help="process only the first N frames")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.command == "manifest":
        build_manifest(cfg)
    elif args.command == "download":
        download_data(cfg)
    elif args.command == "process":
        process_data(cfg, limit=args.limit)
    elif args.command == "analyze":
        analyze(cfg)
    else:
        manifest = build_manifest(cfg)
        download_data(cfg, manifest)
        process_data(cfg, limit=args.limit)
        analyze(cfg)


if __name__ == "__main__":
    main()
