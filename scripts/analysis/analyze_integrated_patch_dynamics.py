# -*- coding: utf-8 -*-
"""
v2.4 Integrated Patch Dynamics and Precursor Warning

This script is additive. It does not modify existing datasets, models, or
training scripts.

Goal:
    Put prepatch indicators, PWSI, dynamic patch indicators, and visible patch
    metrics onto the same event-aligned time axis.

Main output:
    Prepatch spatial organization
    -> Dynamic patch restructuring
    -> Visible patch manifestation
    -> Critical transition
"""

from __future__ import annotations

import argparse
import json
import math
from collections import deque
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def log(msg: str) -> None:
    print(f"[v2.4] {msg}", flush=True)


def load_npz(path: Union[str, Path]) -> Optional[dict]:
    path = Path(path)
    if not path.exists():
        log(f"skip missing file: {path}")
        return None
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def pick_first_key(data: dict, candidates: Sequence[str]) -> Optional[str]:
    for key in candidates:
        if key in data:
            return key
    return None


def decode_names(arr) -> List[str]:
    out = []
    for x in list(arr):
        if isinstance(x, bytes):
            out.append(x.decode("utf-8", errors="ignore"))
        else:
            out.append(str(x))
    return out


def clean_col_name(name: str) -> str:
    name = name.strip().replace(" ", "_").replace("-", "_").replace("/", "_")
    name = name.replace("(", "").replace(")", "").replace("[", "").replace("]", "")
    return name.lower()


def find_feature_names(data: dict, n_features: int, prefix: str) -> List[str]:
    candidate_keys = [
        "feature_names",
        "patch_feature_names",
        "prepatch_feature_names",
        "pwsi_feature_names",
        "names",
        "columns",
    ]
    for key in candidate_keys:
        if key in data:
            names = decode_names(data[key])
            if len(names) == n_features:
                return [clean_col_name(x) for x in names]
    return [f"{prefix}_{i:02d}" for i in range(n_features)]


def get_required_vector(data: dict, candidates: Sequence[str], n: Optional[int] = None) -> np.ndarray:
    key = pick_first_key(data, candidates)
    if key is None:
        raise KeyError(f"Cannot find any of keys: {candidates}")
    arr = np.asarray(data[key])
    if n is not None and len(arr) != n:
        raise ValueError(f"Key {key} has length {len(arr)}, expected {n}")
    return arr


def get_optional_vector(data: dict, candidates: Sequence[str], n: int, fill_value=np.nan) -> np.ndarray:
    key = pick_first_key(data, candidates)
    if key is None:
        return np.full(n, fill_value)
    arr = np.asarray(data[key])
    if arr.ndim == 0:
        return np.full(n, arr.item())
    if len(arr) == n:
        return arr
    return np.full(n, fill_value)


def extract_last_frame(x_img: np.ndarray) -> np.ndarray:
    """
    Convert X_img to one frame per sample.

    Supports:
        (N, T, C, H, W)
        (N, T, H, W)
        (N, C, H, W)
        (N, H, W)
    """
    x_img = np.asarray(x_img)

    if x_img.ndim == 5:
        return x_img[:, -1, 0, :, :]

    if x_img.ndim == 4:
        return x_img[:, -1, :, :]

    if x_img.ndim == 3:
        return x_img

    raise ValueError(f"Unsupported X_img shape: {x_img.shape}")


def extract_last_features(arr: np.ndarray) -> np.ndarray:
    """
    Convert feature arrays to one feature vector per sample.

    Supports:
        (N, T, F)
        (N, F)
        (N,)
    """
    arr = np.asarray(arr)

    if arr.ndim == 3:
        return arr[:, -1, :]

    if arr.ndim == 2:
        return arr

    if arr.ndim == 1:
        return arr.reshape(-1, 1)

    raise ValueError(f"Unsupported feature array shape: {arr.shape}")


def find_feature_array(data: dict) -> Optional[Tuple[str, np.ndarray]]:
    candidate_keys = [
        "X_patch",
        "X_prepatch",
        "X_pwsi",
        "features",
        "patch_features",
        "prepatch_features",
        "pwsi_features",
    ]

    for key in candidate_keys:
        if key in data and np.asarray(data[key]).ndim in (1, 2, 3):
            return key, np.asarray(data[key])

    return None


def connected_components(mask: np.ndarray, connectivity: int = 8) -> List[int]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)

    if connectivity == 4:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        nbrs = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        ]

    sizes: List[int] = []

    for i in range(h):
        for j in range(w):
            if not mask[i, j] or seen[i, j]:
                continue

            q = deque([(i, j)])
            seen[i, j] = True
            size = 0

            while q:
                x, y = q.popleft()
                size += 1

                for dx, dy in nbrs:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w:
                        if mask[nx, ny] and not seen[nx, ny]:
                            seen[nx, ny] = True
                            q.append((nx, ny))

            sizes.append(size)

    return sizes


def binary_edge_count(mask: np.ndarray) -> int:
    mask = np.asarray(mask, dtype=bool)
    edges = np.sum(mask[:, 1:] != mask[:, :-1])
    edges += np.sum(mask[1:, :] != mask[:-1, :])
    return int(edges)


def patch_size_distribution(
    sizes: Sequence[int],
    total_pixels: int,
    bins: np.ndarray,
) -> np.ndarray:
    if len(sizes) == 0:
        return np.zeros(len(bins) - 1, dtype=float)

    ratios = np.asarray(sizes, dtype=float) / float(total_pixels)
    hist, _ = np.histogram(ratios, bins=bins)
    hist = hist.astype(float)

    if hist.sum() <= 0:
        return hist

    return hist / hist.sum()


def compute_visible_patch_metrics(
    frame: np.ndarray,
    threshold: float,
    direction: str,
    connectivity: int,
) -> dict:
    frame = np.asarray(frame, dtype=float)

    if direction == "low":
        mask = frame <= threshold
    else:
        mask = frame >= threshold

    total_pixels = mask.size
    sizes = connected_components(mask, connectivity=connectivity)

    total_area = int(np.sum(mask))
    largest_area = int(max(sizes)) if sizes else 0
    patch_count = int(len(sizes))
    mean_patch_area = float(np.mean(sizes)) if sizes else 0.0

    largest_patch_ratio = largest_area / total_pixels
    total_patch_ratio = total_area / total_pixels

    edge_density = binary_edge_count(mask) / max(1, total_pixels)

    if total_area > 0:
        aggregation_index = largest_area / total_area
    else:
        aggregation_index = 0.0

    return {
        "patch_count": patch_count,
        "total_patch_area": total_area,
        "total_patch_ratio": total_patch_ratio,
        "largest_patch_area": largest_area,
        "largest_patch_ratio": largest_patch_ratio,
        "mean_patch_area": mean_patch_area,
        "edge_density": edge_density,
        "aggregation_index": aggregation_index,
        "visible_patch_metric": largest_patch_ratio,
        "_component_sizes": sizes,
    }


def robust_zscore_by_sim(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()

    for col in cols:
        values = []

        for _, g in out.groupby("sim_id", sort=False):
            x = g[col].astype(float).to_numpy()

            med = np.nanmedian(x)
            mad = np.nanmedian(np.abs(x - med))
            scale = 1.4826 * mad

            if not np.isfinite(scale) or scale < 1e-12:
                scale = np.nanstd(x)

            if not np.isfinite(scale) or scale < 1e-12:
                values.extend(np.zeros(len(x)))
            else:
                values.extend((x - med) / scale)

        out[f"z_{col}"] = values

    return out


def compute_dynamic_patch_indicators(
    df: pd.DataFrame,
    bins: np.ndarray,
) -> pd.DataFrame:
    """
    Add dynamic patch indicators.

    PDSI:
        Patch distribution shift index.

    DPCI:
        Dominant patch component increase.

    DPCR:
        Dominant patch component reduction.

    FAI:
        Fragmentation acceleration index.

    PSII:
        Patch structure instability index.

    BCI:
        Boundary complexity index.

    FPV:
        Front propagation velocity.

    PMR:
        Patch merge rate.

    GCGR:
        Giant connected component growth rate.

    PPI:
        Percolation proximity index.
    """
    rows = []

    for _, g in df.groupby("sim_id", sort=False):
        g = g.sort_values("time_idx").copy().reset_index(drop=True)
        n = len(g)

        pdsi = np.zeros(n)
        dpci = np.zeros(n)
        dpcr = np.zeros(n)
        fai = np.zeros(n)
        bci = g["edge_density"].astype(float).to_numpy()
        fpv = np.zeros(n)
        pmr = np.zeros(n)
        gcgr = np.zeros(n)
        ppi = g["largest_patch_ratio"].astype(float).to_numpy()

        dist_prev = None
        prev_count = None
        prevprev_count = None
        prev_largest = None
        prev_area = None

        for i in range(n):
            sizes = g.loc[i, "_component_sizes"]
            total_pixels = int(g.loc[i, "_total_pixels"])
            dist = patch_size_distribution(sizes, total_pixels, bins)

            count = float(g.loc[i, "patch_count"])
            largest = float(g.loc[i, "largest_patch_ratio"])
            area = float(g.loc[i, "total_patch_ratio"])

            if dist_prev is not None:
                pdsi[i] = float(np.sum(np.abs(dist - dist_prev)))

            if prev_largest is not None:
                delta_largest = largest - prev_largest
                dpci[i] = max(0.0, delta_largest)
                dpcr[i] = max(0.0, -delta_largest)
                gcgr[i] = max(0.0, delta_largest)

            if prev_area is not None:
                fpv[i] = max(0.0, area - prev_area)

            if prev_count is not None and prev_area is not None:
                if count < prev_count and area >= prev_area:
                    pmr[i] = prev_count - count

            if prevprev_count is not None and prev_count is not None:
                accel = count - 2.0 * prev_count + prevprev_count
                fai[i] = max(0.0, accel)

            dist_prev = dist
            prevprev_count = prev_count
            prev_count = count
            prev_largest = largest
            prev_area = area

        g["PDSI"] = pdsi
        g["DPCI"] = dpci
        g["DPCR"] = dpcr
        g["FAI"] = fai
        g["BCI"] = bci
        g["FPV"] = fpv
        g["PMR"] = pmr
        g["GCGR"] = gcgr
        g["PPI"] = ppi

        tmp = robust_zscore_by_sim(
            g,
            ["PDSI", "DPCI", "DPCR", "FAI", "BCI", "FPV", "PMR", "GCGR", "PPI"],
        )

        instability_cols = [
            "z_PDSI",
            "z_DPCI",
            "z_DPCR",
            "z_FAI",
            "z_BCI",
            "z_FPV",
            "z_PMR",
            "z_GCGR",
            "z_PPI",
        ]

        tmp["PSII"] = tmp[instability_cols].clip(lower=0).mean(axis=1)
        rows.append(tmp)

    return pd.concat(rows, axis=0, ignore_index=True)


def build_base_dataframe(
    base: dict,
    system: str,
    seed: int,
    patch_threshold: float,
    patch_direction: str,
    connectivity: int,
) -> pd.DataFrame:
    x_key = pick_first_key(base, ["X_img", "x_img", "images", "X", "frames"])

    if x_key is None:
        raise KeyError("Cannot find X_img/images in base npz.")

    frames = extract_last_frame(base[x_key])
    n = len(frames)

    sim_id = get_required_vector(
        base,
        ["sim_id", "sim_ids", "simulation_id", "simulation_ids"],
        n=n,
    )

    time_idx = get_required_vector(
        base,
        ["time_idx", "time_indices", "t_idx", "t", "time"],
        n=n,
    )

    critical_time = get_optional_vector(
        base,
        [
            "critical_time",
            "critical_times",
            "event_time",
            "event_times",
            "transition_time",
            "transition_times",
        ],
        n=n,
    )

    y_risk = get_optional_vector(
        base,
        ["y_risk", "risk_label", "y", "label", "labels"],
        n=n,
    )

    rows = []

    for i in range(n):
        metrics = compute_visible_patch_metrics(
            frames[i],
            threshold=patch_threshold,
            direction=patch_direction,
            connectivity=connectivity,
        )

        row = {
            "system": system,
            "seed": seed,
            "row_id": i,
            "sim_id": int(sim_id[i]),
            "time_idx": int(time_idx[i]),
            "critical_time": float(critical_time[i]) if np.isfinite(critical_time[i]) else np.nan,
            "y_risk": float(y_risk[i]) if np.isfinite(y_risk[i]) else np.nan,
            "_total_pixels": int(frames[i].size),
        }

        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)

    before = len(df)
    df = (
        df.sort_values(["sim_id", "time_idx", "row_id"])
        .drop_duplicates(["sim_id", "time_idx"], keep="last")
    )

    if len(df) < before:
        log(f"deduplicated base rows: {before} -> {len(df)}")

    df["critical_time"] = df.groupby("sim_id")["critical_time"].transform(
        lambda s: s.dropna().iloc[0] if s.dropna().size else np.nan
    )

    df["relative_time"] = df["time_idx"] - df["critical_time"]

    return df.sort_values(["sim_id", "time_idx"]).reset_index(drop=True)


def merge_feature_file(
    df: pd.DataFrame,
    feature_data: Optional[dict],
    prefix: str,
    strict_alignment: bool,
) -> pd.DataFrame:
    if feature_data is None:
        return df

    arr_info = find_feature_array(feature_data)

    if arr_info is None:
        log(f"skip {prefix}: no recognizable feature array.")
        return df

    arr_key, arr = arr_info
    feats = extract_last_features(arr)
    n = len(feats)

    sim_id = get_optional_vector(
        feature_data,
        ["sim_id", "sim_ids", "simulation_id", "simulation_ids"],
        n=n,
    )

    time_idx = get_optional_vector(
        feature_data,
        ["time_idx", "time_indices", "t_idx", "t", "time"],
        n=n,
    )

    names = find_feature_names(feature_data, feats.shape[1], prefix=prefix)

    out_names = [
        f"{prefix}_{name}" if not name.startswith(prefix) else name
        for name in names
    ]

    if np.all(np.isfinite(sim_id)) and np.all(np.isfinite(time_idx)):
        tmp = pd.DataFrame(feats, columns=out_names)
        tmp["sim_id"] = sim_id.astype(int)
        tmp["time_idx"] = time_idx.astype(int)

        tmp = (
            tmp.sort_values(["sim_id", "time_idx"])
            .drop_duplicates(["sim_id", "time_idx"], keep="last")
        )

        before_cols = set(df.columns)
        merged = df.merge(tmp, on=["sim_id", "time_idx"], how="left")
        new_cols = [c for c in merged.columns if c not in before_cols]

        if new_cols:
            missing_rate = merged[new_cols].isna().mean().mean()
        else:
            missing_rate = 0.0

        log(
            f"merged {prefix} from {arr_key}: "
            f"features={len(new_cols)}, missing_rate={missing_rate:.3f}"
        )

        if strict_alignment and missing_rate > 0.05:
            raise ValueError(
                f"{prefix} alignment missing rate too high: {missing_rate:.3f}"
            )

        return merged

    if strict_alignment:
        raise ValueError(
            f"{prefix} file does not contain sim_id/time_idx; cannot align safely."
        )

    if len(df) != n:
        log(
            f"skip {prefix}: row-order fallback impossible, "
            f"base rows={len(df)}, feature rows={n}"
        )
        return df

    for j, name in enumerate(out_names):
        df[name] = feats[:, j]

    log(f"merged {prefix} by row order from {arr_key}: features={feats.shape[1]}")

    return df


def infer_indicator_group(col: str) -> Optional[str]:
    c = col.lower()

    if c in {"ac1", "variance", "var", "trend"} or "traditional" in c:
        return "Traditional EWS"

    if "pwsi" in c:
        return "PWSI"

    prepatch_tokens = [
        "z_sync",
        "z_conn",
        "z_rigid",
        "z_mode",
        "local_neighbor",
        "sync_edge",
        "moran",
        "geary",
        "boundary",
        "sharpness",
        "gradient",
        "svd",
        "dominant_mode",
        "localization",
        "prepatch",
    ]

    if any(tok in c for tok in prepatch_tokens):
        return "Prepatch"

    dynamic_tokens = [
        "pdsi",
        "dpci",
        "dpcr",
        "fai",
        "psii",
        "bci",
        "fpv",
        "pmr",
        "gcgr",
        "ppi",
    ]

    if any(tok in c for tok in dynamic_tokens):
        return "Dynamic patch"

    visible_tokens = [
        "patch_count",
        "total_patch",
        "largest_patch",
        "mean_patch",
        "edge_density",
        "aggregation_index",
        "visible_patch",
    ]

    if any(tok in c for tok in visible_tokens):
        return "Visible patch"

    return None


def candidate_indicator_columns(df: pd.DataFrame) -> List[str]:
    excluded = {
        "system",
        "seed",
        "row_id",
        "sim_id",
        "time_idx",
        "critical_time",
        "relative_time",
        "y_risk",
        "_total_pixels",
        "_component_sizes",
    }

    cols = []

    for col in df.columns:
        if col in excluded or col.startswith("_"):
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            if infer_indicator_group(col) is not None:
                cols.append(col)

    return cols


def first_persistent_alarm(
    times: np.ndarray,
    values: np.ndarray,
    threshold: float,
    k: int,
) -> Optional[int]:
    ok = np.isfinite(values) & (values > threshold)

    run = 0

    for i, flag in enumerate(ok):
        if flag:
            run += 1
            if run >= k:
                return int(times[i - k + 1])
        else:
            run = 0

    return None


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 1000,
    seed: int = 42,
) -> Tuple[float, float]:
    arr = np.asarray([x for x in values if np.isfinite(x)], dtype=float)

    if len(arr) == 0:
        return np.nan, np.nan

    if len(arr) == 1:
        return float(arr[0]), float(arr[0])

    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)

    for i in range(n_boot):
        means[i] = np.mean(rng.choice(arr, size=len(arr), replace=True))

    return (
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def compute_first_rise(
    df: pd.DataFrame,
    indicator_cols: Sequence[str],
    baseline_start: int,
    baseline_end: int,
    persistent_k: int,
    sigma: float,
    min_baseline_points: int,
):
    detail_rows = []

    for sim_id, g in df.groupby("sim_id", sort=False):
        g = g.sort_values("time_idx").reset_index(drop=True)

        event_time_values = g["critical_time"].dropna().unique()

        if len(event_time_values) == 0:
            continue

        event_time = int(event_time_values[0])
        pre_event = g[g["time_idx"] < event_time]

        if pre_event.empty:
            continue

        baseline = g[
            (g["relative_time"] >= baseline_start)
            & (g["relative_time"] <= baseline_end)
        ]

        if len(baseline) < min_baseline_points:
            m = max(min_baseline_points, int(math.ceil(len(pre_event) * 0.25)))
            baseline = pre_event.sort_values("time_idx").head(m)

        for col in indicator_cols:
            x_base = baseline[col].astype(float).to_numpy()
            x_base = x_base[np.isfinite(x_base)]

            if len(x_base) < 3:
                continue

            mu = float(np.mean(x_base))
            sd = float(np.std(x_base, ddof=1))

            if not np.isfinite(sd) or sd < 1e-12:
                sd = 1e-12

            threshold = mu + sigma * sd

            window = g[g["time_idx"] < event_time]

            alarm = first_persistent_alarm(
                times=window["time_idx"].to_numpy(),
                values=window[col].astype(float).to_numpy(),
                threshold=threshold,
                k=persistent_k,
            )

            detected = 0 if alarm is None else 1
            lead = np.nan if alarm is None else event_time - alarm

            detail_rows.append(
                {
                    "sim_id": sim_id,
                    "indicator": col,
                    "indicator_group": infer_indicator_group(col),
                    "event_time": event_time,
                    "t_alarm": alarm if alarm is not None else np.nan,
                    "detected": detected,
                    "lead_time": lead,
                    "threshold": threshold,
                    "baseline_mean": mu,
                    "baseline_std": sd,
                    "persistent_k": persistent_k,
                    "sigma": sigma,
                }
            )

    detail = pd.DataFrame(detail_rows)

    if detail.empty:
        return detail, pd.DataFrame(), pd.DataFrame()

    summary_rows = []

    for (group, indicator), s in detail.groupby(
        ["indicator_group", "indicator"],
        dropna=False,
    ):
        leads = s.loc[s["detected"] == 1, "lead_time"].astype(float).to_numpy()
        ci_low, ci_high = bootstrap_ci(leads)

        summary_rows.append(
            {
                "indicator_group": group,
                "indicator": indicator,
                "n_sims": int(s["sim_id"].nunique()),
                "detection_rate": float(s["detected"].mean()),
                "lead_mean": float(np.nanmean(leads)) if len(leads) else np.nan,
                "lead_median": float(np.nanmedian(leads)) if len(leads) else np.nan,
                "lead_q25": float(np.nanpercentile(leads, 25)) if len(leads) else np.nan,
                "lead_q75": float(np.nanpercentile(leads, 75)) if len(leads) else np.nan,
                "lead_mean_ci_low": ci_low,
                "lead_mean_ci_high": ci_high,
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["indicator_group", "lead_median", "detection_rate"],
        ascending=[True, False, False],
    )

    stage_rows = []

    for (group, sim_id), s in detail[detail["detected"] == 1].groupby(
        ["indicator_group", "sim_id"]
    ):
        idx = s["t_alarm"].astype(float).idxmin()
        row = s.loc[idx]

        stage_rows.append(
            {
                "sim_id": sim_id,
                "indicator_group": group,
                "t_alarm_group": row["t_alarm"],
                "lead_time_group": row["lead_time"],
                "best_indicator": row["indicator"],
                "event_time": row["event_time"],
            }
        )

    stage_detail = pd.DataFrame(stage_rows)

    return detail, summary, stage_detail


def summarize_stage(stage_detail: pd.DataFrame) -> pd.DataFrame:
    if stage_detail.empty:
        return pd.DataFrame()

    rows = []

    for group, s in stage_detail.groupby("indicator_group"):
        leads = s["lead_time_group"].astype(float).to_numpy()
        ci_low, ci_high = bootstrap_ci(leads)

        rows.append(
            {
                "indicator_group": group,
                "n_detected_sims": int(s["sim_id"].nunique()),
                "lead_mean": float(np.nanmean(leads)),
                "lead_median": float(np.nanmedian(leads)),
                "lead_q25": float(np.nanpercentile(leads, 25)),
                "lead_q75": float(np.nanpercentile(leads, 75)),
                "lead_mean_ci_low": ci_low,
                "lead_mean_ci_high": ci_high,
            }
        )

    order = {
        "Prepatch": 0,
        "PWSI": 1,
        "Dynamic patch": 2,
        "Visible patch": 3,
        "Traditional EWS": 4,
    }

    out = pd.DataFrame(rows)
    out["_order"] = out["indicator_group"].map(order).fillna(99)

    return out.sort_values("_order").drop(columns=["_order"])


def summarize_order(stage_detail: pd.DataFrame) -> pd.DataFrame:
    if stage_detail.empty:
        return pd.DataFrame()

    pivot = stage_detail.pivot_table(
        index="sim_id",
        columns="indicator_group",
        values="t_alarm_group",
        aggfunc="min",
    )

    rows = []
    total = len(pivot)

    patterns = [
        ("Prepatch before Visible patch", ["Prepatch", "Visible patch"]),
        ("PWSI before Visible patch", ["PWSI", "Visible patch"]),
        ("Dynamic patch before Visible patch", ["Dynamic patch", "Visible patch"]),
        (
            "Prepatch before Dynamic before Visible",
            ["Prepatch", "Dynamic patch", "Visible patch"],
        ),
        (
            "PWSI before Dynamic before Visible",
            ["PWSI", "Dynamic patch", "Visible patch"],
        ),
    ]

    for name, cols in patterns:
        if all(c in pivot.columns for c in cols):
            valid = pivot[cols].dropna()

            if len(valid):
                ok = np.ones(len(valid), dtype=bool)

                for a, b in zip(cols[:-1], cols[1:]):
                    ok &= valid[a].to_numpy() <= valid[b].to_numpy()

                prob = float(np.mean(ok))
                n_valid = int(len(valid))
            else:
                prob = np.nan
                n_valid = 0
        else:
            prob = np.nan
            n_valid = 0

        rows.append(
            {
                "order_test": name,
                "n_valid_sims": n_valid,
                "n_total_sims": total,
                "ordering_probability": prob,
            }
        )

    return pd.DataFrame(rows)


def plot_trajectories(
    df: pd.DataFrame,
    out_path: Path,
    max_abs_rel_time: int = 80,
) -> None:
    plot_cols = [
        "prepatch_z_sync",
        "prepatch_z_conn",
        "prepatch_z_rigid",
        "prepatch_z_mode",
        "pwsi_pwsi_equal",
        "PWSI",
        "PSII",
        "PDSI",
        "DPCI",
        "DPCR",
        "visible_patch_metric",
        "largest_patch_ratio",
        "patch_count",
    ]

    existing = [c for c in plot_cols if c in df.columns]

    for c in df.columns:
        lc = c.lower()

        if len(existing) >= 10:
            break

        if any(tok in lc for tok in ["z_sync", "z_conn", "z_mode", "pwsi"]):
            if c not in existing:
                existing.append(c)

    if not existing:
        return

    tmp = df[
        (df["relative_time"] >= -max_abs_rel_time)
        & (df["relative_time"] <= 5)
    ].copy()

    if tmp.empty:
        return

    plt.figure(figsize=(11, 6))

    for col in existing[:10]:
        s = tmp.groupby("relative_time")[col].mean(numeric_only=True)
        y = s.to_numpy(dtype=float)

        if np.nanstd(y) > 1e-12:
            y = (y - np.nanmean(y)) / np.nanstd(y)

        plt.plot(s.index.to_numpy(), y, label=col)

    plt.axvline(0, linestyle="--", linewidth=1)
    plt.xlabel("Relative time to transition")
    plt.ylabel("Standardized indicator value")
    plt.title("Integrated patch dynamics trajectory")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_stage_leads(stage_detail: pd.DataFrame, out_path: Path) -> None:
    if stage_detail.empty:
        return

    groups = [
        "Prepatch",
        "PWSI",
        "Dynamic patch",
        "Visible patch",
        "Traditional EWS",
    ]

    data = []
    labels = []

    for group in groups:
        vals = stage_detail.loc[
            stage_detail["indicator_group"] == group,
            "lead_time_group",
        ].dropna().to_numpy()

        if len(vals):
            data.append(vals)
            labels.append(group)

    if not data:
        return

    plt.figure(figsize=(9, 5))
    plt.boxplot(data, tick_labels=labels, showmeans=True)
    plt.ylabel("Lead time to transition")
    plt.title("First-rise lead time by indicator group")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_ordering(order_summary: pd.DataFrame, out_path: Path) -> None:
    if order_summary.empty:
        return

    s = order_summary.dropna(subset=["ordering_probability"])

    if s.empty:
        return

    plt.figure(figsize=(9, 5))
    x = np.arange(len(s))
    plt.bar(x, s["ordering_probability"].to_numpy())
    plt.xticks(x, s["order_test"].to_list(), rotation=25, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Ordering probability")
    plt.title("Patch formation order tests")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--system", default="vegetation", choices=["vegetation", "seir", "custom"])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--base-npz", required=True)
    parser.add_argument("--prepatch-npz", default="")
    parser.add_argument("--pwsi-npz", default="")

    parser.add_argument("--outdir", default="")

    parser.add_argument("--patch-threshold", type=float, default=0.35)
    parser.add_argument("--patch-direction", choices=["low", "high"], default="low")
    parser.add_argument("--connectivity", type=int, choices=[4, 8], default=8)

    parser.add_argument("--baseline-start", type=int, default=-80)
    parser.add_argument("--baseline-end", type=int, default=-50)
    parser.add_argument("--persistent-k", type=int, default=3)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--min-baseline-points", type=int, default=5)

    parser.add_argument("--strict-alignment", action="store_true")

    args = parser.parse_args()

    if not args.outdir:
        args.outdir = f"results/v2_4_integrated_patch_dynamics/{args.system}_seed{args.seed}"

    outdir = Path(args.outdir)
    figdir = outdir / "figures"

    outdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    log("loading base dataset")
    base = load_npz(args.base_npz)

    if base is None:
        raise FileNotFoundError(args.base_npz)

    log("building visible patch time series")
    df = build_base_dataframe(
        base=base,
        system=args.system,
        seed=args.seed,
        patch_threshold=args.patch_threshold,
        patch_direction=args.patch_direction,
        connectivity=args.connectivity,
    )

    prepatch = load_npz(args.prepatch_npz) if args.prepatch_npz else None
    pwsi = load_npz(args.pwsi_npz) if args.pwsi_npz else None

    df = merge_feature_file(
        df,
        prepatch,
        prefix="prepatch",
        strict_alignment=args.strict_alignment,
    )

    df = merge_feature_file(
        df,
        pwsi,
        prefix="pwsi",
        strict_alignment=args.strict_alignment,
    )

    log("computing dynamic patch indicators")

    bins = np.asarray(
        [0.0, 0.001, 0.003, 0.006, 0.01, 0.02, 0.05, 0.10, 0.20, 1.0]
    )

    df = compute_dynamic_patch_indicators(df, bins=bins)

    indicator_cols = candidate_indicator_columns(df)

    log(f"candidate indicators: {len(indicator_cols)}")

    detail, summary, stage_detail = compute_first_rise(
        df=df,
        indicator_cols=indicator_cols,
        baseline_start=args.baseline_start,
        baseline_end=args.baseline_end,
        persistent_k=args.persistent_k,
        sigma=args.sigma,
        min_baseline_points=args.min_baseline_points,
    )

    stage_summary = summarize_stage(stage_detail)
    order_summary = summarize_order(stage_detail)

    df.drop(columns=["_component_sizes"], errors="ignore").to_csv(
        outdir / "integrated_patch_dynamics_timeseries.csv",
        index=False,
        encoding="utf-8-sig",
    )

    detail.to_csv(
        outdir / "first_rise_all_indicators_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        outdir / "first_rise_all_indicators_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    stage_detail.to_csv(
        outdir / "formation_stage_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )

    stage_summary.to_csv(
        outdir / "formation_stage_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    order_summary.to_csv(
        outdir / "formation_order_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    config = vars(args).copy()
    config["n_timeseries_rows"] = int(len(df))
    config["n_simulations"] = int(df["sim_id"].nunique())
    config["n_indicators"] = int(len(indicator_cols))
    config["indicator_columns"] = indicator_cols

    (outdir / "run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    plot_trajectories(
        df,
        figdir / "integrated_patch_dynamics_trajectory.png",
    )

    plot_stage_leads(
        stage_detail,
        figdir / "formation_stage_lead_boxplot.png",
    )

    plot_ordering(
        order_summary,
        figdir / "formation_order_probability.png",
    )

    log("Formation-stage summary:")

    if not stage_summary.empty:
        print(stage_summary.to_string(index=False))
    else:
        print("(empty)")

    log("Formation-order summary:")

    if not order_summary.empty:
        print(order_summary.to_string(index=False))
    else:
        print("(empty)")

    log(f"saved results to: {outdir}")


if __name__ == "__main__":
    main()
