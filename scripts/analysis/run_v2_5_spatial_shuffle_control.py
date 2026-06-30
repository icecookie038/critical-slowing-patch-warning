# -*- coding: utf-8 -*-
"""
v2.5A Spatial-shuffle control for patch formation process.

Purpose:
    Test whether prepatch / dynamic / visible patch signals depend on spatial
    organization rather than only marginal value distribution.

This script:
    1. Loads an existing NPZ dataset.
    2. Recomputes a compact set of spatial indicators from image frames.
    3. Creates spatial-shuffled controls by permuting pixels within each frame.
    4. Compares original vs shuffled first-rise lead time and detection rate.

It does not modify existing data or model files.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def log(msg: str) -> None:
    print(f"[v2.5A] {msg}", flush=True)


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def pick_key(data: dict, keys: Sequence[str]) -> Optional[str]:
    for k in keys:
        if k in data:
            return k
    return None


def extract_last_frame(x_img: np.ndarray) -> np.ndarray:
    x_img = np.asarray(x_img)

    if x_img.ndim == 5:
        return x_img[:, -1, 0, :, :]
    if x_img.ndim == 4:
        return x_img[:, -1, :, :]
    if x_img.ndim == 3:
        return x_img

    raise ValueError(f"Unsupported image array shape: {x_img.shape}")


def get_vector(data: dict, keys: Sequence[str], n: int, fill=np.nan) -> np.ndarray:
    key = pick_key(data, keys)
    if key is None:
        return np.full(n, fill)

    arr = np.asarray(data[key])

    if arr.ndim == 0:
        return np.full(n, arr.item())

    if len(arr) != n:
        return np.full(n, fill)

    return arr


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

    sizes = []

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


def edge_density(mask: np.ndarray) -> float:
    mask = np.asarray(mask, dtype=bool)
    edges = np.sum(mask[:, 1:] != mask[:, :-1])
    edges += np.sum(mask[1:, :] != mask[:-1, :])
    return float(edges / max(1, mask.size))


def neighbor_corr(frame: np.ndarray) -> float:
    frame = np.asarray(frame, dtype=float)

    pairs = [
        (frame[:-1, :].ravel(), frame[1:, :].ravel()),
        (frame[:, :-1].ravel(), frame[:, 1:].ravel()),
    ]

    vals = []

    for a, b in pairs:
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            continue
        vals.append(np.corrcoef(a, b)[0, 1])

    if not vals:
        return 0.0

    return float(np.nanmean(vals))


def moran_i(frame: np.ndarray) -> float:
    x = np.asarray(frame, dtype=float)
    h, w = x.shape
    mean = np.mean(x)
    z = x - mean
    denom = np.sum(z ** 2)

    if denom < 1e-12:
        return 0.0

    num = 0.0
    weight = 0

    # 4-neighbor Moran's I
    for dx, dy in [(1, 0), (0, 1)]:
        a = z[:h - dx, :w - dy]
        b = z[dx:, dy:]
        num += np.sum(a * b) * 2.0
        weight += a.size * 2

    n = x.size

    if weight == 0:
        return 0.0

    return float((n / weight) * (num / denom))


def gradient_top10_mean(frame: np.ndarray) -> float:
    frame = np.asarray(frame, dtype=float)

    gx = np.diff(frame, axis=1)
    gy = np.diff(frame, axis=0)

    g = np.concatenate([np.abs(gx).ravel(), np.abs(gy).ravel()])

    if g.size == 0:
        return 0.0

    q = np.percentile(g, 90)
    top = g[g >= q]

    if top.size == 0:
        return 0.0

    return float(np.mean(top))


def sync_edge_ratio(frame: np.ndarray, direction: str, threshold: float) -> float:
    x = np.asarray(frame, dtype=float)

    if direction == "high":
        mask = x >= threshold
    else:
        mask = x <= threshold

    a1 = mask[:-1, :]
    b1 = mask[1:, :]
    a2 = mask[:, :-1]
    b2 = mask[:, 1:]

    total = a1.size + a2.size
    if total == 0:
        return 0.0

    same_active = np.sum(a1 & b1) + np.sum(a2 & b2)

    return float(same_active / total)


def visible_metrics(frame: np.ndarray, direction: str, threshold: float, connectivity: int) -> dict:
    if direction == "high":
        mask = frame >= threshold
    else:
        mask = frame <= threshold

    sizes = connected_components(mask, connectivity=connectivity)
    total = mask.size
    area = int(np.sum(mask))
    largest = max(sizes) if sizes else 0
    patch_count = len(sizes)

    return {
        "patch_count": float(patch_count),
        "total_patch_ratio": float(area / total),
        "largest_patch_ratio": float(largest / total),
        "edge_density": edge_density(mask),
        "visible_patch_metric": float(largest / total),
    }


def build_timeseries(
    frames: np.ndarray,
    sim_id: np.ndarray,
    time_idx: np.ndarray,
    critical_time: np.ndarray,
    system: str,
    seed: int,
    variant: str,
    shuffle_run: int,
    direction: str,
    threshold: float,
    connectivity: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []

    for i, frame in enumerate(frames):
        f = np.asarray(frame, dtype=float)

        if variant == "spatial_shuffle":
            flat = f.ravel().copy()
            rng.shuffle(flat)
            f = flat.reshape(f.shape)

        vm = visible_metrics(
            f,
            direction=direction,
            threshold=threshold,
            connectivity=connectivity,
        )

        row = {
            "system": system,
            "seed": seed,
            "variant": variant,
            "shuffle_run": shuffle_run,
            "row_id": i,
            "sim_id": int(sim_id[i]),
            "time_idx": int(time_idx[i]),
            "critical_time": float(critical_time[i]) if np.isfinite(critical_time[i]) else np.nan,
            "prepatch_moran_i": moran_i(f),
            "prepatch_local_neighbor_corr_mean": neighbor_corr(f),
            "prepatch_sync_edge_ratio": sync_edge_ratio(f, direction, threshold),
            "prepatch_gradient_top10_mean": gradient_top10_mean(f),
        }

        row.update(vm)
        rows.append(row)

    df = pd.DataFrame(rows)

    before = len(df)
    df = (
        df.sort_values(["variant", "shuffle_run", "sim_id", "time_idx", "row_id"])
        .drop_duplicates(["variant", "shuffle_run", "sim_id", "time_idx"], keep="last")
    )

    if len(df) < before:
        log(f"deduplicated rows: {before} -> {len(df)}")

    df["critical_time"] = df.groupby(
        ["variant", "shuffle_run", "sim_id"]
    )["critical_time"].transform(
        lambda s: s.dropna().iloc[0] if s.dropna().size else np.nan
    )

    df["relative_time"] = df["time_idx"] - df["critical_time"]

    return df.reset_index(drop=True)


def add_dynamic_metrics(df: pd.DataFrame) -> pd.DataFrame:
    all_rows = []

    group_cols = ["variant", "shuffle_run", "sim_id"]

    for _, g in df.groupby(group_cols, sort=False):
        g = g.sort_values("time_idx").copy().reset_index(drop=True)

        n = len(g)

        pdsi = np.zeros(n)
        dpci = np.zeros(n)
        fpv = np.zeros(n)
        psii = np.zeros(n)

        prev_largest = None
        prev_area = None
        prev_count = None

        for i in range(n):
            largest = float(g.loc[i, "largest_patch_ratio"])
            area = float(g.loc[i, "total_patch_ratio"])
            count = float(g.loc[i, "patch_count"])

            if prev_largest is not None:
                dpci[i] = max(0.0, largest - prev_largest)

            if prev_area is not None:
                fpv[i] = max(0.0, area - prev_area)

            if prev_count is not None:
                pdsi[i] = abs(count - prev_count)

            prev_largest = largest
            prev_area = area
            prev_count = count

        g["DPCI"] = dpci
        g["FPV"] = fpv
        g["PDSI"] = pdsi

        # Simple patch-structure instability index.
        dyn_cols = ["DPCI", "FPV", "PDSI", "edge_density"]
        z_parts = []

        for c in dyn_cols:
            x = g[c].astype(float).to_numpy()
            sd = np.nanstd(x)
            if sd < 1e-12:
                z = np.zeros_like(x)
            else:
                z = (x - np.nanmean(x)) / sd
            z_parts.append(np.maximum(0, z))

        g["PSII"] = np.mean(np.vstack(z_parts), axis=0)

        all_rows.append(g)

    return pd.concat(all_rows, ignore_index=True)


def add_pwsi(df: pd.DataFrame) -> pd.DataFrame:
    all_rows = []

    cols = [
        "prepatch_moran_i",
        "prepatch_local_neighbor_corr_mean",
        "prepatch_sync_edge_ratio",
        "prepatch_gradient_top10_mean",
    ]

    for _, g in df.groupby(["variant", "shuffle_run", "sim_id"], sort=False):
        g = g.sort_values("time_idx").copy().reset_index(drop=True)

        zs = []

        for c in cols:
            x = g[c].astype(float).to_numpy()
            sd = np.nanstd(x)
            if sd < 1e-12:
                z = np.zeros_like(x)
            else:
                z = (x - np.nanmean(x)) / sd
            zs.append(z)

        g["pwsi_control"] = np.mean(np.vstack(zs), axis=0)
        all_rows.append(g)

    return pd.concat(all_rows, ignore_index=True)


def first_persistent_alarm(times, values, threshold, k) -> Optional[int]:
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


def compute_first_rise(
    df: pd.DataFrame,
    indicators: Sequence[str],
    baseline_start: int,
    baseline_end: int,
    sigma: float,
    persistent_k: int,
    min_baseline_points: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []

    group_cols = ["variant", "shuffle_run", "sim_id"]

    for key, g in df.groupby(group_cols, sort=False):
        variant, shuffle_run, sim = key
        g = g.sort_values("time_idx").reset_index(drop=True)

        event_vals = g["critical_time"].dropna().unique()
        if len(event_vals) == 0:
            continue

        event_time = int(event_vals[0])

        baseline = g[
            (g["relative_time"] >= baseline_start)
            & (g["relative_time"] <= baseline_end)
        ]

        pre_event = g[g["time_idx"] < event_time]

        if len(baseline) < min_baseline_points:
            m = max(min_baseline_points, int(math.ceil(len(pre_event) * 0.25)))
            baseline = pre_event.head(m)

        for ind in indicators:
            xbase = baseline[ind].astype(float).to_numpy()
            xbase = xbase[np.isfinite(xbase)]

            if len(xbase) < 3:
                continue

            mu = float(np.mean(xbase))
            sd = float(np.std(xbase, ddof=1))
            if not np.isfinite(sd) or sd < 1e-12:
                sd = 1e-12

            th = mu + sigma * sd
            window = g[g["time_idx"] < event_time]

            alarm = first_persistent_alarm(
                window["time_idx"].to_numpy(),
                window[ind].astype(float).to_numpy(),
                th,
                persistent_k,
            )

            detected = int(alarm is not None)
            lead = np.nan if alarm is None else event_time - alarm

            rows.append({
                "variant": variant,
                "shuffle_run": int(shuffle_run),
                "sim_id": int(sim),
                "indicator": ind,
                "event_time": event_time,
                "t_alarm": alarm if alarm is not None else np.nan,
                "detected": detected,
                "lead_time": lead,
                "threshold": th,
            })

    detail = pd.DataFrame(rows)

    summary_rows = []

    if detail.empty:
        return detail, pd.DataFrame()

    for (variant, ind), s in detail.groupby(["variant", "indicator"]):
        detected = s[s["detected"] == 1]
        leads = detected["lead_time"].astype(float).to_numpy()

        summary_rows.append({
            "variant": variant,
            "indicator": ind,
            "n_sims": int(s["sim_id"].nunique()),
            "detection_rate": float(s["detected"].mean()),
            "lead_mean": float(np.nanmean(leads)) if len(leads) else np.nan,
            "lead_median": float(np.nanmedian(leads)) if len(leads) else np.nan,
            "lead_q25": float(np.nanpercentile(leads, 25)) if len(leads) else np.nan,
            "lead_q75": float(np.nanpercentile(leads, 75)) if len(leads) else np.nan,
        })

    summary = pd.DataFrame(summary_rows)

    return detail, summary


def stage_summary(detail: pd.DataFrame) -> pd.DataFrame:
    stages = {
        "prepatch_spatial_organization": [
            "prepatch_moran_i",
            "prepatch_local_neighbor_corr_mean",
            "prepatch_sync_edge_ratio",
            "prepatch_gradient_top10_mean",
        ],
        "pwsi_control_index": [
            "pwsi_control",
        ],
        "dynamic_patch": [
            "PSII",
            "DPCI",
            "FPV",
            "PDSI",
        ],
        "visible_patch": [
            "patch_count",
            "edge_density",
            "largest_patch_ratio",
            "visible_patch_metric",
        ],
    }

    rows = []

    for (variant, shuffle_run), d in detail.groupby(["variant", "shuffle_run"]):
        for stage, inds in stages.items():
            sub = d[(d["indicator"].isin(inds)) & (d["detected"] == 1)].copy()

            if sub.empty:
                rows.append({
                    "variant": variant,
                    "shuffle_run": shuffle_run,
                    "stage": stage,
                    "n_detected_sims": 0,
                    "detection_rate": 0.0,
                    "lead_mean": np.nan,
                    "lead_median": np.nan,
                    "lead_q25": np.nan,
                    "lead_q75": np.nan,
                })
                continue

            idx = sub.groupby("sim_id")["t_alarm"].idxmin()
            earliest = sub.loc[idx].copy()

            leads = earliest["lead_time"].astype(float).to_numpy()
            n_total = d["sim_id"].nunique()
            n_detected = earliest["sim_id"].nunique()

            rows.append({
                "variant": variant,
                "shuffle_run": int(shuffle_run),
                "stage": stage,
                "n_detected_sims": int(n_detected),
                "n_total_sims": int(n_total),
                "detection_rate": float(n_detected / n_total),
                "lead_mean": float(np.nanmean(leads)) if len(leads) else np.nan,
                "lead_median": float(np.nanmedian(leads)) if len(leads) else np.nan,
                "lead_q25": float(np.nanpercentile(leads, 25)) if len(leads) else np.nan,
                "lead_q75": float(np.nanpercentile(leads, 75)) if len(leads) else np.nan,
            })

    return pd.DataFrame(rows)


def compare_original_shuffle(stage_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for stage, s in stage_df.groupby("stage"):
        original = s[s["variant"] == "original"]
        shuffle = s[s["variant"] == "spatial_shuffle"]

        if original.empty or shuffle.empty:
            continue

        rows.append({
            "stage": stage,
            "original_detection_rate": float(original["detection_rate"].mean()),
            "shuffle_detection_rate_mean": float(shuffle["detection_rate"].mean()),
            "shuffle_detection_rate_std": float(shuffle["detection_rate"].std(ddof=1)),
            "original_lead_median": float(original["lead_median"].mean()),
            "shuffle_lead_median_mean": float(shuffle["lead_median"].mean()),
            "shuffle_lead_median_std": float(shuffle["lead_median"].std(ddof=1)),
            "detection_rate_drop": float(
                original["detection_rate"].mean() - shuffle["detection_rate"].mean()
            ),
            "lead_median_drop": float(
                original["lead_median"].mean() - shuffle["lead_median"].mean()
            ),
        })

    return pd.DataFrame(rows)


def plot_control(stage_df: pd.DataFrame, out_path: Path) -> None:
    stages = [
        "prepatch_spatial_organization",
        "pwsi_control_index",
        "dynamic_patch",
        "visible_patch",
    ]

    tmp = stage_df[stage_df["stage"].isin(stages)].copy()

    if tmp.empty:
        return

    plt.figure(figsize=(10, 5))

    labels = []
    data = []

    for stage in stages:
        for variant in ["original", "spatial_shuffle"]:
            vals = tmp[
                (tmp["stage"] == stage) & (tmp["variant"] == variant)
            ]["lead_median"].dropna().to_numpy()

            if len(vals):
                data.append(vals)
                labels.append(f"{stage}\n{variant}")

    if not data:
        return

    plt.boxplot(data, tick_labels=labels, showmeans=True)
    plt.ylabel("Median lead time")
    plt.title("Spatial-shuffle control for patch-formation indicators")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--system", required=True, choices=["vegetation", "seir"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--base-npz", required=True)
    parser.add_argument("--patch-direction", required=True, choices=["low", "high"])
    parser.add_argument("--patch-threshold", type=float, required=True)
    parser.add_argument("--outdir", required=True)

    parser.add_argument("--n-shuffles", type=int, default=5)
    parser.add_argument("--shuffle-seed", type=int, default=2026)
    parser.add_argument("--connectivity", type=int, choices=[4, 8], default=8)

    parser.add_argument("--baseline-start", type=int, default=-80)
    parser.add_argument("--baseline-end", type=int, default=-50)
    parser.add_argument("--persistent-k", type=int, default=3)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--min-baseline-points", type=int, default=5)

    args = parser.parse_args()

    outdir = Path(args.outdir)
    figdir = outdir / "figures"

    outdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    data = load_npz(Path(args.base_npz))

    x_key = pick_key(data, ["X_img", "x_img", "images", "X", "frames"])
    if x_key is None:
        raise KeyError("Cannot find image array key in NPZ.")

    frames = extract_last_frame(data[x_key])
    n = len(frames)

    sim_id = get_vector(data, ["sim_id", "sim_ids", "simulation_id", "simulation_ids"], n)
    time_idx = get_vector(data, ["time_idx", "time_indices", "t_idx", "t", "time"], n)
    critical_time = get_vector(
        data,
        ["critical_time", "critical_times", "event_time", "event_times", "transition_time", "transition_times"],
        n,
    )

    if np.any(~np.isfinite(sim_id)) or np.any(~np.isfinite(time_idx)):
        raise ValueError("sim_id/time_idx missing; cannot run control safely.")

    variants = []

    rng = np.random.default_rng(args.shuffle_seed)

    log("computing original indicators")

    variants.append(
        build_timeseries(
            frames=frames,
            sim_id=sim_id,
            time_idx=time_idx,
            critical_time=critical_time,
            system=args.system,
            seed=args.seed,
            variant="original",
            shuffle_run=0,
            direction=args.patch_direction,
            threshold=args.patch_threshold,
            connectivity=args.connectivity,
            rng=rng,
        )
    )

    for r in range(args.n_shuffles):
        log(f"computing spatial shuffle run {r + 1}/{args.n_shuffles}")

        variants.append(
            build_timeseries(
                frames=frames,
                sim_id=sim_id,
                time_idx=time_idx,
                critical_time=critical_time,
                system=args.system,
                seed=args.seed,
                variant="spatial_shuffle",
                shuffle_run=r + 1,
                direction=args.patch_direction,
                threshold=args.patch_threshold,
                connectivity=args.connectivity,
                rng=rng,
            )
        )

    df = pd.concat(variants, ignore_index=True)
    df = add_dynamic_metrics(df)
    df = add_pwsi(df)

    indicators = [
        "prepatch_moran_i",
        "prepatch_local_neighbor_corr_mean",
        "prepatch_sync_edge_ratio",
        "prepatch_gradient_top10_mean",
        "pwsi_control",
        "PSII",
        "DPCI",
        "FPV",
        "PDSI",
        "patch_count",
        "edge_density",
        "largest_patch_ratio",
        "visible_patch_metric",
    ]

    detail, summary = compute_first_rise(
        df=df,
        indicators=indicators,
        baseline_start=args.baseline_start,
        baseline_end=args.baseline_end,
        sigma=args.sigma,
        persistent_k=args.persistent_k,
        min_baseline_points=args.min_baseline_points,
    )

    stage = stage_summary(detail)
    effect = compare_original_shuffle(stage)

    df.to_csv(outdir / "spatial_shuffle_timeseries.csv", index=False, encoding="utf-8-sig")
    detail.to_csv(outdir / "spatial_shuffle_first_rise_detail.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(outdir / "spatial_shuffle_indicator_summary.csv", index=False, encoding="utf-8-sig")
    stage.to_csv(outdir / "spatial_shuffle_stage_summary.csv", index=False, encoding="utf-8-sig")
    effect.to_csv(outdir / "spatial_shuffle_effect_summary.csv", index=False, encoding="utf-8-sig")

    config = vars(args)
    (outdir / "run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    plot_control(stage, figdir / "spatial_shuffle_control_lead_boxplot.png")

    log("Spatial-shuffle stage summary:")
    print(stage.to_string(index=False))

    log("Spatial-shuffle effect summary:")
    print(effect.to_string(index=False))

    log(f"saved results to: {outdir}")


if __name__ == "__main__":
    main()
