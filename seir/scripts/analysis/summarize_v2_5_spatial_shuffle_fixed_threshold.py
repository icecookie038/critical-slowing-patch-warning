# -*- coding: utf-8 -*-
"""
v2.5A fixed-threshold spatial-shuffle summary.

Important correction:
    Original and spatial-shuffle variants should not use their own separate
    alarm thresholds. The threshold is estimated from the original baseline
    trajectory and then applied to both original and shuffled variants.

This tests whether the original spatial precursor threshold still works after
spatial organization is destroyed.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd


INDICATORS = [
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


STAGES = {
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


RUNS = [
    (
        "vegetation",
        42,
        "0p50",
        Path("results/v2_5_controls/spatial_shuffle/vegetation_seed42_thr0p50"),
    ),
    (
        "vegetation",
        123,
        "0p50",
        Path("results/v2_5_controls/spatial_shuffle/vegetation_seed123_thr0p50"),
    ),
    (
        "seir",
        2026,
        "0p20",
        Path("results/v2_5_controls/spatial_shuffle/seir_seed2026_thr0p20"),
    ),
]


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


def compute_original_thresholds(
    df: pd.DataFrame,
    baseline_start: int,
    baseline_end: int,
    sigma: float,
    min_baseline_points: int,
) -> pd.DataFrame:
    rows = []

    original = df[(df["variant"] == "original") & (df["shuffle_run"] == 0)].copy()

    for sim_id, g in original.groupby("sim_id"):
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

        for ind in INDICATORS:
            if ind not in baseline.columns:
                continue

            x = baseline[ind].astype(float).to_numpy()
            x = x[np.isfinite(x)]

            if len(x) < 3:
                continue

            mu = float(np.mean(x))
            sd = float(np.std(x, ddof=1))

            if not np.isfinite(sd) or sd < 1e-12:
                sd = 1e-12

            rows.append({
                "sim_id": int(sim_id),
                "indicator": ind,
                "event_time": event_time,
                "original_baseline_mean": mu,
                "original_baseline_std": sd,
                "fixed_threshold": mu + sigma * sd,
            })

    return pd.DataFrame(rows)


def compute_fixed_first_rise(
    df: pd.DataFrame,
    thresholds: pd.DataFrame,
    persistent_k: int,
) -> pd.DataFrame:
    rows = []

    threshold_map = {
        (int(r["sim_id"]), r["indicator"]): float(r["fixed_threshold"])
        for _, r in thresholds.iterrows()
    }

    event_map = {
        int(r["sim_id"]): int(r["event_time"])
        for _, r in thresholds.drop_duplicates("sim_id").iterrows()
    }

    for (variant, shuffle_run, sim_id), g in df.groupby(
        ["variant", "shuffle_run", "sim_id"],
        sort=False,
    ):
        sim_id = int(sim_id)

        if sim_id not in event_map:
            continue

        event_time = event_map[sim_id]
        g = g.sort_values("time_idx").reset_index(drop=True)
        window = g[g["time_idx"] < event_time]

        for ind in INDICATORS:
            if ind not in g.columns:
                continue

            key = (sim_id, ind)

            if key not in threshold_map:
                continue

            alarm = first_persistent_alarm(
                times=window["time_idx"].to_numpy(),
                values=window[ind].astype(float).to_numpy(),
                threshold=threshold_map[key],
                k=persistent_k,
            )

            detected = int(alarm is not None)
            lead = np.nan if alarm is None else event_time - alarm

            rows.append({
                "variant": variant,
                "shuffle_run": int(shuffle_run),
                "sim_id": sim_id,
                "indicator": ind,
                "event_time": event_time,
                "t_alarm": alarm if alarm is not None else np.nan,
                "detected": detected,
                "lead_time": lead,
                "fixed_threshold": threshold_map[key],
            })

    return pd.DataFrame(rows)


def summarize_stage(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (variant, shuffle_run), d in detail.groupby(["variant", "shuffle_run"]):
        n_total = d["sim_id"].nunique()

        for stage, inds in STAGES.items():
            sub = d[(d["indicator"].isin(inds)) & (d["detected"] == 1)].copy()

            if sub.empty:
                rows.append({
                    "variant": variant,
                    "shuffle_run": int(shuffle_run),
                    "stage": stage,
                    "n_total_sims": int(n_total),
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
            n_detected = earliest["sim_id"].nunique()

            rows.append({
                "variant": variant,
                "shuffle_run": int(shuffle_run),
                "stage": stage,
                "n_total_sims": int(n_total),
                "n_detected_sims": int(n_detected),
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
                original["detection_rate"].mean()
                - shuffle["detection_rate"].mean()
            ),
            "lead_median_drop": float(
                original["lead_median"].mean()
                - shuffle["lead_median"].mean()
            ),
        })

    return pd.DataFrame(rows)


def main():
    baseline_start = -80
    baseline_end = -50
    sigma = 2.0
    persistent_k = 3
    min_baseline_points = 5

    all_effects = []
    all_stages = []

    for system, seed, threshold_tag, d in RUNS:
        ts_path = d / "spatial_shuffle_timeseries.csv"

        if not ts_path.exists():
            print(f"Skip missing: {ts_path}")
            continue

        print(f"\nProcessing {system} seed{seed} threshold={threshold_tag}")

        df = pd.read_csv(ts_path)

        thresholds = compute_original_thresholds(
            df=df,
            baseline_start=baseline_start,
            baseline_end=baseline_end,
            sigma=sigma,
            min_baseline_points=min_baseline_points,
        )

        detail = compute_fixed_first_rise(
            df=df,
            thresholds=thresholds,
            persistent_k=persistent_k,
        )

        stage = summarize_stage(detail)
        effect = compare_original_shuffle(stage)

        for x in [thresholds, detail, stage, effect]:
            x["system"] = system
            x["seed"] = seed
            x["threshold"] = threshold_tag

        thresholds.to_csv(
            d / "fixed_original_thresholds.csv",
            index=False,
            encoding="utf-8-sig",
        )

        detail.to_csv(
            d / "fixed_threshold_first_rise_detail.csv",
            index=False,
            encoding="utf-8-sig",
        )

        stage.to_csv(
            d / "fixed_threshold_stage_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

        effect.to_csv(
            d / "fixed_threshold_effect_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

        all_effects.append(effect)
        all_stages.append(stage)

    if not all_effects:
        raise SystemExit("No valid fixed-threshold results found.")

    effects = pd.concat(all_effects, ignore_index=True)
    stages = pd.concat(all_stages, ignore_index=True)

    outdir = Path("results/v2_5_controls/spatial_shuffle_fixed_threshold_summary")
    outdir.mkdir(parents=True, exist_ok=True)

    effects.to_csv(
        outdir / "fixed_threshold_effect_all_systems.csv",
        index=False,
        encoding="utf-8-sig",
    )

    stages.to_csv(
        outdir / "fixed_threshold_stage_all_systems.csv",
        index=False,
        encoding="utf-8-sig",
    )

    cols = [
        "system",
        "seed",
        "threshold",
        "stage",
        "original_detection_rate",
        "shuffle_detection_rate_mean",
        "detection_rate_drop",
        "original_lead_median",
        "shuffle_lead_median_mean",
        "lead_median_drop",
    ]

    print("\n=== Fixed-threshold spatial-shuffle effect summary ===")
    print(effects[cols].to_string(index=False))

    print("\n=== Mean fixed-threshold effect by system and stage ===")
    print(
        effects.groupby(["system", "stage"])[
            ["detection_rate_drop", "lead_median_drop"]
        ].mean().reset_index().to_string(index=False)
    )


if __name__ == "__main__":
    main()
