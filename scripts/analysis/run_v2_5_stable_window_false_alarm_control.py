# -*- coding: utf-8 -*-
"""
v2.5B Stable-window false-alarm control.

Purpose:
    Test whether patch-formation indicators produce alarms in far-from-transition
    stable windows.

This is a negative-control style analysis based on existing v2.4 integrated
patch dynamics results.

Main idea:
    - Far stable window should have low alarm rate.
    - Precursor window should have higher alarm rate.

This script does not modify datasets or model files.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd


RUNS = [
    (
        "vegetation",
        42,
        "0p50",
        Path("results/v2_4_integrated_patch_dynamics/vegetation_seed42_thr0p50"),
    ),
    (
        "vegetation",
        123,
        "0p50",
        Path("results/v2_4_integrated_patch_dynamics/vegetation_seed123_thr0p50"),
    ),
    (
        "seir",
        2026,
        "0p20",
        Path("results/v2_4_integrated_patch_dynamics/seir_seed2026_thr0p20"),
    ),
]


STAGES = {
    "prepatch_spatial_organization": [
        "prepatch_moran_i",
        "prepatch_sync_edge_ratio",
        "prepatch_local_neighbor_corr_mean",
        "prepatch_svd_spectral_gap",
        "prepatch_boundary_sharpness",
        "prepatch_gradient_top10_mean",
    ],
    "pwsi_application_index": [
        "pwsi_equal",
        "pwsi_z_conn",
        "pwsi_z_sync",
        "pwsi_z_rigid",
        "pwsi_importance",
    ],
    "dynamic_patch": [
        "PSII",
        "DPCI",
        "DPCR",
        "FPV",
        "GCGR",
        "PDSI",
        "BCI",
    ],
    "visible_patch": [
        "patch_count",
        "edge_density",
        "largest_patch_ratio",
        "total_patch_ratio",
        "visible_patch_metric",
    ],
}


WINDOWS = [
    ("far_stable_window", -120, -81),
    ("baseline_window", -80, -51),
    ("early_precursor_window", -50, -31),
    ("late_precursor_window", -30, -1),
    ("full_precursor_window", -50, -1),
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


def clean_numeric(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()

    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


def get_available_indicators(df: pd.DataFrame) -> dict:
    available = {}

    for stage, indicators in STAGES.items():
        available[stage] = [x for x in indicators if x in df.columns]

    return available


def compute_thresholds(
    df: pd.DataFrame,
    available: dict,
    sigma: float,
    min_points: int,
) -> pd.DataFrame:
    rows = []

    for sim_id, g in df.groupby("sim_id"):
        g = g.sort_values("time_idx").reset_index(drop=True)

        baseline = g[
            (g["relative_time"] >= -80)
            & (g["relative_time"] <= -51)
        ].copy()

        if len(baseline) < min_points:
            pre_event = g[g["relative_time"] < -1].copy()
            m = max(min_points, int(math.ceil(len(pre_event) * 0.25)))
            baseline = pre_event.head(m)

        if len(baseline) < 3:
            continue

        for stage, indicators in available.items():
            for ind in indicators:
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
                    "stage": stage,
                    "indicator": ind,
                    "baseline_mean": mu,
                    "baseline_std": sd,
                    "threshold": mu + sigma * sd,
                })

    return pd.DataFrame(rows)


def compute_window_alarms(
    df: pd.DataFrame,
    thresholds: pd.DataFrame,
    persistent_k: int,
) -> pd.DataFrame:
    rows = []

    threshold_map = {
        (int(r["sim_id"]), r["indicator"]): float(r["threshold"])
        for _, r in thresholds.iterrows()
    }

    stage_map = {
        r["indicator"]: r["stage"]
        for _, r in thresholds.drop_duplicates("indicator").iterrows()
    }

    for sim_id, g in df.groupby("sim_id"):
        sim_id = int(sim_id)
        g = g.sort_values("time_idx").reset_index(drop=True)

        for window_name, start, end in WINDOWS:
            w = g[
                (g["relative_time"] >= start)
                & (g["relative_time"] <= end)
            ].copy()

            if len(w) < persistent_k:
                continue

            for ind, stage in stage_map.items():
                key = (sim_id, ind)

                if key not in threshold_map or ind not in w.columns:
                    continue

                alarm_time = first_persistent_alarm(
                    times=w["time_idx"].to_numpy(),
                    values=w[ind].astype(float).to_numpy(),
                    threshold=threshold_map[key],
                    k=persistent_k,
                )

                rows.append({
                    "sim_id": sim_id,
                    "window": window_name,
                    "window_start": start,
                    "window_end": end,
                    "stage": stage,
                    "indicator": ind,
                    "detected": int(alarm_time is not None),
                    "t_alarm": alarm_time if alarm_time is not None else np.nan,
                })

    return pd.DataFrame(rows)


def summarize_stage_alarms(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (window, stage, sim_id), s in detail.groupby(["window", "stage", "sim_id"]):
        detected = int(s["detected"].max() > 0)

        if detected:
            t_alarm = float(s.loc[s["detected"] == 1, "t_alarm"].min())
            best_indicator = s.loc[s["detected"] == 1].sort_values("t_alarm").iloc[0]["indicator"]
        else:
            t_alarm = np.nan
            best_indicator = ""

        rows.append({
            "window": window,
            "stage": stage,
            "sim_id": int(sim_id),
            "detected": detected,
            "t_alarm": t_alarm,
            "best_indicator": best_indicator,
        })

    stage_detail = pd.DataFrame(rows)

    summary_rows = []

    for (window, stage), s in stage_detail.groupby(["window", "stage"]):
        summary_rows.append({
            "window": window,
            "stage": stage,
            "n_sims": int(s["sim_id"].nunique()),
            "alarm_rate": float(s["detected"].mean()),
            "n_alarms": int(s["detected"].sum()),
            "most_common_indicator": (
                s.loc[s["detected"] == 1, "best_indicator"].mode().iloc[0]
                if (s["detected"] == 1).any()
                else ""
            ),
        })

    return stage_detail, pd.DataFrame(summary_rows)


def compute_false_alarm_effect(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []

    pivot = summary.pivot_table(
        index="stage",
        columns="window",
        values="alarm_rate",
        aggfunc="mean",
    )

    for stage in pivot.index:
        far = pivot.loc[stage].get("far_stable_window", np.nan)
        early = pivot.loc[stage].get("early_precursor_window", np.nan)
        late = pivot.loc[stage].get("late_precursor_window", np.nan)
        full = pivot.loc[stage].get("full_precursor_window", np.nan)

        rows.append({
            "stage": stage,
            "far_stable_alarm_rate": far,
            "early_precursor_alarm_rate": early,
            "late_precursor_alarm_rate": late,
            "full_precursor_alarm_rate": full,
            "full_minus_far": full - far if np.isfinite(full) and np.isfinite(far) else np.nan,
            "late_minus_far": late - far if np.isfinite(late) and np.isfinite(far) else np.nan,
        })

    return pd.DataFrame(rows)


def main():
    sigma = 2.0
    persistent_k = 3
    min_points = 5

    all_stage_summary = []
    all_stage_detail = []
    all_effects = []

    for system, seed, threshold_tag, run_dir in RUNS:
        ts_path = run_dir / "integrated_patch_dynamics_timeseries.csv"

        if not ts_path.exists():
            print(f"Skip missing: {ts_path}")
            continue

        print(f"\nProcessing {system} seed{seed} threshold={threshold_tag}")

        df = pd.read_csv(ts_path)
        df = clean_numeric(
            df,
            ["sim_id", "time_idx", "critical_time", "relative_time"],
        )

        available = get_available_indicators(df)

        thresholds = compute_thresholds(
            df=df,
            available=available,
            sigma=sigma,
            min_points=min_points,
        )

        detail = compute_window_alarms(
            df=df,
            thresholds=thresholds,
            persistent_k=persistent_k,
        )

        stage_detail, stage_summary = summarize_stage_alarms(detail)
        effect = compute_false_alarm_effect(stage_summary)

        for x in [thresholds, detail, stage_detail, stage_summary, effect]:
            x["system"] = system
            x["seed"] = seed
            x["threshold"] = threshold_tag

        outdir = Path("results/v2_5_controls/stable_window_false_alarm") / f"{system}_seed{seed}_thr{threshold_tag}"
        outdir.mkdir(parents=True, exist_ok=True)

        thresholds.to_csv(outdir / "stable_window_thresholds.csv", index=False, encoding="utf-8-sig")
        detail.to_csv(outdir / "stable_window_indicator_alarm_detail.csv", index=False, encoding="utf-8-sig")
        stage_detail.to_csv(outdir / "stable_window_stage_alarm_detail.csv", index=False, encoding="utf-8-sig")
        stage_summary.to_csv(outdir / "stable_window_stage_alarm_summary.csv", index=False, encoding="utf-8-sig")
        effect.to_csv(outdir / "stable_window_false_alarm_effect.csv", index=False, encoding="utf-8-sig")

        all_stage_summary.append(stage_summary)
        all_stage_detail.append(stage_detail)
        all_effects.append(effect)

    if not all_stage_summary:
        raise SystemExit("No valid v2.4 timeseries files found.")

    summary_all = pd.concat(all_stage_summary, ignore_index=True)
    detail_all = pd.concat(all_stage_detail, ignore_index=True)
    effect_all = pd.concat(all_effects, ignore_index=True)

    outdir = Path("results/v2_5_controls/stable_window_false_alarm_summary")
    outdir.mkdir(parents=True, exist_ok=True)

    summary_all.to_csv(outdir / "stable_window_stage_alarm_summary_all.csv", index=False, encoding="utf-8-sig")
    detail_all.to_csv(outdir / "stable_window_stage_alarm_detail_all.csv", index=False, encoding="utf-8-sig")
    effect_all.to_csv(outdir / "stable_window_false_alarm_effect_all.csv", index=False, encoding="utf-8-sig")

    print("\n=== Stable-window stage alarm summary ===")
    print(
        summary_all[
            ["system", "seed", "threshold", "window", "stage", "n_sims", "alarm_rate", "n_alarms", "most_common_indicator"]
        ].to_string(index=False)
    )

    print("\n=== Stable-window false-alarm effect ===")
    print(
        effect_all[
            ["system", "seed", "threshold", "stage", "far_stable_alarm_rate", "full_precursor_alarm_rate", "late_precursor_alarm_rate", "full_minus_far", "late_minus_far"]
        ].to_string(index=False)
    )

    print("\n=== Mean false-alarm effect by system and stage ===")
    print(
        effect_all.groupby(["system", "stage"])[
            ["far_stable_alarm_rate", "full_precursor_alarm_rate", "late_precursor_alarm_rate", "full_minus_far", "late_minus_far"]
        ].mean().reset_index().to_string(index=False)
    )


if __name__ == "__main__":
    main()
