# -*- coding: utf-8 -*-
"""
v2.5C Event-time permutation control.

Purpose:
    Test whether patch-formation signals are specifically aligned with the true
    transition time.

Method:
    1. Use existing v2.4 integrated_patch_dynamics_timeseries.csv.
    2. Compute stage-level alarm rate in the real precursor window.
    3. Randomly permute critical_time across simulations.
    4. Recompute alarm rates in fake precursor windows.
    5. Compare real precursor alarm rate against permuted-event alarm rate.

Interpretation:
    If real alarm rate >> permuted alarm rate, then indicators are temporally
    aligned with true transitions rather than arbitrary event times.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

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
    ("early_precursor_window", -50, -31),
    ("late_precursor_window", -30, -1),
    ("full_precursor_window", -50, -1),
]

BASELINE_START = -80
BASELINE_END = -51
SIGMA = 2.0
PERSISTENT_K = 3
N_PERMUTATIONS = 100
RANDOM_SEED = 2026


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


def available_stages(df: pd.DataFrame) -> dict:
    out = {}
    for stage, indicators in STAGES.items():
        out[stage] = [x for x in indicators if x in df.columns]
    return out


def compute_real_thresholds(df: pd.DataFrame, stages: dict) -> pd.DataFrame:
    rows = []

    for sim_id, g in df.groupby("sim_id"):
        g = g.sort_values("time_idx").reset_index(drop=True)

        baseline = g[
            (g["relative_time"] >= BASELINE_START)
            & (g["relative_time"] <= BASELINE_END)
        ]

        if len(baseline) < 3:
            continue

        for stage, indicators in stages.items():
            for ind in indicators:
                x = pd.to_numeric(baseline[ind], errors="coerce").to_numpy()
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
                    "threshold": mu + SIGMA * sd,
                })

    return pd.DataFrame(rows)


def detect_stage_window(
    df: pd.DataFrame,
    thresholds: pd.DataFrame,
    event_time_map: dict,
    use_fake_event: bool,
) -> pd.DataFrame:
    threshold_map = {
        (int(r["sim_id"]), r["indicator"]): float(r["threshold"])
        for _, r in thresholds.iterrows()
    }

    stage_map = {
        r["indicator"]: r["stage"]
        for _, r in thresholds.drop_duplicates("indicator").iterrows()
    }

    rows = []

    for sim_id, g in df.groupby("sim_id"):
        sim_id = int(sim_id)

        if sim_id not in event_time_map:
            continue

        event_time = int(event_time_map[sim_id])
        g = g.sort_values("time_idx").copy()
        g["relative_time_control"] = g["time_idx"] - event_time

        for window_name, start, end in WINDOWS:
            w = g[
                (g["relative_time_control"] >= start)
                & (g["relative_time_control"] <= end)
            ]

            if len(w) < PERSISTENT_K:
                continue

            for ind, stage in stage_map.items():
                key = (sim_id, ind)

                if key not in threshold_map or ind not in w.columns:
                    continue

                alarm = first_persistent_alarm(
                    times=w["time_idx"].to_numpy(),
                    values=pd.to_numeric(w[ind], errors="coerce").to_numpy(),
                    threshold=threshold_map[key],
                    k=PERSISTENT_K,
                )

                rows.append({
                    "sim_id": sim_id,
                    "window": window_name,
                    "stage": stage,
                    "indicator": ind,
                    "detected": int(alarm is not None),
                    "t_alarm": alarm if alarm is not None else np.nan,
                    "use_fake_event": use_fake_event,
                })

    return pd.DataFrame(rows)


def summarize_stage_detection(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (window, stage, sim_id), s in detail.groupby(["window", "stage", "sim_id"]):
        detected = int(s["detected"].max() > 0)
        rows.append({
            "window": window,
            "stage": stage,
            "sim_id": int(sim_id),
            "detected": detected,
        })

    stage_detail = pd.DataFrame(rows)

    summary_rows = []

    for (window, stage), s in stage_detail.groupby(["window", "stage"]):
        summary_rows.append({
            "window": window,
            "stage": stage,
            "n_sims": int(s["sim_id"].nunique()),
            "alarm_rate": float(s["detected"].mean()),
        })

    return pd.DataFrame(summary_rows)


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    all_rows = []

    for system, seed, threshold_tag, run_dir in RUNS:
        ts_path = run_dir / "integrated_patch_dynamics_timeseries.csv"

        if not ts_path.exists():
            print(f"Skip missing: {ts_path}")
            continue

        print(f"\nProcessing {system} seed{seed} threshold={threshold_tag}")

        df = pd.read_csv(ts_path)
        df["sim_id"] = pd.to_numeric(df["sim_id"], errors="coerce").astype(int)
        df["time_idx"] = pd.to_numeric(df["time_idx"], errors="coerce")
        df["critical_time"] = pd.to_numeric(df["critical_time"], errors="coerce")
        df["relative_time"] = pd.to_numeric(df["relative_time"], errors="coerce")

        stages = available_stages(df)
        thresholds = compute_real_thresholds(df, stages)

        real_event_map = (
            df.dropna(subset=["critical_time"])
            .groupby("sim_id")["critical_time"]
            .first()
            .astype(int)
            .to_dict()
        )

        real_detail = detect_stage_window(
            df=df,
            thresholds=thresholds,
            event_time_map=real_event_map,
            use_fake_event=False,
        )

        real_summary = summarize_stage_detection(real_detail)
        real_summary["control_type"] = "real_event"
        real_summary["permutation_id"] = 0

        all_summaries = [real_summary]

        sim_ids = np.array(list(real_event_map.keys()))
        event_times = np.array([real_event_map[x] for x in sim_ids])

        for p in range(1, N_PERMUTATIONS + 1):
            permuted_event_times = rng.permutation(event_times)
            fake_event_map = {
                int(sim_id): int(fake_time)
                for sim_id, fake_time in zip(sim_ids, permuted_event_times)
            }

            fake_detail = detect_stage_window(
                df=df,
                thresholds=thresholds,
                event_time_map=fake_event_map,
                use_fake_event=True,
            )

            fake_summary = summarize_stage_detection(fake_detail)
            fake_summary["control_type"] = "permuted_event"
            fake_summary["permutation_id"] = p
            all_summaries.append(fake_summary)

        summary = pd.concat(all_summaries, ignore_index=True)

        rows = []

        for (window, stage), s in summary.groupby(["window", "stage"]):
            real = s[s["control_type"] == "real_event"]["alarm_rate"].mean()
            perm = s[s["control_type"] == "permuted_event"]["alarm_rate"]

            rows.append({
                "system": system,
                "seed": seed,
                "threshold": threshold_tag,
                "window": window,
                "stage": stage,
                "real_alarm_rate": float(real),
                "permuted_alarm_rate_mean": float(perm.mean()),
                "permuted_alarm_rate_std": float(perm.std(ddof=1)),
                "real_minus_permuted": float(real - perm.mean()),
                "permutation_p_ge_real": float(np.mean(perm >= real)),
            })

        effect = pd.DataFrame(rows)

        outdir = Path("results/v2_5_controls/event_time_permutation") / f"{system}_seed{seed}_thr{threshold_tag}"
        outdir.mkdir(parents=True, exist_ok=True)

        summary.to_csv(outdir / "event_time_permutation_alarm_summary.csv", index=False, encoding="utf-8-sig")
        effect.to_csv(outdir / "event_time_permutation_effect.csv", index=False, encoding="utf-8-sig")

        all_rows.append(effect)

    if not all_rows:
        raise SystemExit("No valid v2.4 timeseries files found.")

    all_effect = pd.concat(all_rows, ignore_index=True)

    outdir = Path("results/v2_5_controls/event_time_permutation_summary")
    outdir.mkdir(parents=True, exist_ok=True)

    all_effect.to_csv(outdir / "event_time_permutation_effect_all.csv", index=False, encoding="utf-8-sig")

    print("\n=== Event-time permutation effect summary ===")
    cols = [
        "system",
        "seed",
        "threshold",
        "window",
        "stage",
        "real_alarm_rate",
        "permuted_alarm_rate_mean",
        "real_minus_permuted",
        "permutation_p_ge_real",
    ]
    print(all_effect[cols].to_string(index=False))

    print("\n=== Mean event-time permutation effect by system and stage ===")
    print(
        all_effect.groupby(["system", "window", "stage"])[
            ["real_alarm_rate", "permuted_alarm_rate_mean", "real_minus_permuted", "permutation_p_ge_real"]
        ].mean().reset_index().to_string(index=False)
    )


if __name__ == "__main__":
    main()
