# -*- coding: utf-8 -*-
"""
v2.8A-fixed Patch Formation State Variables

Purpose:
    Build stable 0-1 patch formation state variables:

    P(t): prepatch spatial organization
    D(t): dynamic patch restructuring
    B(t): boundary complexity
    V(t): visible patch manifestation
    F(t): overall patch formation activation

Why fixed:
    The first v2.8A version used raw robust z-scores.
    Some baseline variances were too small, causing extremely large spikes.
    This fixed version clips, compresses, smooths, and rescales indicators
    into comparable 0-1 activation scores.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


RUNS = [
    (
        "vegetation",
        42,
        "0p50",
        Path("results/v2_4_integrated_patch_dynamics/vegetation_seed42_thr0p50/integrated_patch_dynamics_timeseries.csv"),
    ),
    (
        "vegetation",
        123,
        "0p50",
        Path("results/v2_4_integrated_patch_dynamics/vegetation_seed123_thr0p50/integrated_patch_dynamics_timeseries.csv"),
    ),
    (
        "seir",
        2026,
        "0p20",
        Path("results/v2_4_integrated_patch_dynamics/seir_seed2026_thr0p20/integrated_patch_dynamics_timeseries.csv"),
    ),
]


STATE_INDICATORS = {
    "P_state": [
        "prepatch_moran_i",
        "prepatch_local_neighbor_corr_mean",
        "prepatch_sync_edge_ratio",
        "prepatch_boundary_sharpness",
        "prepatch_gradient_top10_mean",
        "prepatch_svd_spectral_gap",
    ],
    "D_state": [
        "PSII",
        "DPCI",
        "DPCR",
        "FPV",
        "GCGR",
        "PDSI",
    ],
    "B_state": [
        "BCI",
        "edge_density",
    ],
    "V_state": [
        "patch_count",
        "largest_patch_ratio",
        "total_patch_ratio",
        "visible_patch_metric",
    ],
}


BASELINE_START = -80
BASELINE_END = -51

PLOT_START = -80
PLOT_END = 5

MAX_Z = 8.0
SMOOTH_WINDOW = 5
ONSET_THRESHOLD = 0.30
ONSET_PERSIST_K = 3

OUTDIR = Path("results/v2_8_patch_formation_dynamics/state_variables_fixed")
TABLE_DIR = OUTDIR / "tables"
FIG_DIR = OUTDIR / "figures"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def log(msg: str):
    print(f"[v2.8A-fixed] {msg}", flush=True)


def safe_numeric(s):
    return pd.to_numeric(s, errors="coerce").astype(float)


def robust_activation_by_sim_baseline(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Convert one raw indicator into 0-1 activation score.

    Procedure:
        1. Use each simulation's early baseline window.
        2. Compute robust z-score.
        3. Apply lower clipping at 0.
        4. Apply upper clipping at MAX_Z.
        5. Compress by log1p.
        6. Map to 0-1.
    """
    result = []

    for _, g in df.groupby("sim_id", sort=False):
        g = g.sort_values("time_idx").copy()

        x = safe_numeric(g[col]).to_numpy()
        finite_x = x[np.isfinite(x)]

        baseline = g[
            (g["relative_time"] >= BASELINE_START)
            & (g["relative_time"] <= BASELINE_END)
        ]

        xb = safe_numeric(baseline[col]).to_numpy()
        xb = xb[np.isfinite(xb)]

        if len(xb) < 5:
            xb = finite_x

        if len(xb) < 5 or len(finite_x) < 5:
            activation = np.zeros_like(x, dtype=float)
            result.extend(activation.tolist())
            continue

        med = np.nanmedian(xb)
        mad = np.nanmedian(np.abs(xb - med))
        scale = 1.4826 * mad

        global_q05 = np.nanquantile(finite_x, 0.05)
        global_q95 = np.nanquantile(finite_x, 0.95)
        global_range = global_q95 - global_q05

        if not np.isfinite(scale) or scale < 1e-12:
            scale = np.nanstd(xb)

        if not np.isfinite(scale) or scale < 1e-12:
            scale = global_range / 6.0

        if not np.isfinite(scale) or scale < 1e-12:
            scale = 1.0

        # Avoid exploding z-score when baseline variance is extremely small.
        if np.isfinite(global_range) and global_range > 1e-12:
            min_scale = global_range * 0.05
            scale = max(scale, min_scale)

        z = (x - med) / scale
        z = np.where(np.isfinite(z), z, 0.0)

        z = np.clip(z, 0.0, MAX_Z)
        activation = np.log1p(z) / np.log1p(MAX_Z)
        activation = np.clip(activation, 0.0, 1.0)

        result.extend(activation.tolist())

    return pd.Series(result, index=df.index)


def smooth_state_by_sim(df: pd.DataFrame, col: str) -> pd.Series:
    values = []

    for _, g in df.groupby("sim_id", sort=False):
        x = safe_numeric(g[col])
        smoothed = (
            x.rolling(window=SMOOTH_WINDOW, center=True, min_periods=1)
            .mean()
            .to_numpy()
        )
        values.extend(smoothed.tolist())

    return pd.Series(values, index=df.index)


def build_states_for_run(system: str, seed: int, threshold: str, path: Path) -> pd.DataFrame:
    if not path.exists():
        log(f"missing file, skipped: {path}")
        return pd.DataFrame()

    log(f"processing {system} seed={seed} threshold={threshold}")

    df = pd.read_csv(path)

    required = ["sim_id", "time_idx", "relative_time"]
    for c in required:
        if c not in df.columns:
            log(f"missing required column {c}, skipped: {path}")
            return pd.DataFrame()

    for c in ["sim_id", "time_idx", "critical_time", "relative_time"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["sim_id", "time_idx", "relative_time"]).copy()
    df["sim_id"] = df["sim_id"].astype(int)
    df = df.sort_values(["sim_id", "time_idx"]).reset_index(drop=True)

    available_map = {}
    for state, cols in STATE_INDICATORS.items():
        available = [c for c in cols if c in df.columns]
        available_map[state] = available
        log(f"{system} {state}: {available}")

    # Indicator-level activation
    for state, cols in available_map.items():
        for col in cols:
            act_col = f"act_{col}"
            df[act_col] = robust_activation_by_sim_baseline(df, col)

    # State-level activation
    for state, cols in available_map.items():
        act_cols = [f"act_{c}" for c in cols if f"act_{c}" in df.columns]

        if len(act_cols) == 0:
            df[state] = np.nan
        else:
            df[state] = df[act_cols].mean(axis=1)

        df[state] = smooth_state_by_sim(df, state)
        df[state] = np.clip(df[state], 0.0, 1.0)

    state_cols = [c for c in ["P_state", "D_state", "B_state", "V_state"] if c in df.columns]

    if state_cols:
        df["F_state"] = df[state_cols].mean(axis=1)
        df["F_state"] = smooth_state_by_sim(df, "F_state")
        df["F_state"] = np.clip(df["F_state"], 0.0, 1.0)
    else:
        df["F_state"] = np.nan

    df["system"] = system
    df["seed"] = seed
    df["threshold"] = threshold

    keep_cols = [
        "system",
        "seed",
        "threshold",
        "sim_id",
        "time_idx",
        "critical_time",
        "relative_time",
        "P_state",
        "D_state",
        "B_state",
        "V_state",
        "F_state",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]

    return df[keep_cols].copy()


def summarize_trajectory(states: pd.DataFrame) -> pd.DataFrame:
    rows = []

    states = states[
        (states["relative_time"] >= PLOT_START)
        & (states["relative_time"] <= PLOT_END)
    ].copy()

    for (system, seed, threshold, t), g in states.groupby(
        ["system", "seed", "threshold", "relative_time"]
    ):
        row = {
            "system": system,
            "seed": seed,
            "threshold": threshold,
            "relative_time": t,
        }

        for col in ["P_state", "D_state", "B_state", "V_state", "F_state"]:
            if col in g.columns:
                row[f"{col}_mean"] = float(g[col].mean())
                row[f"{col}_median"] = float(g[col].median())
                row[f"{col}_q25"] = float(g[col].quantile(0.25))
                row[f"{col}_q75"] = float(g[col].quantile(0.75))

        rows.append(row)

    return pd.DataFrame(rows)


def plot_system_trajectories(summary: pd.DataFrame):
    for system, g in summary.groupby("system"):
        agg_rows = []

        for t, s in g.groupby("relative_time"):
            row = {"relative_time": t}
            for col in ["P_state_mean", "D_state_mean", "B_state_mean", "V_state_mean", "F_state_mean"]:
                if col in s.columns:
                    row[col] = float(s[col].mean())
            agg_rows.append(row)

        agg = pd.DataFrame(agg_rows).sort_values("relative_time")

        plt.figure(figsize=(9, 5))

        plot_map = {
            "P_state_mean": "P: prepatch organization",
            "D_state_mean": "D: dynamic restructuring",
            "B_state_mean": "B: boundary complexity",
            "V_state_mean": "V: visible patch",
            "F_state_mean": "F: overall formation",
        }

        for col, label in plot_map.items():
            if col in agg.columns:
                plt.plot(agg["relative_time"], agg[col], label=label, linewidth=2)

        plt.axvline(0, linestyle="--", linewidth=1)
        plt.ylim(-0.05, 1.05)
        plt.xlabel("Relative time to transition")
        plt.ylabel("State activation, 0-1")
        plt.title(f"Patch formation state trajectories, fixed scale: {system}")
        plt.legend()
        plt.tight_layout()

        plt.savefig(FIG_DIR / f"state_trajectories_{system}_fixed.png", dpi=300)
        plt.close()


def first_sustained_onset(g: pd.DataFrame, state_col: str) -> float:
    g = g.sort_values("relative_time").copy()
    g = g[(g["relative_time"] >= PLOT_START) & (g["relative_time"] <= 0)].copy()

    if g.empty or state_col not in g.columns:
        return np.nan

    x = safe_numeric(g[state_col]).to_numpy()
    t = safe_numeric(g["relative_time"]).to_numpy()

    active = np.isfinite(x) & (x >= ONSET_THRESHOLD)

    if len(active) < ONSET_PERSIST_K:
        return np.nan

    for i in range(0, len(active) - ONSET_PERSIST_K + 1):
        if active[i:i + ONSET_PERSIST_K].all():
            return float(t[i])

    return np.nan


def summarize_onsets(states: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (system, seed, threshold, sim_id), g in states.groupby(
        ["system", "seed", "threshold", "sim_id"]
    ):
        for state_col in ["P_state", "D_state", "B_state", "V_state", "F_state"]:
            onset = first_sustained_onset(g, state_col)
            rows.append({
                "system": system,
                "seed": seed,
                "threshold": threshold,
                "sim_id": sim_id,
                "state": state_col,
                "onset_relative_time": onset,
                "detected": int(np.isfinite(onset)),
            })

    onset_df = pd.DataFrame(rows)

    summary_rows = []

    for (system, state), g in onset_df.groupby(["system", "state"]):
        detected = g[g["detected"] == 1]
        row = {
            "system": system,
            "state": state,
            "n_total": int(len(g)),
            "n_detected": int(len(detected)),
            "detection_rate": float(len(detected) / max(len(g), 1)),
            "onset_mean": float(detected["onset_relative_time"].mean()) if len(detected) else np.nan,
            "onset_median": float(detected["onset_relative_time"].median()) if len(detected) else np.nan,
            "onset_q25": float(detected["onset_relative_time"].quantile(0.25)) if len(detected) else np.nan,
            "onset_q75": float(detected["onset_relative_time"].quantile(0.75)) if len(detected) else np.nan,
        }
        summary_rows.append(row)

    onset_summary = pd.DataFrame(summary_rows)

    onset_df.to_csv(TABLE_DIR / "state_onset_by_simulation_fixed.csv", index=False, encoding="utf-8-sig")
    onset_summary.to_csv(TABLE_DIR / "state_onset_summary_fixed.csv", index=False, encoding="utf-8-sig")

    return onset_summary


def plot_onset_summary(onset_summary: pd.DataFrame):
    if onset_summary.empty:
        return

    plot_df = onset_summary.dropna(subset=["onset_median"]).copy()
    if plot_df.empty:
        return

    order = []
    for system in ["vegetation", "seir"]:
        for state in ["P_state", "D_state", "B_state", "V_state", "F_state"]:
            order.append((system, state))

    plot_df["order_key"] = plot_df.apply(
        lambda r: order.index((r["system"], r["state"])) if (r["system"], r["state"]) in order else 999,
        axis=1,
    )
    plot_df = plot_df.sort_values("order_key")

    labels = [f"{r.system}\n{r.state}" for _, r in plot_df.iterrows()]
    values = plot_df["onset_median"].to_numpy()

    plt.figure(figsize=(10, 5))
    plt.bar(np.arange(len(values)), values)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xticks(np.arange(len(values)), labels, rotation=30, ha="right")
    plt.ylabel("Median onset relative time")
    plt.title(f"First sustained state activation, threshold={ONSET_THRESHOLD}")
    plt.tight_layout()

    plt.savefig(FIG_DIR / "state_onset_timing_summary_fixed.png", dpi=300)
    plt.close()


def main():
    all_states = []

    for system, seed, threshold, path in RUNS:
        out = build_states_for_run(system, seed, threshold, path)
        if not out.empty:
            all_states.append(out)

    if not all_states:
        raise SystemExit("No valid input files found.")

    states = pd.concat(all_states, ignore_index=True)

    states.to_csv(
        TABLE_DIR / "patch_formation_state_timeseries_fixed.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = summarize_trajectory(states)
    summary.to_csv(
        TABLE_DIR / "patch_formation_state_trajectory_summary_fixed.csv",
        index=False,
        encoding="utf-8-sig",
    )

    onset_summary = summarize_onsets(states)

    plot_system_trajectories(summary)
    plot_onset_summary(onset_summary)

    log("fixed state variables generated")
    log(f"output directory: {OUTDIR}")

    print("\n=== Fixed state variables summary ===")
    print(
        states[
            ["system", "seed", "threshold", "P_state", "D_state", "B_state", "V_state", "F_state"]
        ].describe().to_string()
    )

    print("\n=== Fixed onset summary ===")
    print(onset_summary.to_string(index=False))

    print("\n=== Output files ===")
    for p in sorted(OUTDIR.rglob("*")):
        print(p)


if __name__ == "__main__":
    main()
