# -*- coding: utf-8 -*-
"""
v2.8B Lag Correlation Analysis

Goal:
    Quantify temporal leading relationships among patch formation states.

    Main questions:
        Does P(t) lead D(t + lag)?
        Does P(t) lead B(t + lag)?
        Does P(t) lead V(t + lag)?
        Does D(t) lead V(t + lag)?
        Does B(t) lead V(t + lag)?

Input:
    results/v2_8_patch_formation_dynamics/state_variables_fixed/tables/
        patch_formation_state_timeseries_fixed.csv

Output:
    results/v2_8_patch_formation_dynamics/lag_correlation/
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


INPUT = Path(
    "results/v2_8_patch_formation_dynamics/state_variables_fixed/tables/"
    "patch_formation_state_timeseries_fixed.csv"
)

OUTDIR = Path("results/v2_8_patch_formation_dynamics/lag_correlation")
TABLE_DIR = OUTDIR / "tables"
FIG_DIR = OUTDIR / "figures"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


STATE_PAIRS = [
    ("P_state", "D_state"),
    ("P_state", "B_state"),
    ("P_state", "V_state"),
    ("D_state", "V_state"),
    ("B_state", "V_state"),
    ("F_state", "V_state"),
]

MAX_LAG = 30
MIN_POINTS = 20


def log(msg: str):
    print(f"[v2.8B] {msg}", flush=True)


def safe_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)

    if mask.sum() < MIN_POINTS:
        return np.nan

    x = x[mask]
    y = y[mask]

    if np.nanstd(x) < 1e-12 or np.nanstd(y) < 1e-12:
        return np.nan

    return float(np.corrcoef(x, y)[0, 1])


def lag_correlation_for_sim(g: pd.DataFrame, x_col: str, y_col: str, lag: int):
    """
    Positive lag means:
        x(t) is compared with y(t + lag)

    If positive-lag correlation is high,
    x is interpreted as leading y by lag steps.
    """
    g = g.sort_values("relative_time").copy()

    x = g[x_col].to_numpy(dtype=float)
    y = g[y_col].to_numpy(dtype=float)

    if lag > 0:
        x_lag = x[:-lag]
        y_lag = y[lag:]
    elif lag < 0:
        k = abs(lag)
        x_lag = x[k:]
        y_lag = y[:-k]
    else:
        x_lag = x
        y_lag = y

    return safe_corr(x_lag, y_lag)


def compute_lag_correlations(df: pd.DataFrame):
    rows = []

    for (system, seed, threshold, sim_id), g in df.groupby(
        ["system", "seed", "threshold", "sim_id"]
    ):
        for x_col, y_col in STATE_PAIRS:
            if x_col not in g.columns or y_col not in g.columns:
                continue

            for lag in range(-MAX_LAG, MAX_LAG + 1):
                corr = lag_correlation_for_sim(g, x_col, y_col, lag)

                rows.append({
                    "system": system,
                    "seed": seed,
                    "threshold": threshold,
                    "sim_id": sim_id,
                    "x_state": x_col,
                    "y_state": y_col,
                    "lag": lag,
                    "corr": corr,
                })

    return pd.DataFrame(rows)


def summarize_lag_correlations(lag_df: pd.DataFrame):
    summary_rows = []

    for (system, x_state, y_state, lag), g in lag_df.groupby(
        ["system", "x_state", "y_state", "lag"]
    ):
        corr_values = g["corr"].dropna()

        if len(corr_values) == 0:
            continue

        summary_rows.append({
            "system": system,
            "x_state": x_state,
            "y_state": y_state,
            "lag": lag,
            "corr_mean": float(corr_values.mean()),
            "corr_median": float(corr_values.median()),
            "corr_q25": float(corr_values.quantile(0.25)),
            "corr_q75": float(corr_values.quantile(0.75)),
            "n": int(len(corr_values)),
        })

    summary = pd.DataFrame(summary_rows)

    peak_rows = []

    for (system, x_state, y_state), g in summary.groupby(
        ["system", "x_state", "y_state"]
    ):
        g = g.dropna(subset=["corr_mean"]).copy()

        if g.empty:
            continue

        # Main interpretation: prefer positive lags.
        positive = g[g["lag"] >= 0].copy()

        if positive.empty:
            peak = g.loc[g["corr_mean"].idxmax()]
        else:
            peak = positive.loc[positive["corr_mean"].idxmax()]

        peak_rows.append({
            "system": system,
            "x_state": x_state,
            "y_state": y_state,
            "best_positive_lag": int(peak["lag"]),
            "best_positive_corr_mean": float(peak["corr_mean"]),
            "best_positive_corr_median": float(peak["corr_median"]),
            "interpretation": f"{x_state} leads {y_state} by {int(peak['lag'])} time steps",
        })

    peak_summary = pd.DataFrame(peak_rows)

    return summary, peak_summary


def plot_lag_curves(summary: pd.DataFrame):
    if summary.empty:
        return

    for system, g_system in summary.groupby("system"):
        plt.figure(figsize=(10, 6))

        for x_state, y_state in STATE_PAIRS:
            g = g_system[
                (g_system["x_state"] == x_state)
                & (g_system["y_state"] == y_state)
            ].copy()

            if g.empty:
                continue

            g = g.sort_values("lag")
            label = f"{x_state} -> {y_state}"
            plt.plot(g["lag"], g["corr_mean"], linewidth=2, label=label)

        plt.axvline(0, linestyle="--", linewidth=1)
        plt.xlabel("Lag")
        plt.ylabel("Mean correlation")
        plt.title(f"Lag correlation among patch formation states: {system}")
        plt.legend(fontsize=8)
        plt.tight_layout()

        out = FIG_DIR / f"lag_correlation_curves_{system}.png"
        plt.savefig(out, dpi=300)
        plt.close()


def plot_best_lag_summary(peak_summary: pd.DataFrame):
    if peak_summary.empty:
        return

    for system, g in peak_summary.groupby("system"):
        g = g.copy()

        labels = [f"{r.x_state}->{r.y_state}" for _, r in g.iterrows()]
        values = g["best_positive_lag"].to_numpy()

        plt.figure(figsize=(10, 5))
        plt.bar(np.arange(len(values)), values)
        plt.axhline(0, linestyle="--", linewidth=1)
        plt.xticks(np.arange(len(values)), labels, rotation=30, ha="right")
        plt.ylabel("Best positive lag")
        plt.title(f"Best leading lag among patch formation states: {system}")
        plt.tight_layout()

        out = FIG_DIR / f"best_positive_lag_summary_{system}.png"
        plt.savefig(out, dpi=300)
        plt.close()


def main():
    if not INPUT.exists():
        raise SystemExit(f"Missing input file: {INPUT}")

    df = pd.read_csv(INPUT)

    required = [
        "system",
        "seed",
        "threshold",
        "sim_id",
        "relative_time",
        "P_state",
        "D_state",
        "B_state",
        "V_state",
        "F_state",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns: {missing}")

    for c in ["sim_id", "relative_time", "P_state", "D_state", "B_state", "V_state", "F_state"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Focus on pre-transition period.
    df = df[(df["relative_time"] >= -80) & (df["relative_time"] <= 0)].copy()

    lag_df = compute_lag_correlations(df)
    lag_summary, peak_summary = summarize_lag_correlations(lag_df)

    lag_df.to_csv(TABLE_DIR / "lag_correlation_by_simulation.csv", index=False, encoding="utf-8-sig")
    lag_summary.to_csv(TABLE_DIR / "lag_correlation_summary.csv", index=False, encoding="utf-8-sig")
    peak_summary.to_csv(TABLE_DIR / "best_positive_lag_summary.csv", index=False, encoding="utf-8-sig")

    plot_lag_curves(lag_summary)
    plot_best_lag_summary(peak_summary)

    log("lag correlation analysis finished")
    log(f"output directory: {OUTDIR}")

    print("\n=== Best positive lag summary ===")
    print(peak_summary.to_string(index=False))

    print("\n=== Output files ===")
    for p in sorted(OUTDIR.rglob("*")):
        print(p)


if __name__ == "__main__":
    main()
