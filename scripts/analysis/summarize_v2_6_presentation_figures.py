# -*- coding: utf-8 -*-
"""
v2.6B Presentation-quality figures for patch formation process.

This script redraws v2.6 figures with shorter labels and clearer structure
for supervisor reporting.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE = Path("results/v2_6_patch_formation_final_summary")
TABLE_DIR = BASE / "tables"
OUTDIR = BASE / "figures_presentation"
OUTDIR.mkdir(parents=True, exist_ok=True)


def read_csv(path):
    if not path.exists():
        print(f"[WARN] missing: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def short_stage_name(stage):
    s = str(stage)

    if "prepatch" in s:
        return "Prepatch"
    if "pwsi" in s:
        return "PWSI"
    if "dynamic" in s:
        return "Dynamic"
    if "boundary" in s:
        return "Boundary"
    if "visible" in s:
        return "Visible"

    return s


def short_indicator_name(ind):
    s = str(ind)

    if "moran" in s:
        return "Moran's I"
    if "local_neighbor_corr" in s:
        return "Local corr"
    if "pwsi" in s:
        return "PWSI"
    if "sync" in s:
        return "Sync edge"
    return s


def fig1_stage_lead_time():
    df = read_csv(TABLE_DIR / "cross_system_stage_lead_summary.csv")
    if df.empty:
        return

    df = df.copy()
    df["stage_short"] = df["stage"].apply(short_stage_name)

    order = ["Prepatch", "PWSI", "Dynamic", "Boundary", "Visible"]
    rows = []

    for system in ["Vegetation CA", "SEIR"]:
        sub = df[df["system"] == system].copy()

        for stage in order:
            s = sub[sub["stage_short"] == stage]

            if s.empty:
                continue

            rows.append({
                "label": f"{'Veg' if system == 'Vegetation CA' else 'SEIR'}\n{stage}",
                "system": system,
                "stage": stage,
                "median": float(s["lead_median"].mean()),
                "q25": float(s["lead_q25"].mean()),
                "q75": float(s["lead_q75"].mean()),
            })

    p = pd.DataFrame(rows)
    p.to_csv(TABLE_DIR / "presentation_fig1_stage_lead_time.csv", index=False, encoding="utf-8-sig")

    x = np.arange(len(p))
    y = p["median"].to_numpy()
    yerr_low = y - p["q25"].to_numpy()
    yerr_high = p["q75"].to_numpy() - y

    plt.figure(figsize=(10, 5))
    plt.bar(x, y)
    plt.errorbar(x, y, yerr=[yerr_low, yerr_high], fmt="none", capsize=4)
    plt.xticks(x, p["label"], rotation=0)
    plt.ylabel("Median lead time")
    plt.title("Patch-formation stages appear before critical transition")
    plt.tight_layout()
    plt.savefig(OUTDIR / "fig1_stage_lead_time_presentation.png", dpi=300)
    plt.close()


def fig2_order_probability_key():
    df = read_csv(TABLE_DIR / "cross_system_order_tests.csv")
    if df.empty:
        return

    rows = []

    for _, r in df.iterrows():
        system = str(r["system"])
        test = str(r["order_test"])

        use = False

        if system == "Vegetation CA" and "dynamic_no_bci" in test and "visible_fragmentation" in test:
            use = True

        if system == "SEIR" and "dynamic_spreading" in test and "visible_patch" in test:
            use = True

        if not use:
            continue

        rows.append({
            "system": system,
            "seed": int(r["seed"]) if "seed" in r and pd.notna(r["seed"]) else -1,
            "weak": float(r["weak_order_probability_leq"]),
            "strict": float(r["strict_order_probability_lt"]),
            "margin3": float(r["margin3_order_probability"]),
        })

    p = pd.DataFrame(rows)

    if p.empty:
        print("[WARN] no key order rows found")
        return

    grouped = p.groupby("system")[["weak", "strict", "margin3"]].mean().reset_index()
    grouped.to_csv(TABLE_DIR / "presentation_fig2_key_order_probability.csv", index=False, encoding="utf-8-sig")

    labels = grouped["system"].tolist()
    x = np.arange(len(labels))
    width = 0.22

    plt.figure(figsize=(8, 5))
    plt.bar(x - width, grouped["weak"], width, label="Weak order")
    plt.bar(x, grouped["strict"], width, label="Strict order")
    plt.bar(x + width, grouped["margin3"], width, label="Margin ≥ 3")
    plt.xticks(x, labels)
    plt.ylim(0, 1.05)
    plt.ylabel("Order probability")
    plt.title("Key patch-formation order probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "fig2_key_order_probability_presentation.png", dpi=300)
    plt.close()


def fig3_spatial_shuffle():
    df = read_csv(TABLE_DIR / "control_spatial_shuffle_raw_effects.csv")
    if df.empty:
        return

    df = df.copy()

    df = df[
        df["window"].isin(["full_precursor_window", "late_precursor_window"])
        & df["indicator"].isin([
            "prepatch_moran_i",
            "prepatch_local_neighbor_corr_mean",
            "pwsi_control",
        ])
    ].copy()

    if df.empty:
        return

    df["indicator_short"] = df["indicator"].apply(short_indicator_name)

    grouped = (
        df.groupby(["system", "indicator_short"])["original_minus_shuffle"]
        .mean()
        .reset_index()
    )

    grouped.to_csv(TABLE_DIR / "presentation_fig3_spatial_shuffle.csv", index=False, encoding="utf-8-sig")

    systems = ["vegetation", "seir"]
    indicators = ["Moran's I", "Local corr", "PWSI"]

    rows = []

    for system in systems:
        for ind in indicators:
            s = grouped[
                (grouped["system"].astype(str).str.lower() == system)
                & (grouped["indicator_short"] == ind)
            ]

            if s.empty:
                continue

            rows.append({
                "label": f"{'Veg' if system == 'vegetation' else 'SEIR'}\n{ind}",
                "value": float(s["original_minus_shuffle"].mean()),
            })

    p = pd.DataFrame(rows)

    plt.figure(figsize=(8, 5))
    plt.bar(np.arange(len(p)), p["value"])
    plt.xticks(np.arange(len(p)), p["label"], rotation=0)
    plt.ylabel("Original - shuffled")
    plt.title("Spatial shuffling attenuates core spatial indicators")
    plt.tight_layout()
    plt.savefig(OUTDIR / "fig3_spatial_shuffle_presentation.png", dpi=300)
    plt.close()


def fig4_stable_window():
    df = read_csv(TABLE_DIR / "control_stable_window_false_alarm.csv")
    if df.empty:
        return

    df = df.copy()
    df["stage_short"] = df["stage"].apply(short_stage_name)

    keep_stages = ["Dynamic", "Visible", "Prepatch", "PWSI"]
    df = df[df["stage_short"].isin(keep_stages)].copy()

    grouped = (
        df.groupby(["system", "stage_short"])[
            ["far_stable_alarm_rate", "late_precursor_alarm_rate"]
        ]
        .mean()
        .reset_index()
    )

    grouped.to_csv(TABLE_DIR / "presentation_fig4_stable_window.csv", index=False, encoding="utf-8-sig")

    labels = []
    far = []
    late = []

    for _, r in grouped.iterrows():
        sys_short = "Veg" if str(r["system"]).lower() == "vegetation" else "SEIR"
        labels.append(f"{sys_short}\n{r['stage_short']}")
        far.append(r["far_stable_alarm_rate"])
        late.append(r["late_precursor_alarm_rate"])

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, far, width, label="Far stable")
    plt.bar(x + width / 2, late, width, label="Late precursor")
    plt.xticks(x, labels, rotation=0)
    plt.ylim(0, 1.05)
    plt.ylabel("Alarm rate")
    plt.title("Stable-window false-alarm control")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "fig4_stable_window_presentation.png", dpi=300)
    plt.close()


def fig5_event_time_permutation():
    df = read_csv(TABLE_DIR / "control_event_time_permutation.csv")
    if df.empty:
        return

    rows = []

    for _, r in df.iterrows():
        system = str(r["system"]).lower()
        window = str(r["window"])
        stage = short_stage_name(r["stage"])

        use = False

        # Vegetation: early window is the cleanest evidence
        if system == "vegetation" and window == "early_precursor_window":
            use = True

        # SEIR: early prepatch/PWSI, late dynamic/visible
        if system == "seir":
            if window == "early_precursor_window" and stage in ["Prepatch", "PWSI"]:
                use = True
            if window == "late_precursor_window" and stage in ["Dynamic", "Visible"]:
                use = True

        if not use:
            continue

        sys_short = "Veg" if system == "vegetation" else "SEIR"
        win_short = "Early" if window == "early_precursor_window" else "Late"

        rows.append({
            "label": f"{sys_short}\n{win_short}\n{stage}",
            "value": float(r["real_minus_permuted"]),
        })

    p = pd.DataFrame(rows)

    if p.empty:
        return

    # Average duplicated vegetation seeds by label
    p = p.groupby("label")["value"].mean().reset_index()
    p.to_csv(TABLE_DIR / "presentation_fig5_event_time_permutation.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(9, 5))
    plt.bar(np.arange(len(p)), p["value"])
    plt.xticks(np.arange(len(p)), p["label"], rotation=0)
    plt.ylabel("Real alarm rate - permuted alarm rate")
    plt.title("Event-time permutation control")
    plt.tight_layout()
    plt.savefig(OUTDIR / "fig5_event_time_permutation_presentation.png", dpi=300)
    plt.close()


def main():
    fig1_stage_lead_time()
    fig2_order_probability_key()
    fig3_spatial_shuffle()
    fig4_stable_window()
    fig5_event_time_permutation()
    print(f"Saved presentation figures to: {OUTDIR}")


if __name__ == "__main__":
    main()
