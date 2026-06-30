# -*- coding: utf-8 -*-
"""
v2.6 Final patch-formation summary.

Purpose:
    Consolidate v2.4 and v2.5 results into final tables and figures for
    supervisor reporting and paper drafting.

Inputs:
    - v2.4 vegetation paper-stage order results
    - v2.4 SEIR paper-stage order results
    - v2.5 spatial-shuffle raw effects
    - v2.5 stable-window false-alarm control
    - v2.5 event-time permutation control

Outputs:
    results/v2_6_patch_formation_final_summary/
        tables/
        figures/
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


OUTDIR = Path("results/v2_6_patch_formation_final_summary")
TABLE_DIR = OUTDIR / "tables"
FIG_DIR = OUTDIR / "figures"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[WARN] missing file: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def stage_class(stage: str) -> str:
    s = str(stage)

    if "prepatch" in s:
        return "Prepatch organization"
    if "pwsi" in s:
        return "PWSI index"
    if "dynamic" in s:
        return "Dynamic patch"
    if "boundary" in s:
        return "Boundary complexity"
    if "visible" in s:
        return "Visible patch"

    return s


def build_stage_summary() -> pd.DataFrame:
    rows = []

    # Vegetation CA
    veg_stage_path = Path(
        "results/v2_4_integrated_patch_dynamics/"
        "vegetation_paper_stage_order_thr0p50/paper_stage_summary.csv"
    )

    veg = read_csv_safe(veg_stage_path)

    if not veg.empty:
        veg = veg.copy()
        veg["system"] = "Vegetation CA"
        veg["threshold"] = "0p50"
        veg["stage_class"] = veg["stage"].apply(stage_class)
        rows.append(veg)

    # SEIR
    seir_stage_path = Path(
        "results/v2_4_integrated_patch_dynamics/"
        "seir_seed2026_paper_stage_order/paper_stage_summary.csv"
    )

    seir = read_csv_safe(seir_stage_path)

    if not seir.empty:
        seir = seir.copy()
        seir = seir[seir["threshold"].astype(str) == "0p20"].copy()
        seir["system"] = "SEIR"
        seir["seed"] = 2026
        seir["stage_class"] = seir["stage"].apply(stage_class)
        rows.append(seir)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)

    keep = [
        "system",
        "seed",
        "threshold",
        "stage",
        "stage_class",
        "n_detected_sims",
        "lead_mean",
        "lead_median",
        "lead_q25",
        "lead_q75",
        "most_common_indicator",
    ]

    keep = [c for c in keep if c in out.columns]
    out = out[keep].copy()

    out.to_csv(TABLE_DIR / "cross_system_stage_lead_summary.csv", index=False, encoding="utf-8-sig")

    return out


def build_order_summary() -> pd.DataFrame:
    rows = []

    veg_order_path = Path(
        "results/v2_4_integrated_patch_dynamics/"
        "vegetation_paper_stage_order_thr0p50/paper_stage_order_tests.csv"
    )

    veg = read_csv_safe(veg_order_path)

    if not veg.empty:
        veg = veg.copy()
        veg["system"] = "Vegetation CA"
        veg["threshold"] = "0p50"
        rows.append(veg)

    seir_order_path = Path(
        "results/v2_4_integrated_patch_dynamics/"
        "seir_seed2026_paper_stage_order/paper_stage_order_tests.csv"
    )

    seir = read_csv_safe(seir_order_path)

    if not seir.empty:
        seir = seir.copy()
        seir = seir[seir["threshold"].astype(str) == "0p20"].copy()
        seir["system"] = "SEIR"
        seir["seed"] = 2026
        rows.append(seir)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)

    keep = [
        "system",
        "seed",
        "threshold",
        "order_test",
        "n_valid_sims",
        "weak_order_probability_leq",
        "strict_order_probability_lt",
        "margin3_order_probability",
    ]

    keep = [c for c in keep if c in out.columns]
    out = out[keep].copy()

    out.to_csv(TABLE_DIR / "cross_system_order_tests.csv", index=False, encoding="utf-8-sig")

    return out


def build_control_summary() -> pd.DataFrame:
    rows = []

    # Spatial shuffle raw attenuation
    spatial_path = Path(
        "results/v2_5_controls/spatial_shuffle_raw_effects/"
        "core_spatial_organization_shuffle_effects.csv"
    )

    spatial = read_csv_safe(spatial_path)

    if not spatial.empty:
        spatial = spatial.copy()
        spatial["control"] = "Spatial shuffle raw attenuation"
        rows.append(spatial)

    # Stable window false-alarm
    stable_path = Path(
        "results/v2_5_controls/stable_window_false_alarm_summary/"
        "stable_window_false_alarm_effect_all.csv"
    )

    stable = read_csv_safe(stable_path)

    if not stable.empty:
        stable = stable.copy()
        stable["control"] = "Stable-window false-alarm"
        rows.append(stable)

    # Event-time permutation
    perm_path = Path(
        "results/v2_5_controls/event_time_permutation_summary/"
        "event_time_permutation_effect_all.csv"
    )

    perm = read_csv_safe(perm_path)

    if not perm.empty:
        perm = perm.copy()
        perm["control"] = "Event-time permutation"
        rows.append(perm)

    if not rows:
        return pd.DataFrame()

    # Save each raw table also in v2.6 tables
    if not spatial.empty:
        spatial.to_csv(TABLE_DIR / "control_spatial_shuffle_raw_effects.csv", index=False, encoding="utf-8-sig")

    if not stable.empty:
        stable.to_csv(TABLE_DIR / "control_stable_window_false_alarm.csv", index=False, encoding="utf-8-sig")

    if not perm.empty:
        perm.to_csv(TABLE_DIR / "control_event_time_permutation.csv", index=False, encoding="utf-8-sig")

    return pd.concat(rows, ignore_index=True, sort=False)


def plot_stage_lead(stage_df: pd.DataFrame) -> None:
    if stage_df.empty:
        return

    plot_df = stage_df.copy()

    order = [
        "Prepatch organization",
        "PWSI index",
        "Dynamic patch",
        "Boundary complexity",
        "Visible patch",
    ]

    rows = []

    for system in plot_df["system"].dropna().unique():
        sub = plot_df[plot_df["system"] == system].copy()

        for stage in order:
            s = sub[sub["stage_class"] == stage]

            if s.empty:
                continue

            rows.append({
                "system": system,
                "stage_class": stage,
                "lead_median": float(s["lead_median"].mean()),
                "lead_q25": float(s["lead_q25"].mean()),
                "lead_q75": float(s["lead_q75"].mean()),
            })

    p = pd.DataFrame(rows)

    if p.empty:
        return

    p.to_csv(TABLE_DIR / "figure_stage_lead_data.csv", index=False, encoding="utf-8-sig")

    labels = []
    values = []

    for _, r in p.iterrows():
        labels.append(f"{r['system']}\n{r['stage_class']}")
        values.append(r["lead_median"])

    plt.figure(figsize=(12, 5))
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(values)), labels, rotation=35, ha="right")
    plt.ylabel("Median lead time")
    plt.title("Cross-system patch-formation stage lead time")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_v2_6_stage_lead_time.png", dpi=300)
    plt.close()


def plot_order_probability(order_df: pd.DataFrame) -> None:
    if order_df.empty:
        return

    p = order_df.copy()

    p["main_order"] = p["order_test"].astype(str)

    rows = []

    for _, r in p.iterrows():
        rows.append({
            "label": f"{r['system']}\n{r['main_order']}",
            "weak": r.get("weak_order_probability_leq", np.nan),
            "strict": r.get("strict_order_probability_lt", np.nan),
            "margin3": r.get("margin3_order_probability", np.nan),
        })

    d = pd.DataFrame(rows)
    d.to_csv(TABLE_DIR / "figure_order_probability_data.csv", index=False, encoding="utf-8-sig")

    if d.empty:
        return

    x = np.arange(len(d))

    plt.figure(figsize=(12, 5))
    plt.plot(x, d["weak"], marker="o", label="weak order")
    plt.plot(x, d["strict"], marker="o", label="strict order")
    plt.plot(x, d["margin3"], marker="o", label="margin >= 3")
    plt.xticks(x, d["label"], rotation=35, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Order probability")
    plt.title("Patch-formation order probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_v2_6_order_probability.png", dpi=300)
    plt.close()


def plot_spatial_shuffle() -> None:
    path = Path(
        "results/v2_5_controls/spatial_shuffle_raw_effects/"
        "core_spatial_organization_shuffle_effects.csv"
    )

    df = read_csv_safe(path)

    if df.empty:
        return

    core = df[
        df["indicator"].isin([
            "prepatch_moran_i",
            "prepatch_local_neighbor_corr_mean",
            "pwsi_control",
        ])
    ].copy()

    core = core[
        core["window"].isin([
            "full_precursor_window",
            "late_precursor_window",
        ])
    ].copy()

    if core.empty:
        return

    grouped = (
        core.groupby(["system", "indicator"])["original_minus_shuffle"]
        .mean()
        .reset_index()
    )

    grouped.to_csv(TABLE_DIR / "figure_spatial_shuffle_data.csv", index=False, encoding="utf-8-sig")

    labels = [f"{r.system}\n{r.indicator}" for _, r in grouped.iterrows()]
    values = grouped["original_minus_shuffle"].to_numpy()

    plt.figure(figsize=(11, 5))
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(values)), labels, rotation=35, ha="right")
    plt.ylabel("Original - shuffled")
    plt.title("Spatial-shuffle attenuation of core spatial indicators")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_v2_6_spatial_shuffle_raw_effect.png", dpi=300)
    plt.close()


def plot_stable_window() -> None:
    path = Path(
        "results/v2_5_controls/stable_window_false_alarm_summary/"
        "stable_window_false_alarm_effect_all.csv"
    )

    df = read_csv_safe(path)

    if df.empty:
        return

    grouped = (
        df.groupby(["system", "stage"])[
            ["far_stable_alarm_rate", "late_precursor_alarm_rate"]
        ]
        .mean()
        .reset_index()
    )

    grouped.to_csv(TABLE_DIR / "figure_stable_window_data.csv", index=False, encoding="utf-8-sig")

    labels = [f"{r.system}\n{r.stage}" for _, r in grouped.iterrows()]
    x = np.arange(len(grouped))

    plt.figure(figsize=(12, 5))
    plt.plot(x, grouped["far_stable_alarm_rate"], marker="o", label="far stable")
    plt.plot(x, grouped["late_precursor_alarm_rate"], marker="o", label="late precursor")
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Alarm rate")
    plt.title("Stable-window false-alarm control")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_v2_6_stable_window_effect.png", dpi=300)
    plt.close()


def plot_event_time_permutation() -> None:
    path = Path(
        "results/v2_5_controls/event_time_permutation_summary/"
        "event_time_permutation_effect_all.csv"
    )

    df = read_csv_safe(path)

    if df.empty:
        return

    # Use most interpretable windows:
    # Vegetation: early precursor
    # SEIR: early for prepatch/PWSI, late for dynamic/visible
    rows = []

    for _, r in df.iterrows():
        system = str(r["system"])
        window = str(r["window"])
        stage = str(r["stage"])

        use = False

        if system == "vegetation" and window == "early_precursor_window":
            use = True

        if system == "seir":
            if stage in ["prepatch_spatial_organization", "pwsi_application_index"] and window == "early_precursor_window":
                use = True
            if stage in ["dynamic_patch", "visible_patch"] and window == "late_precursor_window":
                use = True

        if use:
            rows.append(r)

    p = pd.DataFrame(rows)

    if p.empty:
        return

    p.to_csv(TABLE_DIR / "figure_event_time_permutation_data.csv", index=False, encoding="utf-8-sig")

    labels = [f"{r.system}\n{r.window}\n{r.stage}" for _, r in p.iterrows()]
    values = p["real_minus_permuted"].to_numpy()

    plt.figure(figsize=(13, 5))
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(values)), labels, rotation=35, ha="right")
    plt.ylabel("Real alarm rate - permuted alarm rate")
    plt.title("Event-time permutation control")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_v2_6_event_time_permutation.png", dpi=300)
    plt.close()


def write_summary_markdown(stage_df: pd.DataFrame, order_df: pd.DataFrame) -> None:
    lines = []

    lines.append("# v2.6 Final Patch-formation Summary")
    lines.append("")
    lines.append("## Main conclusion")
    lines.append("")
    lines.append(
        "The project focus has shifted from pure prediction performance to patch-formation process analysis for precursor warning."
    )
    lines.append("")
    lines.append(
        "Vegetation CA shows a clearer gradual process: prepatch spatial organization -> dynamic restructuring -> visible fragmentation."
    )
    lines.append("")
    lines.append(
        "SEIR shows an early prepatch/PWSI layer followed by rapid dynamic spreading and visible patch manifestation."
    )
    lines.append("")
    lines.append("## Generated outputs")
    lines.append("")
    lines.append("- tables/cross_system_stage_lead_summary.csv")
    lines.append("- tables/cross_system_order_tests.csv")
    lines.append("- figures/fig_v2_6_stage_lead_time.png")
    lines.append("- figures/fig_v2_6_order_probability.png")
    lines.append("- figures/fig_v2_6_spatial_shuffle_raw_effect.png")
    lines.append("- figures/fig_v2_6_stable_window_effect.png")
    lines.append("- figures/fig_v2_6_event_time_permutation.png")
    lines.append("")
    lines.append("## How to interpret")
    lines.append("")
    lines.append("1. Stage lead-time figures show when each patch-formation layer appears before transition.")
    lines.append("2. Order probability figures show whether the stages follow the expected formation sequence.")
    lines.append("3. Spatial-shuffle controls validate that prepatch indicators depend on spatial organization.")
    lines.append("4. Stable-window controls show stage specificity of dynamic and visible patch indicators.")
    lines.append("5. Event-time permutation controls test whether precursor signals align with true event times.")
    lines.append("")

    (OUTDIR / "v2_6_final_patch_formation_summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main():
    print("[v2.6] building stage summary")
    stage_df = build_stage_summary()

    print("[v2.6] building order summary")
    order_df = build_order_summary()

    print("[v2.6] building control summary")
    build_control_summary()

    print("[v2.6] plotting stage lead time")
    plot_stage_lead(stage_df)

    print("[v2.6] plotting order probability")
    plot_order_probability(order_df)

    print("[v2.6] plotting spatial shuffle")
    plot_spatial_shuffle()

    print("[v2.6] plotting stable-window control")
    plot_stable_window()

    print("[v2.6] plotting event-time permutation")
    plot_event_time_permutation()

    write_summary_markdown(stage_df, order_df)

    print(f"[v2.6] saved outputs to: {OUTDIR}")


if __name__ == "__main__":
    main()
