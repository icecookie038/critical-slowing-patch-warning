# -*- coding: utf-8 -*-
"""
Summarize v2.4 representative indicators.

Purpose:
    Avoid group-level inflation caused by selecting the earliest alarm among
    many indicators in the same group.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


CORE_INDICATORS = [
    # Prepatch representative indicators
    "prepatch_moran_i",
    "prepatch_sync_edge_ratio",
    "prepatch_local_neighbor_corr_mean",
    "prepatch_svd_spectral_gap",

    # PWSI representative indicators
    "pwsi_equal",
    "pwsi_z_conn",

    # Dynamic patch representative indicators
    "PSII",
    "BCI",
    "DPCR",
    "PDSI",

    # Visible patch representative indicators
    "edge_density",
    "patch_count",
]


ORDER_TESTS = [
    (
        "moran_i <= PSII <= patch_count",
        ["prepatch_moran_i", "PSII", "patch_count"],
    ),
    (
        "sync_edge <= PSII <= patch_count",
        ["prepatch_sync_edge_ratio", "PSII", "patch_count"],
    ),
    (
        "local_corr <= PSII <= patch_count",
        ["prepatch_local_neighbor_corr_mean", "PSII", "patch_count"],
    ),
    (
        "pwsi_equal <= PSII <= patch_count",
        ["pwsi_equal", "PSII", "patch_count"],
    ),
    (
        "z_conn <= PSII <= patch_count",
        ["pwsi_z_conn", "PSII", "patch_count"],
    ),
    (
        "moran_i <= BCI <= edge_density",
        ["prepatch_moran_i", "BCI", "edge_density"],
    ),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indir",
        default="results/v2_4_integrated_patch_dynamics/vegetation_seed42",
    )
    args = parser.parse_args()

    indir = Path(args.indir)

    summary_path = indir / "first_rise_all_indicators_summary.csv"
    detail_path = indir / "first_rise_all_indicators_detail.csv"

    summary = pd.read_csv(summary_path)
    detail = pd.read_csv(detail_path)

    summary["detection_rate"] = pd.to_numeric(summary["detection_rate"], errors="coerce")
    summary["lead_median"] = pd.to_numeric(summary["lead_median"], errors="coerce")
    summary["lead_mean"] = pd.to_numeric(summary["lead_mean"], errors="coerce")
    summary["lead_q25"] = pd.to_numeric(summary["lead_q25"], errors="coerce")
    summary["lead_q75"] = pd.to_numeric(summary["lead_q75"], errors="coerce")

    reliable = summary[
        (summary["detection_rate"] >= 0.8)
        & summary["lead_median"].notna()
    ].copy()

    reliable = reliable.sort_values(
        ["indicator_group", "lead_median", "detection_rate"],
        ascending=[True, False, False],
    )

    core = summary[summary["indicator"].isin(CORE_INDICATORS)].copy()
    core["indicator"] = pd.Categorical(
        core["indicator"],
        categories=CORE_INDICATORS,
        ordered=True,
    )
    core = core.sort_values("indicator")

    reliable.to_csv(
        indir / "reliable_indicators_detection_ge_0_8.csv",
        index=False,
        encoding="utf-8-sig",
    )

    core.to_csv(
        indir / "core_representative_indicators_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n=== Reliable indicators: detection_rate >= 0.8 ===")
    cols = [
        "indicator_group",
        "indicator",
        "detection_rate",
        "lead_mean",
        "lead_median",
        "lead_q25",
        "lead_q75",
    ]
    print(reliable[cols].to_string(index=False))

    print("\n=== Core representative indicators ===")
    print(core[cols].to_string(index=False))

    d = detail.copy()
    d = d[d["detected"] == 1].copy()
    d["t_alarm"] = pd.to_numeric(d["t_alarm"], errors="coerce")
    d = d.dropna(subset=["t_alarm"])

    rows = []

    for test_name, indicators in ORDER_TESTS:
        sub = d[d["indicator"].isin(indicators)].copy()

        pivot = sub.pivot_table(
            index="sim_id",
            columns="indicator",
            values="t_alarm",
            aggfunc="min",
        )

        if not all(x in pivot.columns for x in indicators):
            rows.append({
                "order_test": test_name,
                "n_valid_sims": 0,
                "ordering_probability": np.nan,
            })
            continue

        valid = pivot[indicators].dropna()

        if len(valid) == 0:
            rows.append({
                "order_test": test_name,
                "n_valid_sims": 0,
                "ordering_probability": np.nan,
            })
            continue

        ok = np.ones(len(valid), dtype=bool)
        for a, b in zip(indicators[:-1], indicators[1:]):
            ok &= valid[a].to_numpy() <= valid[b].to_numpy()

        rows.append({
            "order_test": test_name,
            "n_valid_sims": int(len(valid)),
            "ordering_probability": float(np.mean(ok)),
        })

    order_df = pd.DataFrame(rows)

    order_df.to_csv(
        indir / "core_representative_order_tests.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n=== Core representative order tests ===")
    print(order_df.to_string(index=False))


if __name__ == "__main__":
    main()
