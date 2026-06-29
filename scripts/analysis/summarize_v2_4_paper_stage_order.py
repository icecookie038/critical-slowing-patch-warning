import pandas as pd
import numpy as np
from pathlib import Path

base = Path("results/v2_4_integrated_patch_dynamics")
seeds = [42, 123]
tag = "thr0p50"

STAGES = {
    "prepatch_spatial_organization": [
        "prepatch_moran_i",
        "prepatch_sync_edge_ratio",
        "prepatch_local_neighbor_corr_mean",
        "prepatch_svd_spectral_gap",
    ],
    "dynamic_restructuring_no_bci": [
        "PSII",
        "DPCR",
        "PDSI",
    ],
    "boundary_hardening": [
        "edge_density",
        "prepatch_boundary_sharpness",
    ],
    "visible_fragmentation": [
        "patch_count",
    ],
    "pwsi_application_index": [
        "pwsi_equal",
        "pwsi_z_conn",
    ],
}

ORDER_TESTS = [
    (
        "prepatch <= dynamic_no_bci <= visible_fragmentation",
        [
            "prepatch_spatial_organization",
            "dynamic_restructuring_no_bci",
            "visible_fragmentation",
        ],
    ),
    (
        "prepatch <= boundary_hardening <= visible_fragmentation",
        [
            "prepatch_spatial_organization",
            "boundary_hardening",
            "visible_fragmentation",
        ],
    ),
    (
        "pwsi <= dynamic_no_bci <= visible_fragmentation",
        [
            "pwsi_application_index",
            "dynamic_restructuring_no_bci",
            "visible_fragmentation",
        ],
    ),
]

rows_stage = []
rows_order = []

for seed in seeds:
    d = base / f"vegetation_seed{seed}_{tag}"
    detail_path = d / "first_rise_all_indicators_detail.csv"

    if not detail_path.exists():
        print(f"Skip seed{seed}: missing {detail_path}")
        continue

    detail = pd.read_csv(detail_path)
    detail = detail[detail["detected"] == 1].copy()
    detail["t_alarm"] = pd.to_numeric(detail["t_alarm"], errors="coerce")
    detail["lead_time"] = pd.to_numeric(detail["lead_time"], errors="coerce")
    detail = detail.dropna(subset=["t_alarm", "lead_time"])

    stage_tables = []

    for stage_name, indicators in STAGES.items():
        sub = detail[detail["indicator"].isin(indicators)].copy()

        if sub.empty:
            continue

        idx = sub.groupby("sim_id")["t_alarm"].idxmin()
        stage = sub.loc[idx, ["sim_id", "indicator", "t_alarm", "lead_time"]].copy()
        stage["stage"] = stage_name
        stage["seed"] = seed

        stage_tables.append(stage)

        leads = stage["lead_time"].to_numpy()

        rows_stage.append({
            "seed": seed,
            "stage": stage_name,
            "n_detected_sims": int(stage["sim_id"].nunique()),
            "lead_mean": float(np.mean(leads)),
            "lead_median": float(np.median(leads)),
            "lead_q25": float(np.percentile(leads, 25)),
            "lead_q75": float(np.percentile(leads, 75)),
            "most_common_indicator": stage["indicator"].mode().iloc[0],
        })

    if not stage_tables:
        continue

    stage_all = pd.concat(stage_tables, ignore_index=True)

    pivot = stage_all.pivot_table(
        index="sim_id",
        columns="stage",
        values="t_alarm",
        aggfunc="min",
    )

    for order_name, stages in ORDER_TESTS:
        if not all(s in pivot.columns for s in stages):
            rows_order.append({
                "seed": seed,
                "order_test": order_name,
                "n_valid_sims": 0,
                "weak_order_probability_leq": np.nan,
                "strict_order_probability_lt": np.nan,
                "margin3_order_probability": np.nan,
            })
            continue

        valid = pivot[stages].dropna()

        if len(valid) == 0:
            rows_order.append({
                "seed": seed,
                "order_test": order_name,
                "n_valid_sims": 0,
                "weak_order_probability_leq": np.nan,
                "strict_order_probability_lt": np.nan,
                "margin3_order_probability": np.nan,
            })
            continue

        weak = np.ones(len(valid), dtype=bool)
        strict = np.ones(len(valid), dtype=bool)
        margin3 = np.ones(len(valid), dtype=bool)

        for a, b in zip(stages[:-1], stages[1:]):
            weak &= valid[a].to_numpy() <= valid[b].to_numpy()
            strict &= valid[a].to_numpy() < valid[b].to_numpy()
            margin3 &= (valid[b].to_numpy() - valid[a].to_numpy()) >= 3

        rows_order.append({
            "seed": seed,
            "order_test": order_name,
            "n_valid_sims": int(len(valid)),
            "weak_order_probability_leq": float(np.mean(weak)),
            "strict_order_probability_lt": float(np.mean(strict)),
            "margin3_order_probability": float(np.mean(margin3)),
        })

outdir = base / "vegetation_paper_stage_order_thr0p50"
outdir.mkdir(parents=True, exist_ok=True)

stage_df = pd.DataFrame(rows_stage)
order_df = pd.DataFrame(rows_order)

stage_df.to_csv(outdir / "paper_stage_summary.csv", index=False, encoding="utf-8-sig")
order_df.to_csv(outdir / "paper_stage_order_tests.csv", index=False, encoding="utf-8-sig")

print("\n=== Paper-level stage summary ===")
print(stage_df.to_string(index=False))

print("\n=== Paper-level order tests ===")
print(order_df.to_string(index=False))

print("\n=== Mean order probabilities ===")
print(
    order_df.groupby("order_test")[
        ["weak_order_probability_leq", "strict_order_probability_lt", "margin3_order_probability"]
    ].agg(["mean", "std", "min", "max"]).to_string()
)
