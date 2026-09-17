import pandas as pd
from pathlib import Path

base = Path("results/v2_4_integrated_patch_dynamics")
threshold_tags = ["0p10", "0p20", "0p30", "0p50"]

stage_rows = []
order_rows = []
core_rows = []

for tag in threshold_tags:
    d = base / f"seir_seed2026_thr{tag}"

    stage_path = d / "formation_stage_summary.csv"
    order_path = d / "formation_order_summary.csv"
    core_path = d / "core_representative_indicators_summary.csv"

    if not stage_path.exists():
        print(f"Skip thr{tag}: missing {stage_path}")
        continue
    if not order_path.exists():
        print(f"Skip thr{tag}: missing {order_path}")
        continue
    if not core_path.exists():
        print(f"Skip thr{tag}: missing {core_path}")
        continue

    stage = pd.read_csv(stage_path)
    stage["threshold"] = tag
    stage_rows.append(stage)

    order = pd.read_csv(order_path)
    order["threshold"] = tag
    order_rows.append(order)

    core = pd.read_csv(core_path)
    core["threshold"] = tag
    core_rows.append(core)

if not stage_rows:
    raise SystemExit("No valid threshold results found.")

stage_all = pd.concat(stage_rows, ignore_index=True)
order_all = pd.concat(order_rows, ignore_index=True)
core_all = pd.concat(core_rows, ignore_index=True)

outdir = base / "seir_seed2026_threshold_summary"
outdir.mkdir(parents=True, exist_ok=True)

stage_all.to_csv(outdir / "stage_summary_all_thresholds.csv", index=False, encoding="utf-8-sig")
order_all.to_csv(outdir / "order_summary_all_thresholds.csv", index=False, encoding="utf-8-sig")
core_all.to_csv(outdir / "core_indicators_all_thresholds.csv", index=False, encoding="utf-8-sig")

print("\n=== Stage summary by threshold ===")
print(stage_all.to_string(index=False))

print("\n=== Order summary by threshold ===")
print(order_all.to_string(index=False))

print("\n=== Selected core indicators by threshold ===")
selected = core_all[core_all["indicator"].isin([
    "prepatch_moran_i",
    "prepatch_sync_edge_ratio",
    "pwsi_equal",
    "pwsi_z_conn",
    "PSII",
    "BCI",
    "PDSI",
    "edge_density",
    "patch_count",
])].copy()

cols = [
    "threshold",
    "indicator_group",
    "indicator",
    "detection_rate",
    "lead_median",
    "lead_q25",
    "lead_q75",
]
print(selected[cols].to_string(index=False))
