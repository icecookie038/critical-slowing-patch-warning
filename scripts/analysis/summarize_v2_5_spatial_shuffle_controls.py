import pandas as pd
from pathlib import Path

runs = [
    ("vegetation", 42, "0p50", Path("results/v2_5_controls/spatial_shuffle/vegetation_seed42_thr0p50")),
    ("vegetation", 123, "0p50", Path("results/v2_5_controls/spatial_shuffle/vegetation_seed123_thr0p50")),
    ("seir", 2026, "0p20", Path("results/v2_5_controls/spatial_shuffle/seir_seed2026_thr0p20")),
]

rows = []

for system, seed, threshold, d in runs:
    path = d / "spatial_shuffle_effect_summary.csv"

    if not path.exists():
        print(f"Skip missing: {path}")
        continue

    df = pd.read_csv(path)
    df["system"] = system
    df["seed"] = seed
    df["threshold"] = threshold
    rows.append(df)

if not rows:
    raise SystemExit("No valid spatial-shuffle effect files found.")

all_df = pd.concat(rows, ignore_index=True)

outdir = Path("results/v2_5_controls/spatial_shuffle_summary")
outdir.mkdir(parents=True, exist_ok=True)

all_df.to_csv(outdir / "spatial_shuffle_effect_all_systems.csv", index=False, encoding="utf-8-sig")

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

print("\n=== Spatial-shuffle effect summary across systems ===")
print(all_df[cols].to_string(index=False))

print("\n=== Mean effect by system and stage ===")
print(
    all_df.groupby(["system", "stage"])[
        ["detection_rate_drop", "lead_median_drop"]
    ].mean().reset_index().to_string(index=False)
)
