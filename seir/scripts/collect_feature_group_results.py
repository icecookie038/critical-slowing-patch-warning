from pathlib import Path
import re
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "scripts" / "results"
SUMMARY_DIR = ROOT / "results_summary"

SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


TARGET_SIMS = 300
TARGET_GRID = 64
TARGET_STEPS = 200
TARGET_HORIZON = 30

FEATURE_GROUPS = [
    "image_only",
    "classic_patch_only",
    "dynamic_patch_only",
    "classic_dynamic_patch",
    "full",
]

SEEDS = [42, 123, 2026, 3407, 7777]


METRIC_KEYS = [
    "RMSE",
    "MAE",
    "R2",
    "AUC",
    "AUPRC",
    "ACC",
    "F1",
    "Precision",
    "Recall",
    "Best_Threshold",
    "Positive_Rate",
    "TN",
    "FP",
    "FN",
    "TP",
]


def parse_run_name(name: str):
    pattern = re.compile(
        r"seed(?P<seed>\d+)_"
        r"sims(?P<sims>\d+)_"
        r"L(?P<grid>\d+)_"
        r"steps(?P<steps>\d+)_"
        r"h(?P<horizon>\d+)_"
        r"group(?P<feature_group>.+?)_"
        r"rw(?P<reg_weight>[-+]?\d*\.?\d+)_"
        r"wd(?P<weight_decay>[-+]?\d*\.?\d+)"
    )

    match = pattern.match(name)

    if not match:
        return None

    info = match.groupdict()

    info["seed"] = int(info["seed"])
    info["sims"] = int(info["sims"])
    info["grid"] = int(info["grid"])
    info["steps"] = int(info["steps"])
    info["horizon"] = int(info["horizon"])
    info["reg_weight"] = float(info["reg_weight"])
    info["weight_decay"] = float(info["weight_decay"])

    return info


def parse_metrics_file(path: Path):
    metrics = {}

    if not path.exists():
        return metrics

    text = path.read_text(encoding="utf-8", errors="ignore")

    for line in text.splitlines():
        if ":" not in line:
            continue

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()

        if key in METRIC_KEYS:
            try:
                metrics[key] = float(value)
            except ValueError:
                pass

    return metrics


def collect_results():
    rows = []

    if not RESULTS_DIR.exists():
        raise FileNotFoundError(f"Results directory does not exist: {RESULTS_DIR}")

    for run_dir in RESULTS_DIR.iterdir():
        if not run_dir.is_dir():
            continue

        info = parse_run_name(run_dir.name)

        if info is None:
            continue

        # Exclude small pilot tests such as sims20_L32_steps80_h20
        if info["sims"] != TARGET_SIMS:
            continue

        if info["grid"] != TARGET_GRID:
            continue

        if info["steps"] != TARGET_STEPS:
            continue

        if info["horizon"] != TARGET_HORIZON:
            continue

        if info["feature_group"] not in FEATURE_GROUPS:
            continue

        metric_path = run_dir / "validation_metrics.txt"

        if not metric_path.exists():
            print(f"NO METRICS: {run_dir.name}")
            continue

        metrics = parse_metrics_file(metric_path)

        row = {
            "Seed": info["seed"],
            "Num_Sims": info["sims"],
            "Grid": info["grid"],
            "Sim_Steps": info["steps"],
            "Horizon": info["horizon"],
            "Feature_Group": info["feature_group"],
            "Reg_Weight": info["reg_weight"],
            "Weight_Decay": info["weight_decay"],
            "Run_Dir": str(run_dir),
        }

        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("No valid feature-group results were collected.")

    df = df.sort_values(["Feature_Group", "Seed"]).reset_index(drop=True)

    return df


def check_completeness(df: pd.DataFrame):
    print()
    print("===== Completeness Check =====")

    valid_count = 0

    for seed in SEEDS:
        for group in FEATURE_GROUPS:
            hit = df[
                (df["Seed"] == seed)
                & (df["Feature_Group"] == group)
            ]

            if hit.empty:
                print(f"MISSING seed={seed} group={group}")
            else:
                valid_count += 1

    print(f"Valid feature-group rows: {valid_count}")

    return valid_count


def make_grouped_summary(df: pd.DataFrame):
    metrics = ["AUC", "AUPRC", "F1", "ACC", "Precision", "Recall"]

    existing_metrics = [m for m in metrics if m in df.columns]

    grouped = (
        df.groupby("Feature_Group")[existing_metrics]
        .agg(["mean", "std"])
        .reset_index()
    )

    grouped.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in grouped.columns
    ]

    order = {
        "image_only": 0,
        "classic_patch_only": 1,
        "dynamic_patch_only": 2,
        "classic_dynamic_patch": 3,
        "full": 4,
    }

    grouped["Order"] = grouped["Feature_Group"].map(order)
    grouped = grouped.sort_values("Order").drop(columns=["Order"])

    return grouped


def main():
    df = collect_results()

    valid_count = check_completeness(df)

    output_all = SUMMARY_DIR / "feature_group_summary_all.csv"
    output_grouped = SUMMARY_DIR / "feature_group_summary_grouped.csv"

    grouped = make_grouped_summary(df)

    df.to_csv(output_all, index=False)
    grouped.to_csv(output_grouped, index=False)

    print()
    print("===== Feature Group Summary: All Results =====")
    print(df[["Seed", "Feature_Group", "AUC", "AUPRC", "F1"]].to_string(index=False))

    print()
    print("===== Feature Group Summary: Grouped =====")
    print(grouped.to_string(index=False))

    print()
    print("Saved to:")
    print(output_all)
    print(output_grouped)

    if valid_count != 25:
        print()
        print("WARNING: Expected 25 valid rows for v0.9.")


if __name__ == "__main__":
    main()