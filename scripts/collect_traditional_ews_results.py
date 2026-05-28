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

TARGET_GROUP = "traditional_ews_only"
SEEDS = [42, 123, 2026, 3407, 7777]

METRIC_KEYS = [
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

    for run_dir in RESULTS_DIR.iterdir():
        if not run_dir.is_dir():
            continue

        info = parse_run_name(run_dir.name)

        if info is None:
            continue

        if info["feature_group"] != TARGET_GROUP:
            continue

        if info["sims"] != TARGET_SIMS:
            continue

        if info["grid"] != TARGET_GRID:
            continue

        if info["steps"] != TARGET_STEPS:
            continue

        if info["horizon"] != TARGET_HORIZON:
            continue

        metric_path = run_dir / "validation_metrics.txt"

        if not metric_path.exists():
            print(f"NO METRICS: {run_dir.name}")
            continue

        metrics = parse_metrics_file(metric_path)

        row = {
            "Seed": info["seed"],
            "Method": "Traditional EWS",
            "Feature_Group": info["feature_group"],
            "Num_Sims": info["sims"],
            "Grid": info["grid"],
            "Sim_Steps": info["steps"],
            "Horizon": info["horizon"],
            "Run_Dir": str(run_dir),
        }

        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("No traditional EWS results were collected.")

    df = df.sort_values("Seed").reset_index(drop=True)

    return df


def check_completeness(df: pd.DataFrame):
    print()
    print("===== Traditional EWS Completeness Check =====")

    count = 0

    for seed in SEEDS:
        hit = df[df["Seed"] == seed]

        if hit.empty:
            print(f"MISSING seed={seed}")
        else:
            count += 1

    print(f"Valid traditional EWS rows: {count}")

    return count


def make_grouped_summary(df: pd.DataFrame):
    metrics = ["AUC", "AUPRC", "ACC", "F1", "Precision", "Recall"]

    grouped = (
        df.groupby("Method")[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )

    grouped.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in grouped.columns
    ]

    return grouped


def main():
    df = collect_results()
    count = check_completeness(df)

    grouped = make_grouped_summary(df)

    output_all = SUMMARY_DIR / "traditional_ews_summary_all.csv"
    output_grouped = SUMMARY_DIR / "traditional_ews_summary_grouped.csv"

    df.to_csv(output_all, index=False)
    grouped.to_csv(output_grouped, index=False)

    print()
    print("===== Traditional EWS: All Results =====")
    print(df[["Seed", "AUC", "AUPRC", "F1", "Precision", "Recall"]].to_string(index=False))

    print()
    print("===== Traditional EWS: Grouped =====")
    print(grouped.to_string(index=False))

    print()
    print("Saved to:")
    print(output_all)
    print(output_grouped)

    if count != 5:
        print()
        print("WARNING: Expected 5 traditional EWS rows.")


if __name__ == "__main__":
    main()