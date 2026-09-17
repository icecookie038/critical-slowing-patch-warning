from pathlib import Path
import re
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

RESULT_DIRS = [
    ROOT / "results",
    ROOT / "results_baselines",
    ROOT / "scripts" / "results",
    ROOT / "scripts" / "results_baselines",
]

OUT_DIR = ROOT / "results_summary"
OUT_DIR.mkdir(exist_ok=True)


OUTPUT_ALL = OUT_DIR / "robustness_summary_all.csv"
OUTPUT_GROUPED = OUT_DIR / "robustness_summary_grouped.csv"


COLUMN_ORDER = [
    "Source_File",
    "Result_Type",
    "Seed",
    "Horizon",
    "Feature_Mode",
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
    "LeadTime_Mean",
    "LeadTime_Median",
    "LeadTime_Q25",
    "LeadTime_Q75",
    "Model",
]


METRIC_NAME_MAP = {
    "auc": "AUC",
    "auprc": "AUPRC",
    "acc": "ACC",
    "accuracy": "ACC",
    "f1": "F1",
    "precision": "Precision",
    "recall": "Recall",
    "best threshold": "Best_Threshold",
    "best_threshold": "Best_Threshold",
    "positive rate": "Positive_Rate",
    "positive_rate": "Positive_Rate",
    "tn": "TN",
    "fp": "FP",
    "fn": "FN",
    "tp": "TP",
    "leadtime mean": "LeadTime_Mean",
    "leadtime_mean": "LeadTime_Mean",
    "leadtime median": "LeadTime_Median",
    "leadtime_median": "LeadTime_Median",
    "leadtime q25": "LeadTime_Q25",
    "leadtime_q25": "LeadTime_Q25",
    "leadtime q75": "LeadTime_Q75",
    "leadtime_q75": "LeadTime_Q75",
}


def safe_float(value):
    value = str(value).strip()
    value = value.replace("%", "")
    try:
        return float(value)
    except ValueError:
        return np.nan


def normalize_metric_key(key):
    key = key.strip().lower()
    key = key.replace("_", " ")
    key = re.sub(r"\s+", " ", key)
    return key


def parse_validation_metrics(metric_file: Path) -> dict:
    metrics = {}

    try:
        lines = metric_file.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception as e:
        print(f"[warning] failed to read {metric_file}: {e}")
        return metrics

    for line in lines:
        if ":" not in line:
            continue

        raw_key, raw_value = line.split(":", 1)
        key_norm = normalize_metric_key(raw_key)

        if key_norm in METRIC_NAME_MAP:
            col = METRIC_NAME_MAP[key_norm]
            metrics[col] = safe_float(raw_value)

    return metrics


def extract_info_from_path(metric_file: Path) -> dict:
    folder_name = metric_file.parent.name

    seed = np.nan
    horizon = np.nan
    feature_mode = "unknown"

    seed_match = re.search(r"seed(\d+)", folder_name, flags=re.IGNORECASE)
    if seed_match:
        seed = int(seed_match.group(1))

    horizon_match = re.search(r"(?:^|_)h(\d+)(?:_|$)", folder_name, flags=re.IGNORECASE)
    if horizon_match:
        horizon = int(horizon_match.group(1))

    mode_match = re.search(
        r"_mode(full|img_only|patch_only)(?:_|$)",
        folder_name,
        flags=re.IGNORECASE,
    )
    if mode_match:
        feature_mode = mode_match.group(1)

    try:
        source_file = str(metric_file.parent.relative_to(ROOT))
    except ValueError:
        source_file = str(metric_file.parent)

    return {
        "Source_File": source_file,
        "Result_Type": "deep_model",
        "Seed": seed,
        "Horizon": horizon,
        "Feature_Mode": feature_mode,
        "Model": np.nan,
    }


def collect_metric_files():
    metric_files = []

    for result_dir in RESULT_DIRS:
        if not result_dir.exists():
            print(f"[skip] missing directory: {result_dir}")
            continue

        found = list(result_dir.rglob("validation_metrics.txt"))
        print(f"[scan] {result_dir} -> found {len(found)} validation_metrics.txt")
        metric_files.extend(found)

    return metric_files


def main():
    metric_files = collect_metric_files()

    print(f"\nTotal metric files found: {len(metric_files)}")

    rows = []

    for metric_file in metric_files:
        print(f"[read] {metric_file}")

        row = extract_info_from_path(metric_file)
        metrics = parse_validation_metrics(metric_file)

        row.update(metrics)
        rows.append(row)

    if not rows:
        print("No metric files collected. Nothing to save.")
        return

    df = pd.DataFrame(rows)

    for col in COLUMN_ORDER:
        if col not in df.columns:
            df[col] = np.nan

    df = df[COLUMN_ORDER]

    df = df.sort_values(
        by=["Seed", "Feature_Mode", "Horizon", "Source_File"],
        na_position="last",
    )

    df.to_csv(OUTPUT_ALL, index=False, encoding="utf-8-sig")

    numeric_cols = [
        "AUC",
        "AUPRC",
        "ACC",
        "F1",
        "Precision",
        "Recall",
        "Best_Threshold",
        "Positive_Rate",
        "LeadTime_Mean",
        "LeadTime_Median",
        "LeadTime_Q25",
        "LeadTime_Q75",
    ]

    grouped_cols = ["Result_Type", "Horizon", "Feature_Mode"]

    grouped = (
        df.groupby(grouped_cols, dropna=False)[numeric_cols]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    grouped.columns = [
        "_".join([str(x) for x in col if str(x) != ""]).strip("_")
        for col in grouped.columns
    ]

    grouped.to_csv(OUTPUT_GROUPED, index=False, encoding="utf-8-sig")

    print("\nRobustness result collection finished.")
    print(f"Saved all results to: {OUTPUT_ALL}")
    print(f"Saved grouped summary to: {OUTPUT_GROUPED}")

    print("\nPreview:")
    print(df.head(20))

    print("\nSeeds collected:")
    print(sorted(df["Seed"].dropna().unique()))


if __name__ == "__main__":
    main()