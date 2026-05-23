from pathlib import Path
import re
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

RESULT_DIRS = [
    ROOT / "results",
    ROOT / "results_baselines",
]

OUT_DIR = ROOT / "results_summary"
OUT_DIR.mkdir(exist_ok=True)


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
    "LeadTime_Mean",
    "LeadTime_Median",
    "LeadTime_Q25",
    "LeadTime_Q75",
]


def extract_from_name(path: Path) -> dict:
    text = str(path)

    seed_match = re.search(r"seed(\d+)", text)
    h_match = re.search(r"_h(\d+)", text)
    mode_match = re.search(r"mode([A-Za-z0-9_]+)", text)

    seed = int(seed_match.group(1)) if seed_match else None
    horizon = int(h_match.group(1)) if h_match else None
    mode = mode_match.group(1) if mode_match else None

    if mode is not None:
        mode = mode.replace("_rw0", "")
        mode = mode.replace("_wd0", "")

    return {
        "Seed": seed,
        "Horizon": horizon,
        "Feature_Mode": mode,
    }


def parse_summary_txt(path: Path) -> dict:
    record = {
        "Source_File": str(path.relative_to(ROOT)),
        "Result_Type": "deep_model",
    }
    record.update(extract_from_name(path))

    text = path.read_text(encoding="utf-8", errors="ignore")

    patterns = {
        "AUC": r"AUC:\s*([0-9.]+)",
        "AUPRC": r"AUPRC:\s*([0-9.]+)",
        "ACC": r"ACC:\s*([0-9.]+)",
        "F1": r"F1:\s*([0-9.]+)",
        "Precision": r"Precision:\s*([0-9.]+)",
        "Recall": r"Recall:\s*([0-9.]+)",
        "Best_Threshold": r"Best_Threshold:\s*([0-9.]+)",
        "Positive_Rate": r"Positive_Rate:\s*([0-9.]+)",
        "TN": r"TN:\s*([0-9.]+)",
        "FP": r"FP:\s*([0-9.]+)",
        "FN": r"FN:\s*([0-9.]+)",
        "TP": r"TP:\s*([0-9.]+)",
        "LeadTime_Mean": r"LeadTime_Mean:\s*([0-9.]+)",
        "LeadTime_Median": r"LeadTime_Median:\s*([0-9.]+)",
        "LeadTime_Q25": r"LeadTime_Q25:\s*([0-9.]+)",
        "LeadTime_Q75": r"LeadTime_Q75:\s*([0-9.]+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        record[key] = float(match.group(1)) if match else None

    return record


def collect_deep_results() -> pd.DataFrame:
    rows = []

    for results_dir in RESULT_DIRS:
        if not results_dir.exists():
            continue

        for path in results_dir.rglob("*final_summary.txt"):
            rows.append(parse_summary_txt(path))

    return pd.DataFrame(rows)


def collect_baseline_results() -> pd.DataFrame:
    rows = []

    baseline_root = ROOT / "results_baselines"
    if not baseline_root.exists():
        return pd.DataFrame()

    for csv_path in baseline_root.rglob("baseline_metrics.csv"):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        for _, row in df.iterrows():
            record = {
                "Source_File": str(csv_path.relative_to(ROOT)),
                "Result_Type": "baseline",
                "Model": row.get("Model", None),
                "Feature_Mode": row.get("Feature_Mode", None),
                "Seed": row.get("Seed", None),
                "Horizon": row.get("Horizon", None),
            }

            for key in METRIC_KEYS:
                record[key] = row.get(key, None)

            rows.append(record)

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "AUC",
        "AUPRC",
        "ACC",
        "F1",
        "Precision",
        "Recall",
        "LeadTime_Mean",
        "LeadTime_Median",
        "LeadTime_Q25",
        "LeadTime_Q75",
    ]

    group_cols = ["Result_Type", "Model", "Feature_Mode"]

    existing_group_cols = [c for c in group_cols if c in df.columns]
    existing_metric_cols = [c for c in metric_cols if c in df.columns]

    if df.empty or not existing_metric_cols:
        return pd.DataFrame()

    summary = (
        df.groupby(existing_group_cols, dropna=False)[existing_metric_cols]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    summary.columns = [
        "_".join([str(x) for x in col if str(x) != ""])
        for col in summary.columns.to_flat_index()
    ]

    return summary


def main():
    deep_df = collect_deep_results()
    baseline_df = collect_baseline_results()

    all_df = pd.concat([deep_df, baseline_df], ignore_index=True, sort=False)

    if all_df.empty:
        print("No result files found.")
        print("Please check whether results/ or results_baselines/ contains final_summary.txt or baseline_metrics.csv.")
        return

    all_path = OUT_DIR / "robustness_summary_all.csv"
    grouped_path = OUT_DIR / "robustness_summary_grouped.csv"

    all_df.to_csv(all_path, index=False, encoding="utf-8-sig")

    grouped_df = summarize(all_df)
    grouped_df.to_csv(grouped_path, index=False, encoding="utf-8-sig")

    print("Robustness result collection finished.")
    print(f"Saved all results to: {all_path}")
    print(f"Saved grouped summary to: {grouped_path}")
    print()
    print("Preview:")
    print(all_df.head())


if __name__ == "__main__":
    main()