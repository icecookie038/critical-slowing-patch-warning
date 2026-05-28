from pathlib import Path
import re
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "scripts" / "results"
SUMMARY_DIR = ROOT / "results_summary"
TABLE_DIR = ROOT / "docs" / "paper_tables"

SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


TARGET_SIMS = 300
TARGET_GRID = 64
TARGET_STEPS = 200
TARGET_HORIZON = 30

SEEDS = [42, 123, 2026, 3407, 7777]

FEATURE_GROUPS = [
    "traditional_ews_only",
    "image_only",
    "classic_patch_only",
    "dynamic_patch_only",
    "classic_dynamic_patch",
    "full",
]

DISPLAY_NAMES = {
    "traditional_ews_only": "Traditional EWS",
    "image_only": "Image only",
    "classic_patch_only": "Classic patch",
    "dynamic_patch_only": "Dynamic patch v1",
    "classic_dynamic_patch": "Classic + dynamic patch",
    "full": "Full",
}


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


def read_best_threshold(metrics_path: Path, default: float = 0.5) -> float:
    if not metrics_path.exists():
        return default

    text = metrics_path.read_text(encoding="utf-8", errors="ignore")

    for line in text.splitlines():
        if line.startswith("Best_Threshold"):
            try:
                return float(line.split(":", 1)[1].strip())
            except Exception:
                return default

    return default


def safe_mean(x):
    x = np.asarray(x, dtype=float)

    if x.size == 0:
        return np.nan

    return float(np.mean(x))


def safe_median(x):
    x = np.asarray(x, dtype=float)

    if x.size == 0:
        return np.nan

    return float(np.median(x))


def safe_std(x):
    x = np.asarray(x, dtype=float)

    if x.size <= 1:
        return 0.0 if x.size == 1 else np.nan

    return float(np.std(x, ddof=1))


def analyze_one_run(run_dir: Path, info: dict):
    pred_path = run_dir / "validation_predictions.npz"
    metrics_path = run_dir / "validation_metrics.txt"

    if not pred_path.exists():
        print(f"NO PREDICTIONS: {run_dir.name}")
        return None

    data = np.load(pred_path, allow_pickle=True)

    required = ["y_remaining_true", "y_risk_true", "risk_prob"]

    for key in required:
        if key not in data.files:
            print(f"MISSING KEY {key}: {run_dir.name}")
            return None

    y_remaining = np.asarray(data["y_remaining_true"], dtype=float)
    y_true = np.asarray(data["y_risk_true"], dtype=int)
    risk_prob = np.asarray(data["risk_prob"], dtype=float)

    threshold = read_best_threshold(metrics_path, default=0.5)

    y_pred = (risk_prob >= threshold).astype(int)

    true_risk = y_true == 1
    predicted_alarm = y_pred == 1
    correct_alarm = true_risk & predicted_alarm
    missed_alarm = true_risk & (~predicted_alarm)
    false_alarm = (~true_risk) & predicted_alarm

    correct_lead_times = y_remaining[correct_alarm]
    missed_lead_times = y_remaining[missed_alarm]

    row = {
        "Seed": info["seed"],
        "Feature_Group": info["feature_group"],
        "Method": DISPLAY_NAMES.get(info["feature_group"], info["feature_group"]),
        "Threshold": threshold,
        "Num_Samples": len(y_true),
        "Num_True_Risk": int(true_risk.sum()),
        "Num_Predicted_Alarm": int(predicted_alarm.sum()),
        "Num_Correct_Alarm": int(correct_alarm.sum()),
        "Num_False_Alarm": int(false_alarm.sum()),
        "Num_Missed_Alarm": int(missed_alarm.sum()),
        "Alarm_Recall": float(correct_alarm.sum() / max(true_risk.sum(), 1)),
        "Alarm_Precision": float(correct_alarm.sum() / max(predicted_alarm.sum(), 1)),
        "LeadTime_Mean": safe_mean(correct_lead_times),
        "LeadTime_Median": safe_median(correct_lead_times),
        "LeadTime_Std": safe_std(correct_lead_times),
        "LeadTime_Q25": float(np.percentile(correct_lead_times, 25)) if correct_lead_times.size else np.nan,
        "LeadTime_Q75": float(np.percentile(correct_lead_times, 75)) if correct_lead_times.size else np.nan,
        "MissedLeadTime_Mean": safe_mean(missed_lead_times),
        "Run_Dir": str(run_dir),
    }

    return row


def collect_lead_time_results():
    rows = []

    for run_dir in RESULTS_DIR.iterdir():
        if not run_dir.is_dir():
            continue

        info = parse_run_name(run_dir.name)

        if info is None:
            continue

        if info["feature_group"] not in FEATURE_GROUPS:
            continue

        if info["sims"] != TARGET_SIMS:
            continue

        if info["grid"] != TARGET_GRID:
            continue

        if info["steps"] != TARGET_STEPS:
            continue

        if info["horizon"] != TARGET_HORIZON:
            continue

        row = analyze_one_run(run_dir, info)

        if row is not None:
            rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("No lead time results collected.")

    order = {g: i for i, g in enumerate(FEATURE_GROUPS)}
    df["Order"] = df["Feature_Group"].map(order)
    df = df.sort_values(["Order", "Seed"]).drop(columns=["Order"]).reset_index(drop=True)

    return df


def check_completeness(df: pd.DataFrame):
    print()
    print("===== Lead Time Completeness Check =====")

    valid_count = 0

    for group in FEATURE_GROUPS:
        for seed in SEEDS:
            hit = df[(df["Feature_Group"] == group) & (df["Seed"] == seed)]

            if hit.empty:
                print(f"MISSING group={group} seed={seed}")
            else:
                valid_count += 1

    print(f"Valid lead-time rows: {valid_count}")

    return valid_count


def make_grouped_summary(df: pd.DataFrame):
    metrics = [
        "Alarm_Recall",
        "Alarm_Precision",
        "LeadTime_Mean",
        "LeadTime_Median",
        "LeadTime_Std",
        "LeadTime_Q25",
        "LeadTime_Q75",
    ]

    grouped = (
        df.groupby(["Method", "Feature_Group"])[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )

    grouped.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in grouped.columns
    ]

    order = {g: i for i, g in enumerate(FEATURE_GROUPS)}
    grouped["Order"] = grouped["Feature_Group"].map(order)
    grouped = grouped.sort_values("Order").drop(columns=["Order"]).reset_index(drop=True)

    return grouped


def fmt_mean_std(row, metric: str) -> str:
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    if mean_col not in row or std_col not in row:
        return "-"

    mean = row[mean_col]
    std = row[std_col]

    if pd.isna(mean):
        return "-"

    if pd.isna(std):
        return f"{mean:.2f}"

    return f"{mean:.2f} ± {std:.2f}"


def export_table(grouped: pd.DataFrame):
    rows = []

    for _, row in grouped.iterrows():
        rows.append(
            {
                "Method": row["Method"],
                "Alarm recall": fmt_mean_std(row, "Alarm_Recall"),
                "Alarm precision": fmt_mean_std(row, "Alarm_Precision"),
                "Mean lead time": fmt_mean_std(row, "LeadTime_Mean"),
                "Median lead time": fmt_mean_std(row, "LeadTime_Median"),
                "Lead time Q25": fmt_mean_std(row, "LeadTime_Q25"),
                "Lead time Q75": fmt_mean_std(row, "LeadTime_Q75"),
            }
        )

    table = pd.DataFrame(rows)

    out_csv = TABLE_DIR / "lead_time_proxy_table.csv"
    out_md = TABLE_DIR / "lead_time_proxy_table.md"
    out_tex = TABLE_DIR / "lead_time_proxy_table.tex"

    table.to_csv(out_csv, index=False)

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Lead Time Proxy Table\n\n")
        f.write(
            "This table reports a sample-level lead-time proxy based on "
            "`y_remaining_true` for correctly predicted risk samples. "
            "It is not a strict first-alarm lead time because simulation IDs "
            "and time indices are not stored in the current prediction files.\n\n"
        )
        f.write(table.to_markdown(index=False))
        f.write("\n")

    with open(out_tex, "w", encoding="utf-8") as f:
        f.write(table.to_latex(index=False, escape=False))

    print()
    print("===== Lead Time Proxy Table =====")
    print(table.to_string(index=False))

    print()
    print("Saved table to:")
    print(out_csv)
    print(out_md)
    print(out_tex)


def main():
    df = collect_lead_time_results()
    count = check_completeness(df)

    grouped = make_grouped_summary(df)

    output_all = SUMMARY_DIR / "lead_time_proxy_summary_all.csv"
    output_grouped = SUMMARY_DIR / "lead_time_proxy_summary_grouped.csv"

    df.to_csv(output_all, index=False)
    grouped.to_csv(output_grouped, index=False)

    print()
    print("===== Lead Time Proxy: All Results =====")
    print(
        df[
            [
                "Seed",
                "Method",
                "Alarm_Recall",
                "Alarm_Precision",
                "LeadTime_Mean",
                "LeadTime_Median",
            ]
        ].to_string(index=False)
    )

    print()
    print("===== Lead Time Proxy: Grouped =====")
    print(grouped.to_string(index=False))

    print()
    print("Saved summary to:")
    print(output_all)
    print(output_grouped)

    export_table(grouped)

    expected = len(SEEDS) * len(FEATURE_GROUPS)

    if count != expected:
        print()
        print(f"WARNING: Expected {expected} rows, got {count}.")
        print("Some feature groups or seeds may be missing.")


if __name__ == "__main__":
    main()