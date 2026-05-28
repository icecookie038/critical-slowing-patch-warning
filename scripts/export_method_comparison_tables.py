from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

SUMMARY_DIR = ROOT / "results_summary"
TABLE_DIR = ROOT / "docs" / "paper_tables"

TABLE_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_GROUP_CSV = SUMMARY_DIR / "feature_group_summary_grouped.csv"
TRADITIONAL_EWS_CSV = SUMMARY_DIR / "traditional_ews_summary_grouped.csv"

OUTPUT_CSV = TABLE_DIR / "method_comparison_table.csv"
OUTPUT_MD = TABLE_DIR / "method_comparison_table.md"
OUTPUT_TEX = TABLE_DIR / "method_comparison_table.tex"


FEATURE_GROUP_DISPLAY = {
    "image_only": "Image only",
    "classic_patch_only": "Classic patch",
    "dynamic_patch_only": "Dynamic patch v1",
    "classic_dynamic_patch": "Classic + dynamic patch",
    "full": "Full",
}

FEATURE_GROUP_ORDER = [
    "image_only",
    "classic_patch_only",
    "dynamic_patch_only",
    "classic_dynamic_patch",
    "full",
]


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
        return f"{mean:.3f}"

    return f"{mean:.3f} ± {std:.3f}"


def add_metric_row(rows, method_name: str, row):
    rows.append(
        {
            "Method": method_name,
            "AUC": fmt_mean_std(row, "AUC"),
            "AUPRC": fmt_mean_std(row, "AUPRC"),
            "F1": fmt_mean_std(row, "F1"),
            "Accuracy": fmt_mean_std(row, "ACC"),
            "Precision": fmt_mean_std(row, "Precision"),
            "Recall": fmt_mean_std(row, "Recall"),
        }
    )


def build_table():
    if not FEATURE_GROUP_CSV.exists():
        raise FileNotFoundError(f"Cannot find: {FEATURE_GROUP_CSV}")

    if not TRADITIONAL_EWS_CSV.exists():
        raise FileNotFoundError(f"Cannot find: {TRADITIONAL_EWS_CSV}")

    feature_df = pd.read_csv(FEATURE_GROUP_CSV)
    ews_df = pd.read_csv(TRADITIONAL_EWS_CSV)

    rows = []

    # 1. Traditional EWS first
    if len(ews_df) > 0:
        add_metric_row(rows, "Traditional EWS", ews_df.iloc[0])

    # 2. Feature groups from v0.9
    for group in FEATURE_GROUP_ORDER:
        hit = feature_df[feature_df["Feature_Group"] == group]

        if hit.empty:
            print(f"WARNING: missing feature group: {group}")
            continue

        display_name = FEATURE_GROUP_DISPLAY.get(group, group)
        add_metric_row(rows, display_name, hit.iloc[0])

    table = pd.DataFrame(rows)

    return table


def main():
    table = build_table()

    table.to_csv(OUTPUT_CSV, index=False)

    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("# Method Comparison Table\n\n")
        f.write(table.to_markdown(index=False))
        f.write("\n")

    with open(OUTPUT_TEX, "w", encoding="utf-8") as f:
        f.write(table.to_latex(index=False, escape=False))

    print("===== Method Comparison Table =====")
    print(table.to_string(index=False))

    print()
    print("Saved to:")
    print(OUTPUT_CSV)
    print(OUTPUT_MD)
    print(OUTPUT_TEX)


if __name__ == "__main__":
    main()