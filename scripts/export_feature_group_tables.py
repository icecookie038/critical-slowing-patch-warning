from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_DIR = ROOT / "results_summary"
TABLE_DIR = ROOT / "docs" / "paper_tables"

TABLE_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CSV = SUMMARY_DIR / "feature_group_summary_grouped.csv"

OUTPUT_MD = TABLE_DIR / "feature_group_comparison_table.md"
OUTPUT_TEX = TABLE_DIR / "feature_group_comparison_table.tex"
OUTPUT_CSV = TABLE_DIR / "feature_group_comparison_table.csv"


ORDER = [
    "image_only",
    "classic_patch_only",
    "dynamic_patch_only",
    "classic_dynamic_patch",
    "full",
]


DISPLAY_NAME = {
    "image_only": "Image only",
    "classic_patch_only": "Classic patch only",
    "dynamic_patch_only": "Dynamic patch only",
    "classic_dynamic_patch": "Classic + dynamic patch",
    "full": "Full",
}


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


def build_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    order_map = {name: i for i, name in enumerate(ORDER)}
    df = df.copy()
    df["Order"] = df["Feature_Group"].map(order_map)
    df = df.sort_values("Order")

    for _, row in df.iterrows():
        group = row["Feature_Group"]

        rows.append(
            {
                "Feature group": DISPLAY_NAME.get(group, group),
                "AUC": fmt_mean_std(row, "AUC"),
                "AUPRC": fmt_mean_std(row, "AUPRC"),
                "F1": fmt_mean_std(row, "F1"),
                "Accuracy": fmt_mean_std(row, "ACC"),
                "Precision": fmt_mean_std(row, "Precision"),
                "Recall": fmt_mean_std(row, "Recall"),
            }
        )

    return pd.DataFrame(rows)


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Cannot find input file: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    table = build_table(df)

    table.to_csv(OUTPUT_CSV, index=False)

    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("# Feature Group Comparison Table\n\n")
        f.write(table.to_markdown(index=False))
        f.write("\n")

    with open(OUTPUT_TEX, "w", encoding="utf-8") as f:
        f.write(table.to_latex(index=False, escape=False))

    print("===== Feature Group Paper Table =====")
    print(table.to_string(index=False))

    print()
    print("Saved to:")
    print(OUTPUT_CSV)
    print(OUTPUT_MD)
    print(OUTPUT_TEX)


if __name__ == "__main__":
    main()