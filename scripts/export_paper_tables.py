from pathlib import Path
import pandas as pd
import math


ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT / "results_summary" / "robustness_summary_grouped.csv"
OUT_DIR = ROOT / "docs" / "paper_tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)


METRICS = [
    ("AUC", "AUC"),
    ("AUPRC", "AUPRC"),
    ("F1", "F1"),
    ("Precision", "Precision"),
    ("Recall", "Recall"),
    ("LeadTime_Mean", "Lead time"),
]


def fmt_mean_std(row, metric):
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    mean = row.get(mean_col, None)
    std = row.get(std_col, None)

    if pd.isna(mean):
        return "-"

    if pd.isna(std):
        return f"{mean:.3f}"

    return f"{mean:.3f} ± {std:.3f}"


def clean_model_name(row):
    result_type = row.get("Result_Type", "")

    if result_type == "deep_model":
        return "Deep model"

    model = row.get("Model", "")
    if pd.isna(model) or model == "":
        return "Baseline"

    return str(model)


def clean_feature_mode(row):
    mode = row.get("Feature_Mode", "")

    if pd.isna(mode):
        return "-"

    mode = str(mode)

    mapping = {
        "full": "Full",
        "img_only": "Image-only",
        "patch_only": "Patch-only",
        "stats": "Patch statistics",
    }

    return mapping.get(mode, mode)


def make_paper_table(df):
    rows = []

    for _, row in df.iterrows():
        item = {
            "Model": clean_model_name(row),
            "Input mode": clean_feature_mode(row),
        }

        for metric_key, metric_name in METRICS:
            item[metric_name] = fmt_mean_std(row, metric_key)

        rows.append(item)

    out_df = pd.DataFrame(rows)

    order = {
        ("Deep model", "Full"): 1,
        ("Deep model", "Image-only"): 2,
        ("Deep model", "Patch-only"): 3,
        ("ExtraTrees", "Patch statistics"): 4,
        ("RandomForest", "Patch statistics"): 5,
        ("HistGradientBoosting", "Patch statistics"): 6,
        ("LogisticRegression", "Patch statistics"): 7,
        ("MLP", "Patch statistics"): 8,
    }

    out_df["_order"] = out_df.apply(
        lambda r: order.get((r["Model"], r["Input mode"]), 999),
        axis=1,
    )
    out_df = out_df.sort_values("_order").drop(columns=["_order"])

    return out_df


def df_to_latex(df):
    columns = list(df.columns)

    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Robustness comparison across input modes and baseline models.}")
    latex.append("\\label{tab:robustness_comparison}")
    latex.append("\\begin{tabular}{llllllll}")
    latex.append("\\hline")
    latex.append(" & ".join(columns) + " \\\\")
    latex.append("\\hline")

    for _, row in df.iterrows():
        values = []
        for col in columns:
            value = str(row[col])
            value = value.replace("_", "\\_")
            value = value.replace("±", "$\\pm$")
            values.append(value)
        latex.append(" & ".join(values) + " \\\\")

    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    return "\n".join(latex)


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Cannot find input file: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    paper_df = make_paper_table(df)

    md_path = OUT_DIR / "robustness_comparison_table.md"
    tex_path = OUT_DIR / "robustness_comparison_table.tex"
    csv_path = OUT_DIR / "robustness_comparison_table.csv"

    paper_df.to_markdown(md_path, index=False)
    paper_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    tex_path.write_text(df_to_latex(paper_df), encoding="utf-8")

    print("Paper tables exported.")
    print(f"Markdown table: {md_path}")
    print(f"LaTeX table: {tex_path}")
    print(f"CSV table: {csv_path}")
    print()
    print(paper_df)


if __name__ == "__main__":
    main()