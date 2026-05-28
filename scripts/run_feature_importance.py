from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "scripts" / "data"
SUMMARY_DIR = ROOT / "results_summary"
TABLE_DIR = ROOT / "docs" / "paper_tables"

SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 2026, 3407, 7777]

TARGET_SIMS = 300
TARGET_GRID = 64
TARGET_STEPS = 200
TARGET_HORIZON = 30

CLASSIC_PATCH_FEATURE_NAMES = [
    "patch_count",
    "total_patch_area",
    "largest_patch_area",
    "mean_patch_area",
    "largest_patch_ratio",
    "patch_area_std",
    "patch_area_cv",
    "gini",
    "edge_density",
    "aggregation_index",
    "centroid_x",
    "centroid_y",
    "spread_x",
    "spread_y",
    "perimeter",
    "perimeter_area_ratio",
    "compactness",
    "fragmentation",
    "nearest_patch_distance",
    "patch_density",
]


def find_dataset(seed: int) -> Path:
    patterns = [
        f"patch_dataset_seed{seed}_sims{TARGET_SIMS}_L{TARGET_GRID}_steps{TARGET_STEPS}_seq*_h{TARGET_HORIZON}.npz",
        f"*seed{seed}*sims{TARGET_SIMS}*L{TARGET_GRID}*steps{TARGET_STEPS}*h{TARGET_HORIZON}*.npz",
    ]

    for pattern in patterns:
        candidates = sorted(DATA_DIR.glob(pattern))
        if candidates:
            return candidates[0]

    raise FileNotFoundError(f"Cannot find cached dataset for seed={seed}")


def get_array(data, candidates):
    for key in candidates:
        if key in data.files:
            return data[key], key

    raise KeyError(f"Cannot find any of keys: {candidates}. Available keys: {data.files}")


def cohen_d(x_pos, x_neg, eps=1e-8):
    x_pos = np.asarray(x_pos, dtype=float)
    x_neg = np.asarray(x_neg, dtype=float)

    if x_pos.size < 2 or x_neg.size < 2:
        return 0.0

    mean_pos = np.mean(x_pos)
    mean_neg = np.mean(x_neg)

    var_pos = np.var(x_pos, ddof=1)
    var_neg = np.var(x_neg, ddof=1)

    pooled = np.sqrt((var_pos + var_neg) / 2.0)

    if pooled < eps:
        return 0.0

    return float((mean_pos - mean_neg) / pooled)


def safe_auc(y_true, score):
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score).astype(float)

    if len(np.unique(y_true)) < 2:
        return np.nan

    try:
        auc = roc_auc_score(y_true, score)
    except Exception:
        return np.nan

    # Direction-free AUC: values close to 0 or 1 both mean strong separation.
    return float(max(auc, 1.0 - auc))


def summarize_feature_matrix(X_patch):
    """
    Convert sequence features into sample-level feature summaries.

    Input:
        X_patch shape = (N, T, F)

    Output:
        X_summary shape = (N, F)
    """
    if X_patch.ndim != 3:
        raise ValueError(f"X_patch must have shape (N,T,F), got {X_patch.shape}")

    # Use temporal mean as the main feature-level summary.
    X_summary = np.nanmean(X_patch, axis=1)

    X_summary = np.nan_to_num(
        X_summary,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return X_summary


def get_feature_names(n_features: int):
    if n_features <= len(CLASSIC_PATCH_FEATURE_NAMES):
        return CLASSIC_PATCH_FEATURE_NAMES[:n_features]

    names = CLASSIC_PATCH_FEATURE_NAMES.copy()

    for i in range(len(names), n_features):
        names.append(f"classic_patch_feature_{i:02d}")

    return names


def analyze_seed(seed: int):
    path = find_dataset(seed)

    print()
    print(f"Loading seed={seed}:")
    print(path)

    data = np.load(path, allow_pickle=True)

    print("Available keys:", data.files)

    X_patch, x_key = get_array(
        data,
        ["X_patch", "x_patch", "patch_features", "X_patch_seq"],
    )

    y_risk, y_key = get_array(
        data,
        ["y_risk", "Y_risk", "risk", "y_risk_true"],
    )

    X_patch = np.asarray(X_patch, dtype=float)
    y_risk = np.asarray(y_risk).astype(int)

    print(f"Using X key: {x_key}, shape={X_patch.shape}")
    print(f"Using y key: {y_key}, shape={y_risk.shape}")

    X_summary = summarize_feature_matrix(X_patch)

    if X_summary.shape[0] != y_risk.shape[0]:
        raise ValueError(
            f"Sample mismatch: X_summary={X_summary.shape}, y_risk={y_risk.shape}"
        )

    feature_names = get_feature_names(X_summary.shape[1])

    rows = []

    for j, name in enumerate(feature_names):
        x = X_summary[:, j]

        x_pos = x[y_risk == 1]
        x_neg = x[y_risk == 0]

        mean_pos = float(np.mean(x_pos)) if x_pos.size else np.nan
        mean_neg = float(np.mean(x_neg)) if x_neg.size else np.nan
        diff = mean_pos - mean_neg
        abs_diff = abs(diff)
        d = cohen_d(x_pos, x_neg)
        abs_d = abs(d)
        auc = safe_auc(y_risk, x)

        rows.append(
            {
                "Seed": seed,
                "Feature_Index": j,
                "Feature": name,
                "Mean_Risk": mean_pos,
                "Mean_NonRisk": mean_neg,
                "Difference": diff,
                "Abs_Difference": abs_diff,
                "Cohen_D": d,
                "Abs_Cohen_D": abs_d,
                "Single_Feature_AUC": auc,
            }
        )

    return pd.DataFrame(rows)


def main():
    all_rows = []

    for seed in SEEDS:
        df_seed = analyze_seed(seed)
        all_rows.append(df_seed)

    df = pd.concat(all_rows, ignore_index=True)

    grouped = (
        df.groupby(["Feature_Index", "Feature"])
        .agg(
            Abs_Cohen_D_mean=("Abs_Cohen_D", "mean"),
            Abs_Cohen_D_std=("Abs_Cohen_D", "std"),
            Single_Feature_AUC_mean=("Single_Feature_AUC", "mean"),
            Single_Feature_AUC_std=("Single_Feature_AUC", "std"),
            Abs_Difference_mean=("Abs_Difference", "mean"),
            Abs_Difference_std=("Abs_Difference", "std"),
        )
        .reset_index()
    )

    grouped = grouped.sort_values(
        ["Abs_Cohen_D_mean", "Single_Feature_AUC_mean"],
        ascending=False,
    ).reset_index(drop=True)

    df.to_csv(SUMMARY_DIR / "feature_importance_summary_all.csv", index=False)
    grouped.to_csv(SUMMARY_DIR / "feature_importance_summary_grouped.csv", index=False)

    top_table = grouped.head(15).copy()

    def fmt_mean_std(mean, std, digits=3):
        if pd.isna(std):
            return f"{mean:.{digits}f}"
        return f"{mean:.{digits}f} ± {std:.{digits}f}"

    paper_rows = []

    for _, row in top_table.iterrows():
        paper_rows.append(
            {
                "Rank": len(paper_rows) + 1,
                "Feature": row["Feature"],
                "Abs Cohen d": fmt_mean_std(
                    row["Abs_Cohen_D_mean"],
                    row["Abs_Cohen_D_std"],
                ),
                "Single-feature AUC": fmt_mean_std(
                    row["Single_Feature_AUC_mean"],
                    row["Single_Feature_AUC_std"],
                ),
                "Abs difference": fmt_mean_std(
                    row["Abs_Difference_mean"],
                    row["Abs_Difference_std"],
                ),
            }
        )

    paper_table = pd.DataFrame(paper_rows)

    out_csv = TABLE_DIR / "feature_importance_table.csv"
    out_md = TABLE_DIR / "feature_importance_table.md"
    out_tex = TABLE_DIR / "feature_importance_table.tex"

    paper_table.to_csv(out_csv, index=False)

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Feature Importance Table\n\n")
        f.write(
            "This table reports feature-value diagnostic importance for classic patch features. "
            "Importance is estimated by the separation between risk and non-risk samples using "
            "absolute Cohen's d and single-feature AUC.\n\n"
        )
        f.write(paper_table.to_markdown(index=False))
        f.write("\n")

    with open(out_tex, "w", encoding="utf-8") as f:
        f.write(paper_table.to_latex(index=False, escape=False))

    print()
    print("===== Feature Importance: Top 15 =====")
    print(paper_table.to_string(index=False))

    print()
    print("Saved to:")
    print(SUMMARY_DIR / "feature_importance_summary_all.csv")
    print(SUMMARY_DIR / "feature_importance_summary_grouped.csv")
    print(out_csv)
    print(out_md)
    print(out_tex)


if __name__ == "__main__":
    main()