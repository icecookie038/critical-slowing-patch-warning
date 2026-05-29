from pathlib import Path
import re

import pandas as pd
import matplotlib.pyplot as plt

METHOD_SHORT_NAMES = {
    "Traditional EWS": "Trad. EWS",
    "Image only": "Image",
    "Classic patch": "Classic patch",
    "Dynamic patch v1": "Dynamic v1",
    "Classic + dynamic patch": "Classic+Dyn.",
    "Full": "Full",
}
ROOT = Path(__file__).resolve().parents[1]

TABLE_DIR = ROOT / "docs" / "paper_tables"
FIGURE_DIR = ROOT / "figures" / "main"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


METHOD_TABLE = TABLE_DIR / "method_comparison_table.csv"
LEAD_TIME_TABLE = TABLE_DIR / "lead_time_proxy_table.csv"
FEATURE_IMPORTANCE_TABLE = TABLE_DIR / "feature_importance_table.csv"
METHOD_SHORT_NAMES = {
    "Traditional EWS": "Trad. EWS",
    "Image only": "Image",
    "Classic patch": "Classic patch",
    "Dynamic patch v1": "Dynamic v1",
    "Classic + dynamic patch": "Classic+Dyn.",
    "Full": "Full",
}

def parse_mean_std(value):
    """
    Parse strings like:
        '0.836 ± 0.042'
        '13.81 ± 0.30'
    into:
        mean, std
    """
    if pd.isna(value):
        return None, None

    text = str(value).strip()

    match = re.match(
        r"^\s*([-+]?\d*\.?\d+)\s*±\s*([-+]?\d*\.?\d+)\s*$",
        text,
    )

    if match:
        return float(match.group(1)), float(match.group(2))

    try:
        return float(text), 0.0
    except ValueError:
        return None, None


def save_figure(fig, name):
    png_path = FIGURE_DIR / f"{name}.png"
    pdf_path = FIGURE_DIR / f"{name}.pdf"

    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def plot_method_comparison():
    if not METHOD_TABLE.exists():
        raise FileNotFoundError(f"Cannot find {METHOD_TABLE}")

    df = pd.read_csv(METHOD_TABLE)

    metrics = ["AUC", "AUPRC", "F1"]
    methods = [METHOD_SHORT_NAMES.get(m, m) for m in df["Method"].tolist()]

    x = range(len(methods))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, metric in enumerate(metrics):
        means = []
        stds = []

        for value in df[metric]:
            mean, std = parse_mean_std(value)
            means.append(mean)
            stds.append(std)

        positions = [p + (i - 1) * width for p in x]
        ax.bar(
            positions,
            means,
            width=width,
            yerr=stds,
            capsize=3,
            label=metric,
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Predictive performance across feature groups")
    ax.legend()

    save_figure(fig, "figure_method_comparison_auc_auprc_f1")


def plot_lead_time_proxy():
    if not LEAD_TIME_TABLE.exists():
        raise FileNotFoundError(f"Cannot find {LEAD_TIME_TABLE}")

    df = pd.read_csv(LEAD_TIME_TABLE)

    methods = [METHOD_SHORT_NAMES.get(m, m) for m in df["Method"].tolist()]

    lead_means = []
    lead_stds = []
    precision_means = []
    precision_stds = []

    for value in df["Mean lead time"]:
        mean, std = parse_mean_std(value)
        lead_means.append(mean)
        lead_stds.append(std)

    for value in df["Alarm precision"]:
        mean, std = parse_mean_std(value)
        precision_means.append(mean)
        precision_stds.append(std)

    x = range(len(methods))

    # Figure 5a: lead-time proxy
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        x,
        lead_means,
        yerr=lead_stds,
        capsize=4,
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylabel("Mean lead-time proxy")
    ax.set_ylim(0, max(lead_means) + 2)
    ax.set_title("Mean lead-time proxy across methods")

    save_figure(fig, "figure_lead_time_proxy")

    # Figure 5b: alarm precision
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        x,
        precision_means,
        yerr=precision_stds,
        capsize=4,
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylabel("Alarm precision")
    ax.set_ylim(0, 1.05)
    ax.set_title("Alarm precision across methods")

    save_figure(fig, "figure_alarm_precision")
def plot_feature_importance():
    if not FEATURE_IMPORTANCE_TABLE.exists():
        raise FileNotFoundError(f"Cannot find {FEATURE_IMPORTANCE_TABLE}")

    df = pd.read_csv(FEATURE_IMPORTANCE_TABLE)

    top_n = min(12, len(df))
    df = df.head(top_n).copy()

    features = df["Feature"].tolist()

    means = []
    stds = []

    for value in df["Abs Cohen d"]:
        mean, std = parse_mean_std(value)
        means.append(mean)
        stds.append(std)

    fig, ax = plt.subplots(figsize=(8, 6))

    y = range(len(features))

    ax.barh(
        list(y),
        means,
        xerr=stds,
        capsize=3,
    )

    ax.set_yticks(list(y))
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel("Absolute Cohen's d")
    ax.set_title("Risk/non-risk separation of classic patch features")

    save_figure(fig, "figure_feature_importance")


def main():
    print("Generating paper figures...")
    print(f"Figure directory: {FIGURE_DIR}")

    plot_method_comparison()
    plot_lead_time_proxy()
    plot_feature_importance()

    print()
    print("Done.")


if __name__ == "__main__":
    main()