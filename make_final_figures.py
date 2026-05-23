# make_final_figures.py
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    auc,
    average_precision_score,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def find_latest_prediction_file(root_dir: Path):
    files = list(root_dir.rglob("validation_predictions.npz"))
    if len(files) == 0:
        raise FileNotFoundError(
            f"No validation_predictions.npz found under: {root_dir}"
        )
    latest_file = max(files, key=lambda p: p.stat().st_mtime)
    return latest_file


def load_predictions(pred_file: Path):
    data = np.load(pred_file)

    y_risk_true = data["y_risk_true"].astype(int)
    risk_prob = data["risk_prob"].astype(float)

    if "y_remaining_true" in data:
        y_remaining_true = data["y_remaining_true"].astype(float)
    elif "y_remaining" in data:
        y_remaining_true = data["y_remaining"].astype(float)
    else:
        y_remaining_true = None

    return y_risk_true, risk_prob, y_remaining_true


def search_best_threshold(y_true, y_prob):
    best = {
        "threshold": 0.5,
        "f1": -1.0,
        "acc": 0.0,
        "precision": 0.0,
        "recall": 0.0,
    }

    for th in np.linspace(0.05, 0.95, 91):
        y_pred = (y_prob >= th).astype(int)

        f1 = f1_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)

        if f1 > best["f1"]:
            best = {
                "threshold": float(th),
                "f1": float(f1),
                "acc": float(acc),
                "precision": float(precision),
                "recall": float(recall),
            }

    return best


def plot_roc_curve(y_true, y_prob, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    return roc_auc


def plot_pr_curve(y_true, y_prob, save_path):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, linewidth=2, label=f"AUPRC = {auprc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    return auprc


def plot_confusion_matrix(y_true, y_prob, threshold, save_path):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    plt.figure(figsize=(5.5, 5))
    plt.imshow(cm)
    plt.title(f"Confusion Matrix, threshold = {threshold:.3f}")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks([0, 1], ["Non-risk", "Risk"])
    plt.yticks([0, 1], ["Non-risk", "Risk"])

    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=14,
            )

    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    return cm


def plot_binned_risk(y_true, y_prob, save_path, n_bins=10):
    order = np.argsort(y_prob)
    sorted_prob = y_prob[order]
    sorted_true = y_true[order]

    bins = np.array_split(np.arange(len(sorted_prob)), n_bins)

    bin_ids = []
    mean_pred = []
    observed_rate = []
    counts = []

    for i, b in enumerate(bins, start=1):
        if len(b) == 0:
            continue

        bin_ids.append(i)
        mean_pred.append(float(np.mean(sorted_prob[b])))
        observed_rate.append(float(np.mean(sorted_true[b])))
        counts.append(int(len(b)))

    plt.figure(figsize=(7, 5))
    plt.plot(
        bin_ids,
        mean_pred,
        marker="o",
        linewidth=2,
        label="Mean predicted risk",
    )
    plt.plot(
        bin_ids,
        observed_rate,
        marker="s",
        linewidth=2,
        label="Observed risk rate",
    )

    plt.xlabel("Risk bin, low to high predicted risk")
    plt.ylabel("Risk")
    plt.ylim(-0.05, 1.05)
    plt.title("Binned Risk Calibration Plot")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    return {
        "bin": bin_ids,
        "mean_predicted_risk": mean_pred,
        "observed_risk_rate": observed_rate,
        "count": counts,
    }


def analyze_lead_time(y_true, y_prob, y_remaining, threshold, save_path_txt, save_path_png):
    y_pred = (y_prob >= threshold).astype(int)

    lines = []
    lines.append("===== Lead Time Analysis =====")
    lines.append(f"Threshold: {threshold:.4f}")
    lines.append(f"Total samples: {len(y_true)}")
    lines.append(f"Positive samples: {int(np.sum(y_true == 1))}")
    lines.append(f"Negative samples: {int(np.sum(y_true == 0))}")
    lines.append("")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    lines.append(f"TN: {tn}")
    lines.append(f"FP: {fp}")
    lines.append(f"FN: {fn}")
    lines.append(f"TP: {tp}")
    lines.append("")

    if y_remaining is None:
        lines.append("No y_remaining_true found. Lead time analysis skipped.")
        save_path_txt.write_text("\n".join(lines), encoding="utf-8")
        return None

    tp_mask = (y_pred == 1) & (y_true == 1)
    fp_mask = (y_pred == 1) & (y_true == 0)
    fn_mask = (y_pred == 0) & (y_true == 1)

    tp_lead = y_remaining[tp_mask]
    fp_remain = y_remaining[fp_mask]
    fn_lead = y_remaining[fn_mask]

    lines.append(f"TP count: {len(tp_lead)}")
    lines.append(f"FP count: {len(fp_remain)}")
    lines.append(f"FN count: {len(fn_lead)}")
    lines.append("")

    summary = {}

    if len(tp_lead) > 0:
        summary = {
            "mean": float(np.mean(tp_lead)),
            "median": float(np.median(tp_lead)),
            "min": float(np.min(tp_lead)),
            "max": float(np.max(tp_lead)),
            "q25": float(np.percentile(tp_lead, 25)),
            "q75": float(np.percentile(tp_lead, 75)),
        }

        lines.append("Successful warning lead time, TP only:")
        lines.append(f"Mean lead time:   {summary['mean']:.4f} steps")
        lines.append(f"Median lead time: {summary['median']:.4f} steps")
        lines.append(f"Min lead time:    {summary['min']:.4f} steps")
        lines.append(f"Max lead time:    {summary['max']:.4f} steps")
        lines.append(f"25% quantile:     {summary['q25']:.4f} steps")
        lines.append(f"75% quantile:     {summary['q75']:.4f} steps")
        lines.append("")

        plt.figure(figsize=(7, 5))
        plt.hist(tp_lead, bins=20)
        plt.xlabel("Lead time, remaining steps before critical point")
        plt.ylabel("Count")
        plt.title("Successful Warning Lead Time Distribution")
        plt.tight_layout()
        plt.savefig(save_path_png, dpi=300)
        plt.close()

    if len(fn_lead) > 0:
        lines.append("Missed positive samples, FN only:")
        lines.append(f"Mean remaining steps:   {np.mean(fn_lead):.4f}")
        lines.append(f"Median remaining steps: {np.median(fn_lead):.4f}")
        lines.append("")

    save_path_txt.write_text("\n".join(lines), encoding="utf-8")

    return summary


def save_metrics_summary(save_path, metrics, bin_table):
    lines = []
    lines.append("===== Final Analysis Summary =====")
    lines.append("")

    for k, v in metrics.items():
        if isinstance(v, float):
            lines.append(f"{k}: {v:.6f}")
        else:
            lines.append(f"{k}: {v}")

    lines.append("")
    lines.append("===== Binned Risk Table =====")
    lines.append("bin, mean_predicted_risk, observed_risk_rate, count")

    for b, mp, obs, c in zip(
        bin_table["bin"],
        bin_table["mean_predicted_risk"],
        bin_table["observed_risk_rate"],
        bin_table["count"],
    ):
        lines.append(f"{b}, {mp:.6f}, {obs:.6f}, {c}")

    save_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--result-dir",
        type=str,
        default="results",
        help="Folder containing validation_predictions.npz, or root results folder.",
    )

    parser.add_argument(
        "--prediction-file",
        type=str,
        default=None,
        help="Direct path to validation_predictions.npz. If omitted, use latest under result-dir.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Classification threshold. If omitted, best F1 threshold will be searched.",
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default="final",
        help="Output file prefix.",
    )

    parser.add_argument(
        "--bins",
        type=int,
        default=10,
        help="Number of bins for binned risk plot.",
    )

    args = parser.parse_args()

    result_dir = Path(args.result_dir)

    if args.prediction_file is not None:
        pred_file = Path(args.prediction_file)
    else:
        pred_file = find_latest_prediction_file(result_dir)

    output_dir = pred_file.parent / "final_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Using prediction file:")
    print(pred_file)
    print("Saving final analysis to:")
    print(output_dir)

    y_true, y_prob, y_remaining = load_predictions(pred_file)

    if args.threshold is None:
        best = search_best_threshold(y_true, y_prob)
        threshold = best["threshold"]
    else:
        threshold = float(args.threshold)
        y_pred_tmp = (y_prob >= threshold).astype(int)
        best = {
            "threshold": threshold,
            "f1": float(f1_score(y_true, y_pred_tmp, zero_division=0)),
            "acc": float(accuracy_score(y_true, y_pred_tmp)),
            "precision": float(precision_score(y_true, y_pred_tmp, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred_tmp, zero_division=0)),
        }

    roc_auc = plot_roc_curve(
        y_true,
        y_prob,
        output_dir / f"{args.prefix}_roc_curve.png",
    )

    auprc = plot_pr_curve(
        y_true,
        y_prob,
        output_dir / f"{args.prefix}_pr_curve.png",
    )

    cm = plot_confusion_matrix(
        y_true,
        y_prob,
        threshold,
        output_dir / f"{args.prefix}_confusion_matrix.png",
    )

    bin_table = plot_binned_risk(
        y_true,
        y_prob,
        output_dir / f"{args.prefix}_binned_risk_plot.png",
        n_bins=args.bins,
    )

    lead_summary = analyze_lead_time(
        y_true,
        y_prob,
        y_remaining,
        threshold,
        output_dir / f"{args.prefix}_lead_time_summary.txt",
        output_dir / f"{args.prefix}_lead_time_distribution.png",
    )

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "AUC": float(roc_auc_score(y_true, y_prob)),
        "AUPRC": float(average_precision_score(y_true, y_prob)),
        "ACC": float(accuracy_score(y_true, y_pred)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "Best_Threshold": float(threshold),
        "Positive_Rate": float(np.mean(y_true)),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }

    if lead_summary is not None:
        metrics.update({
            "LeadTime_Mean": lead_summary.get("mean", np.nan),
            "LeadTime_Median": lead_summary.get("median", np.nan),
            "LeadTime_Q25": lead_summary.get("q25", np.nan),
            "LeadTime_Q75": lead_summary.get("q75", np.nan),
        })

    save_metrics_summary(
        output_dir / f"{args.prefix}_final_summary.txt",
        metrics,
        bin_table,
    )

    print("\nGenerated files:")
    print(output_dir / f"{args.prefix}_roc_curve.png")
    print(output_dir / f"{args.prefix}_pr_curve.png")
    print(output_dir / f"{args.prefix}_confusion_matrix.png")
    print(output_dir / f"{args.prefix}_binned_risk_plot.png")
    print(output_dir / f"{args.prefix}_lead_time_distribution.png")
    print(output_dir / f"{args.prefix}_lead_time_summary.txt")
    print(output_dir / f"{args.prefix}_final_summary.txt")


if __name__ == "__main__":
    main()