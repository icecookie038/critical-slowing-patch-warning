# train_patch_baselines.py
# -*- coding: utf-8 -*-

"""
Patch feature baseline models for critical slowing down prediction.

Purpose:
1. Use only patch indicators, not infection images.
2. Train classical machine learning baselines.
3. Compare whether patch indicators themselves are predictive.
4. Save AUC, AUPRC, F1, Precision, Recall, lead time, and binned risk plots.

Recommended command:

python train_patch_baselines.py --data-dir data --feature-mode stats --seed 42
python train_patch_baselines.py --data-dir data --feature-mode stats --seed 123
python train_patch_baselines.py --data-dir data --feature-mode stats --seed 2026

You can also use flatten features:

python train_patch_baselines.py --data-dir data --feature-mode flatten --seed 42
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.neural_network import MLPClassifier


warnings.filterwarnings("ignore")


# ============================================================
# 1. Basic utilities
# ============================================================

def find_latest_dataset(data_dir: Path) -> Path:
    candidates = []
    candidates.extend(list(data_dir.glob("*.npz")))
    candidates.extend(list(data_dir.rglob("*.npz")))

    candidates = [
        p for p in candidates
        if "patch" in p.name.lower() or "dataset" in p.name.lower()
    ]

    if len(candidates) == 0:
        raise FileNotFoundError(
            f"No dataset npz file found under: {data_dir}\n"
            f"Please check whether your dataset is saved in the data folder."
        )

    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest


def pick_key(npz_data, candidates, required=True):
    keys = list(npz_data.keys())
    lower_map = {k.lower(): k for k in keys}

    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]

    for k in keys:
        lk = k.lower()
        for c in candidates:
            if c.lower() in lk:
                return k

    if required:
        raise KeyError(
            f"Cannot find any key from candidates: {candidates}\n"
            f"Available keys are: {keys}"
        )

    return None


def clean_array(x):
    x = np.asarray(x)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


# ============================================================
# 2. Feature construction
# ============================================================

def temporal_slope(x):
    """
    x shape: (N, T, F)
    Return slope shape: (N, F)
    """
    n, t, f = x.shape
    time = np.arange(t, dtype=np.float32)
    time = time - time.mean()
    denom = np.sum(time ** 2) + 1e-8

    x_centered = x - x.mean(axis=1, keepdims=True)
    slope = np.sum(x_centered * time[None, :, None], axis=1) / denom
    return slope


def build_features(x_patch, feature_mode="stats"):
    """
    x_patch can be:
    - (N, T, F)
    - (N, F)

    feature_mode:
    - flatten: directly flatten all time steps
    - last: only use the last time step
    - stats: use mean, std, min, max, last, slope
    """
    x_patch = clean_array(x_patch)

    if x_patch.ndim == 2:
        return x_patch.astype(np.float32)

    if x_patch.ndim != 3:
        raise ValueError(
            f"X_patch should have shape (N, T, F) or (N, F), "
            f"but got shape: {x_patch.shape}"
        )

    if feature_mode == "flatten":
        x = x_patch.reshape(x_patch.shape[0], -1)

    elif feature_mode == "last":
        x = x_patch[:, -1, :]

    elif feature_mode == "stats":
        mean = x_patch.mean(axis=1)
        std = x_patch.std(axis=1)
        xmin = x_patch.min(axis=1)
        xmax = x_patch.max(axis=1)
        last = x_patch[:, -1, :]
        slope = temporal_slope(x_patch)

        x = np.concatenate(
            [mean, std, xmin, xmax, last, slope],
            axis=1,
        )

    else:
        raise ValueError(
            f"Unknown feature_mode: {feature_mode}. "
            f"Use one of: flatten, last, stats"
        )

    x = clean_array(x).astype(np.float32)
    return x


# ============================================================
# 3. Load dataset
# ============================================================

def load_patch_dataset(data_file: Path, feature_mode: str):
    data = np.load(data_file, allow_pickle=True)

    x_patch_key = pick_key(
        data,
        ["X_patch", "x_patch", "patch", "patch_features", "patch_feature"],
        required=True,
    )

    y_key = pick_key(
        data,
        ["y_risk", "risk", "risk_label", "label", "y"],
        required=True,
    )

    remaining_key = pick_key(
        data,
        ["y_remaining", "y_remaining_true", "remaining", "remain"],
        required=False,
    )

    sim_id_key = pick_key(
        data,
        ["sim_id", "sim_ids", "simulation_id", "simulation_ids"],
        required=False,
    )

    x_patch = data[x_patch_key]
    y = data[y_key].astype(int)

    x = build_features(x_patch, feature_mode=feature_mode)

    y_remaining = None
    if remaining_key is not None:
        y_remaining = data[remaining_key].astype(float)

    sim_ids = None
    if sim_id_key is not None:
        sim_ids = data[sim_id_key]

    valid_mask = np.isfinite(x).all(axis=1) & np.isfinite(y)
    if y_remaining is not None:
        valid_mask = valid_mask & np.isfinite(y_remaining)

    x = x[valid_mask]
    y = y[valid_mask]

    if y_remaining is not None:
        y_remaining = y_remaining[valid_mask]

    if sim_ids is not None:
        sim_ids = sim_ids[valid_mask]

    info = {
        "data_file": str(data_file),
        "x_patch_key": x_patch_key,
        "y_key": y_key,
        "remaining_key": remaining_key,
        "sim_id_key": sim_id_key,
        "raw_x_patch_shape": list(x_patch.shape),
        "final_x_shape": list(x.shape),
        "positive_rate": float(np.mean(y)),
        "n_samples": int(len(y)),
    }

    return x, y, y_remaining, sim_ids, info


# ============================================================
# 4. Train / validation split
# ============================================================

def split_dataset(x, y, y_remaining, sim_ids, val_ratio=0.25, seed=42):
    rng = np.random.default_rng(seed)

    if sim_ids is not None:
        unique_ids = np.unique(sim_ids)
        rng.shuffle(unique_ids)

        n_val = max(1, int(len(unique_ids) * val_ratio))
        val_ids = set(unique_ids[:n_val].tolist())

        val_mask = np.array([sid in val_ids for sid in sim_ids])
        train_mask = ~val_mask

        x_train = x[train_mask]
        y_train = y[train_mask]
        x_val = x[val_mask]
        y_val = y[val_mask]

        rem_train = None if y_remaining is None else y_remaining[train_mask]
        rem_val = None if y_remaining is None else y_remaining[val_mask]

        split_info = {
            "split_method": "by_simulation_id",
            "n_train": int(len(y_train)),
            "n_val": int(len(y_val)),
            "train_positive_rate": float(np.mean(y_train)),
            "val_positive_rate": float(np.mean(y_val)),
            "n_unique_sim_ids": int(len(unique_ids)),
            "n_val_sim_ids": int(n_val),
        }

    else:
        indices = np.arange(len(y))

        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_ratio,
            random_state=seed,
            stratify=y,
        )

        x_train = x[train_idx]
        y_train = y[train_idx]
        x_val = x[val_idx]
        y_val = y[val_idx]

        rem_train = None if y_remaining is None else y_remaining[train_idx]
        rem_val = None if y_remaining is None else y_remaining[val_idx]

        split_info = {
            "split_method": "stratified_random",
            "n_train": int(len(y_train)),
            "n_val": int(len(y_val)),
            "train_positive_rate": float(np.mean(y_train)),
            "val_positive_rate": float(np.mean(y_val)),
        }

    return x_train, x_val, y_train, y_val, rem_train, rem_val, split_info


# ============================================================
# 5. Models
# ============================================================

def build_models(seed=42):
    models = {}

    models["LogisticRegression"] = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=seed,
    )

    models["RandomForest"] = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )

    models["ExtraTrees"] = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )

    models["HistGradientBoosting"] = HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.04,
        l2_regularization=0.01,
        random_state=seed,
    )

    models["MLP"] = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=400,
        early_stopping=True,
        random_state=seed,
    )

    # Optional XGBoost
    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
        )
    except Exception:
        pass

    return models


def wrap_model(model):
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


# ============================================================
# 6. Metrics and plots
# ============================================================

def best_threshold_by_f1(y_true, prob):
    best = {
        "threshold": 0.5,
        "f1": -1,
        "precision": 0,
        "recall": 0,
        "acc": 0,
    }

    for th in np.linspace(0.05, 0.95, 91):
        pred = (prob >= th).astype(int)

        f1 = f1_score(y_true, pred, zero_division=0)
        precision = precision_score(y_true, pred, zero_division=0)
        recall = recall_score(y_true, pred, zero_division=0)
        acc = accuracy_score(y_true, pred)

        if f1 > best["f1"]:
            best = {
                "threshold": float(th),
                "f1": float(f1),
                "precision": float(precision),
                "recall": float(recall),
                "acc": float(acc),
            }

    return best


def lead_time_summary(y_true, prob, y_remaining, threshold):
    pred = (prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    result = {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }

    if y_remaining is None:
        result.update({
            "LeadTime_Mean": np.nan,
            "LeadTime_Median": np.nan,
            "LeadTime_Q25": np.nan,
            "LeadTime_Q75": np.nan,
            "FN_Remaining_Mean": np.nan,
            "FN_Remaining_Median": np.nan,
        })
        return result

    tp_mask = (pred == 1) & (y_true == 1)
    fn_mask = (pred == 0) & (y_true == 1)

    tp_lead = y_remaining[tp_mask]
    fn_remain = y_remaining[fn_mask]

    if len(tp_lead) > 0:
        result.update({
            "LeadTime_Mean": float(np.mean(tp_lead)),
            "LeadTime_Median": float(np.median(tp_lead)),
            "LeadTime_Q25": float(np.percentile(tp_lead, 25)),
            "LeadTime_Q75": float(np.percentile(tp_lead, 75)),
        })
    else:
        result.update({
            "LeadTime_Mean": np.nan,
            "LeadTime_Median": np.nan,
            "LeadTime_Q25": np.nan,
            "LeadTime_Q75": np.nan,
        })

    if len(fn_remain) > 0:
        result.update({
            "FN_Remaining_Mean": float(np.mean(fn_remain)),
            "FN_Remaining_Median": float(np.median(fn_remain)),
        })
    else:
        result.update({
            "FN_Remaining_Mean": np.nan,
            "FN_Remaining_Median": np.nan,
        })

    return result


def evaluate_model(y_true, prob, y_remaining=None):
    best = best_threshold_by_f1(y_true, prob)
    th = best["threshold"]
    pred = (prob >= th).astype(int)

    metrics = {
        "AUC": float(roc_auc_score(y_true, prob)),
        "AUPRC": float(average_precision_score(y_true, prob)),
        "ACC": float(accuracy_score(y_true, pred)),
        "F1": float(f1_score(y_true, pred, zero_division=0)),
        "Precision": float(precision_score(y_true, pred, zero_division=0)),
        "Recall": float(recall_score(y_true, pred, zero_division=0)),
        "Best_Threshold": float(th),
        "Positive_Rate": float(np.mean(y_true)),
    }

    metrics.update(lead_time_summary(y_true, prob, y_remaining, th))
    return metrics


def plot_roc_pr(y_true, prob, out_dir, model_name):
    fpr, tpr, _ = roc_curve(y_true, prob)
    precision, recall, _ = precision_recall_curve(y_true, prob)

    auc_value = roc_auc_score(y_true, prob)
    auprc_value = average_precision_score(y_true, prob)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc_value:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{model_name}_roc.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, linewidth=2, label=f"AUPRC = {auprc_value:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Curve - {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{model_name}_pr.png", dpi=300)
    plt.close()


def plot_binned_risk(y_true, prob, out_dir, model_name, n_bins=10):
    order = np.argsort(prob)
    prob_sorted = prob[order]
    y_sorted = y_true[order]

    bins = np.array_split(np.arange(len(y_sorted)), n_bins)

    bin_ids = []
    mean_pred = []
    observed = []
    counts = []

    for i, b in enumerate(bins, start=1):
        bin_ids.append(i)
        mean_pred.append(float(np.mean(prob_sorted[b])))
        observed.append(float(np.mean(y_sorted[b])))
        counts.append(int(len(b)))

    plt.figure(figsize=(7, 5))
    plt.plot(bin_ids, mean_pred, marker="o", linewidth=2, label="Mean predicted risk")
    plt.plot(bin_ids, observed, marker="s", linewidth=2, label="Observed risk rate")
    plt.xlabel("Risk bin, low to high predicted risk")
    plt.ylabel("Risk")
    plt.ylim(-0.05, 1.05)
    plt.title(f"Binned Risk Calibration - {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{model_name}_binned_risk.png", dpi=300)
    plt.close()

    return pd.DataFrame({
        "bin": bin_ids,
        "mean_predicted_risk": mean_pred,
        "observed_risk_rate": observed,
        "count": counts,
    })


# ============================================================
# 7. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Dataset folder. Used when --data-file is not provided.",
    )

    parser.add_argument(
        "--data-file",
        type=str,
        default=None,
        help="Direct path to dataset npz file.",
    )

    parser.add_argument(
        "--feature-mode",
        type=str,
        default="stats",
        choices=["stats", "flatten", "last"],
        help="How to convert temporal patch sequence into tabular features.",
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.25,
        help="Validation ratio.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="results_baselines",
        help="Output folder.",
    )

    args = parser.parse_args()

    if args.data_file is None:
        data_file = find_latest_dataset(Path(args.data_dir))
    else:
        data_file = Path(args.data_file)

    out_dir = Path(args.out_dir) / f"seed{args.seed}_{args.feature_mode}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Using dataset:")
    print(data_file)
    print("Output folder:")
    print(out_dir)

    x, y, y_remaining, sim_ids, data_info = load_patch_dataset(
        data_file=data_file,
        feature_mode=args.feature_mode,
    )

    x_train, x_val, y_train, y_val, rem_train, rem_val, split_info = split_dataset(
        x=x,
        y=y,
        y_remaining=y_remaining,
        sim_ids=sim_ids,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    with open(out_dir / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "data_info": data_info,
                "split_info": split_info,
                "args": vars(args),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\nDataset info:")
    print(json.dumps(data_info, ensure_ascii=False, indent=2))
    print("\nSplit info:")
    print(json.dumps(split_info, ensure_ascii=False, indent=2))

    models = build_models(seed=args.seed)

    all_metrics = []
    all_bin_tables = {}

    for model_name, model in models.items():
        print(f"\n===== Training {model_name} =====")

        clf = wrap_model(model)
        clf.fit(x_train, y_train)

        if hasattr(clf, "predict_proba"):
            prob = clf.predict_proba(x_val)[:, 1]
        else:
            score = clf.decision_function(x_val)
            prob = 1.0 / (1.0 + np.exp(-score))

        prob = np.clip(prob, 0.0, 1.0)

        metrics = evaluate_model(
            y_true=y_val,
            prob=prob,
            y_remaining=rem_val,
        )

        metrics["Model"] = model_name
        metrics["Feature_Mode"] = args.feature_mode
        metrics["Seed"] = args.seed

        all_metrics.append(metrics)

        print(pd.Series(metrics).to_string())

        plot_roc_pr(y_val, prob, out_dir, model_name)
        bin_df = plot_binned_risk(y_val, prob, out_dir, model_name)
        bin_df.to_csv(out_dir / f"{model_name}_binned_risk.csv", index=False)
        all_bin_tables[model_name] = bin_df

        pred_df = pd.DataFrame({
            "y_true": y_val,
            "risk_prob": prob,
        })

        if rem_val is not None:
            pred_df["y_remaining"] = rem_val

        pred_df.to_csv(out_dir / f"{model_name}_predictions.csv", index=False)

    metrics_df = pd.DataFrame(all_metrics)

    preferred_cols = [
        "Model",
        "Feature_Mode",
        "Seed",
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
        "FN_Remaining_Mean",
        "FN_Remaining_Median",
    ]

    metrics_df = metrics_df[[c for c in preferred_cols if c in metrics_df.columns]]
    metrics_df = metrics_df.sort_values(by="AUC", ascending=False)

    metrics_df.to_csv(out_dir / "baseline_metrics.csv", index=False)

    with open(out_dir / "baseline_metrics.txt", "w", encoding="utf-8") as f:
        f.write("===== Patch Baseline Metrics =====\n\n")
        f.write(metrics_df.to_string(index=False))
        f.write("\n\n")

    print("\n===== Final baseline results =====")
    print(metrics_df.to_string(index=False))

    print("\nSaved to:")
    print(out_dir / "baseline_metrics.csv")
    print(out_dir / "baseline_metrics.txt")


if __name__ == "__main__":
    main()