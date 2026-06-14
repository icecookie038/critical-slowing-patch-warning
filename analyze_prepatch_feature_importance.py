# analyze_prepatch_feature_importance.py
# -*- coding: utf-8 -*-
"""
Analyze feature importance for v1.3 pre-patch indicators.

Purpose
-------
This script answers:

1. Which prepatch indicators are most important?
2. Are the dominant indicators from synchronization, connectivity,
   boundary rigidity, or dominant mode locking?
3. Which indicators should be retained for a conservative prepatch set?

Input:
    data/processed/v1_3_prepatch/seir_v1_3_prepatch_h30_seed42.npz

Outputs:
    results_feature_importance/v1_3_prepatch_seed42/
        prepatch_feature_importance_random_forest.csv
        prepatch_feature_importance_extra_trees.csv
        prepatch_feature_importance_mean.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
from sklearn.inspection import permutation_importance


def split_by_sim_id(y, sim_id, val_ratio=0.25, seed=42):
    rng = np.random.default_rng(seed)

    sim_id = np.asarray(sim_id)
    unique_ids = np.unique(sim_id)

    rng.shuffle(unique_ids)

    n_val = max(1, int(len(unique_ids) * val_ratio))
    val_ids = set(unique_ids[:n_val].tolist())

    val_mask = np.array([sid in val_ids for sid in sim_id])
    train_mask = ~val_mask

    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]

    return train_idx, val_idx


def evaluate_model(model, x_val, y_val):
    prob = model.predict_proba(x_val)[:, 1]

    thresholds = np.linspace(0.05, 0.95, 91)
    best_f1 = -1.0
    best_threshold = 0.5

    for th in thresholds:
        pred = (prob >= th).astype(int)
        f1 = f1_score(y_val, pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(th)

    pred = (prob >= best_threshold).astype(int)

    metrics = {
        "AUC": roc_auc_score(y_val, prob),
        "AUPRC": average_precision_score(y_val, prob),
        "ACC": accuracy_score(y_val, pred),
        "F1": f1_score(y_val, pred),
        "Best_Threshold": best_threshold,
    }

    return metrics


def get_feature_group(name: str) -> str:
    if name in {
        "local_neighbor_corr_mean",
        "local_neighbor_corr_max",
        "sync_edge_ratio",
    }:
        return "local_synchronization"

    if name in {
        "moran_i",
        "geary_c",
        "high_state_component_ratio",
    }:
        return "spatial_connectivity"

    if name in {
        "gradient_entropy",
        "boundary_sharpness",
        "gradient_top10_mean",
    }:
        return "boundary_rigidity"

    if name in {
        "svd_mode1_energy_ratio",
        "svd_spectral_gap",
        "svd_mode1_ac1",
        "dominant_mode_stability",
        "dominant_mode_localization",
    }:
        return "dominant_mode_locking"

    return "unknown"


def build_importance_table(
    feature_names,
    impurity_importance,
    permutation_mean,
    permutation_std,
    model_name,
):
    df = pd.DataFrame({
        "feature": feature_names,
        "group": [get_feature_group(x) for x in feature_names],
        "model": model_name,
        "impurity_importance": impurity_importance,
        "permutation_importance_mean": permutation_mean,
        "permutation_importance_std": permutation_std,
    })

    df["impurity_rank"] = df["impurity_importance"].rank(
        ascending=False,
        method="min",
    ).astype(int)

    df["permutation_rank"] = df["permutation_importance_mean"].rank(
        ascending=False,
        method="min",
    ).astype(int)

    df = df.sort_values(
        ["permutation_importance_mean", "impurity_importance"],
        ascending=False,
    ).reset_index(drop=True)

    return df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-file",
        type=str,
        default="data/processed/v1_3_prepatch/seir_v1_3_prepatch_h30_seed42.npz",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="results_feature_importance/v1_3_prepatch_seed42",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.25)
    parser.add_argument("--n-repeats", type=int, default=10)

    args = parser.parse_args()

    data_file = Path(args.data_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    data = np.load(data_file, allow_pickle=True)

    required = ["X_prepatch", "prepatch_feature_names", "y_risk", "sim_id"]
    missing = [k for k in required if k not in data.files]

    if missing:
        raise KeyError(f"Missing keys: {missing}. Available keys: {data.files}")

    x = data["X_prepatch"].astype(np.float32)
    y = data["y_risk"].astype(int)
    sim_id = data["sim_id"]

    feature_names = [str(x) for x in data["prepatch_feature_names"].tolist()]

    print("===== v1.3 prepatch feature importance =====")
    print(f"Data file: {data_file}")
    print(f"X_prepatch: {x.shape}")
    print(f"y positive rate: {np.mean(y):.4f}")
    print(f"n features: {len(feature_names)}")
    print()

    train_idx, val_idx = split_by_sim_id(
        y=y,
        sim_id=sim_id,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    x_train, x_val = x[train_idx], x[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    print(f"Train samples: {len(train_idx)}")
    print(f"Val samples:   {len(val_idx)}")
    print(f"Train positive rate: {np.mean(y_train):.4f}")
    print(f"Val positive rate:   {np.mean(y_val):.4f}")
    print()

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=args.seed,
            n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=args.seed,
            n_jobs=-1,
        ),
    }

    all_importance = []

    for model_name, model in models.items():
        print(f"===== Training {model_name} =====")

        model.fit(x_train, y_train)

        metrics = evaluate_model(model, x_val, y_val)

        print("Validation metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v:.6f}")
        print()

        perm = permutation_importance(
            model,
            x_val,
            y_val,
            scoring="average_precision",
            n_repeats=args.n_repeats,
            random_state=args.seed,
            n_jobs=-1,
        )

        df_imp = build_importance_table(
            feature_names=feature_names,
            impurity_importance=model.feature_importances_,
            permutation_mean=perm.importances_mean,
            permutation_std=perm.importances_std,
            model_name=model_name,
        )

        out_csv = out_dir / f"prepatch_feature_importance_{model_name}.csv"
        df_imp.to_csv(out_csv, index=False)

        print(f"Saved: {out_csv}")
        print("Top features:")
        print(df_imp.head(10).to_string(index=False))
        print()

        all_importance.append(df_imp)

    merged = pd.concat(all_importance, axis=0, ignore_index=True)

    mean_df = (
        merged
        .groupby(["feature", "group"], as_index=False)
        .agg(
            impurity_importance_mean=("impurity_importance", "mean"),
            permutation_importance_mean=("permutation_importance_mean", "mean"),
            permutation_importance_std=("permutation_importance_mean", "std"),
        )
    )

    mean_df["overall_rank"] = mean_df["permutation_importance_mean"].rank(
        ascending=False,
        method="min",
    ).astype(int)

    mean_df = mean_df.sort_values(
        ["permutation_importance_mean", "impurity_importance_mean"],
        ascending=False,
    ).reset_index(drop=True)

    mean_csv = out_dir / "prepatch_feature_importance_mean.csv"
    mean_df.to_csv(mean_csv, index=False)

    group_df = (
        mean_df
        .groupby("group", as_index=False)
        .agg(
            group_permutation_importance=("permutation_importance_mean", "sum"),
            group_impurity_importance=("impurity_importance_mean", "sum"),
            n_features=("feature", "count"),
        )
        .sort_values("group_permutation_importance", ascending=False)
    )

    group_csv = out_dir / "prepatch_feature_group_importance.csv"
    group_df.to_csv(group_csv, index=False)

    print("===== Mean feature importance =====")
    print(mean_df.to_string(index=False))
    print()
    print("===== Group importance =====")
    print(group_df.to_string(index=False))
    print()
    print("Saved:")
    print(mean_csv)
    print(group_csv)


if __name__ == "__main__":
    main()