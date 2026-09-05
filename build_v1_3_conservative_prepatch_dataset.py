# build_v1_3_conservative_prepatch_dataset.py
# -*- coding: utf-8 -*-
"""
Build conservative v1.3 prepatch datasets.

This script selects a conservative subset of prepatch indicators based on
feature importance results.

Input:
    data/processed/v1_3_prepatch/seir_v1_3_prepatch_h30_seed42.npz

Outputs:
    data/processed/v1_3_prepatch/seir_v1_3_conservative_prepatch_only_h30_seed42.npz
    data/processed/v1_3_prepatch/seir_v1_3_visible_conservative_prepatch_h30_seed42.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


CONSERVATIVE_PREPATCH_FEATURES = [
    "local_neighbor_corr_mean",
    "sync_edge_ratio",
    "gradient_entropy",
    "svd_mode1_ac1",
    "geary_c",
    "moran_i",
    "high_state_component_ratio",
    "gradient_top10_mean",
]


def require_key(data, key: str):
    if key not in data.files:
        raise KeyError(f"Missing required key: {key}. Available keys: {data.files}")
    return data[key]


def optional_key(data, key: str, default=None):
    if key in data.files:
        return data[key]
    return default


def save_npz(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-file",
        type=str,
        default="data/processed/v1_3_prepatch/seir_v1_3_prepatch_h30_seed42.npz",
        help="Path to full v1.3 prepatch dataset.",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/processed/v1_3_prepatch",
        help="Output directory.",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon", type=int, default=30)

    args = parser.parse_args()

    data_file = Path(args.data_file)
    out_dir = Path(args.out_dir)

    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    print("===== Build conservative v1.3 prepatch dataset =====")
    print(f"Input:  {data_file}")
    print(f"Output: {out_dir}")

    data = np.load(data_file, allow_pickle=True)

    X_img = require_key(data, "X_img")
    X_patch_original = require_key(data, "X_patch_original")
    X_prepatch = require_key(data, "X_prepatch")
    prepatch_feature_names = require_key(data, "prepatch_feature_names")

    y_risk = require_key(data, "y_risk")
    y_remaining = require_key(data, "y_remaining")
    sim_id = require_key(data, "sim_id")

    time_idx = optional_key(data, "time_idx")
    critical_time = optional_key(data, "critical_time")
    infected_area = optional_key(data, "infected_area")
    dominant_patch = optional_key(data, "dominant_patch")
    reff = optional_key(data, "reff")

    feature_names = [str(x) for x in prepatch_feature_names.tolist()]

    print("\nAvailable prepatch features:")
    for i, name in enumerate(feature_names):
        print(f"{i:02d}: {name}")

    missing = [f for f in CONSERVATIVE_PREPATCH_FEATURES if f not in feature_names]
    if missing:
        raise KeyError(f"Conservative features not found in dataset: {missing}")

    selected_idx = [feature_names.index(f) for f in CONSERVATIVE_PREPATCH_FEATURES]

    X_prepatch_conservative = X_prepatch[:, selected_idx].astype(np.float32)
    conservative_feature_names = np.asarray(CONSERVATIVE_PREPATCH_FEATURES, dtype=object)

    print("\nSelected conservative features:")
    for i, name in zip(selected_idx, CONSERVATIVE_PREPATCH_FEATURES):
        print(f"{i:02d}: {name}")

    print("\nShapes:")
    print(f"X_img:                    {X_img.shape}")
    print(f"X_patch_original:         {X_patch_original.shape}")
    print(f"X_prepatch:               {X_prepatch.shape}")
    print(f"X_prepatch_conservative:  {X_prepatch_conservative.shape}")

    if X_patch_original.ndim == 3:
        n, t, _ = X_patch_original.shape
        repeated_conservative = np.repeat(
            X_prepatch_conservative[:, None, :],
            t,
            axis=1,
        )
        X_visible_conservative = np.concatenate(
            [
                X_patch_original.astype(np.float32),
                repeated_conservative.astype(np.float32),
            ],
            axis=2,
        )
    elif X_patch_original.ndim == 2:
        X_visible_conservative = np.concatenate(
            [
                X_patch_original.astype(np.float32),
                X_prepatch_conservative.astype(np.float32),
            ],
            axis=1,
        )
    else:
        raise ValueError(f"Unexpected X_patch_original shape: {X_patch_original.shape}")

    print(f"X_visible_conservative:   {X_visible_conservative.shape}")

    base_payload = {
        "X_img": X_img.astype(np.float32),
        "X_patch_original": X_patch_original.astype(np.float32),
        "X_prepatch": X_prepatch.astype(np.float32),
        "X_prepatch_conservative": X_prepatch_conservative.astype(np.float32),
        "X_visible_conservative": X_visible_conservative.astype(np.float32),
        "prepatch_feature_names": prepatch_feature_names,
        "conservative_prepatch_feature_names": conservative_feature_names,
        "selected_prepatch_indices": np.asarray(selected_idx, dtype=np.int32),
        "y_risk": y_risk.astype(np.int32),
        "y_remaining": y_remaining.astype(np.float32),
        "sim_id": sim_id,
        "source_data_file": str(data_file),
        "seed": np.asarray(args.seed, dtype=np.int32),
        "horizon": np.asarray(args.horizon, dtype=np.int32),
    }

    if time_idx is not None:
        base_payload["time_idx"] = time_idx
    if critical_time is not None:
        base_payload["critical_time"] = critical_time
    if infected_area is not None:
        base_payload["infected_area"] = infected_area
    if dominant_patch is not None:
        base_payload["dominant_patch"] = dominant_patch
    if reff is not None:
        base_payload["reff"] = reff

    # Dataset 1: conservative prepatch only.
    prepatch_only_payload = dict(base_payload)
    prepatch_only_payload["X_patch"] = X_prepatch_conservative.astype(np.float32)
    prepatch_only_payload["feature_set"] = "conservative_prepatch_only"

    prepatch_only_path = (
        out_dir
        / f"seir_v1_3_conservative_prepatch_only_h{args.horizon}_seed{args.seed}.npz"
    )

    save_npz(prepatch_only_path, prepatch_only_payload)

    # Dataset 2: visible patch + conservative prepatch.
    visible_payload = dict(base_payload)
    visible_payload["X_patch"] = X_visible_conservative.astype(np.float32)
    visible_payload["feature_set"] = "visible_conservative_prepatch"

    visible_path = (
        out_dir
        / f"seir_v1_3_visible_conservative_prepatch_h{args.horizon}_seed{args.seed}.npz"
    )

    save_npz(visible_path, visible_payload)

    summary = {
        "source_data_file": str(data_file),
        "out_dir": str(out_dir),
        "seed": args.seed,
        "horizon": args.horizon,
        "selected_features": CONSERVATIVE_PREPATCH_FEATURES,
        "selected_indices": selected_idx,
        "X_prepatch_shape": list(X_prepatch.shape),
        "X_prepatch_conservative_shape": list(X_prepatch_conservative.shape),
        "X_visible_conservative_shape": list(X_visible_conservative.shape),
        "outputs": {
            "conservative_prepatch_only": str(prepatch_only_path),
            "visible_conservative_prepatch": str(visible_path),
        },
    }

    summary_path = (
        out_dir
        / f"v1_3_conservative_prepatch_summary_h{args.horizon}_seed{args.seed}.json"
    )

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nSummary saved:")
    print(summary_path)

    print("\n===== Done =====")


if __name__ == "__main__":
    main()