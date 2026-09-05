# build_v1_4_pwsi_single_index_dataset.py
# -*- coding: utf-8 -*-
"""
Build v1.4.1 PWSI single-index ablation datasets.

This script separates the compact PWSI representation into single-index datasets:

1. PWSI_equal only
2. PWSI_importance only
3. visible patch + PWSI_equal
4. visible patch + PWSI_importance

Input:
    data/processed/v1_4_pwsi/seir_v1_4_pwsi_only_h30_seed42.npz

Outputs:
    data/processed/v1_4_pwsi_single/
        seir_v1_4_pwsi_equal_only_h30_seed42.npz
        seir_v1_4_pwsi_importance_only_h30_seed42.npz
        seir_v1_4_visible_pwsi_equal_h30_seed42.npz
        seir_v1_4_visible_pwsi_importance_h30_seed42.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


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


def build_visible_single_index(
    X_patch_original: np.ndarray,
    X_single: np.ndarray,
) -> np.ndarray:
    """
    Append one PWSI index to visible patch indicators.

    If X_patch_original is 3D:
        (N, T, F_visible) + (N, 1) -> (N, T, F_visible + 1)

    If X_patch_original is 2D:
        (N, F_visible) + (N, 1) -> (N, F_visible + 1)
    """
    X_patch_original = np.asarray(X_patch_original, dtype=np.float32)
    X_single = np.asarray(X_single, dtype=np.float32)

    if X_single.ndim == 1:
        X_single = X_single[:, None]

    if X_patch_original.ndim == 3:
        n, t, _ = X_patch_original.shape
        repeated_single = np.repeat(X_single[:, None, :], t, axis=1)
        X_visible_single = np.concatenate(
            [
                X_patch_original,
                repeated_single.astype(np.float32),
            ],
            axis=2,
        )
    elif X_patch_original.ndim == 2:
        X_visible_single = np.concatenate(
            [
                X_patch_original,
                X_single.astype(np.float32),
            ],
            axis=1,
        )
    else:
        raise ValueError(f"Unexpected X_patch_original shape: {X_patch_original.shape}")

    return X_visible_single.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-file",
        type=str,
        default="data/processed/v1_4_pwsi/seir_v1_4_pwsi_only_h30_seed42.npz",
        help="Path to v1.4 PWSI dataset.",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/processed/v1_4_pwsi_single",
        help="Output directory.",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon", type=int, default=30)

    args = parser.parse_args()

    data_file = Path(args.data_file)
    out_dir = Path(args.out_dir)

    if not data_file.exists():
        raise FileNotFoundError(f"Input file not found: {data_file}")

    print("===== Build v1.4.1 PWSI single-index datasets =====")
    print(f"Input:  {data_file}")
    print(f"Output: {out_dir}")
    print(f"Seed:   {args.seed}")
    print(f"Horizon:{args.horizon}")

    data = np.load(data_file, allow_pickle=True)

    X_img = require_key(data, "X_img")
    X_patch_original = require_key(data, "X_patch_original")
    X_pwsi = require_key(data, "X_pwsi")
    pwsi_feature_names = require_key(data, "pwsi_feature_names")

    y_risk = require_key(data, "y_risk")
    y_remaining = require_key(data, "y_remaining")
    sim_id = require_key(data, "sim_id")

    time_idx = optional_key(data, "time_idx")
    critical_time = optional_key(data, "critical_time")
    infected_area = optional_key(data, "infected_area")
    dominant_patch = optional_key(data, "dominant_patch")
    reff = optional_key(data, "reff")

    names = [str(x) for x in pwsi_feature_names.tolist()]

    print("\nAvailable PWSI features:")
    for i, name in enumerate(names):
        print(f"{i:02d}: {name}")

    required_features = ["PWSI_equal", "PWSI_importance"]
    missing = [x for x in required_features if x not in names]
    if missing:
        raise KeyError(f"Missing PWSI features: {missing}")

    idx_equal = names.index("PWSI_equal")
    idx_importance = names.index("PWSI_importance")

    X_equal = X_pwsi[:, idx_equal].astype(np.float32).reshape(-1, 1)
    X_importance = X_pwsi[:, idx_importance].astype(np.float32).reshape(-1, 1)

    X_visible_equal = build_visible_single_index(
        X_patch_original=X_patch_original,
        X_single=X_equal,
    )

    X_visible_importance = build_visible_single_index(
        X_patch_original=X_patch_original,
        X_single=X_importance,
    )

    print("\nShapes:")
    print(f"X_img:                {X_img.shape}")
    print(f"X_patch_original:     {X_patch_original.shape}")
    print(f"X_pwsi:               {X_pwsi.shape}")
    print(f"X_equal:              {X_equal.shape}")
    print(f"X_importance:         {X_importance.shape}")
    print(f"X_visible_equal:      {X_visible_equal.shape}")
    print(f"X_visible_importance: {X_visible_importance.shape}")

    base_payload = {
        "X_img": X_img.astype(np.float32),
        "X_patch_original": X_patch_original.astype(np.float32),
        "X_pwsi": X_pwsi.astype(np.float32),
        "pwsi_feature_names": np.asarray(names, dtype=object),
        "y_risk": y_risk.astype(np.int32),
        "y_remaining": y_remaining.astype(np.float32),
        "sim_id": sim_id,
        "seed": np.asarray(args.seed, dtype=np.int32),
        "horizon": np.asarray(args.horizon, dtype=np.int32),
        "source_data_file": str(data_file),
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

    # 1. PWSI_equal only.
    payload_equal = dict(base_payload)
    payload_equal["X_patch"] = X_equal.astype(np.float32)
    payload_equal["single_index_name"] = np.asarray("PWSI_equal", dtype=object)
    payload_equal["feature_set"] = "pwsi_equal_only"

    path_equal = out_dir / f"seir_v1_4_pwsi_equal_only_h{args.horizon}_seed{args.seed}.npz"
    save_npz(path_equal, payload_equal)

    # 2. PWSI_importance only.
    payload_importance = dict(base_payload)
    payload_importance["X_patch"] = X_importance.astype(np.float32)
    payload_importance["single_index_name"] = np.asarray("PWSI_importance", dtype=object)
    payload_importance["feature_set"] = "pwsi_importance_only"

    path_importance = out_dir / f"seir_v1_4_pwsi_importance_only_h{args.horizon}_seed{args.seed}.npz"
    save_npz(path_importance, payload_importance)

    # 3. visible patch + PWSI_equal.
    payload_visible_equal = dict(base_payload)
    payload_visible_equal["X_patch"] = X_visible_equal.astype(np.float32)
    payload_visible_equal["single_index_name"] = np.asarray("PWSI_equal", dtype=object)
    payload_visible_equal["feature_set"] = "visible_pwsi_equal"

    path_visible_equal = out_dir / f"seir_v1_4_visible_pwsi_equal_h{args.horizon}_seed{args.seed}.npz"
    save_npz(path_visible_equal, payload_visible_equal)

    # 4. visible patch + PWSI_importance.
    payload_visible_importance = dict(base_payload)
    payload_visible_importance["X_patch"] = X_visible_importance.astype(np.float32)
    payload_visible_importance["single_index_name"] = np.asarray("PWSI_importance", dtype=object)
    payload_visible_importance["feature_set"] = "visible_pwsi_importance"

    path_visible_importance = out_dir / f"seir_v1_4_visible_pwsi_importance_h{args.horizon}_seed{args.seed}.npz"
    save_npz(path_visible_importance, payload_visible_importance)

    summary = {
        "source_data_file": str(data_file),
        "out_dir": str(out_dir),
        "seed": args.seed,
        "horizon": args.horizon,
        "pwsi_feature_names": names,
        "selected_indices": {
            "PWSI_equal": idx_equal,
            "PWSI_importance": idx_importance,
        },
        "shapes": {
            "X_patch_original": list(X_patch_original.shape),
            "X_equal": list(X_equal.shape),
            "X_importance": list(X_importance.shape),
            "X_visible_equal": list(X_visible_equal.shape),
            "X_visible_importance": list(X_visible_importance.shape),
        },
        "outputs": {
            "pwsi_equal_only": str(path_equal),
            "pwsi_importance_only": str(path_importance),
            "visible_pwsi_equal": str(path_visible_equal),
            "visible_pwsi_importance": str(path_visible_importance),
        },
    }

    summary_path = out_dir / f"v1_4_pwsi_single_index_summary_h{args.horizon}_seed{args.seed}.json"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nSummary saved:")
    print(summary_path)

    print("\n===== Done =====")


if __name__ == "__main__":
    main()