# build_v1_4_pwsi_dataset.py
# -*- coding: utf-8 -*-
"""
Build v1.4 PWSI datasets.

PWSI = Pre-patch Warning Signal Index

This script converts the 14 prepatch indicators into group-level scores
and two integrated warning indices:

1. PWSI_equal:
   equal-weight average of four prepatch groups.

2. PWSI_importance:
   importance-weighted average of four prepatch groups,
   based on v1.3 feature-importance results.

Input:
    data/processed/v1_3_prepatch/seir_v1_3_prepatch_h30_seed42.npz

Outputs:
    data/processed/v1_4_pwsi/seir_v1_4_pwsi_only_h30_seed42.npz
    data/processed/v1_4_pwsi/seir_v1_4_visible_pwsi_h30_seed42.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PREPATCH_GROUPS = {
    "Z_sync": [
        "local_neighbor_corr_mean",
        "local_neighbor_corr_max",
        "sync_edge_ratio",
    ],
    "Z_conn": [
        "moran_i",
        "geary_c",
        "high_state_component_ratio",
    ],
    "Z_rigid": [
        "gradient_entropy",
        "boundary_sharpness",
        "gradient_top10_mean",
    ],
    "Z_mode": [
        "svd_mode1_energy_ratio",
        "svd_spectral_gap",
        "svd_mode1_ac1",
        "dominant_mode_stability",
        "dominant_mode_localization",
    ],
}


# Direction adjustment.
# Positive direction means larger value indicates stronger warning signal.
# Geary's C is reversed because lower Geary's C usually means stronger spatial autocorrelation.
FEATURE_DIRECTIONS = {
    "local_neighbor_corr_mean": 1.0,
    "local_neighbor_corr_max": 1.0,
    "sync_edge_ratio": 1.0,

    "moran_i": 1.0,
    "geary_c": -1.0,
    "high_state_component_ratio": 1.0,

    "gradient_entropy": 1.0,
    "boundary_sharpness": 1.0,
    "gradient_top10_mean": 1.0,

    "svd_mode1_energy_ratio": 1.0,
    "svd_spectral_gap": 1.0,
    "svd_mode1_ac1": 1.0,
    "dominant_mode_stability": 1.0,
    "dominant_mode_localization": 1.0,
}


# Group importance values from v1.3 feature-importance analysis.
GROUP_IMPORTANCE = {
    "Z_sync": 0.093480,
    "Z_rigid": 0.052703,
    "Z_conn": 0.050721,
    "Z_mode": 0.032717,
}


def require_key(data, key: str):
    if key not in data.files:
        raise KeyError(f"Missing required key: {key}. Available keys: {data.files}")
    return data[key]


def optional_key(data, key: str, default=None):
    if key in data.files:
        return data[key]
    return default


def robust_zscore(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Robust z-score using median and IQR.

    This avoids excessive influence from extreme simulation states.
    """
    x = np.asarray(x, dtype=np.float32)

    med = np.nanmedian(x, axis=0, keepdims=True)
    q25 = np.nanpercentile(x, 25, axis=0, keepdims=True)
    q75 = np.nanpercentile(x, 75, axis=0, keepdims=True)
    iqr = q75 - q25

    z = (x - med) / (iqr + eps)
    z = np.nan_to_num(z, nan=0.0, posinf=5.0, neginf=-5.0)
    z = np.clip(z, -5.0, 5.0)

    return z.astype(np.float32)


def build_group_scores(
    X_prepatch: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, list[str], dict]:
    """
    Build four group-level prepatch scores:
        Z_sync, Z_conn, Z_rigid, Z_mode
    """
    X = np.asarray(X_prepatch, dtype=np.float32).copy()

    # Direction adjustment before normalization.
    for j, name in enumerate(feature_names):
        direction = FEATURE_DIRECTIONS.get(name, 1.0)
        X[:, j] = X[:, j] * direction

    X_z = robust_zscore(X)

    group_scores = []
    group_info = {}

    for group_name, members in PREPATCH_GROUPS.items():
        missing = [m for m in members if m not in feature_names]
        if missing:
            raise KeyError(f"Missing features for {group_name}: {missing}")

        idx = [feature_names.index(m) for m in members]
        score = np.mean(X_z[:, idx], axis=1)

        group_scores.append(score.astype(np.float32))
        group_info[group_name] = {
            "features": members,
            "indices": idx,
        }

    X_group = np.stack(group_scores, axis=1).astype(np.float32)
    group_names = list(PREPATCH_GROUPS.keys())

    return X_group, group_names, group_info


def build_pwsi(X_group: np.ndarray, group_names: list[str]) -> tuple[np.ndarray, list[str], dict]:
    """
    Build PWSI_equal and PWSI_importance from group-level scores.
    """
    X_group = np.asarray(X_group, dtype=np.float32)

    pwsi_equal = np.mean(X_group, axis=1)

    raw_weights = np.asarray(
        [GROUP_IMPORTANCE[g] for g in group_names],
        dtype=np.float32,
    )
    weights = raw_weights / np.sum(raw_weights)

    pwsi_importance = np.sum(X_group * weights[None, :], axis=1)

    X_pwsi = np.column_stack(
        [
            X_group,
            pwsi_equal,
            pwsi_importance,
        ]
    ).astype(np.float32)

    pwsi_feature_names = group_names + [
        "PWSI_equal",
        "PWSI_importance",
    ]

    info = {
        "group_names": group_names,
        "group_importance": {g: float(GROUP_IMPORTANCE[g]) for g in group_names},
        "normalized_group_weights": {
            g: float(w) for g, w in zip(group_names, weights)
        },
        "pwsi_feature_names": pwsi_feature_names,
    }

    return X_pwsi, pwsi_feature_names, info


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
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/processed/v1_4_pwsi",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon", type=int, default=30)

    args = parser.parse_args()

    data_file = Path(args.data_file)
    out_dir = Path(args.out_dir)

    if not data_file.exists():
        raise FileNotFoundError(f"Input file not found: {data_file}")

    print("===== Build v1.4 PWSI dataset =====")
    print(f"Input: {data_file}")
    print(f"Output directory: {out_dir}")
    print(f"Seed: {args.seed}")
    print(f"Horizon: {args.horizon}")

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

    print("\nInput shapes:")
    print(f"X_img:             {X_img.shape}")
    print(f"X_patch_original:  {X_patch_original.shape}")
    print(f"X_prepatch:        {X_prepatch.shape}")
    print(f"y_risk:            {y_risk.shape}")
    print(f"risk ratio:        {float(np.mean(y_risk)):.4f}")

    X_group, group_names, group_info = build_group_scores(
        X_prepatch=X_prepatch,
        feature_names=feature_names,
    )

    X_pwsi, pwsi_feature_names, pwsi_info = build_pwsi(
        X_group=X_group,
        group_names=group_names,
    )

    print("\nPWSI features:")
    for i, name in enumerate(pwsi_feature_names):
        print(f"{i:02d}: {name}")

    print("\nPWSI shape:")
    print(f"X_pwsi: {X_pwsi.shape}")

    if X_patch_original.ndim == 3:
        n, t, _ = X_patch_original.shape
        repeated_pwsi = np.repeat(X_pwsi[:, None, :], t, axis=1)
        X_visible_pwsi = np.concatenate(
            [
                X_patch_original.astype(np.float32),
                repeated_pwsi.astype(np.float32),
            ],
            axis=2,
        )
    elif X_patch_original.ndim == 2:
        X_visible_pwsi = np.concatenate(
            [
                X_patch_original.astype(np.float32),
                X_pwsi.astype(np.float32),
            ],
            axis=1,
        )
    else:
        raise ValueError(f"Unexpected X_patch_original shape: {X_patch_original.shape}")

    print(f"X_visible_pwsi: {X_visible_pwsi.shape}")

    base_payload = {
        "X_img": X_img.astype(np.float32),
        "X_patch_original": X_patch_original.astype(np.float32),
        "X_prepatch": X_prepatch.astype(np.float32),
        "X_group": X_group.astype(np.float32),
        "X_pwsi": X_pwsi.astype(np.float32),
        "X_visible_pwsi": X_visible_pwsi.astype(np.float32),
        "prepatch_feature_names": np.asarray(feature_names, dtype=object),
        "pwsi_feature_names": np.asarray(pwsi_feature_names, dtype=object),
        "pwsi_group_names": np.asarray(group_names, dtype=object),
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

    # Dataset 1: PWSI only.
    pwsi_only_payload = dict(base_payload)
    pwsi_only_payload["X_patch"] = X_pwsi.astype(np.float32)
    pwsi_only_payload["feature_set"] = "pwsi_only"

    pwsi_only_path = (
        out_dir / f"seir_v1_4_pwsi_only_h{args.horizon}_seed{args.seed}.npz"
    )
    save_npz(pwsi_only_path, pwsi_only_payload)

    # Dataset 2: visible patch + PWSI.
    visible_pwsi_payload = dict(base_payload)
    visible_pwsi_payload["X_patch"] = X_visible_pwsi.astype(np.float32)
    visible_pwsi_payload["feature_set"] = "visible_pwsi"

    visible_pwsi_path = (
        out_dir / f"seir_v1_4_visible_pwsi_h{args.horizon}_seed{args.seed}.npz"
    )
    save_npz(visible_pwsi_path, visible_pwsi_payload)

    summary = {
        "source_data_file": str(data_file),
        "seed": args.seed,
        "horizon": args.horizon,
        "pwsi_feature_names": pwsi_feature_names,
        "group_info": group_info,
        "pwsi_info": pwsi_info,
        "X_group_shape": list(X_group.shape),
        "X_pwsi_shape": list(X_pwsi.shape),
        "X_visible_pwsi_shape": list(X_visible_pwsi.shape),
        "outputs": {
            "pwsi_only": str(pwsi_only_path),
            "visible_pwsi": str(visible_pwsi_path),
        },
    }

    summary_path = out_dir / f"v1_4_pwsi_summary_h{args.horizon}_seed{args.seed}.json"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nSummary saved:")
    print(summary_path)

    print("\n===== Done =====")


if __name__ == "__main__":
    main()