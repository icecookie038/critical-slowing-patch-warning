# build_v1_3_prepatch_dataset.py
# -*- coding: utf-8 -*-
"""
Build v1.3 pre-patch indicator datasets from v1.2 label-fix dataset.

Input:
    data/processed/v1_2_label_fix/seir_v1_2_h30_seed42.npz

Output:
    data/processed/v1_3_prepatch/seir_v1_3_prepatch_h30_seed42.npz
    data/processed/v1_3_prepatch/seir_v1_3_prepatch_only_h30_seed42.npz
    data/processed/v1_3_prepatch/seir_v1_3_visible_prepatch_h30_seed42.npz

Design:
    1. prepatch_h30:
       Keeps original X_img, X_patch, X_prepatch, X_patch_combined.

    2. prepatch_only:
       Sets X_patch = X_prepatch.
       This allows train_patch_baselines.py to directly train on pre-patch indicators.

    3. visible_prepatch:
       Sets X_patch = concat(original X_patch, repeated X_prepatch over T).
       This allows visible patch + pre-patch indicators to be trained together.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json

import numpy as np

from prepatch_indicators import (
    compute_prepatch_indicators_batch,
    feature_names,
)


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
        required=True,
        help="Path to v1.2 label-fix dataset npz.",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/processed/v1_3_prepatch",
        help="Output directory for v1.3 prepatch datasets.",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--sync-threshold", type=float, default=0.6)

    args = parser.parse_args()

    data_file = Path(args.data_file)
    out_dir = Path(args.out_dir)

    if not data_file.exists():
        raise FileNotFoundError(f"Input dataset not found: {data_file}")

    print("===== Build v1.3 pre-patch dataset =====")
    print(f"Input dataset: {data_file}")
    print(f"Output dir:    {out_dir}")

    data = np.load(data_file, allow_pickle=True)

    print("\nAvailable keys:")
    print(data.files)

    X_img = require_key(data, "X_img")
    X_patch = require_key(data, "X_patch")
    y_risk = require_key(data, "y_risk")
    y_remaining = require_key(data, "y_remaining")
    sim_id = require_key(data, "sim_id")

    time_idx = optional_key(data, "time_idx")
    critical_time = optional_key(data, "critical_time")

    infected_area = optional_key(data, "infected_area")
    dominant_patch = optional_key(data, "dominant_patch")
    reff = optional_key(data, "reff")

    print("\nInput shapes:")
    print(f"X_img:        {X_img.shape}")
    print(f"X_patch:      {X_patch.shape}")
    print(f"y_risk:       {y_risk.shape}")
    print(f"y_remaining:  {y_remaining.shape}")
    print(f"sim_id:       {sim_id.shape}")

    if time_idx is not None:
        print(f"time_idx:     {time_idx.shape}")

    if critical_time is not None:
        print(f"critical_time:{critical_time.shape}")

    # Compute pre-patch indicators from X_img.
    X_prepatch = compute_prepatch_indicators_batch(
        X_img,
        sync_threshold=args.sync_threshold,
        verbose=True,
    )

    names = np.asarray(feature_names(), dtype=object)

    print("\nPre-patch indicators:")
    print(f"X_prepatch: {X_prepatch.shape}")
    print("Feature names:")
    print(feature_names())

    # Build combined sequence feature:
    # original X_patch: (N,T,F)
    # X_prepatch:       (N,P)
    # repeated:         (N,T,P)
    if X_patch.ndim == 3:
        n, t, _ = X_patch.shape
        repeated_prepatch = np.repeat(X_prepatch[:, None, :], t, axis=1)
        X_patch_combined = np.concatenate(
            [X_patch.astype(np.float32), repeated_prepatch.astype(np.float32)],
            axis=2,
        )
    elif X_patch.ndim == 2:
        X_patch_combined = np.concatenate(
            [X_patch.astype(np.float32), X_prepatch.astype(np.float32)],
            axis=1,
        )
    else:
        raise ValueError(f"Unexpected X_patch shape: {X_patch.shape}")

    print(f"X_patch_combined: {X_patch_combined.shape}")

    base_payload = {
        "X_img": X_img.astype(np.float32),
        "X_patch_original": X_patch.astype(np.float32),
        "X_prepatch": X_prepatch.astype(np.float32),
        "X_patch_combined": X_patch_combined.astype(np.float32),
        "prepatch_feature_names": names,
        "y_risk": y_risk.astype(np.int32),
        "y_remaining": y_remaining.astype(np.float32),
        "sim_id": sim_id,
        "source_data_file": str(data_file),
        "sync_threshold": np.asarray(args.sync_threshold, dtype=np.float32),
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

    # 1. Full prepatch dataset with all representations.
    full_path = out_dir / f"seir_v1_3_prepatch_h{args.horizon}_seed{args.seed}.npz"
    save_npz(full_path, base_payload)

    # 2. Prepatch-only dataset:
    # train_patch_baselines.py reads X_patch, so set X_patch = X_prepatch.
    prepatch_only_payload = dict(base_payload)
    prepatch_only_payload["X_patch"] = X_prepatch.astype(np.float32)
    prepatch_only_payload["feature_set"] = "prepatch_only"

    prepatch_only_path = out_dir / f"seir_v1_3_prepatch_only_h{args.horizon}_seed{args.seed}.npz"
    save_npz(prepatch_only_path, prepatch_only_payload)

    # 3. Visible + prepatch dataset:
    # train_patch_baselines.py reads X_patch, so set X_patch = X_patch_combined.
    visible_prepatch_payload = dict(base_payload)
    visible_prepatch_payload["X_patch"] = X_patch_combined.astype(np.float32)
    visible_prepatch_payload["feature_set"] = "visible_prepatch"

    visible_prepatch_path = out_dir / f"seir_v1_3_visible_prepatch_h{args.horizon}_seed{args.seed}.npz"
    save_npz(visible_prepatch_path, visible_prepatch_payload)

    summary = {
        "source_data_file": str(data_file),
        "out_dir": str(out_dir),
        "seed": args.seed,
        "horizon": args.horizon,
        "sync_threshold": args.sync_threshold,
        "n_samples": int(len(y_risk)),
        "X_img_shape": list(X_img.shape),
        "X_patch_shape": list(X_patch.shape),
        "X_prepatch_shape": list(X_prepatch.shape),
        "X_patch_combined_shape": list(X_patch_combined.shape),
        "positive_rate": float(np.mean(y_risk)),
        "feature_names": feature_names(),
        "outputs": {
            "full": str(full_path),
            "prepatch_only": str(prepatch_only_path),
            "visible_prepatch": str(visible_prepatch_path),
        },
    }

    summary_path = out_dir / f"v1_3_prepatch_summary_h{args.horizon}_seed{args.seed}.json"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nSummary saved:")
    print(summary_path)

    print("\n===== Done =====")


if __name__ == "__main__":
    main()