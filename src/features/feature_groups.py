"""
Feature group definitions for v0.9 SEIR experiments.

v0.7:
    full / img_only / patch_only

v0.9:
    image_only
    classic_patch_only
    dynamic_patch_only
    classic_dynamic_patch
    full

This module only builds feature matrices.
It does not train models.
It does not normalize using the whole dataset.
It does not modify v0.7 baseline results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd


FeatureGroupName = Literal[
    "image_only",
    "classic_patch_only",
    "dynamic_patch_only",
    "classic_dynamic_patch",
    "full",
]

SystemMode = Literal["expansion", "collapse"]


CLASSIC_PATCH_COLUMNS = [
    "patch_count",
    "largest_patch_area",
    "mean_patch_area",
    "largest_patch_ratio",
    "gini",
    "jsd",
    "spatial_aggregation",
    "ac1",
    "variance",
    "trend",
]

DYNAMIC_PATCH_COLUMNS_EXPANSION = [
    "pdsi",
    "dpci",
    "fai",
    "psii",
]

DYNAMIC_PATCH_COLUMNS_COLLAPSE = [
    "pdsi",
    "dpcr",
    "fai",
    "psii",
]


@dataclass
class FeatureGroupOutput:
    """Output container for one feature group."""

    group_name: str
    matrix: np.ndarray
    feature_names: list[str]
    image_dim: int
    classic_patch_dim: int
    dynamic_patch_dim: int
    total_dim: int


def _as_2d_array(x: np.ndarray, name: str) -> np.ndarray:
    """
    Convert an input feature array into shape (N, D).

    If x has shape:
    - (N, D), keep it.
    - (N, H, W), flatten to (N, H*W).
    - (N, H, W, C), flatten to (N, H*W*C).
    """
    arr = np.asarray(x)

    if arr.ndim < 2:
        raise ValueError(f"{name} must have at least 2 dimensions, got {arr.shape}")

    if arr.ndim == 2:
        out = arr
    else:
        out = arr.reshape(arr.shape[0], -1)

    out = out.astype(np.float32)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    return out


def _normalize_column_name(name: str) -> str:
    """
    Normalize column names for robust matching.

    Example:
        "Largest Patch Area" -> "largest_patch_area"
        "LargestPatchArea"   -> "largestpatcharea"
    """
    return (
        str(name)
        .strip()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .lower()
    )


def _build_column_lookup(df: pd.DataFrame) -> dict[str, str]:
    """
    Map normalized column names to original column names.
    """
    lookup: dict[str, str] = {}

    for col in df.columns:
        lookup[_normalize_column_name(col)] = col

    return lookup


def _select_dataframe_columns(
    df: pd.DataFrame,
    requested_columns: list[str],
    strict: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Select columns from a DataFrame using robust normalized matching.

    If strict=False, missing columns are skipped.
    If strict=True, missing columns raise an error.
    """
    lookup = _build_column_lookup(df)

    selected_original_cols: list[str] = []
    selected_feature_names: list[str] = []
    missing: list[str] = []

    for col in requested_columns:
        key = _normalize_column_name(col)

        if key in lookup:
            selected_original_cols.append(lookup[key])
            selected_feature_names.append(col)
        else:
            missing.append(col)

    if missing and strict:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    if not selected_original_cols:
        raise ValueError(
            f"No requested columns found. Requested={requested_columns}, "
            f"available={list(df.columns)}"
        )

    return df[selected_original_cols], selected_feature_names


def _dataframe_to_matrix(
    df: pd.DataFrame,
    columns: list[str],
    strict: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """
    Convert selected DataFrame columns into a numeric matrix.
    """
    selected_df, feature_names = _select_dataframe_columns(
        df=df,
        requested_columns=columns,
        strict=strict,
    )

    matrix = selected_df.astype(float).to_numpy(dtype=np.float32)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

    return matrix, feature_names


def get_dynamic_columns(mode: SystemMode = "expansion") -> list[str]:
    """
    Return dynamic patch columns for different ecological systems.

    SEIR expansion:
        PDSI + DPCI + FAI + PSII

    Vegetation degradation / healthy patch collapse:
        PDSI + DPCR + FAI + PSII
    """
    if mode == "expansion":
        return DYNAMIC_PATCH_COLUMNS_EXPANSION.copy()

    if mode == "collapse":
        return DYNAMIC_PATCH_COLUMNS_COLLAPSE.copy()

    raise ValueError("mode must be 'expansion' or 'collapse'")


def build_feature_group_matrix(
    group_name: FeatureGroupName,
    image_features: Optional[np.ndarray] = None,
    classic_patch_features: Optional[pd.DataFrame | np.ndarray] = None,
    dynamic_patch_features: Optional[pd.DataFrame | np.ndarray] = None,
    mode: SystemMode = "expansion",
    strict_columns: bool = False,
) -> FeatureGroupOutput:
    """
    Build one feature matrix according to the requested feature group.

    Parameters
    ----------
    group_name:
        One of:
        - image_only
        - classic_patch_only
        - dynamic_patch_only
        - classic_dynamic_patch
        - full

    image_features:
        Image feature array. Shape can be:
        - (N, D)
        - (N, H, W)
        - (N, H, W, C)

    classic_patch_features:
        Classic patch feature DataFrame or array.

    dynamic_patch_features:
        Dynamic patch feature DataFrame or array.

    mode:
        "expansion" for SEIR.
        "collapse" for vegetation degradation.

    strict_columns:
        If True, missing named columns raise errors.
        If False, available matching columns are used.

    Returns
    -------
    FeatureGroupOutput
        Contains matrix, feature names, and dimensions.
    """
    parts: list[np.ndarray] = []
    feature_names: list[str] = []

    image_dim = 0
    classic_dim = 0
    dynamic_dim = 0

    needs_image = group_name in ["image_only", "full"]
    needs_classic = group_name in ["classic_patch_only", "classic_dynamic_patch", "full"]
    needs_dynamic = group_name in ["dynamic_patch_only", "classic_dynamic_patch", "full"]

    if needs_image:
        if image_features is None:
            raise ValueError(f"group {group_name} requires image_features")

        image_matrix = _as_2d_array(image_features, name="image_features")
        image_dim = image_matrix.shape[1]

        parts.append(image_matrix)
        feature_names.extend([f"image_{i}" for i in range(image_dim)])

    if needs_classic:
        if classic_patch_features is None:
            raise ValueError(f"group {group_name} requires classic_patch_features")

        if isinstance(classic_patch_features, pd.DataFrame):
            classic_matrix, classic_names = _dataframe_to_matrix(
                df=classic_patch_features,
                columns=CLASSIC_PATCH_COLUMNS,
                strict=strict_columns,
            )
        else:
            classic_matrix = _as_2d_array(
                classic_patch_features,
                name="classic_patch_features",
            )
            classic_names = [
                CLASSIC_PATCH_COLUMNS[i]
                if i < len(CLASSIC_PATCH_COLUMNS)
                else f"classic_patch_{i}"
                for i in range(classic_matrix.shape[1])
            ]

        classic_dim = classic_matrix.shape[1]
        parts.append(classic_matrix)
        feature_names.extend(classic_names)

    if needs_dynamic:
        if dynamic_patch_features is None:
            raise ValueError(f"group {group_name} requires dynamic_patch_features")

        dynamic_columns = get_dynamic_columns(mode=mode)

        if isinstance(dynamic_patch_features, pd.DataFrame):
            dynamic_matrix, dynamic_names = _dataframe_to_matrix(
                df=dynamic_patch_features,
                columns=dynamic_columns,
                strict=strict_columns,
            )
        else:
            dynamic_matrix = _as_2d_array(
                dynamic_patch_features,
                name="dynamic_patch_features",
            )
            dynamic_names = [
                dynamic_columns[i]
                if i < len(dynamic_columns)
                else f"dynamic_patch_{i}"
                for i in range(dynamic_matrix.shape[1])
            ]

        dynamic_dim = dynamic_matrix.shape[1]
        parts.append(dynamic_matrix)
        feature_names.extend(dynamic_names)

    if not parts:
        raise ValueError(f"No feature parts were selected for group: {group_name}")

    n_rows = [p.shape[0] for p in parts]

    if len(set(n_rows)) != 1:
        raise ValueError(
            f"Feature parts have inconsistent sample counts: {n_rows}. "
            "All feature groups must have the same N."
        )

    matrix = np.concatenate(parts, axis=1).astype(np.float32)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

    return FeatureGroupOutput(
        group_name=group_name,
        matrix=matrix,
        feature_names=feature_names,
        image_dim=image_dim,
        classic_patch_dim=classic_dim,
        dynamic_patch_dim=dynamic_dim,
        total_dim=matrix.shape[1],
    )


def build_all_feature_groups(
    image_features: Optional[np.ndarray] = None,
    classic_patch_features: Optional[pd.DataFrame | np.ndarray] = None,
    dynamic_patch_features: Optional[pd.DataFrame | np.ndarray] = None,
    mode: SystemMode = "expansion",
    strict_columns: bool = False,
) -> dict[str, FeatureGroupOutput]:
    """
    Build all five v0.9 feature groups.
    """
    group_names: list[FeatureGroupName] = [
        "image_only",
        "classic_patch_only",
        "dynamic_patch_only",
        "classic_dynamic_patch",
        "full",
    ]

    outputs: dict[str, FeatureGroupOutput] = {}

    for group in group_names:
        outputs[group] = build_feature_group_matrix(
            group_name=group,
            image_features=image_features,
            classic_patch_features=classic_patch_features,
            dynamic_patch_features=dynamic_patch_features,
            mode=mode,
            strict_columns=strict_columns,
        )

    return outputs


def summarize_feature_groups(
    outputs: dict[str, FeatureGroupOutput],
) -> pd.DataFrame:
    """
    Summarize feature dimensions for all groups.
    """
    rows = []

    for name, output in outputs.items():
        rows.append(
            {
                "Feature_Group": name,
                "Image_Dim": output.image_dim,
                "Classic_Patch_Dim": output.classic_patch_dim,
                "Dynamic_Patch_Dim": output.dynamic_patch_dim,
                "Total_Dim": output.total_dim,
                "Num_Features": len(output.feature_names),
            }
        )

    return pd.DataFrame(rows)


def _demo() -> None:
    """
    Quick self-test.
    """
    n = 20

    image = np.random.rand(n, 8, 8).astype(np.float32)

    classic = pd.DataFrame(
        {
            "patch_count": np.random.rand(n),
            "largest_patch_area": np.random.rand(n),
            "mean_patch_area": np.random.rand(n),
            "largest_patch_ratio": np.random.rand(n),
            "gini": np.random.rand(n),
            "jsd": np.random.rand(n),
            "spatial_aggregation": np.random.rand(n),
            "ac1": np.random.rand(n),
            "variance": np.random.rand(n),
            "trend": np.random.rand(n),
        }
    )

    dynamic = pd.DataFrame(
        {
            "pdsi": np.random.rand(n),
            "dpci": np.random.rand(n),
            "dpcr": np.random.rand(n),
            "fai": np.random.rand(n),
            "psii": np.random.rand(n),
        }
    )

    outputs = build_all_feature_groups(
        image_features=image,
        classic_patch_features=classic,
        dynamic_patch_features=dynamic,
        mode="expansion",
    )

    summary = summarize_feature_groups(outputs)

    print(summary)

    for name, output in outputs.items():
        print(name, output.matrix.shape, output.feature_names[:5])


if __name__ == "__main__":
    _demo()