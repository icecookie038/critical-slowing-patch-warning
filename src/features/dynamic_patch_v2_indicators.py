"""
Dynamic Patch Indicators v2.

This module implements spreading- and connectivity-aware dynamic patch indicators:

BCI   Boundary Complexity Index
FPV   Front Propagation Velocity
PMR   Patch Merging Rate
GCGR  Giant Component Growth Rate
PPI   Percolation Proximity Index

These indicators are designed for spatial spreading systems such as SEIR.
They do not replace v0.8 dynamic indicators. They extend them.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque

import numpy as np
import pandas as pd


DYNAMIC_PATCH_V2_COLUMNS = [
    "bci",
    "fpv",
    "pmr",
    "gcgr",
    "ppi",
]


@dataclass
class PatchState:
    time: int
    patch_count: int
    total_patch_area: float
    largest_patch_area: float
    largest_patch_ratio: float
    total_perimeter: float
    bci: float
    ppi: float
    spanning_x: int
    spanning_y: int


def _to_mask(field: np.ndarray, threshold: float = 0.5, positive: bool = True) -> np.ndarray:
    arr = np.asarray(field)

    if arr.ndim != 2:
        raise ValueError(f"Each field must be 2D, got shape={arr.shape}")

    if positive:
        return arr >= threshold

    return arr <= threshold


def _neighbors(y: int, x: int, h: int, w: int, connectivity: int = 8):
    directions_4 = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
    ]

    directions_8 = directions_4 + [
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ]

    directions = directions_8 if connectivity == 8 else directions_4

    for dy, dx in directions:
        ny = y + dy
        nx = x + dx

        if 0 <= ny < h and 0 <= nx < w:
            yield ny, nx


def _label_components(mask: np.ndarray, connectivity: int = 8):
    """
    Label connected components without requiring scipy.

    Returns:
        labels: integer array with 0 as background
        sizes: list of component sizes
    """
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape

    labels = np.zeros((h, w), dtype=np.int32)
    sizes: list[int] = []

    current_label = 0

    for y in range(h):
        for x in range(w):
            if not mask[y, x]:
                continue

            if labels[y, x] != 0:
                continue

            current_label += 1
            q = deque()
            q.append((y, x))
            labels[y, x] = current_label
            size = 0

            while q:
                cy, cx = q.popleft()
                size += 1

                for ny, nx in _neighbors(cy, cx, h, w, connectivity=connectivity):
                    if mask[ny, nx] and labels[ny, nx] == 0:
                        labels[ny, nx] = current_label
                        q.append((ny, nx))

            sizes.append(size)

    return labels, sizes


def _perimeter_edges(mask: np.ndarray) -> float:
    """
    Count exposed 4-neighborhood boundary edges.

    This is a stable grid-based perimeter approximation.
    """
    mask = np.asarray(mask, dtype=bool)

    if mask.size == 0 or not mask.any():
        return 0.0

    padded = np.pad(mask, pad_width=1, mode="constant", constant_values=False)

    center = padded[1:-1, 1:-1]
    up = padded[:-2, 1:-1]
    down = padded[2:, 1:-1]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]

    perimeter = (
        center & (~up)
    ).sum() + (
        center & (~down)
    ).sum() + (
        center & (~left)
    ).sum() + (
        center & (~right)
    ).sum()

    return float(perimeter)


def _spanning_flags(mask: np.ndarray) -> tuple[int, int]:
    """
    Return whether the infected region spans the grid.

    spanning_x = left-right connection exists
    spanning_y = top-bottom connection exists

    This is a simple whole-mask check. It does not require the same connected
    component to span both sides, but it is useful as a weak percolation signal.
    """
    mask = np.asarray(mask, dtype=bool)

    if mask.size == 0 or not mask.any():
        return 0, 0

    left = mask[:, 0].any()
    right = mask[:, -1].any()
    top = mask[0, :].any()
    bottom = mask[-1, :].any()

    spanning_x = int(left and right)
    spanning_y = int(top and bottom)

    return spanning_x, spanning_y


def _largest_component_mask(labels: np.ndarray, sizes: list[int]) -> np.ndarray:
    if not sizes:
        return np.zeros_like(labels, dtype=bool)

    largest_label = int(np.argmax(sizes)) + 1

    return labels == largest_label


def _compute_patch_state(
    mask: np.ndarray,
    time: int,
    connectivity: int = 8,
    eps: float = 1e-8,
) -> PatchState:
    labels, sizes = _label_components(mask, connectivity=connectivity)

    patch_count = len(sizes)
    total_area = float(np.sum(sizes)) if sizes else 0.0
    largest_area = float(max(sizes)) if sizes else 0.0
    grid_area = float(mask.shape[0] * mask.shape[1])

    largest_ratio = largest_area / (total_area + eps) if total_area > 0 else 0.0

    total_perimeter = _perimeter_edges(mask)

    # BCI: boundary complexity. Normalized by area to avoid area-only dominance.
    bci = (total_perimeter ** 2) / (total_area + eps) if total_area > 0 else 0.0

    # PPI: simple percolation proximity based on largest component size.
    ppi = largest_area / (grid_area + eps)

    spanning_x, spanning_y = _spanning_flags(mask)

    return PatchState(
        time=time,
        patch_count=patch_count,
        total_patch_area=total_area,
        largest_patch_area=largest_area,
        largest_patch_ratio=largest_ratio,
        total_perimeter=total_perimeter,
        bci=bci,
        ppi=ppi,
        spanning_x=spanning_x,
        spanning_y=spanning_y,
    )


def extract_dynamic_patch_v2_indicators(
    field_series: np.ndarray,
    threshold: float = 0.5,
    positive: bool = True,
    connectivity: int = 8,
    lag: int = 1,
    eps: float = 1e-8,
) -> pd.DataFrame:
    """
    Extract dynamic patch v2 indicators from a spatial field series.

    Parameters
    ----------
    field_series:
        Array with shape (T, H, W).

    threshold:
        Threshold used to convert the field into a binary patch mask.

    positive:
        If True, values >= threshold are treated as target patches.
        If False, values <= threshold are treated as target patches.

    connectivity:
        4 or 8 connected neighborhood.

    lag:
        Temporal lag used for dynamic indicators.

    Returns
    -------
    DataFrame
        Contains state variables and dynamic v2 indicators.
    """
    arr = np.asarray(field_series)

    if arr.ndim != 3:
        raise ValueError(f"field_series must have shape (T,H,W), got {arr.shape}")

    if lag < 1:
        raise ValueError("lag must be >= 1")

    states: list[PatchState] = []

    for t in range(arr.shape[0]):
        mask = _to_mask(arr[t], threshold=threshold, positive=positive)
        state = _compute_patch_state(
            mask=mask,
            time=t,
            connectivity=connectivity,
            eps=eps,
        )
        states.append(state)

    rows = []

    for t, state in enumerate(states):
        if t - lag >= 0:
            prev = states[t - lag]
        else:
            prev = None

        if prev is None:
            fpv = 0.0
            pmr = 0.0
            gcgr = 0.0
        else:
            # FPV: front propagation velocity using equivalent radius.
            radius = np.sqrt(state.total_patch_area / np.pi) if state.total_patch_area > 0 else 0.0
            prev_radius = np.sqrt(prev.total_patch_area / np.pi) if prev.total_patch_area > 0 else 0.0
            fpv = max(0.0, radius - prev_radius)

            # PMR: patch merging rate. Only count merging when total area does not shrink.
            if state.total_patch_area >= prev.total_patch_area:
                pmr = max(0.0, float(prev.patch_count - state.patch_count))
            else:
                pmr = 0.0

            # GCGR: growth of giant component ratio.
            gcgr = max(0.0, state.largest_patch_ratio - prev.largest_patch_ratio)

        rows.append(
            {
                "time": state.time,
                "patch_count": state.patch_count,
                "total_patch_area": state.total_patch_area,
                "largest_patch_area": state.largest_patch_area,
                "largest_patch_ratio": state.largest_patch_ratio,
                "total_perimeter": state.total_perimeter,
                "spanning_x": state.spanning_x,
                "spanning_y": state.spanning_y,
                "bci": state.bci,
                "fpv": fpv,
                "pmr": pmr,
                "gcgr": gcgr,
                "ppi": state.ppi,
            }
        )

    df = pd.DataFrame(rows)

    for col in DYNAMIC_PATCH_V2_COLUMNS:
        df[col] = df[col].replace([np.inf, -np.inf], 0.0).fillna(0.0)

    return df


def extract_dynamic_patch_v2_feature_matrix(
    field_series: np.ndarray,
    threshold: float = 0.5,
    positive: bool = True,
    connectivity: int = 8,
    lag: int = 1,
    feature_columns: list[str] | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Return v2 dynamic feature matrix and full indicator DataFrame.

    Default feature matrix columns:
        BCI + FPV + PMR + GCGR + PPI
    """
    if feature_columns is None:
        feature_columns = DYNAMIC_PATCH_V2_COLUMNS

    df = extract_dynamic_patch_v2_indicators(
        field_series=field_series,
        threshold=threshold,
        positive=positive,
        connectivity=connectivity,
        lag=lag,
    )

    missing = [c for c in feature_columns if c not in df.columns]

    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    matrix = df[feature_columns].to_numpy(dtype=np.float32)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

    return matrix, df


def _build_demo_series(t_steps: int = 60, grid: int = 64) -> np.ndarray:
    """
    Build a simple spreading demo with patch expansion and merging.
    """
    yy, xx = np.mgrid[0:grid, 0:grid]
    series = np.zeros((t_steps, grid, grid), dtype=np.float32)

    centers = [
        (18, 18),
        (45, 20),
        (32, 45),
    ]

    for t in range(t_steps):
        field = np.zeros((grid, grid), dtype=np.float32)

        radius = 2 + 0.35 * t

        for cy, cx in centers:
            dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
            field[dist <= radius] = 1.0

        if t > 30:
            bridge = (np.abs(yy - 32) < 3) & (xx > 18) & (xx < 46)
            field[bridge] = 1.0

        series[t] = field

    return series


def _demo() -> None:
    demo = _build_demo_series()

    matrix, df = extract_dynamic_patch_v2_feature_matrix(
        field_series=demo,
        threshold=0.5,
        positive=True,
        connectivity=8,
        lag=1,
    )

    print("Dynamic patch v2 feature matrix:", matrix.shape)
    print()
    print(df.head())
    print()
    print(df[["time", "bci", "fpv", "pmr", "gcgr", "ppi"]].tail())


if __name__ == "__main__":
    _demo()