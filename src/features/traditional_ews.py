"""
Traditional Early-Warning Signal features.

This module implements traditional temporal and spatial EWS indicators
for spatial spreading systems.

Indicators:
    total_infected
    temporal_variance
    temporal_ac1
    temporal_skewness
    return_rate
    kendall_trend
    spatial_variance
    moran_i
    spatial_skewness
    patch_size_slope
"""

from __future__ import annotations

from collections import deque
import numpy as np
import pandas as pd


TRADITIONAL_EWS_COLUMNS = [
    "total_infected",
    "temporal_variance",
    "temporal_ac1",
    "temporal_skewness",
    "return_rate",
    "kendall_trend",
    "spatial_variance",
    "moran_i",
    "spatial_skewness",
    "patch_size_slope",
]


def _safe_float(x, default: float = 0.0) -> float:
    try:
        value = float(x)
    except Exception:
        return default

    if not np.isfinite(value):
        return default

    return value


def _skewness(x: np.ndarray, eps: float = 1e-8) -> float:
    x = np.asarray(x, dtype=float)

    if x.size < 3:
        return 0.0

    mean = np.mean(x)
    std = np.std(x)

    if std < eps:
        return 0.0

    z = (x - mean) / std
    return _safe_float(np.mean(z ** 3))


def _ac1(x: np.ndarray, eps: float = 1e-8) -> float:
    x = np.asarray(x, dtype=float)

    if x.size < 3:
        return 0.0

    x0 = x[:-1]
    x1 = x[1:]

    if np.std(x0) < eps or np.std(x1) < eps:
        return 0.0

    return _safe_float(np.corrcoef(x0, x1)[0, 1])


def _kendall_tau_with_time(x: np.ndarray) -> float:
    """
    Kendall trend between time index and values.
    Implemented without scipy for portability.
    """
    x = np.asarray(x, dtype=float)
    n = x.size

    if n < 3:
        return 0.0

    concordant = 0
    discordant = 0

    for i in range(n - 1):
        for j in range(i + 1, n):
            diff = x[j] - x[i]

            if diff > 0:
                concordant += 1
            elif diff < 0:
                discordant += 1

    denom = n * (n - 1) / 2

    if denom <= 0:
        return 0.0

    return _safe_float((concordant - discordant) / denom)


def _to_mask(field: np.ndarray, threshold: float = 0.5, positive: bool = True) -> np.ndarray:
    arr = np.asarray(field)

    if arr.ndim != 2:
        raise ValueError(f"Each spatial field must be 2D, got shape={arr.shape}")

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


def _component_sizes(mask: np.ndarray, connectivity: int = 8) -> list[int]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape

    visited = np.zeros_like(mask, dtype=bool)
    sizes = []

    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue

            q = deque()
            q.append((y, x))
            visited[y, x] = True
            size = 0

            while q:
                cy, cx = q.popleft()
                size += 1

                for ny, nx in _neighbors(cy, cx, h, w, connectivity=connectivity):
                    if mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        q.append((ny, nx))

            sizes.append(size)

    return sizes


def _patch_size_slope(mask: np.ndarray, connectivity: int = 8, eps: float = 1e-8) -> float:
    """
    Estimate patch-size distribution slope using log(rank)-log(size).

    If fewer than two patches exist, return 0.
    """
    sizes = _component_sizes(mask, connectivity=connectivity)

    if len(sizes) < 2:
        return 0.0

    sizes = np.array(sorted(sizes, reverse=True), dtype=float)
    ranks = np.arange(1, len(sizes) + 1, dtype=float)

    valid = sizes > 0

    if valid.sum() < 2:
        return 0.0

    x = np.log(ranks[valid] + eps)
    y = np.log(sizes[valid] + eps)

    try:
        slope = np.polyfit(x, y, 1)[0]
    except Exception:
        return 0.0

    return _safe_float(slope)


def _moran_i(field: np.ndarray, eps: float = 1e-8) -> float:
    """
    Moran's I using 4-neighborhood adjacency.

    This is a lightweight implementation for grid fields.
    """
    x = np.asarray(field, dtype=float)
    h, w = x.shape
    mean = np.mean(x)
    dev = x - mean

    denom = np.sum(dev ** 2)

    if denom < eps:
        return 0.0

    numerator = 0.0
    weight_sum = 0.0

    # horizontal neighbors
    numerator += np.sum(dev[:, :-1] * dev[:, 1:])
    weight_sum += h * (w - 1)

    # vertical neighbors
    numerator += np.sum(dev[:-1, :] * dev[1:, :])
    weight_sum += (h - 1) * w

    n = h * w

    if weight_sum <= 0:
        return 0.0

    moran = (n / weight_sum) * (numerator / denom)

    return _safe_float(moran)


def extract_traditional_ews_indicators(
    field_series: np.ndarray,
    threshold: float = 0.5,
    positive: bool = True,
    rolling_window: int = 10,
    connectivity: int = 8,
) -> pd.DataFrame:
    """
    Extract traditional EWS indicators from a spatial field series.

    Parameters
    ----------
    field_series:
        Array with shape (T, H, W).

    threshold:
        Threshold used to define infected / target cells.

    positive:
        If True, values >= threshold are target cells.
        If False, values <= threshold are target cells.

    rolling_window:
        Number of time steps used for temporal EWS indicators.

    Returns
    -------
    DataFrame
        One row per time step.
    """
    arr = np.asarray(field_series, dtype=float)

    if arr.ndim != 3:
        raise ValueError(f"field_series must have shape (T,H,W), got {arr.shape}")

    if rolling_window < 2:
        raise ValueError("rolling_window must be >= 2")

    t_steps = arr.shape[0]

    masks = np.stack(
        [
            _to_mask(arr[t], threshold=threshold, positive=positive)
            for t in range(t_steps)
        ],
        axis=0,
    )

    # Use infected fraction as the main scalar state variable.
    total_infected = masks.mean(axis=(1, 2)).astype(float)

    rows = []

    for t in range(t_steps):
        start = max(0, t - rolling_window + 1)
        window = total_infected[start : t + 1]

        current_field = arr[t]
        current_mask = masks[t]

        temporal_variance = np.var(window) if window.size >= 2 else 0.0
        temporal_ac1 = _ac1(window)
        temporal_skewness = _skewness(window)
        return_rate = 1.0 - temporal_ac1
        kendall_trend = _kendall_tau_with_time(window)

        spatial_variance = np.var(current_field)
        moran_i = _moran_i(current_field)
        spatial_skewness = _skewness(current_field.reshape(-1))
        patch_size_slope = _patch_size_slope(
            current_mask,
            connectivity=connectivity,
        )

        row = {
            "time": t,
            "total_infected": _safe_float(total_infected[t]),
            "temporal_variance": _safe_float(temporal_variance),
            "temporal_ac1": _safe_float(temporal_ac1),
            "temporal_skewness": _safe_float(temporal_skewness),
            "return_rate": _safe_float(return_rate),
            "kendall_trend": _safe_float(kendall_trend),
            "spatial_variance": _safe_float(spatial_variance),
            "moran_i": _safe_float(moran_i),
            "spatial_skewness": _safe_float(spatial_skewness),
            "patch_size_slope": _safe_float(patch_size_slope),
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    for col in TRADITIONAL_EWS_COLUMNS:
        df[col] = df[col].replace([np.inf, -np.inf], 0.0).fillna(0.0)

    return df


def extract_traditional_ews_feature_matrix(
    field_series: np.ndarray,
    threshold: float = 0.5,
    positive: bool = True,
    rolling_window: int = 10,
    connectivity: int = 8,
    feature_columns: list[str] | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Return traditional EWS feature matrix and full indicator DataFrame.

    Output:
        matrix shape = (T, F)
    """
    if feature_columns is None:
        feature_columns = TRADITIONAL_EWS_COLUMNS

    df = extract_traditional_ews_indicators(
        field_series=field_series,
        threshold=threshold,
        positive=positive,
        rolling_window=rolling_window,
        connectivity=connectivity,
    )

    missing = [c for c in feature_columns if c not in df.columns]

    if missing:
        raise ValueError(f"Missing traditional EWS columns: {missing}")

    matrix = df[feature_columns].to_numpy(dtype=np.float32)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

    return matrix, df


def _build_demo_series(t_steps: int = 60, grid: int = 64) -> np.ndarray:
    yy, xx = np.mgrid[0:grid, 0:grid]
    series = np.zeros((t_steps, grid, grid), dtype=np.float32)

    centers = [
        (20, 20),
        (44, 22),
        (32, 44),
    ]

    for t in range(t_steps):
        field = np.zeros((grid, grid), dtype=np.float32)

        radius = 2 + 0.25 * t

        for cy, cx in centers:
            dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
            field[dist <= radius] = 1.0

        if t > 35:
            bridge = (np.abs(yy - 32) <= 2) & (xx > 18) & (xx < 46)
            field[bridge] = 1.0

        series[t] = field

    return series


def _demo() -> None:
    demo = _build_demo_series()

    matrix, df = extract_traditional_ews_feature_matrix(
        field_series=demo,
        threshold=0.5,
        positive=True,
        rolling_window=10,
        connectivity=8,
    )

    print("Traditional EWS feature matrix:", matrix.shape)
    print()
    print(df.head())
    print()
    print(df[["time"] + TRADITIONAL_EWS_COLUMNS].tail())


if __name__ == "__main__":
    _demo()