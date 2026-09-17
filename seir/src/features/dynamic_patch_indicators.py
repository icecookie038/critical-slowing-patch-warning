"""
Dynamic patch indicators for ecological early-warning analysis.

v0.8 dynamic patch indicators:

1. PDSI : Patch-size Distribution Shift Index
2. DPCI : Dominant Patch Change / Growth Index
3. DPCR : Dominant Patch Collapse Rate
4. FAI  : Fragmentation Acceleration Index
5. PSII : Patch-Structure Instability Index

Important design principle:
- v0.1-v0.7 classic patch indicators are not modified.
- This file only adds new dynamic indicators from v0.8.
- PDSI bins are fixed across the whole time series.
- PSII baseline statistics must be estimated only from baseline / training period.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Literal, Optional

import numpy as np
import pandas as pd


EPS = 1e-8


Connectivity = Literal[4, 8]
SystemMode = Literal["expansion", "collapse"]


@dataclass
class PatchSnapshot:
    """Patch information at one time step."""

    time: int
    patch_areas: np.ndarray
    patch_count: int
    total_patch_area: float
    largest_patch_area: float
    mean_patch_area: float
    largest_patch_ratio: float
    fragmentation_density: float


def to_binary_mask_series(
    field_series: np.ndarray,
    threshold: float = 0.5,
    positive: bool = True,
) -> np.ndarray:
    """
    Convert a spatial field sequence into binary masks.

    Parameters
    ----------
    field_series:
        Array with shape (T, H, W). It can be binary masks or continuous fields.
    threshold:
        Threshold for binarization.
    positive:
        If True, pixels >= threshold are treated as patch pixels.
        If False, pixels <= threshold are treated as patch pixels.

    Returns
    -------
    masks:
        Boolean array with shape (T, H, W).
    """
    arr = np.asarray(field_series)

    if arr.ndim != 3:
        raise ValueError(
            f"field_series must have shape (T, H, W), but got shape {arr.shape}"
        )

    if arr.dtype == bool:
        return arr.copy()

    if positive:
        return arr >= threshold
    return arr <= threshold


def _neighbor_offsets(connectivity: Connectivity = 8) -> list[tuple[int, int]]:
    if connectivity == 4:
        return [(-1, 0), (1, 0), (0, -1), (0, 1)]

    if connectivity == 8:
        return [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]

    raise ValueError("connectivity must be 4 or 8")


def connected_patch_areas(
    mask: np.ndarray,
    connectivity: Connectivity = 8,
) -> np.ndarray:
    """
    Compute connected patch areas from one binary mask.

    Parameters
    ----------
    mask:
        Boolean or binary array with shape (H, W).
    connectivity:
        4 or 8. Ecological patch analysis commonly uses 8-connectivity.

    Returns
    -------
    areas:
        1D array of patch areas in pixels.
    """
    binary = np.asarray(mask).astype(bool)

    if binary.ndim != 2:
        raise ValueError(f"mask must have shape (H, W), but got shape {binary.shape}")

    height, width = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    offsets = _neighbor_offsets(connectivity)
    areas: list[int] = []

    for row in range(height):
        for col in range(width):
            if not binary[row, col] or visited[row, col]:
                continue

            area = 0
            queue: deque[tuple[int, int]] = deque()
            queue.append((row, col))
            visited[row, col] = True

            while queue:
                r, c = queue.popleft()
                area += 1

                for dr, dc in offsets:
                    nr, nc = r + dr, c + dc
                    if nr < 0 or nr >= height or nc < 0 or nc >= width:
                        continue

                    if binary[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        queue.append((nr, nc))

            areas.append(area)

    if not areas:
        return np.array([], dtype=float)

    return np.asarray(areas, dtype=float)


def make_patch_size_bins(
    image_shape: tuple[int, int],
    max_area: Optional[int] = None,
) -> np.ndarray:
    """
    Create fixed patch-size bins.

    The bins must be fixed across the whole time series.
    Do not recompute bins at each time step.

    For a 64x64 grid, bins will look like:
    [0, 1, 2, 4, 8, 16, ..., 4096, 4097]

    Parameters
    ----------
    image_shape:
        Spatial shape, usually (H, W).
    max_area:
        Optional maximum possible patch area. If None, H * W is used.

    Returns
    -------
    bins:
        1D array of histogram bin edges.
    """
    if max_area is None:
        max_area = int(image_shape[0] * image_shape[1])

    if max_area < 1:
        raise ValueError("max_area must be positive")

    edges = [0, 1]
    value = 2

    while value < max_area:
        edges.append(value)
        value *= 2

    edges.append(max_area)
    edges.append(max_area + 1)

    bins = np.asarray(sorted(set(edges)), dtype=float)

    if len(bins) < 3:
        bins = np.asarray([0, 1, max_area + 1], dtype=float)

    return bins


def patch_size_distribution(
    patch_areas: np.ndarray,
    bins: np.ndarray,
) -> np.ndarray:
    """
    Convert patch areas into a normalized patch-size distribution.

    Parameters
    ----------
    patch_areas:
        1D patch area array.
    bins:
        Fixed histogram bins.

    Returns
    -------
    distribution:
        Probability distribution over patch-size bins.
    """
    areas = np.asarray(patch_areas, dtype=float)

    hist, _ = np.histogram(areas, bins=bins)

    hist = hist.astype(float)
    total = hist.sum()

    if total <= 0:
        return np.ones(len(bins) - 1, dtype=float) / max(len(bins) - 1, 1)

    return hist / total


def jensen_shannon_distance(
    p: np.ndarray,
    q: np.ndarray,
) -> float:
    """
    Jensen-Shannon distance between two probability distributions.

    The output is bounded between 0 and 1 when log base 2 is used.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    p = p / (p.sum() + EPS)
    q = q / (q.sum() + EPS)

    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        valid = a > 0
        return float(np.sum(a[valid] * np.log2((a[valid] + EPS) / (b[valid] + EPS))))

    js_divergence = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    js_divergence = max(js_divergence, 0.0)

    return float(np.sqrt(js_divergence))


def compute_patch_snapshots(
    mask_series: np.ndarray,
    connectivity: Connectivity = 8,
) -> list[PatchSnapshot]:
    """
    Compute basic patch information for every time step.

    This does not replace v0.1-v0.7 classic patch features.
    It is only used internally by v0.8 dynamic indicators.
    """
    masks = np.asarray(mask_series).astype(bool)

    if masks.ndim != 3:
        raise ValueError(f"mask_series must have shape (T, H, W), got {masks.shape}")

    snapshots: list[PatchSnapshot] = []

    for t in range(masks.shape[0]):
        areas = connected_patch_areas(masks[t], connectivity=connectivity)

        patch_count = int(len(areas))
        total_area = float(np.sum(areas)) if patch_count > 0 else 0.0
        largest_area = float(np.max(areas)) if patch_count > 0 else 0.0
        mean_area = float(np.mean(areas)) if patch_count > 0 else 0.0
        largest_ratio = largest_area / (total_area + EPS)
        fragmentation_density = patch_count / (total_area + EPS)

        snapshots.append(
            PatchSnapshot(
                time=t,
                patch_areas=areas,
                patch_count=patch_count,
                total_patch_area=total_area,
                largest_patch_area=largest_area,
                mean_patch_area=mean_area,
                largest_patch_ratio=largest_ratio,
                fragmentation_density=float(fragmentation_density),
            )
        )

    return snapshots


def compute_pdsi(
    patch_area_series: list[np.ndarray],
    bins: np.ndarray,
    lag: int = 1,
) -> np.ndarray:
    """
    PDSI: Patch-size Distribution Shift Index.

    Definition:
        PDSI(t) = JSD(P_t, P_{t-lag})

    where P_t is the patch-size distribution at time t.

    Larger PDSI means stronger temporal shift in patch-size structure.
    """
    if lag < 1:
        raise ValueError("lag must be >= 1")

    total_steps = len(patch_area_series)
    distributions = [
        patch_size_distribution(areas, bins=bins) for areas in patch_area_series
    ]

    pdsi = np.zeros(total_steps, dtype=float)

    for t in range(total_steps):
        if t < lag:
            pdsi[t] = 0.0
        else:
            pdsi[t] = jensen_shannon_distance(distributions[t], distributions[t - lag])

    return pdsi


def compute_dpci(
    largest_patch_area: np.ndarray,
    lag: int = 1,
) -> np.ndarray:
    """
    DPCI: Dominant Patch Change / Growth Index.

    For SEIR infection spreading, the dominant infected patch often expands.
    Therefore DPCI focuses on positive growth of the largest patch.

    Definition:
        DPCI(t) = max(0, L_t - L_{t-lag}) / max(L_{t-lag}, 1)

    where L_t is the largest patch area at time t.
    """
    if lag < 1:
        raise ValueError("lag must be >= 1")

    largest = np.asarray(largest_patch_area, dtype=float)
    dpci = np.zeros_like(largest, dtype=float)

    for t in range(len(largest)):
        if t < lag:
            dpci[t] = 0.0
            continue

        previous = largest[t - lag]
        current = largest[t]
        dpci[t] = max(0.0, current - previous) / max(previous, 1.0)

    return dpci


def compute_dpcr(
    largest_patch_area: np.ndarray,
    lag: int = 1,
) -> np.ndarray:
    """
    DPCR: Dominant Patch Collapse Rate.

    DPCR is mainly used for vegetation degradation or healthy-patch collapse.
    It should not be directly used as the main SEIR expansion indicator.

    Definition:
        DPCR(t) = max(0, L_{t-lag} - L_t) / max(L_{t-lag}, 1)

    where L_t is the largest healthy or target patch area at time t.
    """
    if lag < 1:
        raise ValueError("lag must be >= 1")

    largest = np.asarray(largest_patch_area, dtype=float)
    dpcr = np.zeros_like(largest, dtype=float)

    for t in range(len(largest)):
        if t < lag:
            dpcr[t] = 0.0
            continue

        previous = largest[t - lag]
        current = largest[t]
        dpcr[t] = max(0.0, previous - current) / max(previous, 1.0)

    return dpcr


def compute_fai(
    fragmentation_density: np.ndarray,
    lag: int = 1,
) -> np.ndarray:
    """
    FAI: Fragmentation Acceleration Index.

    Fragmentation density:
        F(t) = patch_count(t) / total_patch_area(t)

    Acceleration:
        FAI(t) = max(0, F(t) - 2F(t-lag) + F(t-2lag))

    Larger FAI indicates accelerated fragmentation.
    """
    if lag < 1:
        raise ValueError("lag must be >= 1")

    frag = np.asarray(fragmentation_density, dtype=float)
    fai = np.zeros_like(frag, dtype=float)

    for t in range(len(frag)):
        if t < 2 * lag:
            fai[t] = 0.0
            continue

        acceleration = frag[t] - 2.0 * frag[t - lag] + frag[t - 2 * lag]
        fai[t] = max(0.0, acceleration)

    return fai


def compute_psii(
    metrics_df: pd.DataFrame,
    mode: SystemMode = "expansion",
    baseline_end: Optional[int] = None,
    baseline_indices: Optional[Iterable[int]] = None,
    min_std: float = 0.05,
    z_clip: float = 5.0,
) -> np.ndarray:
    """
    PSII: Patch-Structure Instability Index.

    For SEIR infection expansion:
        PSII = mean positive clipped z-score of [PDSI, DPCI, FAI]

    For vegetation degradation / collapse:
        PSII = mean positive clipped z-score of [PDSI, DPCR, FAI]

    Why min_std and z_clip are needed:
    - In early baseline periods, dynamic indicators may be exactly zero.
    - If baseline std is nearly zero, ordinary z-score can explode.
    - min_std prevents numerical explosion.
    - z_clip prevents one indicator from dominating the composite index.

    Important:
    The mean and std are still estimated only from baseline / training period.
    Do not use the whole time series to fit PSII normalization.
    """
    if mode == "expansion":
        columns = ["pdsi", "dpci", "fai"]
    elif mode == "collapse":
        columns = ["pdsi", "dpcr", "fai"]
    else:
        raise ValueError("mode must be 'expansion' or 'collapse'")

    for col in columns:
        if col not in metrics_df.columns:
            raise ValueError(f"metrics_df does not contain required column: {col}")

    n = len(metrics_df)

    if baseline_indices is not None:
        baseline_idx = np.asarray(list(baseline_indices), dtype=int)
    else:
        if baseline_end is None:
            baseline_end = max(10, int(0.25 * n))

        baseline_end = int(np.clip(baseline_end, 1, n))
        baseline_idx = np.arange(baseline_end)

    baseline = metrics_df.iloc[baseline_idx][columns].astype(float)
    values = metrics_df[columns].astype(float)

    mean = baseline.mean(axis=0)
    std = baseline.std(axis=0, ddof=0)

    # Prevent z-score explosion when baseline is too stable.
    std = std.clip(lower=min_std)

    z = (values - mean) / std

    # Only positive instability signals are accumulated.
    positive_z = z.clip(lower=0.0, upper=z_clip)

    psii = positive_z.mean(axis=1).to_numpy(dtype=float)

    psii = np.nan_to_num(psii, nan=0.0, posinf=0.0, neginf=0.0)

    return psii
def extract_dynamic_patch_indicators(
    field_series: np.ndarray,
    threshold: float = 0.5,
    positive: bool = True,
    connectivity: Connectivity = 8,
    lag: int = 1,
    mode: SystemMode = "expansion",
    baseline_end: Optional[int] = None,
    bins: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Main interface for v0.8 dynamic patch indicators.

    Parameters
    ----------
    field_series:
        Array with shape (T, H, W). Binary masks or continuous spatial fields.
    threshold:
        Threshold used when field_series is continuous.
    positive:
        True means target patches are pixels >= threshold.
        False means target patches are pixels <= threshold.
    connectivity:
        4 or 8 connected neighborhood.
    lag:
        Temporal lag for dynamic indicators.
    mode:
        "expansion" for SEIR infected patch expansion.
        "collapse" for vegetation healthy patch collapse.
    baseline_end:
        Baseline period end for PSII z-score.
    bins:
        Fixed patch-size bins. If None, bins are generated from image size.

    Returns
    -------
    df:
        DataFrame with classic internal states and dynamic indicators.

    Columns
    -------
    time
    patch_count
    total_patch_area
    largest_patch_area
    mean_patch_area
    largest_patch_ratio
    fragmentation_density
    pdsi
    dpci
    dpcr
    fai
    psii
    """
    masks = to_binary_mask_series(
        field_series=field_series,
        threshold=threshold,
        positive=positive,
    )

    total_steps, height, width = masks.shape

    if bins is None:
        bins = make_patch_size_bins(image_shape=(height, width))

    snapshots = compute_patch_snapshots(
        mask_series=masks,
        connectivity=connectivity,
    )

    patch_area_series = [s.patch_areas for s in snapshots]

    df = pd.DataFrame(
        {
            "time": [s.time for s in snapshots],
            "patch_count": [s.patch_count for s in snapshots],
            "total_patch_area": [s.total_patch_area for s in snapshots],
            "largest_patch_area": [s.largest_patch_area for s in snapshots],
            "mean_patch_area": [s.mean_patch_area for s in snapshots],
            "largest_patch_ratio": [s.largest_patch_ratio for s in snapshots],
            "fragmentation_density": [s.fragmentation_density for s in snapshots],
        }
    )

    df["pdsi"] = compute_pdsi(
        patch_area_series=patch_area_series,
        bins=bins,
        lag=lag,
    )

    df["dpci"] = compute_dpci(
        largest_patch_area=df["largest_patch_area"].to_numpy(dtype=float),
        lag=lag,
    )

    df["dpcr"] = compute_dpcr(
        largest_patch_area=df["largest_patch_area"].to_numpy(dtype=float),
        lag=lag,
    )

    df["fai"] = compute_fai(
        fragmentation_density=df["fragmentation_density"].to_numpy(dtype=float),
        lag=lag,
    )

    df["psii"] = compute_psii(
        metrics_df=df,
        mode=mode,
        baseline_end=baseline_end,
    )

    df.attrs["bins"] = bins
    df.attrs["mode"] = mode
    df.attrs["lag"] = lag
    df.attrs["connectivity"] = connectivity
    df.attrs["threshold"] = threshold
    df.attrs["positive"] = positive

    if len(df) != total_steps:
        raise RuntimeError("Dynamic indicator length does not match input time steps")

    return df


def extract_dynamic_feature_matrix(
    field_series: np.ndarray,
    threshold: float = 0.5,
    positive: bool = True,
    connectivity: Connectivity = 8,
    lag: int = 1,
    mode: SystemMode = "expansion",
    baseline_end: Optional[int] = None,
    bins: Optional[np.ndarray] = None,
    feature_columns: Optional[list[str]] = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Extract dynamic patch feature matrix for model input.

    Default dynamic feature columns:
        pdsi, dpci, dpcr, fai, psii

    For SEIR v0.9, usually use:
        pdsi, dpci, fai, psii

    For vegetation degradation, usually use:
        pdsi, dpcr, fai, psii
    """
    df = extract_dynamic_patch_indicators(
        field_series=field_series,
        threshold=threshold,
        positive=positive,
        connectivity=connectivity,
        lag=lag,
        mode=mode,
        baseline_end=baseline_end,
        bins=bins,
    )

    if feature_columns is None:
        feature_columns = ["pdsi", "dpci", "dpcr", "fai", "psii"]

    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    matrix = df[feature_columns].to_numpy(dtype=np.float32)

    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

    return matrix, df


def _build_demo_series(
    total_steps: int = 40,
    size: int = 64,
) -> np.ndarray:
    """
    Build a small artificial expanding patch sequence for quick testing.
    This is only for checking whether the file runs correctly.
    """
    masks = np.zeros((total_steps, size, size), dtype=bool)

    center = size // 2

    for t in range(total_steps):
        radius = 2 + t // 2
        rr, cc = np.ogrid[:size, :size]
        disk = (rr - center) ** 2 + (cc - center) ** 2 <= radius**2
        masks[t] = disk

        if 15 <= t < 30:
            masks[t, 8:12, 8:12] = True
            masks[t, 48:52, 10:14] = True

        if t >= 25:
            masks[t, 10:13, 45:48] = True
            masks[t, 40:43, 42:45] = True

    return masks


if __name__ == "__main__":
    demo = _build_demo_series()

    indicators = extract_dynamic_patch_indicators(
        field_series=demo,
        threshold=0.5,
        positive=True,
        connectivity=8,
        lag=1,
        mode="expansion",
        baseline_end=10,
    )

    print(indicators.head())
    print()
    print(indicators[["time", "pdsi", "dpci", "dpcr", "fai", "psii"]].tail())