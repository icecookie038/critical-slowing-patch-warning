# prepatch_indicators.py
# -*- coding: utf-8 -*-
"""
v1.3 pre-patch spatial organization indicators.

Purpose
-------
These indicators are designed to capture spatial organization before visible
patches become dominant.

The first implementation focuses on low-cost, reproducible indicators:

1. Local synchronization
   - local_neighbor_corr_mean
   - local_neighbor_corr_max
   - sync_edge_ratio

2. Spatial connectivity
   - moran_i
   - geary_c
   - high_state_component_ratio

3. Boundary rigidity
   - gradient_entropy
   - boundary_sharpness
   - gradient_top10_mean

4. Dominant mode locking
   - svd_mode1_energy_ratio
   - svd_spectral_gap
   - svd_mode1_ac1
   - dominant_mode_stability
   - dominant_mode_localization
"""

from __future__ import annotations

import numpy as np


EPS = 1e-8


PREPATCH_FEATURE_NAMES = [
    # Local synchronization
    "local_neighbor_corr_mean",
    "local_neighbor_corr_max",
    "sync_edge_ratio",

    # Spatial connectivity
    "moran_i",
    "geary_c",
    "high_state_component_ratio",

    # Boundary rigidity
    "gradient_entropy",
    "boundary_sharpness",
    "gradient_top10_mean",

    # Dominant mode locking
    "svd_mode1_energy_ratio",
    "svd_spectral_gap",
    "svd_mode1_ac1",
    "dominant_mode_stability",
    "dominant_mode_localization",
]


def _as_time_grid(x_seq: np.ndarray) -> np.ndarray:
    """
    Convert one sample into shape (T, H, W).

    Supported input:
    - (T, C, H, W)
    - (T, H, W)
    """
    x_seq = np.asarray(x_seq, dtype=np.float32)

    if x_seq.ndim == 4:
        # (T, C, H, W), usually C = 1
        if x_seq.shape[1] == 1:
            x_seq = x_seq[:, 0, :, :]
        else:
            # If multi-channel, average channels.
            x_seq = x_seq.mean(axis=1)

    if x_seq.ndim != 3:
        raise ValueError(f"Expected shape (T,H,W) or (T,C,H,W), got {x_seq.shape}")

    x_seq = np.nan_to_num(x_seq, nan=0.0, posinf=0.0, neginf=0.0)
    return x_seq.astype(np.float32)


def _safe_ac1(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)

    if len(x) < 3:
        return 0.0

    a = x[:-1]
    b = x[1:]

    if np.std(a) < EPS or np.std(b) < EPS:
        return 0.0

    c = np.corrcoef(a, b)[0, 1]

    if not np.isfinite(c):
        return 0.0

    return float(c)


def _neighbor_temporal_corr(x_seq: np.ndarray):
    """
    Compute temporal Pearson correlation for 4-neighbor grid edges.

    Returns
    -------
    corr_all:
        1D array of local neighbor correlations.
    """
    x = _as_time_grid(x_seq)
    t, h, w = x.shape

    if t < 3:
        return np.zeros(1, dtype=np.float32)

    # Horizontal neighbor pairs: (i,j) and (i,j+1)
    a_h = x[:, :, :-1]
    b_h = x[:, :, 1:]

    # Vertical neighbor pairs: (i,j) and (i+1,j)
    a_v = x[:, :-1, :]
    b_v = x[:, 1:, :]

    def corr_pair(a, b):
        # a, b: (T, ...)
        a = a.astype(np.float64)
        b = b.astype(np.float64)

        a_mean = a.mean(axis=0, keepdims=True)
        b_mean = b.mean(axis=0, keepdims=True)

        ac = a - a_mean
        bc = b - b_mean

        num = np.sum(ac * bc, axis=0)
        den = np.sqrt(np.sum(ac * ac, axis=0) * np.sum(bc * bc, axis=0)) + EPS

        corr = num / den
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        return corr.reshape(-1)

    corr_h = corr_pair(a_h, b_h)
    corr_v = corr_pair(a_v, b_v)

    corr_all = np.concatenate([corr_h, corr_v]).astype(np.float32)

    if corr_all.size == 0:
        corr_all = np.zeros(1, dtype=np.float32)

    return corr_all


def local_synchronization_features(
    x_seq: np.ndarray,
    sync_threshold: float = 0.6,
) -> dict:
    """
    Local synchronization indicators based on temporal correlation between
    neighboring cells.
    """
    corr = _neighbor_temporal_corr(x_seq)

    return {
        "local_neighbor_corr_mean": float(np.mean(corr)),
        "local_neighbor_corr_max": float(np.max(corr)),
        "sync_edge_ratio": float(np.mean(corr >= sync_threshold)),
    }


def spatial_connectivity_features(x_frame: np.ndarray) -> dict:
    """
    Spatial autocorrelation and high-state connected component indicators.

    x_frame:
        One 2D state map, usually the last frame in the input window.
    """
    x = np.asarray(x_frame, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    h, w = x.shape
    n = h * w

    x_mean = float(np.mean(x))
    xc = x - x_mean
    den = float(np.sum(xc * xc)) + EPS

    # 4-neighbor undirected edges: right + down.
    x_right_a = xc[:, :-1]
    x_right_b = xc[:, 1:]

    x_down_a = xc[:-1, :]
    x_down_b = xc[1:, :]

    edge_prod_sum = float(np.sum(x_right_a * x_right_b) + np.sum(x_down_a * x_down_b))
    edge_count = int(x_right_a.size + x_down_a.size)

    moran_i = (n / max(edge_count, 1)) * edge_prod_sum / den

    diff_sq_sum = float(
        np.sum((x[:, :-1] - x[:, 1:]) ** 2)
        + np.sum((x[:-1, :] - x[1:, :]) ** 2)
    )

    geary_c = ((n - 1) / (2.0 * max(edge_count, 1))) * diff_sq_sum / den

    # High-state component ratio.
    # This is a lightweight proxy for emerging spatial connectivity.
    threshold = x_mean + float(np.std(x))
    mask = x >= threshold
    high_state_component_ratio = _largest_component_ratio(mask)

    return {
        "moran_i": float(moran_i),
        "geary_c": float(geary_c),
        "high_state_component_ratio": float(high_state_component_ratio),
    }


def _largest_component_ratio(mask: np.ndarray) -> float:
    """
    Largest 4-connected component ratio among True cells.

    Returns component_size / total_number_of_cells.
    """
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape

    total_cells = h * w
    if total_cells == 0 or not mask.any():
        return 0.0

    visited = np.zeros_like(mask, dtype=bool)
    largest = 0

    for i in range(h):
        for j in range(w):
            if not mask[i, j] or visited[i, j]:
                continue

            stack = [(i, j)]
            visited[i, j] = True
            size = 0

            while stack:
                r, c = stack.pop()
                size += 1

                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w:
                        if mask[nr, nc] and not visited[nr, nc]:
                            visited[nr, nc] = True
                            stack.append((nr, nc))

            if size > largest:
                largest = size

    return float(largest / total_cells)


def boundary_rigidity_features(x_frame: np.ndarray, n_bins: int = 32) -> dict:
    """
    Boundary rigidity indicators based on spatial gradient concentration.
    """
    x = np.asarray(x_frame, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    gx = np.abs(x[:, 1:] - x[:, :-1]).reshape(-1)
    gy = np.abs(x[1:, :] - x[:-1, :]).reshape(-1)

    g = np.concatenate([gx, gy])

    if g.size == 0:
        return {
            "gradient_entropy": 0.0,
            "boundary_sharpness": 0.0,
            "gradient_top10_mean": 0.0,
        }

    g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)

    g_min = float(np.min(g))
    g_max = float(np.max(g))

    if g_max - g_min < EPS:
        entropy = 0.0
    else:
        hist, _ = np.histogram(g, bins=n_bins, range=(g_min, g_max), density=False)
        p = hist.astype(np.float64)
        p = p / (p.sum() + EPS)
        p = p[p > 0]
        entropy = -float(np.sum(p * np.log(p + EPS)))

    boundary_sharpness = float(np.mean(g))

    k = max(1, int(0.10 * len(g)))
    top10 = np.partition(g, -k)[-k:]
    gradient_top10_mean = float(np.mean(top10))

    return {
        "gradient_entropy": entropy,
        "boundary_sharpness": boundary_sharpness,
        "gradient_top10_mean": gradient_top10_mean,
    }


def dominant_mode_features(x_seq: np.ndarray) -> dict:
    """
    Dominant mode locking indicators using SVD on the recent spatial snapshots.

    x_seq:
        Shape (T,H,W) or (T,C,H,W)
    """
    x = _as_time_grid(x_seq)
    t, h, w = x.shape

    x_flat = x.reshape(t, -1).astype(np.float64)

    # Remove spatial mean at each time step.
    x_flat = x_flat - x_flat.mean(axis=1, keepdims=True)

    # If almost all zero, avoid unstable SVD.
    if np.std(x_flat) < EPS:
        return {
            "svd_mode1_energy_ratio": 0.0,
            "svd_spectral_gap": 0.0,
            "svd_mode1_ac1": 0.0,
            "dominant_mode_stability": 0.0,
            "dominant_mode_localization": 0.0,
        }

    try:
        u, s, vt = np.linalg.svd(x_flat, full_matrices=False)
    except np.linalg.LinAlgError:
        return {
            "svd_mode1_energy_ratio": 0.0,
            "svd_spectral_gap": 0.0,
            "svd_mode1_ac1": 0.0,
            "dominant_mode_stability": 0.0,
            "dominant_mode_localization": 0.0,
        }

    s2 = s * s
    total_energy = float(np.sum(s2)) + EPS

    mode1_energy_ratio = float(s2[0] / total_energy)

    if len(s) >= 2:
        spectral_gap = float(s[0] / (s[1] + EPS))
    else:
        spectral_gap = float(s[0])

    mode1_time_coeff = u[:, 0] * s[0]
    mode1_ac1 = _safe_ac1(mode1_time_coeff)

    mode1_vec = vt[0, :]
    abs_mode = np.abs(mode1_vec)

    # Localization: high value means energy concentrated on fewer cells.
    l2 = float(np.sum(abs_mode ** 2)) + EPS
    l1 = float(np.sum(abs_mode)) + EPS
    participation_ratio = (l1 ** 2) / l2
    dominant_mode_localization = float(1.0 / (participation_ratio + EPS))

    # Stability: compare dominant spatial mode from first half and second half.
    dominant_mode_stability = _mode_stability_between_halves(x_flat)

    return {
        "svd_mode1_energy_ratio": mode1_energy_ratio,
        "svd_spectral_gap": spectral_gap,
        "svd_mode1_ac1": mode1_ac1,
        "dominant_mode_stability": dominant_mode_stability,
        "dominant_mode_localization": dominant_mode_localization,
    }


def _mode_stability_between_halves(x_flat: np.ndarray) -> float:
    t = x_flat.shape[0]

    if t < 4:
        return 0.0

    mid = t // 2
    a = x_flat[:mid]
    b = x_flat[mid:]

    if np.std(a) < EPS or np.std(b) < EPS:
        return 0.0

    try:
        _, _, vta = np.linalg.svd(a, full_matrices=False)
        _, _, vtb = np.linalg.svd(b, full_matrices=False)
    except np.linalg.LinAlgError:
        return 0.0

    va = vta[0]
    vb = vtb[0]

    denom = (np.linalg.norm(va) * np.linalg.norm(vb)) + EPS
    cos = float(np.dot(va, vb) / denom)

    # SVD signs are arbitrary, so use absolute cosine similarity.
    return float(abs(cos))


def compute_prepatch_indicators_for_sequence(
    x_seq: np.ndarray,
    sync_threshold: float = 0.6,
) -> np.ndarray:
    """
    Compute all v1.3 pre-patch indicators for one sample.

    Parameters
    ----------
    x_seq:
        One sample of X_img.
        Supported shape:
        - (T,C,H,W)
        - (T,H,W)

    Returns
    -------
    features:
        1D np.ndarray with order PREPATCH_FEATURE_NAMES.
    """
    x = _as_time_grid(x_seq)
    last_frame = x[-1]

    out = {}

    out.update(local_synchronization_features(x, sync_threshold=sync_threshold))
    out.update(spatial_connectivity_features(last_frame))
    out.update(boundary_rigidity_features(last_frame))
    out.update(dominant_mode_features(x))

    values = [out[name] for name in PREPATCH_FEATURE_NAMES]
    values = np.asarray(values, dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    return values


def compute_prepatch_indicators_batch(
    X_img: np.ndarray,
    sync_threshold: float = 0.6,
    verbose: bool = True,
) -> np.ndarray:
    """
    Compute pre-patch indicators for all samples.

    X_img shape:
    - (N,T,C,H,W)
    - (N,T,H,W)

    Returns:
    - X_prepatch: (N, n_prepatch_features)
    """
    X_img = np.asarray(X_img)
    n = X_img.shape[0]

    feats = np.zeros((n, len(PREPATCH_FEATURE_NAMES)), dtype=np.float32)

    iterator = range(n)

    if verbose:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="Computing pre-patch indicators")
        except Exception:
            pass

    for i in iterator:
        feats[i] = compute_prepatch_indicators_for_sequence(
            X_img[i],
            sync_threshold=sync_threshold,
        )

    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats


def feature_names() -> list[str]:
    return list(PREPATCH_FEATURE_NAMES)


if __name__ == "__main__":
    # Minimal smoke test.
    rng = np.random.default_rng(42)
    x = rng.random((10, 1, 64, 64), dtype=np.float32)

    feat = compute_prepatch_indicators_for_sequence(x)

    print("Feature names:")
    print(PREPATCH_FEATURE_NAMES)
    print("Feature shape:", feat.shape)
    print("Feature values:")
    print(feat)