# build_v2_0_vegetation_dataset.py
# -*- coding: utf-8 -*-
"""
Build v2.0 vegetation cellular automaton early-warning dataset.

This script generates a vegetation degradation dataset compatible with the
existing v1.2/v1.3/v1.4 pipeline.

Output keys are intentionally aligned with the SEIR dataset format:

    X_img
    X_patch
    y_remaining
    y_risk
    sim_id
    time_idx
    critical_time
    infected_area      # alias of degraded_area for compatibility
    dominant_patch     # dominant degraded patch ratio
    reff               # alias of stress for compatibility

Additional vegetation-specific keys:

    vegetation_cover
    degraded_area
    stress
    patch_feature_names
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import deque

import numpy as np

from vegetation_degradation_model import (
    sample_params,
    simulate_vegetation_ca,
)


try:
    from scipy import ndimage
except Exception:
    ndimage = None


PATCH_FEATURE_NAMES = [
    "std_vegetation",
    "var_vegetation",
    "largest_degraded_patch_ratio",
    "degraded_patch_count_ratio",
    "mean_degraded_patch_area_ratio",
    "largest_patch_to_degraded_area",
    "degraded_edge_density",
    "spatial_aggregation",
    "neighbor_autocorr",
    "gradient_mean",
    "gradient_top10_mean",
    "gradient_entropy",
]


def neighbor_mean_8(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)

    total = (
        np.roll(x, 1, axis=0)
        + np.roll(x, -1, axis=0)
        + np.roll(x, 1, axis=1)
        + np.roll(x, -1, axis=1)
        + np.roll(np.roll(x, 1, axis=0), 1, axis=1)
        + np.roll(np.roll(x, 1, axis=0), -1, axis=1)
        + np.roll(np.roll(x, -1, axis=0), 1, axis=1)
        + np.roll(np.roll(x, -1, axis=0), -1, axis=1)
    )

    return total / 8.0


def component_stats_fallback(binary: np.ndarray) -> tuple[int, int, float]:
    """
    Fallback connected-component statistics using pure Python BFS.

    Return:
        patch_count
        largest_area
        mean_area
    """
    binary = np.asarray(binary, dtype=bool)
    L0, L1 = binary.shape

    visited = np.zeros_like(binary, dtype=bool)
    patch_areas = []

    for i in range(L0):
        for j in range(L1):
            if not binary[i, j] or visited[i, j]:
                continue

            q = deque()
            q.append((i, j))
            visited[i, j] = True
            area = 0

            while q:
                x, y = q.popleft()
                area += 1

                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx = x + dx
                    ny = y + dy

                    if nx < 0 or nx >= L0 or ny < 0 or ny >= L1:
                        continue

                    if binary[nx, ny] and not visited[nx, ny]:
                        visited[nx, ny] = True
                        q.append((nx, ny))

            patch_areas.append(area)

    if not patch_areas:
        return 0, 0, 0.0

    patch_areas = np.asarray(patch_areas, dtype=np.float32)

    return int(len(patch_areas)), int(np.max(patch_areas)), float(np.mean(patch_areas))


def component_stats(binary: np.ndarray) -> tuple[int, int, float]:
    """
    Connected-component statistics for degraded patches.

    Return:
        patch_count
        largest_area
        mean_area
    """
    binary = np.asarray(binary, dtype=bool)

    if ndimage is None:
        return component_stats_fallback(binary)

    structure = np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=np.int32,
    )

    labeled, n_components = ndimage.label(binary, structure=structure)

    if n_components == 0:
        return 0, 0, 0.0

    areas = np.bincount(labeled.ravel())[1:].astype(np.float32)

    return int(n_components), int(np.max(areas)), float(np.mean(areas))


def gradient_entropy(grad: np.ndarray, n_bins: int = 16) -> float:
    grad = np.asarray(grad, dtype=np.float32)

    if float(np.max(grad)) <= 1e-8:
        return 0.0

    hist, _ = np.histogram(grad.ravel(), bins=n_bins, range=(0.0, float(np.max(grad))))
    p = hist.astype(np.float32)
    p = p / (float(np.sum(p)) + 1e-8)
    p = p[p > 0]

    return float(-np.sum(p * np.log(p + 1e-8)))


def frame_patch_features(
    V: np.ndarray,
    healthy_threshold: float = 0.50,
    degraded_threshold: float = 0.35,
) -> np.ndarray:
    """
    Compute visible spatial patch indicators from one vegetation frame.

    V:
        vegetation state in [0, 1]

    Larger degradation signal is represented by D = 1 - V.
    """
    V = np.asarray(V, dtype=np.float32)
    L0, L1 = V.shape
    n_cells = float(L0 * L1)

    healthy_map = V >= healthy_threshold
    degraded_map = V <= degraded_threshold

    vegetation_cover = float(np.mean(healthy_map))
    degraded_area = float(np.mean(degraded_map))

    mean_v = float(np.mean(V))
    std_v = float(np.std(V))
    var_v = float(np.var(V))

    p10, p25, p50, p75, p90 = np.percentile(V, [10, 25, 50, 75, 90])

    patch_count, largest_area, mean_area = component_stats(degraded_map)

    largest_patch_ratio = float(largest_area) / n_cells
    patch_count_ratio = float(patch_count) / n_cells
    mean_patch_area_ratio = float(mean_area) / n_cells

    if degraded_area > 1e-8:
        largest_patch_to_degraded_area = largest_patch_ratio / degraded_area
    else:
        largest_patch_to_degraded_area = 0.0

    degraded_float = degraded_map.astype(np.float32)
    neigh_degraded = neighbor_mean_8(degraded_float)

    if degraded_area > 1e-8:
        spatial_aggregation = float(np.mean(neigh_degraded[degraded_map]))
    else:
        spatial_aggregation = 0.0

    # Edge density: neighboring degraded/non-degraded disagreement.
    horizontal_edge = degraded_map[:, 1:] != degraded_map[:, :-1]
    vertical_edge = degraded_map[1:, :] != degraded_map[:-1, :]
    degraded_edge_density = float(
        (np.sum(horizontal_edge) + np.sum(vertical_edge))
        / max(1.0, (L0 * (L1 - 1) + (L0 - 1) * L1))
    )

    # Neighbor autocorrelation of degradation intensity.
    D = 1.0 - V
    D_centered = D - float(np.mean(D))
    D_neighbor = neighbor_mean_8(D)
    D_neighbor_centered = D_neighbor - float(np.mean(D_neighbor))

    denom = float(np.std(D_centered) * np.std(D_neighbor_centered)) + 1e-8
    neighbor_autocorr = float(np.mean(D_centered * D_neighbor_centered) / denom)

    gy, gx = np.gradient(V)
    grad = np.sqrt(gx ** 2 + gy ** 2).astype(np.float32)

    gradient_mean = float(np.mean(grad))
    gradient_top10_mean = float(np.mean(np.sort(grad.ravel())[-max(1, grad.size // 10):]))
    grad_entropy = gradient_entropy(grad)

    features = np.asarray(
        [
            std_v,
            var_v,
            largest_patch_ratio,
            patch_count_ratio,
            mean_patch_area_ratio,
            largest_patch_to_degraded_area,
            degraded_edge_density,
            spatial_aggregation,
            neighbor_autocorr,
            gradient_mean,
            gradient_top10_mean,
            grad_entropy,
        ],
        dtype=np.float32,
    )

    return features


def compute_patch_feature_series(V_series: np.ndarray) -> np.ndarray:
    """
    Compute visible patch indicators for all frames in one simulation.
    """
    T = V_series.shape[0]

    X = np.zeros((T, len(PATCH_FEATURE_NAMES)), dtype=np.float32)

    for t in range(T):
        X[t] = frame_patch_features(V_series[t])

    return X


def build_samples_from_simulation(
    sim_output: dict,
    sim_idx: int,
    input_seq_len: int,
    horizon: int,
) -> dict | None:
    """
    Convert one vegetation CA simulation into supervised early-warning samples.
    """
    event_time = int(sim_output["event_time"])

    if event_time <= input_seq_len:
        return None

    V_series = sim_output["V_series"].astype(np.float32)
    stress = sim_output["stress"].astype(np.float32)
    vegetation_cover = sim_output["vegetation_cover"].astype(np.float32)
    degraded_area = sim_output["degraded_area"].astype(np.float32)
    dominant_patch = sim_output["dominant_degraded_patch"].astype(np.float32)

    patch_features = compute_patch_feature_series(V_series)

    X_img_list = []
    X_patch_list = []
    y_remaining_list = []
    y_risk_list = []
    sim_id_list = []
    time_idx_list = []
    critical_time_list = []
    vegetation_cover_list = []
    degraded_area_list = []
    dominant_patch_list = []
    stress_list = []

    for t in range(input_seq_len - 1, event_time):
        start = t - input_seq_len + 1
        end = t + 1

        V_seq = V_series[start:end]

        # Use degradation intensity as image input.
        D_seq = 1.0 - V_seq
        D_seq = D_seq[:, None, :, :].astype(np.float32)

        patch_seq = patch_features[start:end].astype(np.float32)

        remaining = int(event_time - t)
        risk = int(0 < remaining <= horizon)

        X_img_list.append(D_seq)
        X_patch_list.append(patch_seq)
        y_remaining_list.append(float(remaining))
        y_risk_list.append(int(risk))
        sim_id_list.append(int(sim_idx))
        time_idx_list.append(int(t))
        critical_time_list.append(int(event_time))
        vegetation_cover_list.append(float(vegetation_cover[t]))
        degraded_area_list.append(float(degraded_area[t]))
        dominant_patch_list.append(float(dominant_patch[t]))
        stress_list.append(float(stress[t]))

    return {
        "X_img": X_img_list,
        "X_patch": X_patch_list,
        "y_remaining": y_remaining_list,
        "y_risk": y_risk_list,
        "sim_id": sim_id_list,
        "time_idx": time_idx_list,
        "critical_time": critical_time_list,
        "vegetation_cover": vegetation_cover_list,
        "degraded_area": degraded_area_list,
        "dominant_patch": dominant_patch_list,
        "stress": stress_list,
    }


def extend_list_dict(target: dict, source: dict):
    for k, v in source.items():
        target[k].extend(v)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num-sims", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--L", type=int, default=64)
    parser.add_argument("--sim-steps", type=int, default=200)
    parser.add_argument("--input-seq-len", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=30)

    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/processed/v2_0_vegetation_ca",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    all_samples = {
        "X_img": [],
        "X_patch": [],
        "y_remaining": [],
        "y_risk": [],
        "sim_id": [],
        "time_idx": [],
        "critical_time": [],
        "vegetation_cover": [],
        "degraded_area": [],
        "dominant_patch": [],
        "stress": [],
    }

    sim_summaries = []
    valid_sim_count = 0

    print("===== Build v2.0 vegetation CA dataset =====")
    print(f"num_sims: {args.num_sims}")
    print(f"seed: {args.seed}")
    print(f"L: {args.L}")
    print(f"sim_steps: {args.sim_steps}")
    print(f"input_seq_len: {args.input_seq_len}")
    print(f"horizon: {args.horizon}")
    print()

    for sim_idx in range(args.num_sims):
        sim_seed = int(rng.integers(0, 2**31 - 1))
        params = sample_params(rng=rng, L=args.L, sim_steps=args.sim_steps)

        sim_output = simulate_vegetation_ca(seed=sim_seed, params=params)
        event_time = int(sim_output["event_time"])

        sim_summaries.append(
            {
                "sim_idx": int(sim_idx),
                "sim_seed": int(sim_seed),
                "event_time": int(event_time),
                "cover0": float(sim_output["vegetation_cover"][0]),
                "cover_end": float(sim_output["vegetation_cover"][-1]),
                "degraded0": float(sim_output["degraded_area"][0]),
                "degraded_end": float(sim_output["degraded_area"][-1]),
                "dominant_patch_end": float(sim_output["dominant_degraded_patch"][-1]),
            }
        )

        sample_dict = build_samples_from_simulation(
            sim_output=sim_output,
            sim_idx=sim_idx,
            input_seq_len=args.input_seq_len,
            horizon=args.horizon,
        )

        if sample_dict is None:
            continue

        valid_sim_count += 1
        extend_list_dict(all_samples, sample_dict)

        if (sim_idx + 1) % 20 == 0:
            print(
                f"sim {sim_idx + 1:04d}/{args.num_sims} | "
                f"valid_sims={valid_sim_count} | "
                f"event_time={event_time}"
            )

    if len(all_samples["X_img"]) == 0:
        raise RuntimeError("No valid samples generated. Check vegetation CA parameters.")

    X_img = np.asarray(all_samples["X_img"], dtype=np.float32)
    X_patch = np.asarray(all_samples["X_patch"], dtype=np.float32)
    y_remaining = np.asarray(all_samples["y_remaining"], dtype=np.float32)
    y_risk = np.asarray(all_samples["y_risk"], dtype=np.int32)
    sim_id = np.asarray(all_samples["sim_id"], dtype=np.int32)
    time_idx = np.asarray(all_samples["time_idx"], dtype=np.int32)
    critical_time = np.asarray(all_samples["critical_time"], dtype=np.int32)
    vegetation_cover = np.asarray(all_samples["vegetation_cover"], dtype=np.float32)
    degraded_area = np.asarray(all_samples["degraded_area"], dtype=np.float32)
    dominant_patch = np.asarray(all_samples["dominant_patch"], dtype=np.float32)
    stress = np.asarray(all_samples["stress"], dtype=np.float32)

    # Compatibility aliases with the existing SEIR pipeline.
    infected_area = degraded_area.copy()
    reff = stress.copy()

    out_path = out_dir / f"vegetation_ca_h{args.horizon}_seed{args.seed}.npz"

    np.savez_compressed(
        out_path,
        X_img=X_img,
        X_patch=X_patch,
        y_remaining=y_remaining,
        y_risk=y_risk,
        sim_id=sim_id,
        time_idx=time_idx,
        critical_time=critical_time,
        infected_area=infected_area,
        dominant_patch=dominant_patch,
        reff=reff,
        vegetation_cover=vegetation_cover,
        degraded_area=degraded_area,
        stress=stress,
        patch_feature_names=np.asarray(PATCH_FEATURE_NAMES, dtype=object),
        seed=np.asarray(args.seed, dtype=np.int32),
        horizon=np.asarray(args.horizon, dtype=np.int32),
        input_seq_len=np.asarray(args.input_seq_len, dtype=np.int32),
        sim_steps=np.asarray(args.sim_steps, dtype=np.int32),
    )

    summary_path = out_dir / f"vegetation_ca_h{args.horizon}_seed{args.seed}_summary.json"

    event_times = [
        int(x["event_time"])
        for x in sim_summaries
        if int(x["event_time"]) >= 0
    ]

    summary = {
        "num_sims": args.num_sims,
        "valid_sim_count": valid_sim_count,
        "num_samples": int(X_img.shape[0]),
        "risk_ratio": float(np.mean(y_risk)),
        "event_time_mean": float(np.mean(event_times)) if event_times else None,
        "event_time_min": int(np.min(event_times)) if event_times else None,
        "event_time_max": int(np.max(event_times)) if event_times else None,
        "X_img_shape": list(X_img.shape),
        "X_patch_shape": list(X_patch.shape),
        "output_file": str(out_path),
        "patch_feature_names": PATCH_FEATURE_NAMES,
        "sim_summaries": sim_summaries,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print("===== Dataset finished =====")
    print(f"Saved to: {out_path}")
    print(f"Summary: {summary_path}")
    print(f"Valid simulations: {valid_sim_count} / {args.num_sims}")
    print(f"X_img: {X_img.shape}")
    print(f"X_patch: {X_patch.shape}")
    print(f"y_remaining: {y_remaining.shape}")
    print(f"y_risk: {y_risk.shape}")
    print(f"risk ratio: {float(np.mean(y_risk)):.4f}")
    print(f"mean critical_time: {float(np.mean(critical_time)):.2f}")
    print(f"min critical_time: {int(np.min(critical_time))}")
    print(f"max critical_time: {int(np.max(critical_time))}")


if __name__ == "__main__":
    main()