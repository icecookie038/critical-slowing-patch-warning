# vegetation_degradation_model.py
# -*- coding: utf-8 -*-
"""
Spatial cellular automaton vegetation degradation model.

Purpose
-------
This model is designed for v2.0 cross-system validation.

It simulates a spatial vegetation degradation process in which:
1. Each grid cell has a vegetation state V(i, j, t) in [0, 1].
2. External stress gradually increases over time.
3. Local vegetation provides positive feedback.
4. Degraded neighborhoods increase local degradation pressure.
5. A collapse-like event is defined by vegetation cover loss and degraded-area expansion.

This model is not intended to replace the SEIR model.
It serves as a second ecological spatial transition system for validating:
    - visible patch indicators
    - prepatch spatial organization indicators
    - PWSI
    - strict first-alarm analysis
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class VegetationCAParams:
    L: int = 64
    sim_steps: int = 200

    init_mean: float = 0.82
    init_noise: float = 0.06
    heterogeneity_strength: float = 0.10

    stress_start: float = 0.010
    stress_end: float = 0.075
    stress_curve: float = 1.55

    growth_rate: float = 0.030
    recovery_rate: float = 0.012
    facilitation_strength: float = 0.045
    facilitation_threshold: float = 0.48

    spread_strength: float = 0.95
    noise_strength: float = 0.010

    healthy_threshold: float = 0.50
    degraded_threshold: float = 0.35

    theta_cover: float = 0.50
    theta_degraded: float = 0.40
    persistent_k: int = 3

def neighbor_mean_8(x: np.ndarray) -> np.ndarray:
    """
    Periodic 8-neighbor mean.

    This is used to represent local spatial interaction.
    """
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


def make_spatial_field(
    L: int,
    rng: np.random.Generator,
    smooth_steps: int = 8,
) -> np.ndarray:
    """
    Generate a smooth spatial heterogeneity field.
    """
    field = rng.normal(0.0, 1.0, size=(L, L)).astype(np.float32)

    for _ in range(smooth_steps):
        field = 0.55 * field + 0.45 * neighbor_mean_8(field)

    field = field - float(np.mean(field))
    field = field / (float(np.std(field)) + 1e-6)

    return field.astype(np.float32)


def largest_component_ratio(binary: np.ndarray) -> float:
    """
    Compute the largest connected-component ratio in a binary degraded map.

    Connectivity:
        4-neighbor connectivity.

    Return:
        largest_component_area / total_grid_area
    """
    binary = np.asarray(binary, dtype=bool)
    L0, L1 = binary.shape

    visited = np.zeros_like(binary, dtype=bool)
    largest = 0

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

            largest = max(largest, area)

    return float(largest) / float(L0 * L1)


def find_event_time(
    vegetation_cover: np.ndarray,
    degraded_area: np.ndarray,
    theta_cover: float,
    theta_degraded: float,
    persistent_k: int,
) -> int:
    """
    Event-based transition time.

    The ecological degradation event is defined as the first time when:
        vegetation_cover <= theta_cover
        and degraded_area >= theta_degraded

    for persistent_k consecutive steps.

    Return:
        event time if found;
        -1 otherwise.
    """
    condition = (vegetation_cover <= theta_cover) & (degraded_area >= theta_degraded)

    n = len(condition)
    k = int(persistent_k)

    for t in range(0, n - k + 1):
        if bool(np.all(condition[t : t + k])):
            return int(t)

    return -1


def simulate_vegetation_ca(
    seed: int,
    params: VegetationCAParams | None = None,
) -> dict:
    """
    Simulate one spatial vegetation degradation trajectory.

    Return dictionary keys:
        V_series
        stress
        vegetation_cover
        degraded_area
        dominant_degraded_patch
        event_time
        params
        seed
    """
    if params is None:
        params = VegetationCAParams()

    rng = np.random.default_rng(seed)

    L = params.L
    T = params.sim_steps

    resistance_field = make_spatial_field(L=L, rng=rng, smooth_steps=8)
    initial_field = make_spatial_field(L=L, rng=rng, smooth_steps=6)

    local_resistance = 1.0 + params.heterogeneity_strength * resistance_field
    local_resistance = np.clip(local_resistance, 0.65, 1.35).astype(np.float32)

    V = params.init_mean + params.init_noise * initial_field
    V = V + rng.normal(0.0, params.init_noise * 0.35, size=(L, L)).astype(np.float32)
    V = np.clip(V, 0.0, 1.0).astype(np.float32)

    V_series = np.zeros((T, L, L), dtype=np.float32)
    stress_series = np.zeros(T, dtype=np.float32)
    vegetation_cover = np.zeros(T, dtype=np.float32)
    degraded_area = np.zeros(T, dtype=np.float32)
    dominant_degraded_patch = np.zeros(T, dtype=np.float32)

    for t in range(T):
        progress = t / max(1, T - 1)
        stress = params.stress_start + (
            params.stress_end - params.stress_start
        ) * (progress ** params.stress_curve)

        stress_series[t] = stress
        V_series[t] = V

        healthy_map = V >= params.healthy_threshold
        degraded_map = V <= params.degraded_threshold

        vegetation_cover[t] = float(np.mean(healthy_map))
        degraded_area[t] = float(np.mean(degraded_map))
        dominant_degraded_patch[t] = largest_component_ratio(degraded_map)

        if t == T - 1:
            break

        neigh = neighbor_mean_8(V)
        degraded_neigh = neighbor_mean_8(1.0 - V)

        baseline_growth = params.growth_rate * V * (1.0 - V)

        facilitation = (
            params.facilitation_strength
            * (neigh - params.facilitation_threshold)
            * V
            * (1.0 - V)
        )

        recovery = params.recovery_rate * neigh * (1.0 - V)

        stress_effect = (
            stress
            * V
            * (1.0 + params.spread_strength * degraded_neigh)
            / local_resistance
        )

        stochastic_noise = rng.normal(
            0.0,
            params.noise_strength,
            size=(L, L),
        ).astype(np.float32)

        dV = baseline_growth + facilitation + recovery - stress_effect + stochastic_noise

        V = V + dV
        V = np.clip(V, 0.0, 1.0).astype(np.float32)

    event_time = find_event_time(
        vegetation_cover=vegetation_cover,
        degraded_area=degraded_area,
        theta_cover=params.theta_cover,
        theta_degraded=params.theta_degraded,
        persistent_k=params.persistent_k,
    )

    return {
        "V_series": V_series,
        "stress": stress_series,
        "vegetation_cover": vegetation_cover,
        "degraded_area": degraded_area,
        "dominant_degraded_patch": dominant_degraded_patch,
        "event_time": int(event_time),
        "params": asdict(params),
        "seed": int(seed),
    }


def sample_params(
    rng: np.random.Generator,
    L: int = 64,
    sim_steps: int = 200,
) -> VegetationCAParams:
    """
    Sample one parameter set for a vegetation degradation simulation.

    Parameter ranges are chosen to generate a mixture of:
        - gradual degradation
        - spatial fragmentation
        - collapse-like events
        - non-event trajectories
    """
    return VegetationCAParams(
        L=L,
        sim_steps=sim_steps,
        init_mean=float(rng.uniform(0.78, 0.88)),
        init_noise=float(rng.uniform(0.035, 0.075)),
        heterogeneity_strength=float(rng.uniform(0.06, 0.14)),
        stress_start=float(rng.uniform(0.006, 0.014)),
        stress_end=float(rng.uniform(0.055, 0.090)),
        stress_curve=float(rng.uniform(1.25, 1.95)),
        growth_rate=float(rng.uniform(0.024, 0.038)),
        recovery_rate=float(rng.uniform(0.008, 0.016)),
        facilitation_strength=float(rng.uniform(0.035, 0.060)),
        facilitation_threshold=float(rng.uniform(0.44, 0.52)),
        spread_strength=float(rng.uniform(0.65, 1.25)),
        noise_strength=float(rng.uniform(0.006, 0.014)),
        healthy_threshold=0.50,
        degraded_threshold=0.35,
        theta_cover=0.50,
        theta_degraded=0.40,
        persistent_k=3,
    )


def run_smoke_test(num_sims: int, seed: int, L: int, sim_steps: int) -> list[dict]:
    """
    Run a small smoke test and print simulation summaries.
    """
    rng = np.random.default_rng(seed)

    results = []

    print("===== Vegetation CA smoke test =====")
    print(f"num_sims: {num_sims}")
    print(f"seed: {seed}")
    print(f"L: {L}")
    print(f"sim_steps: {sim_steps}")
    print()

    for sim_idx in range(num_sims):
        sim_seed = int(rng.integers(0, 2**31 - 1))
        params = sample_params(rng=rng, L=L, sim_steps=sim_steps)

        out = simulate_vegetation_ca(seed=sim_seed, params=params)
        results.append(out)

        cover0 = float(out["vegetation_cover"][0])
        cover_end = float(out["vegetation_cover"][-1])
        degraded0 = float(out["degraded_area"][0])
        degraded_end = float(out["degraded_area"][-1])
        event_time = int(out["event_time"])

        print(
            f"sim={sim_idx:03d} "
            f"seed={sim_seed} "
            f"event_time={event_time:4d} "
            f"cover0={cover0:.3f} "
            f"cover_end={cover_end:.3f} "
            f"degraded0={degraded0:.3f} "
            f"degraded_end={degraded_end:.3f} "
            f"dominant_patch_end={float(out['dominant_degraded_patch'][-1]):.3f}"
        )

    valid_events = sum(1 for x in results if int(x["event_time"]) >= 0)

    print()
    print("===== Smoke test summary =====")
    print(f"Valid event simulations: {valid_events} / {num_sims}")

    if valid_events > 0:
        event_times = [int(x["event_time"]) for x in results if int(x["event_time"]) >= 0]
        print(f"Mean event time: {float(np.mean(event_times)):.2f}")
        print(f"Min event time: {int(np.min(event_times))}")
        print(f"Max event time: {int(np.max(event_times))}")

    return results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num-sims", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--L", type=int, default=64)
    parser.add_argument("--sim-steps", type=int, default=200)

    args = parser.parse_args()

    run_smoke_test(
        num_sims=args.num_sims,
        seed=args.seed,
        L=args.L,
        sim_steps=args.sim_steps,
    )


if __name__ == "__main__":
    main()