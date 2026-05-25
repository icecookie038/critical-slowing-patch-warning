"""
v0.8 Dynamic Patch Indicator Pilot

Purpose:
- Run a small pilot test for dynamic patch indicators.
- Do not modify v0.1-v0.7 baseline code.
- Do not retrain models in this stage.
- Only calculate and visualize PDSI / DPCI / DPCR / FAI / PSII.

Outputs:
- results_summary/dynamic_patch_pilot_metrics.csv
- figures/dynamic_patch_pilot/dynamic_indicators_trend.png
- figures/dynamic_patch_pilot/patch_state_summary.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from features.dynamic_patch_indicators import (  # noqa: E402
    extract_dynamic_patch_indicators,
)


def build_demo_seir_like_series(
    total_steps: int = 80,
    size: int = 64,
    critical_time: int = 55,
    seed: int = 42,
) -> np.ndarray:
    """
    Build a synthetic SEIR-like expanding infected patch series.

    Design:
    - Early stage: almost stable and slow expansion.
    - Pre-critical stage: increasing small fragments and patch-size shifts.
    - Post-critical stage: rapid dominant patch expansion.

    This is only for v0.8 pilot visualization.
    It does not replace the real SEIR simulation.
    """
    rng = np.random.default_rng(seed)
    masks = np.zeros((total_steps, size, size), dtype=bool)

    center_r = size // 2
    center_c = size // 2
    rr, cc = np.ogrid[:size, :size]

    for t in range(total_steps):
        # Stage 1: stable early baseline
        if t < 20:
            radius = 5

        # Stage 2: slow pre-critical expansion
        elif t < critical_time:
            radius = 5 + int(0.18 * (t - 20))

        # Stage 3: rapid post-critical expansion
        else:
            radius = 5 + int(0.18 * (critical_time - 20)) + int(0.85 * (t - critical_time))

        main_patch = (rr - center_r) ** 2 + (cc - center_c) ** 2 <= radius**2
        masks[t] = main_patch

        # Stable small patches appear after the baseline period
        if t >= 25:
            masks[t, 8:11, 8:11] = True

        if t >= 32:
            masks[t, 50:53, 12:15] = True

        if t >= 40:
            masks[t, 12:15, 48:51] = True

        # Increasing fragmentation mainly before the critical point
        if t >= critical_time - 12:
            n_fragments = min(24, 2 + 2 * (t - (critical_time - 12)))

            for _ in range(n_fragments):
                r = rng.integers(3, size - 4)
                c = rng.integers(3, size - 4)
                masks[t, r : r + 2, c : c + 2] = True

    return masks

def load_field_series(input_path: str | None) -> np.ndarray:
    """
    Load field series from .npy or .npz.

    If input_path is None, use demo synthetic series.

    Supported .npz keys:
    - field_series
    - masks
    - infected
    - infected_series
    - X
    """
    if input_path is None:
        return build_demo_seir_like_series()

    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    if path.suffix.lower() == ".npy":
        arr = np.load(path)
        return np.asarray(arr)

    if path.suffix.lower() == ".npz":
        data = np.load(path)

        candidate_keys = [
            "field_series",
            "masks",
            "infected",
            "infected_series",
            "X",
        ]

        for key in candidate_keys:
            if key in data:
                return np.asarray(data[key])

        raise KeyError(
            f"No supported key found in {path}. "
            f"Available keys: {list(data.keys())}. "
            f"Expected one of: {candidate_keys}"
        )

    raise ValueError("Only .npy and .npz input files are supported.")


def ensure_3d_series(arr: np.ndarray) -> np.ndarray:
    """
    Make sure the input is shaped as (T, H, W).

    If the input has shape (T, H, W, C), use the first channel.
    """
    arr = np.asarray(arr)

    if arr.ndim == 3:
        return arr

    if arr.ndim == 4:
        return arr[..., 0]

    raise ValueError(f"Expected shape (T,H,W) or (T,H,W,C), but got {arr.shape}")


def plot_dynamic_indicators(
    df: pd.DataFrame,
    output_path: Path,
    critical_time: int | None = None,
) -> None:
    """
    Plot PDSI / DPCI / DPCR / FAI / PSII trend.

    A rolling mean is used for visualization only.
    The raw values are still saved in CSV.
    """
    plot_df = df.copy()

    smooth_cols = ["pdsi", "dpci", "dpcr", "fai", "psii"]

    for col in smooth_cols:
        plot_df[col] = plot_df[col].rolling(window=3, min_periods=1).mean()

    plt.figure(figsize=(11, 6))

    plt.plot(plot_df["time"], plot_df["pdsi"], label="PDSI")
    plt.plot(plot_df["time"], plot_df["dpci"], label="DPCI")
    plt.plot(plot_df["time"], plot_df["dpcr"], label="DPCR")
    plt.plot(plot_df["time"], plot_df["fai"], label="FAI")
    plt.plot(plot_df["time"], plot_df["psii"], label="PSII")

    if critical_time is not None:
        plt.axvline(critical_time, linestyle="--", label="Critical time")

    plt.xlabel("Time")
    plt.ylabel("Indicator value")
    plt.title("v0.8 Dynamic Patch Indicators")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_patch_state_summary(
    df: pd.DataFrame,
    output_path: Path,
    critical_time: int | None = None,
) -> None:
    """
    Plot internal patch state variables.
    """
    plt.figure(figsize=(11, 6))

    plt.plot(df["time"], df["patch_count"], label="Patch count")
    plt.plot(df["time"], df["largest_patch_area"], label="Largest patch area")
    plt.plot(df["time"], df["total_patch_area"], label="Total patch area")
    plt.plot(df["time"], df["largest_patch_ratio"], label="Largest patch ratio")

    if critical_time is not None:
        plt.axvline(critical_time, linestyle="--", label="Critical time")

    plt.xlabel("Time")
    plt.ylabel("Patch state value")
    plt.title("Patch State Summary")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run v0.8 dynamic patch indicator pilot."
    )

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Optional .npy or .npz file with shape (T,H,W). If omitted, demo data is used.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for binary masks.",
    )

    parser.add_argument(
        "--lag",
        type=int,
        default=1,
        help="Temporal lag for dynamic indicators.",
    )

    parser.add_argument(
        "--baseline-end",
        type=int,
        default=15,
        help="Baseline end index for PSII z-score.",
    )

    parser.add_argument(
        "--critical-time",
        type=int,
        default=55,
        help="Critical time used only for plotting a reference line.",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="expansion",
        choices=["expansion", "collapse"],
        help="Use expansion for SEIR and collapse for vegetation degradation.",
    )

    args = parser.parse_args()

    figures_dir = PROJECT_ROOT / "figures" / "dynamic_patch_pilot"
    summary_dir = PROJECT_ROOT / "results_summary"

    figures_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    field_series = load_field_series(args.input)
    field_series = ensure_3d_series(field_series)

    print("Input field series shape:", field_series.shape)

    df = extract_dynamic_patch_indicators(
        field_series=field_series,
        threshold=args.threshold,
        positive=True,
        connectivity=8,
        lag=args.lag,
        mode=args.mode,
        baseline_end=args.baseline_end,
    )

    output_csv = summary_dir / "dynamic_patch_pilot_metrics.csv"
    output_trend = figures_dir / "dynamic_indicators_trend.png"
    output_patch_summary = figures_dir / "patch_state_summary.png"

    df.to_csv(output_csv, index=False)

    plot_dynamic_indicators(
        df=df,
        output_path=output_trend,
        critical_time=args.critical_time,
    )

    plot_patch_state_summary(
        df=df,
        output_path=output_patch_summary,
        critical_time=args.critical_time,
    )

    print()
    print("===== Dynamic patch indicator pilot finished =====")
    print("Saved metrics to:")
    print(output_csv)
    print()
    print("Saved figures to:")
    print(output_trend)
    print(output_patch_summary)
    print()
    print("Preview:")
    print(df[["time", "pdsi", "dpci", "dpcr", "fai", "psii"]].head())
    print()
    print(df[["time", "pdsi", "dpci", "dpcr", "fai", "psii"]].tail())


if __name__ == "__main__":
    main()