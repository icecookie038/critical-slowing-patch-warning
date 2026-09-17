import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def plot_prepatch_trajectory(traj, out_path):
    indicators = [
        "Z_sync",
        "Z_conn",
        "Z_rigid",
        "Z_mode",
        "PWSI_equal",
    ]

    plt.figure(figsize=(9, 5.5))

    for ind in indicators:
        mean_col = f"{ind}_mean"
        sem_col = f"{ind}_sem"

        if mean_col not in traj.columns:
            continue

        x = traj["relative_time"]
        y = traj[mean_col]
        sem = traj[sem_col] if sem_col in traj.columns else 0

        plt.plot(x, y, linewidth=2, label=ind)
        plt.fill_between(x, y - sem, y + sem, alpha=0.15)

    plt.axvline(0, linestyle="--", linewidth=1)
    plt.axvline(-30, linestyle=":", linewidth=1)
    plt.xlabel("Time to critical transition")
    plt.ylabel("Baseline-standardized indicator value")
    plt.title("Event-aligned prepatch indicator dynamics")
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_visible_patch_trajectory(traj, out_path):
    ind = "visible_patch_metric_z"

    mean_col = f"{ind}_mean"
    sem_col = f"{ind}_sem"

    if mean_col not in traj.columns:
        raise ValueError(f"Cannot find {mean_col} in trajectory file.")

    plt.figure(figsize=(8, 5))

    x = traj["relative_time"]
    y = traj[mean_col]
    sem = traj[sem_col] if sem_col in traj.columns else 0

    plt.plot(x, y, linewidth=2, label="visible patch metric")
    plt.fill_between(x, y - sem, y + sem, alpha=0.15)

    plt.axvline(0, linestyle="--", linewidth=1)
    plt.axvline(-30, linestyle=":", linewidth=1)
    plt.xlabel("Time to critical transition")
    plt.ylabel("Baseline-standardized visible patch metric")
    plt.title("Event-aligned visible patch dynamics")
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_first_rise(summary, out_path):
    order = [
        "Z_conn",
        "Z_mode",
        "Z_rigid",
        "Z_sync",
        "PWSI_equal",
        "visible_patch_metric_z",
    ]

    labels = {
        "Z_conn": "Z_conn",
        "Z_mode": "Z_mode",
        "Z_rigid": "Z_rigid",
        "Z_sync": "Z_sync",
        "PWSI_equal": "PWSI",
        "visible_patch_metric_z": "Visible patch",
    }

    df = summary[summary["indicator"].isin(order)].copy()
    df["indicator"] = pd.Categorical(df["indicator"], categories=order, ordered=True)
    df = df.sort_values("indicator")

    plt.figure(figsize=(9, 5.2))

    x_labels = [labels.get(x, x) for x in df["indicator"].astype(str)]
    bars = plt.bar(x_labels, df["lead_median"])

    for bar, lead, det in zip(bars, df["lead_median"], df["detection_rate"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.6,
            f"{lead:.0f}\nDR={det:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.ylabel("Median lead to transition")
    plt.xlabel("Indicator")
    plt.title("First-rise time comparison")
    plt.xticks(rotation=25, ha="right")
    plt.ylim(0, max(df["lead_median"]) + 8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=str, required=True)
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    fig_dir = result_dir / "figures_clean"
    fig_dir.mkdir(parents=True, exist_ok=True)

    trajectory_file = result_dir / "event_aligned_indicator_trajectory.csv"
    first_rise_file = result_dir / "first_rise_time_summary.csv"

    traj = pd.read_csv(trajectory_file)
    summary = pd.read_csv(first_rise_file)

    plot_prepatch_trajectory(
        traj,
        fig_dir / "clean_event_aligned_prepatch_trajectory.png",
    )

    plot_visible_patch_trajectory(
        traj,
        fig_dir / "clean_event_aligned_visible_patch_trajectory.png",
    )

    plot_first_rise(
        summary,
        fig_dir / "clean_first_rise_time_comparison.png",
    )

    print("Clean patch formation figures saved to:")
    print(fig_dir)


if __name__ == "__main__":
    main()