#!/usr/bin/env python3
"""Summarize the salt-marsh resolution, scale, and regularization audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESOLUTION_DIRECTORIES = {
    0.25: "salt_marsh_resolution_0.25m",
    1.0: "salt_marsh_resolution_1m",
    5.0: "salt_marsh_resolution_5m",
    10.0: "salt_marsh_resolution_10m",
}

ALPHA_DIRECTORIES = {
    0.01: "salt_marsh_5m_alpha_0.01",
    0.10: "salt_marsh_resolution_5m",
    1.00: "salt_marsh_5m_alpha_1",
    10.0: "salt_marsh_5m_alpha_10",
}

CANDIDATE_ALPHA_DIRECTORIES = {
    0.01: "salt_marsh_5m_128m_alpha_0.01",
    0.10: "salt_marsh_resolution_5m_multiscale",
    1.00: "salt_marsh_5m_128m_alpha_1",
    10.0: "salt_marsh_5m_128m_alpha_10",
}

MULTISCALE_DIRECTORY = "salt_marsh_resolution_5m_multiscale"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=Path("results/v3_6_published_csd_benchmark"),
    )
    return parser.parse_args()


def read_model_metrics(path: Path) -> pd.DataFrame:
    metrics_path = path / "cross_site_model_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    return pd.read_csv(metrics_path)


def collect_resolution_audit(root: Path) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for resolution, dirname in RESOLUTION_DIRECTORIES.items():
        metrics = read_model_metrics(root / dirname)
        subset = metrics.loc[
            metrics["model"].isin(
                ["stress_plus_static_patch", "stress_plus_all_patch"]
            )
        ]
        for row in subset.itertuples(index=False):
            records.append(
                {
                    "patch_resolution_m": resolution,
                    "train_site": row.train_site,
                    "test_site": row.test_site,
                    "model": row.model,
                    "poisson_deviance": row.poisson_deviance,
                    "deviance_change_vs_stress_percent": (
                        row.deviance_change_vs_stress_percent
                    ),
                    "weighted_log_rmse": row.weighted_log_rmse,
                    "log_rmse_change_vs_stress_percent": (
                        row.log_rmse_change_vs_stress_percent
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def collect_alpha_audit(root: Path) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for alpha, dirname in ALPHA_DIRECTORIES.items():
        metrics = read_model_metrics(root / dirname)
        subset = metrics.loc[metrics["model"] == "stress_plus_all_patch"]
        for row in subset.itertuples(index=False):
            records.append(
                {
                    "patch_resolution_m": 5.0,
                    "ridge_alpha": alpha,
                    "train_site": row.train_site,
                    "test_site": row.test_site,
                    "deviance_change_vs_stress_percent": (
                        row.deviance_change_vs_stress_percent
                    ),
                    "log_rmse_change_vs_stress_percent": (
                        row.log_rmse_change_vs_stress_percent
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def collect_candidate_alpha_audit(root: Path) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for alpha, dirname in CANDIDATE_ALPHA_DIRECTORIES.items():
        metrics = read_model_metrics(root / dirname)
        subset = metrics.loc[
            (metrics["model"] == "stress_plus_all_patch")
            & (metrics["block_width_m"] == 128.0)
        ]
        for row in subset.itertuples(index=False):
            records.append(
                {
                    "patch_resolution_m": 5.0,
                    "block_width_m": 128.0,
                    "ridge_alpha": alpha,
                    "train_site": row.train_site,
                    "test_site": row.test_site,
                    "deviance_change_vs_stress_percent": (
                        row.deviance_change_vs_stress_percent
                    ),
                    "log_rmse_change_vs_stress_percent": (
                        row.log_rmse_change_vs_stress_percent
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def collect_scale_audit(root: Path) -> pd.DataFrame:
    metrics = read_model_metrics(root / MULTISCALE_DIRECTORY)
    return metrics.loc[
        metrics["model"] == "stress_plus_all_patch",
        [
            "block_width_m",
            "train_site",
            "test_site",
            "deviance_change_vs_stress_percent",
            "log_rmse_change_vs_stress_percent",
        ],
    ].reset_index(drop=True)


def collect_candidate_support(root: Path) -> pd.DataFrame:
    support_path = root / MULTISCALE_DIRECTORY / "supported_local_units.csv"
    if not support_path.exists():
        raise FileNotFoundError(support_path)
    support = pd.read_csv(support_path)
    return support.loc[support["block_width_m"] == 128.0].reset_index(drop=True)


def _both_directions_pass(
    frame: pd.DataFrame,
    group_column: str,
    metric: str,
    cutoff: float,
) -> dict[float, bool]:
    return {
        float(key): bool(len(group) == 2 and (group[metric] < cutoff).all())
        for key, group in frame.groupby(group_column)
    }


def build_decision(
    resolution: pd.DataFrame,
    alpha: pd.DataFrame,
    scale: pd.DataFrame,
    candidate_alpha: pd.DataFrame,
    candidate_support: pd.DataFrame,
) -> dict[str, object]:
    dynamic = resolution.loc[resolution["model"] == "stress_plus_all_patch"]
    resolution_deviance = _both_directions_pass(
        dynamic,
        "patch_resolution_m",
        "deviance_change_vs_stress_percent",
        -5.0,
    )
    resolution_both_losses = {
        float(key): bool(
            len(group) == 2
            and (group["deviance_change_vs_stress_percent"] < -5.0).all()
            and (group["log_rmse_change_vs_stress_percent"] < 0.0).all()
        )
        for key, group in dynamic.groupby("patch_resolution_m")
    }
    alpha_deviance = _both_directions_pass(
        alpha,
        "ridge_alpha",
        "deviance_change_vs_stress_percent",
        -5.0,
    )
    scale_deviance = _both_directions_pass(
        scale,
        "block_width_m",
        "deviance_change_vs_stress_percent",
        -5.0,
    )
    candidate_both_losses = {
        float(key): bool(
            len(group) == 2
            and (group["deviance_change_vs_stress_percent"] < -5.0).all()
            and (group["log_rmse_change_vs_stress_percent"] < 0.0).all()
        )
        for key, group in candidate_alpha.groupby("ridge_alpha")
    }
    candidate_data_pass = bool(
        len(candidate_support) == 2
        and (candidate_support["supported_units"] >= 100).all()
        and (candidate_support["represented_blocks"] >= 20).all()
    )
    return {
        "resolutions_tested_m": sorted(resolution_deviance),
        "resolutions_passing_two_way_deviance": [
            key for key, passed in resolution_deviance.items() if passed
        ],
        "resolutions_passing_two_way_deviance_and_log_rmse": [
            key for key, passed in resolution_both_losses.items() if passed
        ],
        "five_metre_alphas_tested": sorted(alpha_deviance),
        "five_metre_alphas_passing_two_way_deviance": [
            key for key, passed in alpha_deviance.items() if passed
        ],
        "five_metre_block_widths_passing_two_way_deviance": [
            key for key, passed in scale_deviance.items() if passed
        ],
        "candidate_5m_128m_alphas_passing_both_losses": [
            key for key, passed in candidate_both_losses.items() if passed
        ],
        "candidate_5m_128m_supported_blocks": {
            str(row.site): int(row.represented_blocks)
            for row in candidate_support.itertuples(index=False)
        },
        "candidate_5m_128m_data_requirement_pass": candidate_data_pass,
        "candidate_5m_128m_status": "exploratory; pre-register for new data",
        "higher_native_resolution_rescues_patch_model": False,
        "scale_sensitivity_present": True,
        "precision_is_sole_explanation": False,
        "patch_primary_detector": "NO-GO",
        "patch_spatial_explanation": "GO",
    }


def make_figure(
    resolution: pd.DataFrame,
    alpha: pd.DataFrame,
    scale: pd.DataFrame,
    candidate_alpha: pd.DataFrame,
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    colors = {"Hellegat": "#b23a48", "Paulina": "#277da1"}

    ax = axes[0]
    dynamic = resolution.loc[resolution["model"] == "stress_plus_all_patch"]
    for test_site, group in dynamic.groupby("test_site"):
        group = group.sort_values("patch_resolution_m")
        ax.plot(
            group["patch_resolution_m"],
            group["deviance_change_vs_stress_percent"],
            marker="o",
            linewidth=2,
            color=colors[str(test_site)],
            label=f"held out {test_site}",
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(-5, color="0.5", linewidth=0.8, linestyle="--")
    ax.set_xscale("log")
    ax.set_xticks([0.25, 1, 5, 10], ["0.25", "1", "5", "10"])
    ax.set_xlabel("Patch resolution (m)")
    ax.set_ylabel("Deviance change vs stress-only (%)")
    ax.set_title("A  Pixel resolution")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    for test_site, group in alpha.groupby("test_site"):
        group = group.sort_values("ridge_alpha")
        ax.plot(
            group["ridge_alpha"],
            group["deviance_change_vs_stress_percent"],
            marker="o",
            linewidth=2,
            color=colors[str(test_site)],
            label=f"{test_site}, 64 m blocks",
        )
    for test_site, group in candidate_alpha.groupby("test_site"):
        group = group.sort_values("ridge_alpha")
        ax.plot(
            group["ridge_alpha"],
            group["deviance_change_vs_stress_percent"],
            marker="s",
            linewidth=1.6,
            linestyle="--",
            alpha=0.70,
            color=colors[str(test_site)],
            label=f"{test_site}, 128 m blocks",
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(-5, color="0.5", linewidth=0.8, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("Ridge alpha at 5 m")
    ax.set_title("B  Regularization")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[2]
    for test_site, group in scale.groupby("test_site"):
        group = group.sort_values("block_width_m")
        ax.plot(
            group["block_width_m"],
            group["deviance_change_vs_stress_percent"],
            marker="o",
            linewidth=2,
            color=colors[str(test_site)],
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(-5, color="0.5", linewidth=0.8, linestyle="--")
    ax.set_xticks([32, 64, 128])
    ax.set_xlabel("Analysis block width (m)")
    ax.set_title("C  Analysis scale at 5 m")

    fig.suptitle(
        "Salt-marsh patch increment is resolution- and model-sensitive",
        fontsize=14,
    )
    fig.savefig(output, dpi=220)
    plt.close(fig)


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    root: Path,
    resolution: pd.DataFrame,
    alpha: pd.DataFrame,
    scale: pd.DataFrame,
    candidate_alpha: pd.DataFrame,
    candidate_support: pd.DataFrame,
    decision: dict[str, object],
) -> None:
    dynamic = resolution.loc[
        resolution["model"] == "stress_plus_all_patch",
        [
            "patch_resolution_m",
            "train_site",
            "test_site",
            "deviance_change_vs_stress_percent",
            "log_rmse_change_vs_stress_percent",
        ],
    ]
    report = f"""# 盐沼斑块精度与尺度审计

## 结论

原始 0.25 m 数据没有挽救斑块增量，因此此前失败不能归因于 1 m 聚合造成的信息损失。5 m 在预设 `alpha=0.1` 下出现双向 deviance 改善，但该信号没有同时改善两向 log-RMSE，并且对正则化强度和分析单元宽度敏感。

这更符合“存在可能的生态尺度/去噪窗口”，而不是“空间精度越高越能检测临界慢化”。斑块继续作为空间解释和尺度敏感性结果，不升级为主要检测器。

## 像元分辨率

{markdown_table(dynamic)}

## 5 m 正则化敏感性

{markdown_table(alpha)}

## 5 m 分析单元敏感性

{markdown_table(scale)}

## 事后候选：5 m 像元 × 128 m 单元

该组合在 `alpha=0.01` 和 `0.1` 时同时改善两向 deviance 与 log-RMSE，但在更强正则化下消失。更重要的是，它只含 Hellegat 10 个、Paulina 15 个空间块，未达到预设的每站至少 20 个空间块要求。因此它是下一数据集应预注册检验的尺度候选，不是当前的确证结果。

{markdown_table(candidate_alpha)}

{markdown_table(candidate_support)}

## 可复现判定

```json
{json.dumps(decision, ensure_ascii=False, indent=2)}
```
"""
    (root / "salt_marsh_precision_audit.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.benchmark_dir
    resolution = collect_resolution_audit(root)
    alpha = collect_alpha_audit(root)
    scale = collect_scale_audit(root)
    candidate_alpha = collect_candidate_alpha_audit(root)
    candidate_support = collect_candidate_support(root)
    decision = build_decision(
        resolution,
        alpha,
        scale,
        candidate_alpha,
        candidate_support,
    )

    resolution.to_csv(root / "salt_marsh_resolution_audit.csv", index=False)
    alpha.to_csv(root / "salt_marsh_regularization_audit.csv", index=False)
    scale.to_csv(root / "salt_marsh_scale_audit.csv", index=False)
    candidate_alpha.to_csv(
        root / "salt_marsh_candidate_scale_regularization_audit.csv",
        index=False,
    )
    candidate_support.to_csv(
        root / "salt_marsh_candidate_scale_support.csv",
        index=False,
    )
    (root / "salt_marsh_precision_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    make_figure(
        resolution,
        alpha,
        scale,
        candidate_alpha,
        root / "salt_marsh_precision_audit.png",
    )
    write_report(
        root,
        resolution,
        alpha,
        scale,
        candidate_alpha,
        candidate_support,
        decision,
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
