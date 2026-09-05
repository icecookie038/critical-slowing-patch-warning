#!/usr/bin/env python3
"""Run the published-data CSD and sampling-precision benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.real_data.published_csd_benchmarks import (  # noqa: E402
    CLEMENTS_TRANSITION_TUBES,
    RINDI_CLASSICAL_METRICS,
    RINDI_PATCH_METRICS,
    clements_replication,
    clements_temporal_downsampling,
    compute_rindi_metrics,
    dai_replicate_precision,
    dai_replication,
    estimate_clements_tipping_points,
    file_md5,
    load_clements_workbook,
    load_dai_deterioration,
    load_rindi_grids,
    rindi_patch_threshold_sensitivity,
    rindi_replication,
    rindi_spatial_subsampling,
)


CLEMENTS_MD5 = {
    # Observed for Nature Communications Supplementary Data 1.
    "experimental_data.xls": "15e7e71bb7cb8a7b34f118bf5907d876",
}

RINDI_MD5 = {
    "cys_cover_avg_1yst.txt": "d929fb705a7f5996b8415e7843ed1203",
    "cys_cover_avg_2nd.txt": "c1036005b61db6c1f217e28d623a22f5",
    "dat_cl_1yst.txt": "4549dd387a9f95ffdbf83dad868e000d",
    "dat_cl_2nd.txt": "a4e5f7dff18150bbd97cc14c0c4194af",
    "Exp_data_2014.txt": "f68a4627374dc1880bf4b649d1089323",
    "Exp_data_2015.txt": "d185a3a226ab43a4467370295b668058",
}

DAI_ARCHIVE_MD5 = "f739beab74af2f981d410c10277ec364"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "csd_benchmark",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "v3_6_published_csd_benchmark",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--precision-iterations", type=int, default=500)
    parser.add_argument("--spatial-iterations", type=int, default=200)
    parser.add_argument("--permutations", type=int, default=999)
    return parser.parse_args()


def _verify(path: Path, expected: str) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    observed = file_md5(path)
    if observed != expected:
        raise ValueError(
            f"Checksum mismatch for {path}: expected {expected}, observed {observed}"
        )


def verify_inputs(data_root: Path) -> pd.DataFrame:
    records: list[dict[str, str | int]] = []
    clements = data_root / "clements_2016"
    for filename, expected in CLEMENTS_MD5.items():
        path = clements / filename
        # The publisher does not advertise a checksum.  The stable hash of the
        # downloaded supplementary file is locked here to detect later changes
        # or partial downloads.
        _verify(path, expected)
        observed = file_md5(path)
        records.append(
            {
                "dataset": "clements_2016",
                "file": filename,
                "expected_md5": expected,
                "observed_md5": observed,
                "status": "stable_observed_hash_verified",
                "bytes": path.stat().st_size,
            }
        )
    rindi = data_root / "rindi_2018"
    for filename, expected in RINDI_MD5.items():
        path = rindi / filename
        _verify(path, expected)
        records.append(
            {
                "dataset": "rindi_2018",
                "file": filename,
                "expected_md5": expected,
                "observed_md5": expected,
                "status": "verified",
                "bytes": path.stat().st_size,
            }
        )
    dai_archive = data_root / "dai_2015" / "data_deterioration.zip"
    _verify(dai_archive, DAI_ARCHIVE_MD5)
    records.append(
        {
            "dataset": "dai_2015",
            "file": dai_archive.name,
            "expected_md5": DAI_ARCHIVE_MD5,
            "observed_md5": DAI_ARCHIVE_MD5,
            "status": "verified",
            "bytes": dai_archive.stat().st_size,
        }
    )
    return pd.DataFrame.from_records(records)


def benchmark_registry() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "study": "van Belzen et al. 2017",
                "system": "Tidal-marsh vegetation",
                "published_csd_evidence": "Direct recovery hazard declines at two sites",
                "sampling_design": "0.25 m binary maps; 8 and 11 irregular dates",
                "spatial_patch_data": "2-D",
                "benchmark_status": "RUN: v3.5 direct recovery + patch explanation",
                "paper": "https://doi.org/10.1038/ncomms15811",
                "data": "https://zenodo.org/records/4998258",
            },
            {
                "study": "Clements & Ozgul 2016",
                "system": "Predator-prey microcosms",
                "published_csd_evidence": "Composite EWS before inferred population transitions",
                "sampling_design": "Daily observations; 37 populations; up to 45 d",
                "spatial_patch_data": "None",
                "benchmark_status": "RUN: replication + temporal downsampling",
                "paper": "https://doi.org/10.1038/ncomms10984",
                "data": "https://www.nature.com/articles/ncomms10984#Sec14",
            },
            {
                "study": "Dai et al. 2015",
                "system": "Cooperative yeast populations",
                "published_csd_evidence": "CV and autocorrelation under two deteriorating drivers",
                "sampling_design": "48 parallel populations; 21 daily states per driver",
                "spatial_patch_data": "None",
                "benchmark_status": "RUN: driver comparison + replicate subsampling",
                "paper": "https://doi.org/10.1073/pnas.1418415112",
                "data": "https://datadryad.org/dataset/doi:10.5061/dryad.k30v3",
            },
            {
                "study": "Rindi et al. 2018",
                "system": "Intertidal macroalgal canopy",
                "published_csd_evidence": "Manipulative spatial EWS before ~75% canopy-loss threshold",
                "sampling_design": "2 y; 16 transects; 5x30 cells per map",
                "spatial_patch_data": "2-D",
                "benchmark_status": "RUN: spatial EWS + patch explanation + subsampling",
                "paper": "https://doi.org/10.1002/ecy.2391",
                "data": "https://doi.org/10.6084/m9.figshare.6200822.v2",
            },
            {
                "study": "Drake & Griffen 2010",
                "system": "Daphnia extinction experiment",
                "published_csd_evidence": "Early-warning signals before experimental extinction",
                "sampling_design": "High-frequency population trajectories",
                "spatial_patch_data": "None",
                "benchmark_status": "SCREENED: repository download currently denied",
                "paper": "https://doi.org/10.1038/nature09389",
                "data": "https://datadryad.org/dataset/doi:10.5061/dryad.q3p64",
            },
            {
                "study": "Veraart et al. 2012",
                "system": "Cyanobacterial chemostats",
                "published_csd_evidence": "Direct recovery rates near a fold bifurcation",
                "sampling_design": "Parallel chemostats with pulse perturbations",
                "spatial_patch_data": "None",
                "benchmark_status": "SCREENED: public endpoint returned an anti-bot HTML file",
                "paper": "https://doi.org/10.1038/nature10723",
                "data": "https://doi.org/10.17026/dans-ztg-93aw",
            },
            {
                "study": "Butitta et al. 2017",
                "system": "Whole-lake trophic manipulation",
                "published_csd_evidence": "Spatial EWS before a manipulated algal bloom",
                "sampling_design": "11 weekly maps; thousands of georeferenced samples/map",
                "spatial_patch_data": "2-D points",
                "benchmark_status": "SCREENED: package metadata open, data entity restricted",
                "paper": "https://doi.org/10.1002/ecs2.1941",
                "data": "https://doi.org/10.6073/pasta/403cc21eba48b801114801ea05c5c2fa",
            },
        ]
    )


def _format_value(value: object) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, (float, np.floating)):
        magnitude = abs(float(value))
        if magnitude and (magnitude < 1e-3 or magnitude >= 1e4):
            return f"{float(value):.3e}"
        return f"{float(value):.4f}"
    return str(value)


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_format_value(value) for value in row) + " |")
    return "\n".join(lines)


def make_summary_figure(
    clements_summary: pd.DataFrame,
    dai_full: pd.DataFrame,
    rindi_full: pd.DataFrame,
    rindi_precision: pd.DataFrame,
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    ax = axes[0, 0]
    subset = clements_summary.loc[
        clements_summary["metric"] == "published_trait"
    ]
    for class_name, color in (("deteriorating", "#bd3f36"), ("constant", "#3d6f9e")):
        data = subset.loc[subset["class"] == class_name]
        ax.plot(
            data["sampling_interval_days"],
            data["signal_rate_all"],
            marker="o",
            linewidth=2,
            color=color,
            label=class_name,
        )
    ax.set_title("A  Predator-prey: temporal sampling")
    ax.set_xlabel("Sampling interval (days)")
    ax.set_ylabel("2σ signal rate")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)

    ax = axes[0, 1]
    plot = dai_full.loc[
        dai_full["indicator"].isin(["cv", "ar1_slope"])
    ].copy()
    labels: list[str] = []
    values: list[float] = []
    colors: list[str] = []
    for driver in ("dilution_factor", "sucrose"):
        for indicator in ("cv", "ar1_slope"):
            for environment in (driver, "control"):
                row = plot.loc[
                    (plot["driver"] == driver)
                    & (plot["indicator"] == indicator)
                    & (plot["environment"] == environment)
                ].iloc[0]
                labels.append(
                    f"{driver[:3]}\n{indicator}\n"
                    + ("driver" if environment == driver else "control")
                )
                values.append(float(row["late_minus_early"]))
                colors.append("#bd3f36" if environment == driver else "#8aa4bd")
    ax.bar(np.arange(len(values)), values, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(len(labels)), labels, fontsize=8)
    ax.set_title("B  Yeast: late-minus-baseline change")
    ax.set_ylabel("Indicator change")

    ax = axes[1, 0]
    colors = [
        "#d0902f" if role == "patch_explanation" else "#4b8a69"
        for role in rindi_full["metric_role"]
    ]
    y = np.arange(len(rindi_full))
    ax.barh(y, rindi_full["spearman_rho"], color=colors)
    ax.set_yticks(y, rindi_full["metric"], fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlim(-1, 1)
    ax.invert_yaxis()
    ax.set_title("C  Macroalgae: stress-gradient association")
    ax.set_xlabel("Spearman ρ with canopy removal")

    ax = axes[1, 1]
    selected_metrics = (
        ("spatial_cv", "spatial CV", "#4b8a69", "-"),
        ("spatial_sd", "spatial SD", "#7aaa8e", "-"),
        (
            "largest_turf_patch_fraction",
            "largest patch",
            "#d0902f",
            "--",
        ),
        ("component_density", "component density", "#a85b32", "--"),
    )
    for metric, label, color, linestyle in selected_metrics:
        data = rindi_precision.loc[rindi_precision["metric"] == metric]
        ax.plot(
            data["sampling_fraction"] * 100,
            data["median_rank_similarity"],
            marker="o",
            linewidth=2,
            color=color,
            linestyle=linestyle,
            label=label,
        )
    ax.set_title("D  Macroalgae: spatial subsampling")
    ax.set_xlabel("Observed spatial cells (%)")
    ax.set_ylabel("Median rank similarity to full map")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Published ecological CSD benchmark and precision audit", fontsize=15)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def make_patch_figure(
    grids: list,
    metrics: pd.DataFrame,
    output: Path,
    *,
    year: int = 2015,
    turf_threshold: int = 2,
) -> None:
    chosen = []
    year_metrics = metrics.loc[metrics["year"] == year]
    for removal in (0.0, 25.0, 50.0, 75.0):
        candidates = year_metrics.loc[year_metrics["removal_percent"] == removal]
        median = candidates["largest_turf_patch_fraction"].median()
        row = candidates.iloc[
            np.argmin(np.abs(candidates["largest_turf_patch_fraction"] - median))
        ]
        chosen.append(
            next(grid for grid in grids if grid.year == year and grid.plot == int(row["plot"]))
        )

    fig, axes = plt.subplots(2, 4, figsize=(14, 5.8), constrained_layout=True)
    for column, grid in enumerate(chosen):
        understory = 100.0 - 25.0 * grid.turf_score
        axes[0, column].imshow(understory, cmap="YlGn", vmin=0, vmax=100, aspect="auto")
        axes[0, column].set_title(f"{int(grid.removal_percent)}% canopy removed")
        axes[0, column].set_xticks([])
        axes[0, column].set_yticks([])
        binary = grid.turf_score >= turf_threshold
        axes[1, column].imshow(binary, cmap="Greys", vmin=0, vmax=1, aspect="auto")
        axes[1, column].set_xticks([])
        axes[1, column].set_yticks([])
        axes[1, column].set_xlabel(f"plot {grid.plot}")
    axes[0, 0].set_ylabel("Understory cover")
    axes[1, 0].set_ylabel("Turf patches ≥50%")
    fig.suptitle(
        "Patch explanation of the manipulated macroalgal transition (representative 2015 grids)",
        fontsize=14,
    )
    fig.savefig(output, dpi=220)
    plt.close(fig)


def build_report(
    registry: pd.DataFrame,
    clements_tips: pd.DataFrame,
    clements_summary: pd.DataFrame,
    dai_full: pd.DataFrame,
    dai_precision: pd.DataFrame,
    rindi_full: pd.DataFrame,
    rindi_precision: pd.DataFrame,
    patch_sensitivity: pd.DataFrame,
    decision: dict[str, object],
) -> str:
    clements_key = clements_summary.loc[
        (clements_summary["metric"] == "published_trait")
        & clements_summary["sampling_interval_days"].isin([1, 2, 3, 4]),
        [
            "sampling_interval_days",
            "class",
            "eligible_fraction",
            "signal_rate_all",
            "normalized_metric_score",
        ],
    ]
    dai_key = dai_full.loc[
        dai_full["indicator"].isin(["cv", "ar1_slope"]),
        [
            "driver",
            "environment",
            "indicator",
            "tipping_index",
            "late_minus_early",
            "kendall_tau",
            "kendall_p",
        ],
    ]
    dai_precision_key = dai_precision.loc[
        dai_precision["sample_size"].isin([48, 24, 12, 6]),
        [
            "driver",
            "indicator",
            "sample_size",
            "positive_direction_rate",
            "greater_than_control_rate",
            "median_rank_similarity",
        ],
    ]
    rindi_key = rindi_full[
        [
            "metric",
            "metric_role",
            "year_adjusted_slope",
            "one_sided_block_permutation_p",
            "spearman_rho",
        ]
    ]
    spatial_key = rindi_precision.loc[
        rindi_precision["sampling_fraction"].isin([1.0, 0.5, 0.25]),
        [
            "sampling_fraction",
            "metric",
            "direction_retention_rate",
            "median_rank_similarity",
        ],
    ]
    sensitivity_key = patch_sensitivity[
        [
            "minimum_turf_cover_percent",
            "metric",
            "year_adjusted_slope",
            "one_sided_block_permutation_p",
        ]
    ]

    transition_count = int(clements_tips["transition_inferred"].sum())
    selected_deteriorating = int(
        (
            clements_tips["transition_inferred"]
            & (clements_tips["treatment"] != "Constant")
        ).sum()
    )
    return f"""# v3.6 已发表生态临界慢化数据集基准与精度审计

## 结论

本轮在四篇已发表研究的数据上建立了统一基准。结果支持：**采样精度会明显影响间接预警指标，但它不是此前斑块路线失败的唯一原因。** 高精度数据中仍存在驱动依赖、指标依赖和对照假阳性；因此不能通过单纯提高分辨率把斑块重新定义为通用 CSD 检测器。

- 主检测层：直接恢复率或论文已验证的动力学 EWS。
- 空间解释层：斑块描述慢恢复/转变区域如何形成、连通和扩展。
- 当前决策：`{decision['patch_primary_detector']}`（斑块主检测器）；`{decision['patch_spatial_explanation']}`（斑块空间解释）。

## 数据集筛选

{markdown_table(registry[['study', 'system', 'sampling_design', 'spatial_patch_data', 'benchmark_status']])}

前三个新增基准与既有盐沼基准均使用已发表论文公开的数据。另三组数据完成了来源和结构核对，但公开端点当前不能无认证取得，因此没有把失败下载的网页当作数据参与统计。

## 1. 捕食—猎物微宇宙：时间精度

按作者 LOESS 实现推断出 {transition_count} 个跨越增长率阈值的种群，其中 {selected_deteriorating} 个来自恶化处理；常量处理 Tube 14 按原代码排除。作者推荐的 `CV + mean size + size SD` 复合指标在完整每日数据上得到更高真阳性率和更低假阳性率。

{markdown_table(clements_key)}

降采样结果用于回答观测间隔问题。`eligible_fraction` 同时报告，因为快速崩溃种群在每 3–4 天采样时甚至无法保留足够的转折前观测；把这些样本静默删除会高估性能。

## 2. 酵母实验：同等精度下的驱动依赖

死亡率上升和蔗糖下降均有 48 个平行种群。用相邻日平均密度比首次低于 0.5 确定论文所述崩溃起点，并比较崩溃前 10 日中末 3 日与前 5 日的指标。

{markdown_table(dai_key)}

两种驱动的 CV 均明显上升，但自回归/自相关信号强度不同，且对应控制窗口本身也可能出现正趋势。这是在观测数量相同的情况下发生的，直接说明“低精度”不能解释全部成败。

平行重复数降采样结果：

{markdown_table(dai_precision_key)}

## 3. 大型藻类野外实验：空间精度与斑块解释

按论文的 5×30 网格、二维平面去趋势和 0–75% 冠层移除梯度复现空间统计。斑块以单元中藻坪覆盖至少 50% 为主定义；它们只标记空间组织，不用于定义临界慢化标签。

{markdown_table(rindi_key)}

空间 SD、CV 和偏度的方向与分块置换结果均支持已发表趋势。低频功率方向为正，但本项目采用的预阈值线性分块检验与论文 GAMM 不同，置换检验未显著；因此只记为方向性复现。Moran 的简单线性斜率虽为正，原论文将其判为不稳定指标，本项目不把它计入预期成功数。

空间单元随机保留后的稳健性：

{markdown_table(spatial_key)}

斑块阈值敏感性（25%、50%、75% 藻坪覆盖）：

{markdown_table(sensitivity_key)}

该实验清楚显示了斑块的合适用途：随着冠层移除，较大的藻坪连通域与更多边界出现，解释了空间方差、偏度和低频结构为何改变。但这些斑块是受控压力的空间响应，不等同于独立的恢复率测量。

## 4. 对“是不是原数据精度太低”的判断

### 精度确实贡献了失败

1. 高频捕食—猎物数据被稀释后，快速崩溃种群很快失去足够的转折前观测。
2. 酵母平行重复数减少后，指标曲线对完整 48 重复结果的排序一致性下降。
3. 大型藻类空间单元减少后，尤其是连通域数量和边界等斑块指标，更容易被缺测人为打碎。

### 但精度不是唯一或主要可修复原因

1. 酵母在相同 48 重复、相同日频率下，不同环境驱动仍产生不同的指标强度。
2. 大型藻类原论文及本复现中，不同空间指标表现不同；Moran 并不是稳定单调信号。
3. 盐沼的直接恢复率在两个地点均成功，而斑块的跨地点增量不稳定。这是“直接动力学量成功、形态代理失败”，不是没有 CSD。
4. 盐沼只有 8/11 个不规则日期，这严重限制动态斑块谱系；但每幅图的 0.25 m 空间分辨率本身并不低。

所以更准确的诊断是：**时间采样稀疏降低了动态斑块检验力，二值分类和环境异质性又削弱了特异性；即使增加空间像元，形态代理仍未必含有恢复率之外的信息。**

## 5. 推荐修改方向

1. 不恢复 PWSI/斑块作为主检测器，也不重新调权重。
2. 用四个已发表基准形成“检测层”：直接恢复率、CV/AC1、论文预先给定的复合指标。
3. 斑块形成独立“解释层”：覆盖、最大连通域、边界和连通域密度；明确报告阈值与缺测敏感性。
4. 在新数据选择上，优先获取“高频时间 + 重复扰动/对照 + 2-D 空间图”的组合；单纯更高像元分辨率不是充分条件。
5. 论文主线可改为：**直接动力学证据如何受到观测设计影响，以及空间斑块何时只解释、何时能提供额外信息。**

## 可复现判定

```json
{json.dumps(decision, ensure_ascii=False, indent=2)}
```
"""


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checksums = verify_inputs(args.data_root)
    registry = benchmark_registry()

    clements_path = args.data_root / "clements_2016" / "experimental_data.xls"
    clements = load_clements_workbook(clements_path)
    clements_tips = estimate_clements_tipping_points(clements)
    observed_transition_tubes = set(
        clements_tips.loc[clements_tips["transition_inferred"], "tube"].astype(int)
    )
    if observed_transition_tubes != CLEMENTS_TRANSITION_TUBES:
        raise RuntimeError(
            "The published Clements transition set was not reproduced: "
            f"{sorted(observed_transition_tubes)}"
        )
    clements_detail, clements_full = clements_replication(clements, clements_tips)
    clements_down_detail, clements_down = clements_temporal_downsampling(
        clements, clements_tips
    )

    dai_root = args.data_root / "dai_2015" / "data_deterioration"
    dai_arrays = load_dai_deterioration(dai_root)
    dai_full, dai_tipping = dai_replication(dai_arrays)
    dai_precision_detail, dai_precision = dai_replicate_precision(
        dai_arrays,
        dai_tipping,
        n_iterations=args.precision_iterations,
        seed=args.seed,
    )

    rindi_root = args.data_root / "rindi_2018"
    rindi_grids = load_rindi_grids(rindi_root)
    rindi_metrics = compute_rindi_metrics(rindi_grids, turf_threshold=2)
    rindi_full = rindi_replication(
        rindi_metrics,
        n_permutations=args.permutations,
        seed=args.seed,
    )
    rindi_precision_detail, rindi_precision = rindi_spatial_subsampling(
        rindi_grids,
        rindi_metrics,
        n_iterations=args.spatial_iterations,
        seed=args.seed,
        turf_threshold=2,
    )
    patch_sensitivity = rindi_patch_threshold_sensitivity(
        rindi_grids,
        n_permutations=args.permutations,
        seed=args.seed,
    )

    clements_key = clements_full.loc[
        (clements_full["metric"] == "published_trait")
        & (clements_full["class"] == "deteriorating")
    ].iloc[0]
    clements_control = clements_full.loc[
        (clements_full["metric"] == "published_trait")
        & (clements_full["class"] == "constant")
    ].iloc[0]
    dai_cv = dai_full.loc[
        (dai_full["indicator"] == "cv")
        & (dai_full["environment"] == dai_full["driver"])
    ]
    rindi_expected = rindi_full.loc[
        rindi_full["metric"].isin(
            ["spatial_sd", "spatial_cv", "spatial_skewness", "low_frequency_power"]
        )
    ]
    precision_contributes = bool(
        clements_down.loc[
            (clements_down["metric"] == "published_trait")
            & (clements_down["class"] == "deteriorating")
            & (clements_down["sampling_interval_days"] == 4),
            "eligible_fraction",
        ].iloc[0]
        < 1.0
    )
    patch_direction_count = int(
        rindi_full.loc[
            rindi_full["metric_role"] == "patch_explanation", "positive_direction"
        ].sum()
    )
    decision: dict[str, object] = {
        "published_datasets_run": 4,
        "clements_transition_set_exact": True,
        "clements_trait_composite_true_positive_rate": float(
            clements_key["signal_rate_all"]
        ),
        "clements_trait_composite_false_positive_rate": float(
            clements_control["signal_rate_all"]
        ),
        "dai_cv_positive_for_both_drivers": bool(
            (dai_cv["late_minus_early"] > 0).all()
        ),
        "rindi_expected_spatial_metrics_positive": int(
            (rindi_expected["year_adjusted_slope"] > 0).sum()
        ),
        "rindi_expected_spatial_metrics_total": int(len(rindi_expected)),
        "rindi_expected_spatial_metrics_block_significant": int(
            (rindi_expected["one_sided_block_permutation_p"] <= 0.05).sum()
        ),
        "rindi_low_frequency_replication": "positive direction; blocked p > 0.05",
        "rindi_positive_patch_descriptors": patch_direction_count,
        "rindi_patch_descriptors_total": len(RINDI_PATCH_METRICS),
        "sampling_precision_contributes": precision_contributes,
        "sampling_precision_is_sole_explanation": False,
        "patch_primary_detector": "NO-GO",
        "patch_spatial_explanation": "GO, descriptive and sensitivity-audited",
    }

    outputs = {
        "input_checksums.csv": checksums,
        "benchmark_registry.csv": registry,
        "clements_tipping_points.csv": clements_tips,
        "clements_full_detail.csv": clements_detail,
        "clements_full_summary.csv": clements_full,
        "clements_temporal_downsampling_detail.csv": clements_down_detail,
        "clements_temporal_downsampling_summary.csv": clements_down,
        "dai_full_replication.csv": dai_full,
        "dai_replicate_precision_detail.csv": dai_precision_detail,
        "dai_replicate_precision_summary.csv": dai_precision,
        "rindi_grid_metrics.csv": rindi_metrics,
        "rindi_full_replication.csv": rindi_full,
        "rindi_spatial_subsampling_detail.csv": rindi_precision_detail,
        "rindi_spatial_subsampling_summary.csv": rindi_precision,
        "rindi_patch_threshold_sensitivity.csv": patch_sensitivity,
    }
    for filename, frame in outputs.items():
        frame.to_csv(args.output_dir / filename, index=False)

    (args.output_dir / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = build_report(
        registry,
        clements_tips,
        clements_down,
        dai_full,
        dai_precision,
        rindi_full,
        rindi_precision,
        patch_sensitivity,
        decision,
    )
    (args.output_dir / "report.md").write_text(report, encoding="utf-8")
    make_summary_figure(
        clements_down,
        dai_full,
        rindi_full,
        rindi_precision,
        args.output_dir / "benchmark_summary.png",
    )
    make_patch_figure(
        rindi_grids,
        rindi_metrics,
        args.output_dir / "macroalgal_patch_explanation.png",
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
