from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.tcg_case_control import (
    _exact_pair_permutation_p,
    _pair_bootstrap_auc_ci,
    lopo_predict,
)
from src.tcg_pilot import ROOT


DEFAULT_CONFIG = ROOT / "configs" / "pilot_v0_3_lead_time.json"

PATH_KEYS = (
    "frame_features_path",
    "metrics_path",
    "predictions_path",
    "case_features_path",
    "report_path",
    "figure_path",
)

CLOUD_FEATURES = ("cold_cloud_fraction", "very_cold_fraction")
ORGANIZATION_FEATURES = (
    "radial_concentration",
    "azimuthal_symmetry",
    "largest_patch_fraction",
    "cold_centroid_offset_km",
)


def load_lead_time_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in PATH_KEYS:
        cfg[key] = str((ROOT / cfg[key]).resolve())
    return cfg


def aggregate_cases_at_cutoff(
    frames: pd.DataFrame, cutoff_hour: int, trailing_window_hours: int
) -> pd.DataFrame:
    """Aggregate only information available at or before a forecast cutoff."""
    if cutoff_hour > -3 or cutoff_hour < -72:
        raise ValueError("cutoff_hour must be between -72 and -3")
    available = frames[frames["hours_to_endpoint"].le(cutoff_hour)].copy()
    records = []
    for sid, case in available.groupby("sid", sort=True):
        case = case.sort_values("hours_to_endpoint")
        latest_hour = int(case["hours_to_endpoint"].max())
        if latest_hour != cutoff_hour:
            raise ValueError(
                f"case {sid} has no frame exactly at cutoff {cutoff_hour} h"
            )
        trailing = case[
            case["hours_to_endpoint"].ge(
                cutoff_hour - int(trailing_window_hours) + 3
            )
        ]
        endpoint = pd.to_datetime(case["endpoint_time"].iloc[0], utc=True)
        day = float(endpoint.dayofyear)
        record = {
            "sid": str(sid),
            "name": case["name"].iloc[0],
            "outcome": int(case["outcome"].iloc[0]),
            "pair_id": int(case["pair_id"].iloc[0]),
            "endpoint_time": endpoint,
            "endpoint_type": (
                case["endpoint_type"].iloc[0]
                if "endpoint_type" in case.columns
                else "unspecified"
            ),
            "cutoff_hour": int(cutoff_hour),
            "n_available_frames": int(len(case)),
            "n_trailing_frames": int(len(trailing)),
            "median_lat": float(case["center_lat"].median()),
            "median_lon": float(case["center_lon"].median()),
            "endpoint_day_sin": math.sin(2 * math.pi * day / 365.25),
            "endpoint_day_cos": math.cos(2 * math.pi * day / 365.25),
        }
        for feature in (*CLOUD_FEATURES, *ORGANIZATION_FEATURES):
            record[f"{feature}__trailing_median"] = float(
                trailing[feature].median()
            )
        records.append(record)
    result = pd.DataFrame(records).sort_values(
        ["pair_id", "outcome"], ascending=[True, False]
    )
    if len(result) != frames["sid"].nunique():
        raise ValueError(
            f"cutoff {cutoff_hour} lost cases: "
            f"{len(result)}/{frames['sid'].nunique()}"
        )
    return result


def lead_time_model_features() -> dict[str, list[str]]:
    cloud = [f"{name}__trailing_median" for name in CLOUD_FEATURES]
    organization = [
        f"{name}__trailing_median" for name in ORGANIZATION_FEATURES
    ]
    return {
        "metadata_only": [
            "median_lat",
            "median_lon",
            "endpoint_day_sin",
            "endpoint_day_cos",
        ],
        "cloud_area_only": cloud,
        "organization_only": organization,
        "cloud_plus_organization": [*cloud, *organization],
    }


def evaluate_lead_time(frames: pd.DataFrame, cfg: dict):
    from sklearn.metrics import average_precision_score, roc_auc_score

    all_cases = []
    metric_rows = []
    prediction_frames = []
    for cutoff in cfg["cutoff_hours_to_endpoint"]:
        cases = aggregate_cases_at_cutoff(
            frames, int(cutoff), int(cfg["trailing_window_hours"])
        )
        all_cases.append(cases)
        for model, features in lead_time_model_features().items():
            prediction = lopo_predict(cases, features, model, cfg)
            prediction["cutoff_hour"] = int(cutoff)
            auc = float(
                roc_auc_score(
                    prediction["outcome"], prediction["probability"]
                )
            )
            ci_low, ci_high = _pair_bootstrap_auc_ci(prediction, cfg)
            p_value, permutations = _exact_pair_permutation_p(
                cases, features, auc, cfg
            )
            metric_rows.append(
                {
                    "cutoff_hour": int(cutoff),
                    "lead_hours": int(-cutoff),
                    "model": model,
                    "n_cases": len(cases),
                    "n_pairs": cases["pair_id"].nunique(),
                    "n_features": len(features),
                    "roc_auc": auc,
                    "pair_bootstrap_ci_low": ci_low,
                    "pair_bootstrap_ci_high": ci_high,
                    "pr_auc": float(
                        average_precision_score(
                            prediction["outcome"], prediction["probability"]
                        )
                    ),
                    "exact_within_pair_permutation_p_one_sided": p_value,
                    "n_exact_permutations": permutations,
                }
            )
            prediction_frames.append(prediction)
    return (
        pd.concat(all_cases, ignore_index=True),
        pd.DataFrame(metric_rows),
        pd.concat(prediction_frames, ignore_index=True),
    )


def _plot_lead_time(metrics: pd.DataFrame, cfg: dict) -> None:
    import matplotlib.pyplot as plt

    labels = {
        "metadata_only": "Metadata",
        "cloud_area_only": "Cold-cloud area",
        "organization_only": "Organization",
        "cloud_plus_organization": "Cloud + organization",
    }
    colors = {
        "metadata_only": "#888888",
        "cloud_area_only": "#cc3311",
        "organization_only": "#3366aa",
        "cloud_plus_organization": "#228833",
    }
    figure, axis = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)
    for model, block in metrics.groupby("model", sort=False):
        block = block.sort_values("cutoff_hour")
        axis.plot(
            block["cutoff_hour"], block["roc_auc"], marker="o",
            linewidth=2, color=colors[model], label=labels[model],
        )
    axis.axhline(0.5, color="black", linestyle="--", linewidth=1)
    axis.set_ylim(0, 1.03)
    axis.set_xticks(cfg["cutoff_hours_to_endpoint"])
    axis.set_xlabel("Latest observation available (hours to endpoint)")
    axis.set_ylabel("Leave-one-pair-out ROC-AUC")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, ncol=2)
    path = Path(cfg["figure_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _earliest_sustained_high_auc(metrics: pd.DataFrame, model: str) -> str:
    block = metrics[metrics["model"].eq(model)].sort_values("cutoff_hour")
    for _, row in block.iterrows():
        current_and_later = block[block["cutoff_hour"].ge(row["cutoff_hour"])]
        if current_and_later["roc_auc"].ge(0.8).all():
            return (
                f"终点前 {int(row['lead_hours'])} h"
                f"（该点 AUC={row['roc_auc']:.3f}，随后均不低于 0.80）"
            )
    return "未持续达到 0.80"


def write_lead_time_report(metrics: pd.DataFrame, cfg: dict) -> Path:
    cold = metrics[metrics["model"].eq("cloud_area_only")]
    organization = metrics[metrics["model"].eq("organization_only")]
    min_cold_p = float(
        cold["exact_within_pair_permutation_p_one_sided"].min()
    )
    text = f"""# v0.3 云团指标提前量决策曲线

## 方法

对每个终点前截断时刻，仅保留当时及更早的 Himawari 帧；特征为最近 {cfg['trailing_window_hours']} h 的中位数。训练和测试仍按完整扰动匹配对分组，任何截断之后的图像均不参与该时刻预测。

## 结果

{metrics.to_markdown(index=False)}

冷云面积最早持续达到 AUC 0.80 的截断点：{_earliest_sustained_high_auc(metrics, 'cloud_area_only')}。

组织指标最早持续达到 AUC 0.80 的截断点：{_earliest_sustained_high_auc(metrics, 'organization_only')}。单个较早时点的偶然高值不被视为稳定提前量。

冷云模型各截断点中最小精确单侧 p 值为 {min_cold_p:.4f}。由于只有 5 对案例，0.0625 已是可能的最小值；因此曲线只能回答“信号大致在何时出现”，不能给出稳定业务提前量。

## 判读边界

- 如果较早截断点接近随机、临近终点才升高，当前方法更像短临生成诊断，而不是 72 h 预报。
- 如果组织模型未稳定超过冷云面积模型，则尚不能把论文创新放在组织动力学或临界慢化上。
- 元数据曲线若持续较高，仍说明地域和季节匹配不足，必须扩大未发展云团样本。
"""
    path = Path(cfg["report_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def run_lead_time_analysis(cfg: dict):
    frames = pd.read_csv(
        cfg["frame_features_path"],
        parse_dates=["observation_time", "endpoint_time"],
    )
    cases, metrics, predictions = evaluate_lead_time(frames, cfg)
    cases.to_csv(cfg["case_features_path"], index=False)
    metrics.to_csv(cfg["metrics_path"], index=False)
    predictions.to_csv(cfg["predictions_path"], index=False)
    _plot_lead_time(metrics, cfg)
    report = write_lead_time_report(metrics, cfg)
    print(
        metrics.pivot(index="cutoff_hour", columns="model", values="roc_auc")
        .round(3)
        .to_string()
    )
    print(f"report -> {report}")
    return metrics, predictions


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Causal lead-time curve for cloud and organization features"
    )
    parser.add_argument("command", choices=["analyze"])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    arguments = parser.parse_args(argv)
    cfg = load_lead_time_config(arguments.config)
    run_lead_time_analysis(cfg)


if __name__ == "__main__":
    main()
