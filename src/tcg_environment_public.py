from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from src.tcg_environment import (
    aggregate_environment_cases,
    compute_environment_features,
    environment_model_features,
    evaluate_environment_models,
)
from src.tcg_case_control import lopo_predict
from src.tcg_pilot import ROOT


DEFAULT_CONFIG = ROOT / "configs" / "pilot_v0_3_public_era5.json"

PATH_KEYS = (
    "case_control_manifest_path",
    "cloud_frame_features_path",
    "environment_frame_features_path",
    "merged_frame_features_path",
    "environment_case_features_path",
    "environment_metrics_path",
    "environment_predictions_path",
    "sensitivity_path",
    "source_manifest_path",
    "vorticity_validation_path",
    "environment_report_path",
    "model_figure_path",
    "trajectory_figure_path",
)

REMOTE_VARIABLES = {
    "u": {
        "parameter": "128_131_u",
        "grid": "ll025uv",
        "dataset": "U",
        "levels": (200, 850),
    },
    "v": {
        "parameter": "128_132_v",
        "grid": "ll025uv",
        "dataset": "V",
        "levels": (200, 850),
    },
    "r": {
        "parameter": "128_157_r",
        "grid": "ll025sc",
        "dataset": "R",
        "levels": (600, 700, 850),
    },
}

DIRECT_VORTICITY_SPEC = {
    "parameter": "128_138_vo",
    "grid": "ll025sc",
    "dataset": "VO",
}


def load_public_environment_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in PATH_KEYS:
        cfg[key] = str((ROOT / cfg[key]).resolve())
    return cfg


def select_anchor_manifest(manifest: pd.DataFrame, anchor_hours: list[int]) -> pd.DataFrame:
    work = manifest.copy()
    work["observation_time"] = pd.to_datetime(
        work["observation_time"], utc=True, errors="raise"
    )
    anchors = work[work["hours_to_endpoint"].isin(anchor_hours)].copy()
    expected = work["sid"].nunique() * len(anchor_hours)
    if len(anchors) != expected:
        counts = anchors.groupby("sid")["hours_to_endpoint"].nunique()
        raise ValueError(
            f"incomplete anchor selection: {len(anchors)}/{expected}; "
            f"per-case counts={counts.to_dict()}"
        )
    duplicates = anchors.duplicated(["sid", "hours_to_endpoint"]).any()
    if duplicates:
        raise ValueError("anchor selection contains duplicate case/hour rows")
    return anchors.sort_values(["sid", "observation_time"]).reset_index(drop=True)


def era5_object_key(variable: str, observation_time: pd.Timestamp) -> str:
    if variable not in REMOTE_VARIABLES:
        raise ValueError(f"unsupported public ERA5 variable: {variable}")
    time = pd.Timestamp(observation_time)
    if time.tzinfo is None:
        time = time.tz_localize("UTC")
    else:
        time = time.tz_convert("UTC")
    spec = REMOTE_VARIABLES[variable]
    day = time.strftime("%Y%m%d")
    month = time.strftime("%Y%m")
    filename = (
        f"e5.oper.an.pl.{spec['parameter']}.{spec['grid']}."
        f"{day}00_{day}23.nc"
    )
    return f"e5.oper.an.pl/{month}/{filename}"


def _coordinate_slice(values: np.ndarray, lower: float, upper: float) -> slice:
    selected = np.flatnonzero((values >= lower) & (values <= upper))
    if len(selected) < 3:
        raise ValueError(
            f"fewer than three coordinates inside [{lower}, {upper}]"
        )
    return slice(int(selected.min()), int(selected.max()) + 1)


def _read_remote_variable(task: dict) -> dict:
    """Read one hourly ERA5 chunk from the anonymous NSF NCAR S3 mirror."""
    import h5py
    import s3fs

    variable = task["variable"]
    spec = REMOTE_VARIABLES[variable]
    fs = s3fs.S3FileSystem(
        anon=True,
        client_kwargs={"region_name": task["region"]},
        skip_instance_cache=True,
    )
    s3_path = f"{task['bucket']}/{task['object_key']}"
    info = fs.info(s3_path)
    with fs.open(
        s3_path,
        "rb",
        block_size=int(task["block_size_bytes"]),
        cache_type="readahead",
    ) as remote:
        with h5py.File(remote, "r") as handle:
            levels = np.asarray(handle["level"][:], dtype=float)
            latitudes = np.asarray(handle["latitude"][:], dtype=float)
            longitudes = np.asarray(handle["longitude"][:], dtype=float)
            margin = float(task["area_margin_deg"])
            lat_slice = _coordinate_slice(
                latitudes,
                float(task["center_lat"]) - margin,
                float(task["center_lat"]) + margin,
            )
            lon_slice = _coordinate_slice(
                longitudes,
                float(task["center_lon"]) - margin,
                float(task["center_lon"]) + margin,
            )
            level_indices = []
            for target in spec["levels"]:
                matches = np.flatnonzero(np.isclose(levels, target))
                if len(matches) != 1:
                    raise ValueError(
                        f"pressure level {target} hPa missing from {task['object_key']}"
                    )
                level_indices.append(int(matches[0]))
            values = np.asarray(
                handle[spec["dataset"]][
                    int(task["hour_utc"]), level_indices, lat_slice, lon_slice
                ],
                dtype=float,
            )
            values[np.abs(values) > 1.0e10] = np.nan
            selected_lats = latitudes[lat_slice]
            selected_lons = longitudes[lon_slice]
    return {
        "task_id": task["task_id"],
        "sid": task["sid"],
        "observation_time": task["observation_time"],
        "variable": variable,
        "levels": list(spec["levels"]),
        "values": values,
        "latitudes": selected_lats,
        "longitudes": selected_lons,
        "object_key": task["object_key"],
        "object_size_bytes": int(info["size"]),
        "object_etag": str(info.get("ETag", "")).strip('"'),
    }


def _read_direct_vorticity(task: dict) -> dict:
    """Read ERA5's archived 850-hPa vorticity for an independent spot check."""
    import h5py
    import s3fs

    time = pd.Timestamp(task["observation_time"])
    day = time.strftime("%Y%m%d")
    month = time.strftime("%Y%m")
    spec = DIRECT_VORTICITY_SPEC
    object_key = (
        f"e5.oper.an.pl/{month}/e5.oper.an.pl.{spec['parameter']}."
        f"{spec['grid']}.{day}00_{day}23.nc"
    )
    fs = s3fs.S3FileSystem(
        anon=True,
        client_kwargs={"region_name": task["region"]},
        skip_instance_cache=True,
    )
    with fs.open(
        f"{task['bucket']}/{object_key}",
        "rb",
        block_size=int(task["block_size_bytes"]),
        cache_type="readahead",
    ) as remote:
        with h5py.File(remote, "r") as handle:
            levels = np.asarray(handle["level"][:], dtype=float)
            latitudes = np.asarray(handle["latitude"][:], dtype=float)
            longitudes = np.asarray(handle["longitude"][:], dtype=float)
            level_index = int(np.flatnonzero(np.isclose(levels, 850))[0])
            margin = float(task["area_margin_deg"])
            lat_slice = _coordinate_slice(
                latitudes,
                float(task["center_lat"]) - margin,
                float(task["center_lat"]) + margin,
            )
            lon_slice = _coordinate_slice(
                longitudes,
                float(task["center_lon"]) - margin,
                float(task["center_lon"]) + margin,
            )
            values = np.asarray(
                handle[spec["dataset"]][
                    int(time.hour), level_index, lat_slice, lon_slice
                ],
                dtype=float,
            )
            values[np.abs(values) > 1.0e10] = np.nan
            lons, lats = np.meshgrid(
                longitudes[lon_slice], latitudes[lat_slice]
            )
    from src.tcg_environment import _weighted_radius_mean

    direct_mean = _weighted_radius_mean(
        values,
        lons,
        lats,
        float(task["center_lon"]),
        float(task["center_lat"]),
        float(task["radius_km"]),
    )
    return {**task, "direct_mean_s1": direct_mean, "object_key": object_key}


def spherical_relative_vorticity(
    u_ms: np.ndarray,
    v_ms: np.ndarray,
    latitudes_deg: np.ndarray,
    longitudes_deg: np.ndarray,
) -> np.ndarray:
    """Calculate vertical relative vorticity on a regular latitude-longitude grid."""
    u = np.asarray(u_ms, dtype=float)
    v = np.asarray(v_ms, dtype=float)
    latitudes = np.asarray(latitudes_deg, dtype=float)
    longitudes = np.asarray(longitudes_deg, dtype=float)
    if u.shape != v.shape or u.shape != (len(latitudes), len(longitudes)):
        raise ValueError("wind arrays and coordinates have incompatible shapes")
    if min(u.shape) < 3:
        raise ValueError("vorticity calculation requires at least a 3x3 grid")
    earth_radius_m = 6_371_000.0
    phi = np.deg2rad(latitudes)
    lam = np.deg2rad(longitudes)
    cos_phi = np.cos(phi)[:, None]
    dv_dlambda = np.gradient(v, lam, axis=1, edge_order=2)
    d_ucosphi_dphi = np.gradient(
        u * cos_phi, phi, axis=0, edge_order=2
    )
    return (dv_dlambda - d_ucosphi_dphi) / (
        earth_radius_m * cos_phi
    )


def _build_tasks(anchors: pd.DataFrame, cfg: dict) -> list[dict]:
    tasks = []
    block_size = int(cfg["remote_block_size_mb"] * 1024 * 1024)
    for row in anchors.itertuples(index=False):
        observation_time = pd.Timestamp(row.observation_time)
        for variable in REMOTE_VARIABLES:
            tasks.append(
                {
                    "task_id": f"{row.sid}_{observation_time:%Y%m%d%H}_{variable}",
                    "sid": str(row.sid),
                    "observation_time": observation_time.isoformat(),
                    "variable": variable,
                    "bucket": cfg["bucket"],
                    "region": cfg["region"],
                    "object_key": era5_object_key(variable, observation_time),
                    "hour_utc": int(observation_time.hour),
                    "center_lat": float(row.center_lat),
                    "center_lon": float(row.center_lon),
                    "area_margin_deg": float(cfg["area_margin_deg"]),
                    "block_size_bytes": block_size,
                }
            )
    return tasks


def _assemble_fields(parts: dict[str, dict]) -> tuple[dict, np.ndarray, np.ndarray]:
    if set(parts) != set(REMOTE_VARIABLES):
        raise ValueError(f"incomplete remote fields: {sorted(parts)}")
    reference = parts["u"]
    latitudes = np.asarray(reference["latitudes"], dtype=float)
    longitudes = np.asarray(reference["longitudes"], dtype=float)
    for variable, part in parts.items():
        if not np.array_equal(latitudes, part["latitudes"]):
            raise ValueError(f"latitude mismatch for {variable}")
        if not np.array_equal(longitudes, part["longitudes"]):
            raise ValueError(f"longitude mismatch for {variable}")
    levels = np.asarray([200, 600, 700, 850], dtype=float)
    shape = (len(levels), len(latitudes), len(longitudes))
    fields = {name: np.full(shape, np.nan) for name in ("u", "v", "r", "vo")}
    for variable in ("u", "v", "r"):
        for source_index, pressure in enumerate(parts[variable]["levels"]):
            target_index = int(np.flatnonzero(np.isclose(levels, pressure))[0])
            fields[variable][target_index] = parts[variable]["values"][source_index]
    fields["vo"][3] = spherical_relative_vorticity(
        fields["u"][3], fields["v"][3], latitudes, longitudes
    )
    lons, lats = np.meshgrid(longitudes, latitudes)
    return fields, lons, lats


def process_public_environment(cfg: dict) -> pd.DataFrame:
    manifest = pd.read_csv(cfg["case_control_manifest_path"])
    anchors = select_anchor_manifest(
        manifest, [int(value) for value in cfg["anchor_hours_to_endpoint"]]
    )
    tasks = _build_tasks(anchors, cfg)
    results = []
    with ProcessPoolExecutor(max_workers=int(cfg["process_workers"])) as pool:
        futures = [pool.submit(_read_remote_variable, task) for task in tasks]
        for position, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if position % 10 == 0 or position == len(futures):
                print(
                    f"public ERA5 chunks processed {position}/{len(futures)}",
                    flush=True,
                )
    grouped: dict[tuple[str, str], dict[str, dict]] = {}
    source_rows = []
    for result in results:
        key = (result["sid"], result["observation_time"])
        grouped.setdefault(key, {})[result["variable"]] = result
        source_rows.append(
            {
                "sid": result["sid"],
                "observation_time": result["observation_time"],
                "variable": result["variable"],
                "bucket": cfg["bucket"],
                "object_key": result["object_key"],
                "object_size_bytes": result["object_size_bytes"],
                "object_etag": result["object_etag"],
                "access_date": cfg["source_access_date"],
            }
        )
    metadata = anchors.copy()
    metadata["observation_time_key"] = metadata["observation_time"].map(
        lambda value: pd.Timestamp(value).isoformat()
    )
    rows = []
    for row in metadata.itertuples(index=False):
        key = (str(row.sid), row.observation_time_key)
        fields, lons, lats = _assemble_fields(grouped[key])
        features = compute_environment_features(
            fields,
            cfg["pressure_levels_hpa"],
            lons,
            lats,
            float(row.center_lon),
            float(row.center_lat),
            cfg["analysis_radii_km"],
        )
        rows.append(
            {
                "sid": str(row.sid),
                "name": row.name,
                "outcome": int(row.outcome),
                "pair_id": int(row.pair_id),
                "endpoint_time": row.endpoint_time,
                "endpoint_type": row.endpoint_type,
                "observation_time": row.observation_time,
                "hours_to_endpoint": int(row.hours_to_endpoint),
                "center_lat": float(row.center_lat),
                "center_lon": float(row.center_lon),
                **features,
            }
        )
    environment = pd.DataFrame(rows).sort_values(["sid", "observation_time"])
    feature_columns = [
        column
        for column in environment.columns
        if column.startswith(("vorticity_", "rh_", "shear_"))
    ]
    if environment[feature_columns].isna().any().any():
        raise ValueError("public ERA5 feature table contains missing values")
    Path(cfg["environment_frame_features_path"]).parent.mkdir(
        parents=True, exist_ok=True
    )
    environment.to_csv(cfg["environment_frame_features_path"], index=False)
    pd.DataFrame(source_rows).sort_values(
        ["sid", "observation_time", "variable"]
    ).to_csv(cfg["source_manifest_path"], index=False)

    cloud = pd.read_csv(cfg["cloud_frame_features_path"])
    cloud["observation_time"] = pd.to_datetime(
        cloud["observation_time"], utc=True, errors="raise"
    )
    environment["observation_time"] = pd.to_datetime(
        environment["observation_time"], utc=True, errors="raise"
    )
    merged = cloud.merge(
        environment[["sid", "observation_time", *feature_columns]],
        on=["sid", "observation_time"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(environment):
        raise ValueError(
            f"cloud/environment anchor merge incomplete: "
            f"{len(merged)}/{len(environment)}"
        )
    merged.to_csv(cfg["merged_frame_features_path"], index=False)
    return merged


def validate_public_vorticity(cfg: dict) -> pd.DataFrame:
    features = pd.read_csv(cfg["environment_frame_features_path"])
    selected = pd.concat(
        [
            features[
                features["outcome"].eq(1)
                & features["hours_to_endpoint"].eq(-3)
            ].sort_values("sid").head(1),
            features[
                features["outcome"].eq(0)
                & features["hours_to_endpoint"].eq(-3)
            ].sort_values("sid").head(1),
            features[
                features["outcome"].eq(1)
                & features["hours_to_endpoint"].eq(-72)
            ].sort_values("sid").tail(1),
        ],
        ignore_index=True,
    )
    suffix = f"{int(cfg['primary_radius_km'])}km"
    derived_column = f"vorticity_850_mean_s1__{suffix}"
    tasks = []
    for row in selected.itertuples(index=False):
        tasks.append(
            {
                "sid": str(row.sid),
                "outcome": int(row.outcome),
                "hours_to_endpoint": int(row.hours_to_endpoint),
                "observation_time": str(row.observation_time),
                "center_lat": float(row.center_lat),
                "center_lon": float(row.center_lon),
                "derived_mean_s1": float(getattr(row, derived_column)),
                "radius_km": int(cfg["primary_radius_km"]),
                "bucket": cfg["bucket"],
                "region": cfg["region"],
                "area_margin_deg": float(cfg["area_margin_deg"]),
                "block_size_bytes": int(
                    cfg["remote_block_size_mb"] * 1024 * 1024
                ),
            }
        )
    with ProcessPoolExecutor(max_workers=len(tasks)) as pool:
        results = list(pool.map(_read_direct_vorticity, tasks))
    validation = pd.DataFrame(results)
    validation["relative_difference"] = (
        validation["derived_mean_s1"] - validation["direct_mean_s1"]
    ) / validation["direct_mean_s1"]
    validation.to_csv(cfg["vorticity_validation_path"], index=False)
    print(
        validation[
            [
                "sid",
                "hours_to_endpoint",
                "direct_mean_s1",
                "derived_mean_s1",
                "relative_difference",
            ]
        ].to_string(index=False)
    )
    return validation


def evaluate_public_sensitivity(cases: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    from sklearn.metrics import roc_auc_score

    rows = []
    samples = {
        "all_5_pairs": cases,
        "drop_pair_5": cases[cases["pair_id"].ne(5)],
    }
    for radius in cfg["analysis_radii_km"]:
        radius_cfg = {**cfg, "primary_radius_km": int(radius)}
        for sample_name, sample in samples.items():
            for model, feature_names in environment_model_features(
                radius_cfg
            ).items():
                prediction = lopo_predict(
                    sample, feature_names, model, radius_cfg
                )
                rows.append(
                    {
                        "sample": sample_name,
                        "radius_km": int(radius),
                        "model": model,
                        "n_cases": len(sample),
                        "n_pairs": sample["pair_id"].nunique(),
                        "roc_auc": float(
                            roc_auc_score(
                                prediction["outcome"],
                                prediction["probability"],
                            )
                        ),
                    }
                )
    sensitivity = pd.DataFrame(rows)
    sensitivity.to_csv(cfg["sensitivity_path"], index=False)
    return sensitivity


def _plot_results(frames: pd.DataFrame, metrics: pd.DataFrame, cfg: dict) -> None:
    import matplotlib.pyplot as plt

    suffix = f"{int(cfg['primary_radius_km'])}km"
    variables = [
        (f"vorticity_850_mean_s1__{suffix}", "850-hPa relative vorticity (s$^{-1}$)"),
        (f"rh_600_850_mean_pct__{suffix}", "600--850-hPa RH (%)"),
        (f"shear_200_850_ms__{suffix}", "200--850-hPa shear (m s$^{-1}$)"),
    ]
    colors = {0: "#3366aa", 1: "#cc3311"}
    labels = {0: "Nondeveloped", 1: "Developed"}
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    for axis, (column, title) in zip(axes, variables, strict=True):
        for outcome in (0, 1):
            group = frames[frames["outcome"].eq(outcome)]
            for _, case in group.groupby("sid"):
                case = case.sort_values("hours_to_endpoint")
                axis.plot(
                    case["hours_to_endpoint"], case[column],
                    color=colors[outcome], alpha=0.25, linewidth=1,
                )
            mean = group.groupby("hours_to_endpoint")[column].mean().sort_index()
            axis.plot(
                mean.index, mean.values, color=colors[outcome], linewidth=2.5,
                marker="o", label=labels[outcome],
            )
        axis.set_title(title)
        axis.set_xlabel("Hours to endpoint")
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    trajectory_path = Path(cfg["trajectory_figure_path"])
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(trajectory_path, dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.5, 4.2), constrained_layout=True)
    ordered = metrics.sort_values("roc_auc")
    axis.barh(ordered["model"], ordered["roc_auc"], color="#4477aa")
    axis.axvline(0.5, color="black", linestyle="--", linewidth=1)
    axis.set_xlim(0, 1.03)
    axis.set_xlabel("Leave-one-pair-out ROC-AUC")
    for position, value in enumerate(ordered["roc_auc"]):
        axis.text(min(value + 0.015, 0.98), position, f"{value:.2f}", va="center")
    model_path = Path(cfg["model_figure_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(model_path, dpi=180)
    plt.close(figure)


def write_public_environment_report(
    cases: pd.DataFrame,
    metrics: pd.DataFrame,
    sensitivity: pd.DataFrame,
    cfg: dict,
) -> Path:
    indexed = metrics.set_index("model")
    metadata_auc = float(indexed.loc["metadata_only", "roc_auc"])
    environment_auc = float(indexed.loc["environment_only", "roc_auc"])
    cloud_auc = float(indexed.loc["cloud_area_only", "roc_auc"])
    combined_auc = float(indexed.loc["environment_plus_cloud", "roc_auc"])
    full_auc = float(
        indexed.loc["environment_plus_cloud_plus_organization", "roc_auc"]
    )
    cloud_increment = combined_auc - environment_auc
    organization_increment = full_auc - combined_auc
    if organization_increment >= 0.05:
        gate = (
            "组织指标在这一小样本上显示正向增量，但仍需扩大并重新匹配负样本后复现，"
            "才能进入表征学习。"
        )
    else:
        gate = (
            "组织指标没有显示至少 0.05 的稳定增量，本阶段不支持直接进入 DynCL、"
            "SINDy 或临界慢化结论。"
        )
    validation_path = Path(cfg["vorticity_validation_path"])
    if validation_path.exists():
        validation = pd.read_csv(validation_path)
        validation_text = (
            "以 ERA5 直接归档的 VO 变量对 3 个锚点交叉校验，球面风场导数法的"
            f"最大绝对相对差为 {validation['relative_difference'].abs().max():.2%}。"
        )
    else:
        validation_text = "直接 VO 变量交叉校验尚未运行。"
    text = f"""# v0.3 匿名开放 ERA5 稀疏环境基线

## 数据与范围

- 轨迹：{len(cases)} 条（{cases['pair_id'].nunique()} 个暂定匹配对）。
- 时间锚点：{', '.join(map(str, cfg['anchor_hours_to_endpoint']))} h。
- 环境量：850 hPa 相对涡度、600/700/850 hPa 平均相对湿度、200--850 hPa 风切变。
- 空间尺度：{', '.join(map(str, cfg['analysis_radii_km']))} km，主分析为 {cfg['primary_radius_km']} km。
- 来源：NSF NCAR Curated ERA5 on AWS，匿名远程分块读取。

本实验是四时次稀疏 Pilot，不等同于原计划的完整 3 小时 ERA5 序列。它的作用是先检查此前冷云信号是否仅由大尺度环境解释。

## 留一匹配对结果

{metrics.to_markdown(index=False)}

## 增量诊断

- 元数据模型 AUC：{metadata_auc:.3f}。
- 仅环境模型 AUC：{environment_auc:.3f}。
- 仅冷云面积模型 AUC：{cloud_auc:.3f}。
- 环境 + 冷云 AUC：{combined_auc:.3f}，相对仅环境变化 {cloud_increment:+.3f}。
- 加入组织指标后 AUC：{full_auc:.3f}，相对环境 + 冷云变化 {organization_increment:+.3f}。

{gate}

## 敏感性分析

{sensitivity.to_markdown(index=False)}

300 km 与 500 km 的总体结论相近。删除匹配最差的第 5 对后指标普遍升高，这不能视为性能证明，反而说明当前结果对样本组成仍然敏感。

## 涡度计算交叉校验

{validation_text}

## 统计边界

只有 5 对案例，成对标签的精确置换检验最小单侧 p 值为 0.0625。AUC 的高低只能用于方法筛选，不能作为可发表的泛化性能。当前未发展样本仍存在地域、季节和“已被最佳路径追踪”的选择偏差。

## 下一决策

1. 扩大并客观追踪开放洋面的未发展云团，改善纬度、月份和环境匹配；
2. 在扩大样本上重复“仅环境 / 环境 + 冷云 / 环境 + 组织”消融；
3. 只有组织变量在分组跨年份验证中重复提供增量，才构造潜在组织状态并做动力学识别；
4. 临界慢化检验必须晚于状态变量有效性、空模型和假阳性控制。
"""
    path = Path(cfg["environment_report_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def analyze_public_environment(cfg: dict):
    frames = pd.read_csv(
        cfg["merged_frame_features_path"],
        parse_dates=["observation_time", "endpoint_time"],
    )
    cases = aggregate_environment_cases(frames, cfg)
    metrics, predictions = evaluate_environment_models(cases, cfg)
    sensitivity = evaluate_public_sensitivity(cases, cfg)
    cases.to_csv(cfg["environment_case_features_path"], index=False)
    metrics.to_csv(cfg["environment_metrics_path"], index=False)
    predictions.to_csv(cfg["environment_predictions_path"], index=False)
    _plot_results(frames, metrics, cfg)
    report = write_public_environment_report(
        cases, metrics, sensitivity, cfg
    )
    print(metrics.to_string(index=False))
    print(f"report -> {report}")
    return metrics, predictions


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Sparse anonymous NSF NCAR ERA5 environment pilot"
    )
    parser.add_argument(
        "command",
        choices=["process", "validate-vorticity", "analyze", "run-all"],
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    arguments = parser.parse_args(argv)
    cfg = load_public_environment_config(arguments.config)
    if arguments.command in ("process", "run-all"):
        process_public_environment(cfg)
    if arguments.command in ("validate-vorticity", "run-all"):
        validate_public_vorticity(cfg)
    if arguments.command in ("analyze", "run-all"):
        analyze_public_environment(cfg)


if __name__ == "__main__":
    main()
