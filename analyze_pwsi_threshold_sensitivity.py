import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def decode_names(arr):
    names = []
    for x in arr:
        if isinstance(x, bytes):
            names.append(x.decode("utf-8"))
        else:
            names.append(str(x))
    return names


def find_feature_matrix(data):
    candidate_keys = [
        "X_patch",
        "X_pwsi",
        "X_features",
        "X",
    ]

    for key in candidate_keys:
        if key in data:
            return key, data[key]

    raise KeyError(
        "Cannot find feature matrix. Expected one of: "
        "X_patch, X_pwsi, X_features, X"
    )


def find_feature_names(data):
    candidate_keys = [
        "patch_feature_names",
        "feature_names",
        "pwsi_feature_names",
        "prepatch_feature_names",
    ]

    for key in candidate_keys:
        if key in data:
            return decode_names(data[key])

    return None


def select_pwsi_feature(X, feature_names, feature_name):
    feature_dim = X.shape[-1]

    # If the feature matrix has only one feature, it must be the selected PWSI feature.
    # This is the case for files such as:
    # seir_v1_4_pwsi_equal_only_h30_seed42.npz
    if feature_dim == 1:
        return 0

    if feature_names is not None:
        lowered = [name.lower() for name in feature_names]
        target = feature_name.lower()

        if target in lowered:
            idx = lowered.index(target)
            if idx < feature_dim:
                return idx

        for i, name in enumerate(lowered):
            if ("pwsi_equal" in name or "pwsi" in name) and i < feature_dim:
                return i

    if feature_dim == 6:
        # Expected order:
        # Z_sync, Z_conn, Z_rigid, Z_mode, PWSI_equal, PWSI_importance
        return 4

    raise ValueError(
        f"Cannot identify PWSI feature automatically. "
        f"feature_dim={feature_dim}, feature_names={feature_names}"
    )

    if feature_names is not None:
        lowered = [name.lower() for name in feature_names]
        target = feature_name.lower()

        if target in lowered:
            return lowered.index(target)

        for i, name in enumerate(lowered):
            if "pwsi_equal" in name or "pwsi" in name:
                return i

    if feature_dim == 1:
        return 0

    if feature_dim == 6:
        # Expected order:
        # Z_sync, Z_conn, Z_rigid, Z_mode, PWSI_equal, PWSI_importance
        return 4

    raise ValueError(
        f"Cannot identify PWSI feature automatically. "
        f"feature_dim={feature_dim}, feature_names={feature_names}"
    )


def extract_feature_values(X, feature_idx):
    if X.ndim == 3:
        # Shape: (n_samples, seq_len, n_features)
        return X[:, -1, feature_idx].astype(float)
    elif X.ndim == 2:
        # Shape: (n_samples, n_features)
        return X[:, feature_idx].astype(float)
    else:
        raise ValueError(f"Unsupported X shape: {X.shape}")


def infer_warning_direction(values, y_risk, time_idx, critical_time, horizon):
    safe_mask = (y_risk == 0) & ((critical_time - time_idx) > horizon)
    risk_mask = y_risk == 1

    if safe_mask.sum() < 10 or risk_mask.sum() < 10:
        return "high"

    safe_mean = np.nanmean(values[safe_mask])
    risk_mean = np.nanmean(values[risk_mask])

    if risk_mean >= safe_mean:
        return "high"
    return "low"


def compute_threshold(values, rule_name, direction):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    mean = float(np.mean(values))
    std = float(np.std(values))

    if direction == "high":
        if rule_name == "mean_plus_2std":
            return mean + 2.0 * std
        if rule_name == "mean_plus_2_5std":
            return mean + 2.5 * std
        if rule_name == "mean_plus_3std":
            return mean + 3.0 * std
        if rule_name == "q95":
            return float(np.quantile(values, 0.95))
        if rule_name == "q99":
            return float(np.quantile(values, 0.99))

    if direction == "low":
        if rule_name == "mean_plus_2std":
            return mean - 2.0 * std
        if rule_name == "mean_plus_2_5std":
            return mean - 2.5 * std
        if rule_name == "mean_plus_3std":
            return mean - 3.0 * std
        if rule_name == "q95":
            return float(np.quantile(values, 0.05))
        if rule_name == "q99":
            return float(np.quantile(values, 0.01))

    raise ValueError(f"Unknown rule: {rule_name}")


def find_first_persistent_alarm(times, values, threshold, direction, persistent_k):
    if not np.isfinite(threshold):
        return None

    if direction == "high":
        alarm_flags = values >= threshold
    else:
        alarm_flags = values <= threshold

    alarm_flags = np.asarray(alarm_flags, dtype=bool)

    for i in range(0, len(alarm_flags) - persistent_k + 1):
        if np.all(alarm_flags[i:i + persistent_k]):
            return int(times[i])

    return None


def summarize_rule(
    df,
    rule_name,
    horizon,
    persistent_k,
    direction,
    min_baseline_points,
):
    per_sim_records = []

    for sim_id, g in df.groupby("sim_id"):
        g = g.sort_values("time_idx").copy()

        times = g["time_idx"].to_numpy()
        values = g["pwsi"].to_numpy()
        critical_time = int(g["critical_time"].iloc[0])

        safe_mask = (critical_time - times) > horizon
        baseline_values = values[safe_mask]

        if len(baseline_values) < min_baseline_points:
            pre_event_mask = times < critical_time
            pre_event_values = values[pre_event_mask]

            if len(pre_event_values) >= min_baseline_points:
                n_base = max(min_baseline_points, int(len(pre_event_values) * 0.3))
                baseline_values = pre_event_values[:n_base]
            else:
                baseline_values = values[: max(min_baseline_points, min(len(values), 5))]

        threshold = compute_threshold(baseline_values, rule_name, direction)
        first_alarm_time = find_first_persistent_alarm(
            times=times,
            values=values,
            threshold=threshold,
            direction=direction,
            persistent_k=persistent_k,
        )

        if first_alarm_time is None:
            lead_time = np.nan
            valid_alarm = 0
            prewindow_alarm = 0
            late_alarm = 0
            missed = 1
        else:
            lead_time = critical_time - first_alarm_time
            valid_alarm = int((lead_time > 0) and (lead_time <= horizon))
            prewindow_alarm = int(lead_time > horizon)
            late_alarm = int(lead_time <= 0)
            missed = int(valid_alarm == 0)

        per_sim_records.append(
            {
                "sim_id": sim_id,
                "rule_name": rule_name,
                "threshold": threshold,
                "direction": direction,
                "critical_time": critical_time,
                "first_alarm_time": first_alarm_time,
                "lead_time": lead_time,
                "valid_alarm": valid_alarm,
                "missed": missed,
                "prewindow_alarm": prewindow_alarm,
                "late_alarm": late_alarm,
            }
        )

    detail = pd.DataFrame(per_sim_records)

    valid_leads = detail.loc[detail["valid_alarm"] == 1, "lead_time"].dropna()

    summary = {
        "rule_name": rule_name,
        "direction": direction,
        "threshold_mean": detail["threshold"].mean(),
        "threshold_std": detail["threshold"].std(),
        "n_sims": len(detail),
        "valid_window_alarm_rate": detail["valid_alarm"].mean(),
        "miss_rate": detail["missed"].mean(),
        "prewindow_alarm_rate": detail["prewindow_alarm"].mean(),
        "late_alarm_rate": detail["late_alarm"].mean(),
        "lead_mean": valid_leads.mean() if len(valid_leads) > 0 else np.nan,
        "lead_median": valid_leads.median() if len(valid_leads) > 0 else np.nan,
        "lead_q25": valid_leads.quantile(0.25) if len(valid_leads) > 0 else np.nan,
        "lead_q75": valid_leads.quantile(0.75) if len(valid_leads) > 0 else np.nan,
    }

    return summary, detail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--feature-name", type=str, default="PWSI_equal")
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--persistent-k", type=int, default=3)
    parser.add_argument("--min-baseline-points", type=int, default=10)
    parser.add_argument(
        "--direction",
        type=str,
        default="auto",
        choices=["auto", "high", "low"],
        help="high means alarm when PWSI is high; low means alarm when PWSI is low.",
    )
    args = parser.parse_args()

    data_file = Path(args.data_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(data_file, allow_pickle=True)

    required_keys = ["sim_id", "time_idx", "critical_time", "y_risk"]
    for key in required_keys:
        if key not in data:
            raise KeyError(f"Missing required key: {key}")

    matrix_key, X = find_feature_matrix(data)
    feature_names = find_feature_names(data)
    feature_idx = select_pwsi_feature(X, feature_names, args.feature_name)
    pwsi_values = extract_feature_values(X, feature_idx)

    sim_id = data["sim_id"]
    time_idx = data["time_idx"]
    critical_time = data["critical_time"]
    y_risk = data["y_risk"]

    if args.direction == "auto":
        direction = infer_warning_direction(
            values=pwsi_values,
            y_risk=y_risk,
            time_idx=time_idx,
            critical_time=critical_time,
            horizon=args.horizon,
        )
    else:
        direction = args.direction

    df = pd.DataFrame(
        {
            "sim_id": sim_id,
            "time_idx": time_idx,
            "critical_time": critical_time,
            "y_risk": y_risk,
            "pwsi": pwsi_values,
        }
    )

    rules = [
        "mean_plus_2std",
        "mean_plus_2_5std",
        "mean_plus_3std",
        "q95",
        "q99",
    ]

    summaries = []
    all_details = []

    for rule in rules:
        summary, detail = summarize_rule(
            df=df,
            rule_name=rule,
            horizon=args.horizon,
            persistent_k=args.persistent_k,
            direction=direction,
            min_baseline_points=args.min_baseline_points,
        )
        summaries.append(summary)
        all_details.append(detail)

    summary_df = pd.DataFrame(summaries)
    detail_df = pd.concat(all_details, ignore_index=True)

    summary_path = out_dir / "pwsi_threshold_sensitivity_summary.csv"
    detail_path = out_dir / "pwsi_threshold_sensitivity_detail.csv"

    summary_df.to_csv(summary_path, index=False)
    detail_df.to_csv(detail_path, index=False)

    print("\n===== PWSI Threshold Sensitivity =====")
    print(f"Data file: {data_file}")
    print(f"Feature matrix: {matrix_key}")
    print(f"Feature index: {feature_idx}")
    print(f"Feature names: {feature_names}")
    print(f"Warning direction: {direction}")
    print(f"Output summary: {summary_path}")
    print(f"Output detail: {detail_path}")

    display_cols = [
        "rule_name",
        "valid_window_alarm_rate",
        "miss_rate",
        "lead_mean",
        "lead_median",
        "prewindow_alarm_rate",
        "late_alarm_rate",
        "threshold_mean",
    ]
    print(summary_df[display_cols].sort_values("lead_mean", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()