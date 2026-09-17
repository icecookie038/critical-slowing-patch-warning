import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def neighbor_pairs_4neigh(H, W):
    pairs = []

    for i in range(H):
        for j in range(W):
            idx = i * W + j

            if i + 1 < H:
                pairs.append((idx, (i + 1) * W + j))

            if j + 1 < W:
                pairs.append((idx, i * W + (j + 1)))

    return np.asarray(pairs, dtype=np.int64)


def compute_neighbor_correlations(seq):
    """
    seq shape: (T, H, W)
    return: correlations for all 4-neighbor edges
    """
    T, H, W = seq.shape
    X = seq.reshape(T, H * W)

    pairs = neighbor_pairs_4neigh(H, W)
    a = X[:, pairs[:, 0]]
    b = X[:, pairs[:, 1]]

    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)

    numerator = np.sum(a * b, axis=0)
    denominator = np.sqrt(np.sum(a * a, axis=0) * np.sum(b * b, axis=0))

    corr = np.zeros_like(numerator, dtype=np.float32)
    valid = denominator > 1e-8
    corr[valid] = numerator[valid] / denominator[valid]

    corr = np.clip(corr, -1.0, 1.0)
    return corr


def compute_sync_edge_ratio(seq, corr_threshold):
    corr = compute_neighbor_correlations(seq)
    return float(np.mean(corr >= corr_threshold))


def extract_sequences(X_img):
    """
    Expected X_img shape:
    (n_samples, T, 1, H, W)
    or
    (n_samples, T, H, W)
    """
    if X_img.ndim == 5:
        return X_img[:, :, 0, :, :]
    if X_img.ndim == 4:
        return X_img
    raise ValueError(f"Unsupported X_img shape: {X_img.shape}")


def compute_threshold(values, rule_name="mean_plus_2std"):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    mean = float(np.mean(values))
    std = float(np.std(values))

    if rule_name == "mean_plus_2std":
        return mean + 2.0 * std
    if rule_name == "mean_plus_2_5std":
        return mean + 2.5 * std
    if rule_name == "q95":
        return float(np.quantile(values, 0.95))

    raise ValueError(f"Unknown threshold rule: {rule_name}")


def find_first_persistent_alarm(times, values, threshold, persistent_k):
    if not np.isfinite(threshold):
        return None

    alarm_flags = np.asarray(values >= threshold, dtype=bool)

    for i in range(0, len(alarm_flags) - persistent_k + 1):
        if np.all(alarm_flags[i:i + persistent_k]):
            return int(times[i])

    return None


def summarize_corr_threshold(
    df,
    corr_threshold,
    horizon,
    persistent_k,
    threshold_rule,
    min_baseline_points,
):
    records = []

    for sim_id, g in df.groupby("sim_id"):
        g = g.sort_values("time_idx").copy()

        times = g["time_idx"].to_numpy()
        values = g["sync_edge_ratio"].to_numpy()
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

        alarm_threshold = compute_threshold(baseline_values, threshold_rule)

        first_alarm_time = find_first_persistent_alarm(
            times=times,
            values=values,
            threshold=alarm_threshold,
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

        records.append(
            {
                "sim_id": sim_id,
                "corr_threshold": corr_threshold,
                "threshold_rule": threshold_rule,
                "alarm_threshold": alarm_threshold,
                "critical_time": critical_time,
                "first_alarm_time": first_alarm_time,
                "lead_time": lead_time,
                "valid_alarm": valid_alarm,
                "missed": missed,
                "prewindow_alarm": prewindow_alarm,
                "late_alarm": late_alarm,
            }
        )

    detail = pd.DataFrame(records)
    valid_leads = detail.loc[detail["valid_alarm"] == 1, "lead_time"].dropna()

    summary = {
        "corr_threshold": corr_threshold,
        "threshold_rule": threshold_rule,
        "n_sims": len(detail),
        "valid_window_alarm_rate": detail["valid_alarm"].mean(),
        "miss_rate": detail["missed"].mean(),
        "lead_mean": valid_leads.mean() if len(valid_leads) > 0 else np.nan,
        "lead_median": valid_leads.median() if len(valid_leads) > 0 else np.nan,
        "lead_q25": valid_leads.quantile(0.25) if len(valid_leads) > 0 else np.nan,
        "lead_q75": valid_leads.quantile(0.75) if len(valid_leads) > 0 else np.nan,
        "prewindow_alarm_rate": detail["prewindow_alarm"].mean(),
        "late_alarm_rate": detail["late_alarm"].mean(),
        "alarm_threshold_mean": detail["alarm_threshold"].mean(),
        "alarm_threshold_std": detail["alarm_threshold"].std(),
    }

    return summary, detail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--persistent-k", type=int, default=3)
    parser.add_argument("--threshold-rule", type=str, default="mean_plus_2std")
    parser.add_argument("--min-baseline-points", type=int, default=10)
    parser.add_argument(
        "--corr-thresholds",
        type=float,
        nargs="+",
        default=[0.4, 0.5, 0.6, 0.7, 0.8],
    )
    args = parser.parse_args()

    data_file = Path(args.data_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(data_file, allow_pickle=True)

    required_keys = ["X_img", "sim_id", "time_idx", "critical_time"]
    for key in required_keys:
        if key not in data:
            raise KeyError(f"Missing required key: {key}")

    X_seq = extract_sequences(data["X_img"])
    sim_id = data["sim_id"]
    time_idx = data["time_idx"]
    critical_time = data["critical_time"]

    print(f"Data file: {data_file}")
    print(f"X_seq shape: {X_seq.shape}")
    print(f"Number of samples: {len(X_seq)}")
    print(f"Correlation thresholds: {args.corr_thresholds}")

    all_summaries = []
    all_details = []

    for corr_th in args.corr_thresholds:
        print(f"\nComputing sync_edge_ratio for corr_threshold = {corr_th}")

        sync_values = np.zeros(len(X_seq), dtype=np.float32)

        for idx in range(len(X_seq)):
            if idx % 1000 == 0:
                print(f"  sample {idx} / {len(X_seq)}")

            sync_values[idx] = compute_sync_edge_ratio(
                seq=X_seq[idx],
                corr_threshold=corr_th,
            )

        df = pd.DataFrame(
            {
                "sim_id": sim_id,
                "time_idx": time_idx,
                "critical_time": critical_time,
                "sync_edge_ratio": sync_values,
            }
        )

        summary, detail = summarize_corr_threshold(
            df=df,
            corr_threshold=corr_th,
            horizon=args.horizon,
            persistent_k=args.persistent_k,
            threshold_rule=args.threshold_rule,
            min_baseline_points=args.min_baseline_points,
        )

        all_summaries.append(summary)
        all_details.append(detail)

    summary_df = pd.DataFrame(all_summaries)
    detail_df = pd.concat(all_details, ignore_index=True)

    summary_path = out_dir / "correlation_threshold_sensitivity_summary.csv"
    detail_path = out_dir / "correlation_threshold_sensitivity_detail.csv"

    summary_df.to_csv(summary_path, index=False)
    detail_df.to_csv(detail_path, index=False)

    print("\n===== Correlation Threshold Sensitivity Summary =====")
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary to: {summary_path}")
    print(f"Saved detail to: {detail_path}")


if __name__ == "__main__":
    main()