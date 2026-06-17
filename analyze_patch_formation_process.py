import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PREPATCH_GROUPS = {
    "Z_sync": [
        "local_neighbor_corr_mean",
        "local_neighbor_corr_max",
        "sync_edge_ratio",
    ],
    "Z_conn": [
        "moran_i",
        "geary_c",
        "high_state_component_ratio",
    ],
    "Z_rigid": [
        "gradient_entropy",
        "boundary_sharpness",
        "gradient_top10_mean",
    ],
    "Z_mode": [
        "svd_mode1_energy_ratio",
        "svd_spectral_gap",
        "svd_mode1_ac1",
        "dominant_mode_stability",
        "dominant_mode_localization",
    ],
}

# Indicators whose lower values usually indicate stronger organization.
# We multiply them by -1 so that larger values consistently mean stronger prepatch organization.
INVERSE_DIRECTION_FEATURES = {
    "geary_c",
    "gradient_entropy",
}


def decode_names(arr):
    names = []
    for x in arr:
        if isinstance(x, bytes):
            names.append(x.decode("utf-8"))
        else:
            names.append(str(x))
    return names


def find_name_key(data, candidates):
    for key in candidates:
        if key in data:
            return key
    return None


def load_feature_names(data):
    candidates = [
        "prepatch_feature_names",
        "feature_names",
        "patch_feature_names",
        "pwsi_feature_names",
    ]
    key = find_name_key(data, candidates)
    if key is None:
        return None
    return decode_names(data[key])


def load_prepatch_matrix(data):
    if "X_prepatch" in data:
        return data["X_prepatch"]

    if "X_patch" in data:
        X = data["X_patch"]
        if X.ndim == 3:
            return X[:, -1, :]
        if X.ndim == 2:
            return X

    if "X" in data:
        X = data["X"]
        if X.ndim == 3:
            return X[:, -1, :]
        if X.ndim == 2:
            return X

    raise KeyError("Cannot find prepatch feature matrix. Expected X_prepatch, X_patch, or X.")


def load_visible_patch_metric(raw_data, preferred_metric):
    if "X_patch" not in raw_data:
        raise KeyError("Raw data must contain X_patch for visible patch metric.")

    X_patch = raw_data["X_patch"]
    if X_patch.ndim != 3:
        raise ValueError(f"Expected X_patch shape (n_samples, seq_len, n_features), got {X_patch.shape}")

    names = load_feature_names(raw_data)
    if names is None:
        raise KeyError("Cannot find patch feature names in raw data.")

    lowered = [n.lower() for n in names]
    preferred = preferred_metric.lower()

    if preferred in lowered:
        idx = lowered.index(preferred)
    else:
        candidate_names = [
            "largest_degraded_patch_ratio",
            "largest_patch_ratio",
            "largest_patch_to_degraded_area",
            "spatial_aggregation",
        ]
        idx = None
        for c in candidate_names:
            if c in lowered:
                idx = lowered.index(c)
                preferred_metric = names[idx]
                break

        if idx is None:
            raise ValueError(
                f"Cannot find visible patch metric. "
                f"preferred={preferred_metric}, available={names}"
            )

    values = X_patch[:, -1, idx].astype(float)
    return values, preferred_metric


def load_pwsi_equal(pwsi_data):
    names = load_feature_names(pwsi_data)

    if "X_patch" in pwsi_data:
        X = pwsi_data["X_patch"]
    elif "X_pwsi" in pwsi_data:
        X = pwsi_data["X_pwsi"]
    elif "X" in pwsi_data:
        X = pwsi_data["X"]
    else:
        raise KeyError("Cannot find PWSI feature matrix. Expected X_patch, X_pwsi, or X.")

    if X.ndim == 3:
        X_last = X[:, -1, :]
    elif X.ndim == 2:
        X_last = X
    else:
        raise ValueError(f"Unsupported PWSI matrix shape: {X.shape}")

    feature_dim = X_last.shape[-1]

    if feature_dim == 1:
        return X_last[:, 0].astype(float)

    if names is not None:
        lowered = [n.lower() for n in names]
        if "pwsi_equal" in lowered:
            return X_last[:, lowered.index("pwsi_equal")].astype(float)

        for i, n in enumerate(lowered):
            if "pwsi_equal" in n:
                return X_last[:, i].astype(float)

    if feature_dim == 6:
        # Expected order:
        # Z_sync, Z_conn, Z_rigid, Z_mode, PWSI_equal, PWSI_importance
        return X_last[:, 4].astype(float)

    raise ValueError(f"Cannot identify PWSI_equal. feature_dim={feature_dim}, names={names}")


def build_base_dataframe(prepatch_data, raw_data):
    required = ["sim_id", "time_idx", "critical_time"]
    for key in required:
        if key not in prepatch_data:
            raise KeyError(f"Missing required key in prepatch data: {key}")

    df = pd.DataFrame(
        {
            "sim_id": prepatch_data["sim_id"],
            "time_idx": prepatch_data["time_idx"],
            "critical_time": prepatch_data["critical_time"],
        }
    )

    df["relative_time"] = df["time_idx"] - df["critical_time"]

    if "y_risk" in prepatch_data:
        df["y_risk"] = prepatch_data["y_risk"]

    visible_values, visible_name = load_visible_patch_metric(
        raw_data=raw_data,
        preferred_metric="largest_degraded_patch_ratio",
    )
    df["visible_patch_metric_raw"] = visible_values

    return df, visible_name


def add_prepatch_features(df, prepatch_data):
    X = load_prepatch_matrix(prepatch_data)
    names = load_feature_names(prepatch_data)

    if names is None:
        default_names = [
            "local_neighbor_corr_mean",
            "local_neighbor_corr_max",
            "sync_edge_ratio",
            "moran_i",
            "geary_c",
            "high_state_component_ratio",
            "gradient_entropy",
            "boundary_sharpness",
            "gradient_top10_mean",
            "svd_mode1_energy_ratio",
            "svd_spectral_gap",
            "svd_mode1_ac1",
            "dominant_mode_stability",
            "dominant_mode_localization",
        ]
        if X.shape[1] == len(default_names):
            names = default_names
        else:
            names = [f"feature_{i}" for i in range(X.shape[1])]

    if X.shape[0] != len(df):
        raise ValueError(f"Feature rows {X.shape[0]} do not match dataframe rows {len(df)}.")

    for i, name in enumerate(names):
        clean_name = str(name)
        values = X[:, i].astype(float)

        if clean_name in INVERSE_DIRECTION_FEATURES:
            values = -values

        df[clean_name] = values

    return df, names


def add_pwsi_if_available(df, pwsi_file):
    if pwsi_file is None:
        return df

    pwsi_file = Path(pwsi_file)
    if not pwsi_file.exists():
        print(f"[Warning] PWSI file not found: {pwsi_file}. PWSI will be computed from groups.")
        return df

    pwsi_data = np.load(pwsi_file, allow_pickle=True)

    required = ["sim_id", "time_idx"]
    for key in required:
        if key not in pwsi_data:
            print(f"[Warning] PWSI data missing {key}. PWSI will be computed from groups.")
            return df

    pwsi_values = load_pwsi_equal(pwsi_data)

    pwsi_df = pd.DataFrame(
        {
            "sim_id": pwsi_data["sim_id"],
            "time_idx": pwsi_data["time_idx"],
            "PWSI_equal_loaded": pwsi_values,
        }
    )

    df = df.merge(pwsi_df, on=["sim_id", "time_idx"], how="left")
    return df


def standardize_columns_by_baseline(df, cols, baseline_start, baseline_end):
    out = df.copy()
    baseline_mask = (out["relative_time"] >= baseline_start) & (out["relative_time"] <= baseline_end)

    if baseline_mask.sum() < 20:
        baseline_mask = out["relative_time"] <= baseline_end

    if baseline_mask.sum() < 20:
        baseline_mask = np.ones(len(out), dtype=bool)

    for col in cols:
        values = out[col].astype(float)
        base = values[baseline_mask]
        mean = float(np.nanmean(base))
        std = float(np.nanstd(base))

        if not np.isfinite(std) or std < 1e-8:
            std = 1.0

        out[col + "_z"] = (values - mean) / std

    return out


def compute_group_scores(df, feature_names):
    out = df.copy()

    for group_name, group_features in PREPATCH_GROUPS.items():
        z_cols = []
        for f in group_features:
            if f in out.columns:
                z_col = f + "_z"
                if z_col in out.columns:
                    z_cols.append(z_col)

        if len(z_cols) == 0:
            print(f"[Warning] No features found for {group_name}.")
            out[group_name] = np.nan
        else:
            out[group_name] = out[z_cols].mean(axis=1)

    if "PWSI_equal_loaded" in out.columns and out["PWSI_equal_loaded"].notna().sum() > 0:
        # Standardize loaded PWSI for trajectory visualization.
        out = standardize_columns_by_baseline(
            out,
            cols=["PWSI_equal_loaded"],
            baseline_start=-60,
            baseline_end=-40,
        )
        out["PWSI_equal"] = out["PWSI_equal_loaded_z"]
    else:
        group_cols = ["Z_sync", "Z_conn", "Z_rigid", "Z_mode"]
        available = [c for c in group_cols if c in out.columns]
        out["PWSI_equal"] = out[available].mean(axis=1)

    return out


def build_event_aligned_trajectory(df, indicators, align_start, align_end):
    sub = df[(df["relative_time"] >= align_start) & (df["relative_time"] <= align_end)].copy()

    rows = []
    for rel_t, g in sub.groupby("relative_time"):
        row = {
            "relative_time": rel_t,
            "n_samples": len(g),
            "n_sims": g["sim_id"].nunique(),
        }

        for ind in indicators:
            values = g[ind].astype(float)
            row[f"{ind}_mean"] = float(np.nanmean(values))
            row[f"{ind}_std"] = float(np.nanstd(values))
            row[f"{ind}_sem"] = float(np.nanstd(values) / np.sqrt(np.isfinite(values).sum())) if np.isfinite(values).sum() > 0 else np.nan

        rows.append(row)

    traj = pd.DataFrame(rows).sort_values("relative_time")
    return traj


def first_rise_for_series(times, values, baseline_mask, threshold_sigma, persistent_k):
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=int)

    baseline_values = values[baseline_mask]
    baseline_values = baseline_values[np.isfinite(baseline_values)]

    if len(baseline_values) < 5:
        return np.nan, np.nan

    mean = float(np.nanmean(baseline_values))
    std = float(np.nanstd(baseline_values))
    if not np.isfinite(std) or std < 1e-8:
        std = 1.0

    threshold = mean + threshold_sigma * std

    flags = values >= threshold

    for i in range(0, len(flags) - persistent_k + 1):
        if times[i] >= 0:
            continue

        if np.all(flags[i:i + persistent_k]):
            return int(times[i]), threshold

    return np.nan, threshold


def compute_first_rise_times(df, indicators, baseline_start, baseline_end, threshold_sigma, persistent_k):
    records = []

    for sim_id, g in df.groupby("sim_id"):
        g = g.sort_values("relative_time")
        times = g["relative_time"].to_numpy()

        baseline_mask = (times >= baseline_start) & (times <= baseline_end)
        if baseline_mask.sum() < 5:
            baseline_mask = times <= baseline_end

        for ind in indicators:
            values = g[ind].to_numpy()
            first_rise_time, threshold = first_rise_for_series(
                times=times,
                values=values,
                baseline_mask=baseline_mask,
                threshold_sigma=threshold_sigma,
                persistent_k=persistent_k,
            )

            records.append(
                {
                    "sim_id": sim_id,
                    "indicator": ind,
                    "first_rise_relative_time": first_rise_time,
                    "lead_to_transition": -first_rise_time if np.isfinite(first_rise_time) else np.nan,
                    "threshold": threshold,
                }
            )

    detail = pd.DataFrame(records)

    summary_rows = []
    for ind, g in detail.groupby("indicator"):
        leads = g["lead_to_transition"].dropna()

        summary_rows.append(
            {
                "indicator": ind,
                "n_sims": g["sim_id"].nunique(),
                "detection_rate": float(g["lead_to_transition"].notna().mean()),
                "lead_mean": float(leads.mean()) if len(leads) > 0 else np.nan,
                "lead_median": float(leads.median()) if len(leads) > 0 else np.nan,
                "lead_q25": float(leads.quantile(0.25)) if len(leads) > 0 else np.nan,
                "lead_q75": float(leads.quantile(0.75)) if len(leads) > 0 else np.nan,
                "first_rise_time_mean": float(g["first_rise_relative_time"].dropna().mean()) if g["first_rise_relative_time"].notna().sum() > 0 else np.nan,
                "first_rise_time_median": float(g["first_rise_relative_time"].dropna().median()) if g["first_rise_relative_time"].notna().sum() > 0 else np.nan,
            }
        )

    summary = pd.DataFrame(summary_rows)
    return summary, detail


def plot_event_aligned_trajectories(traj, indicators, out_path):
    plt.figure(figsize=(10, 6))

    for ind in indicators:
        mean_col = f"{ind}_mean"
        sem_col = f"{ind}_sem"

        if mean_col not in traj.columns:
            continue

        x = traj["relative_time"].to_numpy()
        y = traj[mean_col].to_numpy()
        sem = traj[sem_col].to_numpy() if sem_col in traj.columns else np.zeros_like(y)

        plt.plot(x, y, label=ind)
        plt.fill_between(x, y - sem, y + sem, alpha=0.15)

    plt.axvline(0, linestyle="--", linewidth=1)
    plt.xlabel("Time to critical transition")
    plt.ylabel("Baseline-standardized indicator value")
    plt.title("Event-aligned prepatch dynamics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_first_rise_summary(summary, out_path):
    order = [
        "Z_sync",
        "Z_conn",
        "Z_rigid",
        "Z_mode",
        "PWSI_equal",
        "visible_patch_metric_z",
    ]

    plot_df = summary.copy()
    plot_df = plot_df[plot_df["indicator"].isin(order)].copy()
    plot_df["indicator"] = pd.Categorical(plot_df["indicator"], categories=order, ordered=True)
    plot_df = plot_df.sort_values("indicator")

    plt.figure(figsize=(9, 5))
    plt.bar(plot_df["indicator"].astype(str), plot_df["lead_median"])
    plt.ylabel("Median lead to transition")
    plt.xlabel("Indicator")
    plt.title("First-rise time comparison")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def extract_sequences(X_img):
    if X_img.ndim == 5:
        return X_img[:, :, 0, :, :]
    if X_img.ndim == 4:
        return X_img
    raise ValueError(f"Unsupported X_img shape: {X_img.shape}")


def plot_typical_simulation(raw_data, df, out_path, relative_times):
    if "X_img" not in raw_data:
        print("[Warning] raw_data has no X_img. Skip typical simulation plot.")
        return

    X_seq = extract_sequences(raw_data["X_img"])

    raw_index_df = pd.DataFrame(
        {
            "row_index": np.arange(len(raw_data["sim_id"])),
            "sim_id": raw_data["sim_id"],
            "time_idx": raw_data["time_idx"],
            "critical_time": raw_data["critical_time"],
        }
    )
    raw_index_df["relative_time"] = raw_index_df["time_idx"] - raw_index_df["critical_time"]

    # Choose a simulation with critical time close to median and enough requested time points.
    sim_summary = raw_index_df.groupby("sim_id")["critical_time"].first().reset_index()
    median_ct = sim_summary["critical_time"].median()
    sim_summary["dist_to_median"] = (sim_summary["critical_time"] - median_ct).abs()
    candidates = sim_summary.sort_values("dist_to_median")["sim_id"].tolist()

    selected = None
    selected_rows = []

    for sim_id in candidates:
        g = raw_index_df[raw_index_df["sim_id"] == sim_id].copy()
        rows = []

        for rt in relative_times:
            gg = g.iloc[(g["relative_time"] - rt).abs().argsort()[:1]]
            if len(gg) == 1:
                rows.append(gg.iloc[0])

        if len(rows) == len(relative_times):
            selected = sim_id
            selected_rows = rows
            break

    if selected is None:
        print("[Warning] Could not select typical simulation.")
        return

    n = len(selected_rows)
    plt.figure(figsize=(2.6 * n, 3.0))

    for i, row in enumerate(selected_rows):
        sample_idx = int(row["row_index"])
        img = X_seq[sample_idx, -1, :, :]

        ax = plt.subplot(1, n, i + 1)
        ax.imshow(img)
        ax.set_title(f"t = {int(row['relative_time'])}")
        ax.axis("off")

    plt.suptitle(f"Typical vegetation CA spatial evolution, sim_id={selected}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepatch-file", type=str, required=True)
    parser.add_argument("--raw-data-file", type=str, required=True)
    parser.add_argument("--pwsi-file", type=str, default=None)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--align-start", type=int, default=-60)
    parser.add_argument("--align-end", type=int, default=0)
    parser.add_argument("--baseline-start", type=int, default=-60)
    parser.add_argument("--baseline-end", type=int, default=-40)
    parser.add_argument("--threshold-sigma", type=float, default=2.0)
    parser.add_argument("--persistent-k", type=int, default=2)
    args = parser.parse_args()

    prepatch_file = Path(args.prepatch_file)
    raw_data_file = Path(args.raw_data_file)
    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    prepatch_data = np.load(prepatch_file, allow_pickle=True)
    raw_data = np.load(raw_data_file, allow_pickle=True)

    df, visible_name = build_base_dataframe(prepatch_data, raw_data)
    df, prepatch_names = add_prepatch_features(df, prepatch_data)
    df = add_pwsi_if_available(df, args.pwsi_file)

    # Standardize individual prepatch features and visible patch metric.
    prepatch_feature_cols = [n for n in prepatch_names if n in df.columns]
    standardize_cols = prepatch_feature_cols + ["visible_patch_metric_raw"]

    df = standardize_columns_by_baseline(
        df,
        cols=standardize_cols,
        baseline_start=args.baseline_start,
        baseline_end=args.baseline_end,
    )

    df["visible_patch_metric_z"] = df["visible_patch_metric_raw_z"]

    df = compute_group_scores(df, prepatch_names)

    indicators = [
        "Z_sync",
        "Z_conn",
        "Z_rigid",
        "Z_mode",
        "PWSI_equal",
        "visible_patch_metric_z",
    ]

    full_df_path = out_dir / "patch_formation_indicator_table.csv"
    df.to_csv(full_df_path, index=False)

    trajectory = build_event_aligned_trajectory(
        df=df,
        indicators=indicators,
        align_start=args.align_start,
        align_end=args.align_end,
    )

    trajectory_path = out_dir / "event_aligned_indicator_trajectory.csv"
    trajectory.to_csv(trajectory_path, index=False)

    first_rise_summary, first_rise_detail = compute_first_rise_times(
        df=df,
        indicators=indicators,
        baseline_start=args.baseline_start,
        baseline_end=args.baseline_end,
        threshold_sigma=args.threshold_sigma,
        persistent_k=args.persistent_k,
    )

    first_rise_summary_path = out_dir / "first_rise_time_summary.csv"
    first_rise_detail_path = out_dir / "first_rise_time_detail.csv"

    first_rise_summary.to_csv(first_rise_summary_path, index=False)
    first_rise_detail.to_csv(first_rise_detail_path, index=False)

    plot_event_aligned_trajectories(
        traj=trajectory,
        indicators=indicators,
        out_path=fig_dir / "event_aligned_indicator_trajectories.png",
    )

    plot_first_rise_summary(
        summary=first_rise_summary,
        out_path=fig_dir / "first_rise_time_summary.png",
    )

    plot_typical_simulation(
        raw_data=raw_data,
        df=df,
        out_path=fig_dir / "typical_spatial_evolution.png",
        relative_times=[-50, -40, -30, -20, -10, 0],
    )

    print("\n===== Patch Formation Process Analysis Completed =====")
    print(f"Prepatch file: {prepatch_file}")
    print(f"Raw data file: {raw_data_file}")
    print(f"PWSI file: {args.pwsi_file}")
    print(f"Visible patch metric: {visible_name}")
    print(f"Output table: {full_df_path}")
    print(f"Trajectory table: {trajectory_path}")
    print(f"First-rise summary: {first_rise_summary_path}")
    print(f"First-rise detail: {first_rise_detail_path}")
    print(f"Figures saved to: {fig_dir}")

    print("\n===== First-rise summary =====")
    print(first_rise_summary.to_string(index=False))


if __name__ == "__main__":
    main()