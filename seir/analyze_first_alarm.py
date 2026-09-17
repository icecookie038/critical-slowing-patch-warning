from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def analyze_first_alarm(
    pred_file: Path,
    out_file: Path | None = None,
    persistent_k: int = 1,
    horizon: float = 30.0,
):
    df = pd.read_csv(pred_file)

    required = [
        "sim_id",
        "time_idx",
        "critical_time",
        "y_remaining",
        "y_true",
        "risk_prob",
        "threshold",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["sim_id"] = df["sim_id"].astype(str)
    df["time_idx"] = df["time_idx"].astype(float)
    df["critical_time"] = df["critical_time"].astype(float)
    df["y_remaining"] = df["y_remaining"].astype(float)
    df["y_true"] = df["y_true"].astype(int)
    df["risk_prob"] = df["risk_prob"].astype(float)

    threshold = float(df["threshold"].iloc[0])
    rows = []

    for sim_id, g in df.groupby("sim_id"):
        g = g.sort_values("time_idx").reset_index(drop=True)

        critical_time = float(g["critical_time"].iloc[0])
        has_positive_window = bool((g["y_true"] == 1).any())

        g["alarm"] = (g["risk_prob"] >= threshold).astype(int)
        g["lead_time"] = critical_time - g["time_idx"]

        # any first alarm: first alarm anywhere in the available trajectory
        any_first_alarm_time = np.nan
        any_first_alarm_prob = np.nan
        any_first_alarm_lead = np.nan

        alarm_arr = g["alarm"].values

        def find_first_persistent(mask_arr):
            if persistent_k <= 1:
                idx = np.where(mask_arr)[0]
                return None if len(idx) == 0 else int(idx[0])

            for i in range(0, len(mask_arr) - persistent_k + 1):
                if mask_arr[i:i + persistent_k].sum() == persistent_k:
                    return int(i)
            return None

        any_mask = alarm_arr == 1
        any_i = find_first_persistent(any_mask)

        if any_i is not None:
            any_first_alarm_time = float(g.loc[any_i, "time_idx"])
            any_first_alarm_prob = float(g.loc[any_i, "risk_prob"])
            any_first_alarm_lead = float(g.loc[any_i, "lead_time"])

        # valid alarm: alarm inside the defined warning horizon
        valid_mask = (
            (g["alarm"].values == 1)
            & (g["lead_time"].values > 0)
            & (g["lead_time"].values <= horizon)
        )
        valid_i = find_first_persistent(valid_mask.astype(int))

        valid_first_alarm_time = np.nan
        valid_first_alarm_prob = np.nan
        valid_first_alarm_lead = np.nan

        if valid_i is not None:
            valid_first_alarm_time = float(g.loc[valid_i, "time_idx"])
            valid_first_alarm_prob = float(g.loc[valid_i, "risk_prob"])
            valid_first_alarm_lead = float(g.loc[valid_i, "lead_time"])

        # pre-window alarms: alarms earlier than the official horizon
        prewindow_mask = (
            (g["alarm"].values == 1)
            & (g["lead_time"].values > horizon)
        )
        prewindow_i = find_first_persistent(prewindow_mask.astype(int))

        prewindow_alarm = prewindow_i is not None

        # late alarms: at or after critical time
        late_mask = (
            (g["alarm"].values == 1)
            & (g["lead_time"].values <= 0)
        )
        late_i = find_first_persistent(late_mask.astype(int))

        late_alarm = late_i is not None

        rows.append({
            "sim_id": sim_id,
            "critical_time": critical_time,
            "has_positive_window": has_positive_window,

            "any_alarmed": any_i is not None,
            "any_first_alarm_time": any_first_alarm_time,
            "any_first_alarm_prob": any_first_alarm_prob,
            "any_first_alarm_lead": any_first_alarm_lead,

            "valid_window_alarmed": valid_i is not None,
            "valid_first_alarm_time": valid_first_alarm_time,
            "valid_first_alarm_prob": valid_first_alarm_prob,
            "valid_first_alarm_lead": valid_first_alarm_lead,

            "prewindow_alarm": prewindow_alarm,
            "late_alarm": late_alarm,

            "threshold": threshold,
            "horizon": horizon,
            "n_windows": len(g),
            "persistent_k": persistent_k,
        })

    alarm_df = pd.DataFrame(rows)

    event_df = alarm_df[alarm_df["has_positive_window"]].copy()

    any_alarm_df = event_df[event_df["any_alarmed"]].copy()
    valid_alarm_df = event_df[event_df["valid_window_alarmed"]].copy()
    prewindow_df = event_df[event_df["prewindow_alarm"]].copy()
    late_df = event_df[event_df["late_alarm"]].copy()

    summary = {
        "pred_file": str(pred_file),
        "threshold": threshold,
        "horizon": horizon,
        "persistent_k": persistent_k,

        "n_simulations": int(len(alarm_df)),
        "n_event_simulations": int(len(event_df)),

        "n_any_alarm_event_simulations": int(len(any_alarm_df)),
        "any_alarm_rate": float(len(any_alarm_df) / len(event_df)) if len(event_df) else np.nan,
        "any_first_alarm_lead_mean": float(any_alarm_df["any_first_alarm_lead"].mean()) if len(any_alarm_df) else np.nan,
        "any_first_alarm_lead_median": float(any_alarm_df["any_first_alarm_lead"].median()) if len(any_alarm_df) else np.nan,
        "any_first_alarm_lead_q25": float(any_alarm_df["any_first_alarm_lead"].quantile(0.25)) if len(any_alarm_df) else np.nan,
        "any_first_alarm_lead_q75": float(any_alarm_df["any_first_alarm_lead"].quantile(0.75)) if len(any_alarm_df) else np.nan,

        "n_valid_window_alarm_event_simulations": int(len(valid_alarm_df)),
        "valid_window_alarm_rate": float(len(valid_alarm_df) / len(event_df)) if len(event_df) else np.nan,
        "valid_window_miss_rate": float(1 - len(valid_alarm_df) / len(event_df)) if len(event_df) else np.nan,
        "valid_first_alarm_lead_mean": float(valid_alarm_df["valid_first_alarm_lead"].mean()) if len(valid_alarm_df) else np.nan,
        "valid_first_alarm_lead_median": float(valid_alarm_df["valid_first_alarm_lead"].median()) if len(valid_alarm_df) else np.nan,
        "valid_first_alarm_lead_q25": float(valid_alarm_df["valid_first_alarm_lead"].quantile(0.25)) if len(valid_alarm_df) else np.nan,
        "valid_first_alarm_lead_q75": float(valid_alarm_df["valid_first_alarm_lead"].quantile(0.75)) if len(valid_alarm_df) else np.nan,

        "n_prewindow_alarm_event_simulations": int(len(prewindow_df)),
        "prewindow_alarm_rate": float(len(prewindow_df) / len(event_df)) if len(event_df) else np.nan,

        "n_late_alarm_event_simulations": int(len(late_df)),
        "late_alarm_rate": float(len(late_df) / len(event_df)) if len(event_df) else np.nan,
    }

    summary_df = pd.DataFrame([summary])

    if out_file is None:
        out_file = pred_file.with_name(
            pred_file.stem.replace(
                "_predictions",
                f"_first_alarm_k{persistent_k}_h{int(horizon)}"
            ) + ".csv"
        )

    summary_file = out_file.with_name(out_file.stem + "_summary.csv")

    alarm_df.to_csv(out_file, index=False)
    summary_df.to_csv(summary_file, index=False)

    print("===== First alarm summary =====")
    print(summary_df.to_string(index=False))
    print()
    print("Saved:")
    print(out_file)
    print(summary_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-file", type=str, required=True)
    parser.add_argument("--out-file", type=str, default=None)
    parser.add_argument("--persistent-k", type=int, default=1)
    parser.add_argument("--horizon", type=float, default=30.0)
    args = parser.parse_args()

    analyze_first_alarm(
        pred_file=Path(args.pred_file),
        out_file=None if args.out_file is None else Path(args.out_file),
        persistent_k=args.persistent_k,
        horizon=args.horizon,
    )


if __name__ == "__main__":
    main()