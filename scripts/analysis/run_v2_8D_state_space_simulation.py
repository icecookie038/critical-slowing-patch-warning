# -*- coding: utf-8 -*-
"""
v2.8D Minimal Coupled State-Space Simulation

Goal:
    Build and validate a minimal coupled state-space model for patch formation dynamics.

State vector:
    X(t) = [P(t), D(t), B(t), V(t), F(t)]

Models:
    1. diagonal baseline:
        each state is predicted only from itself.

    2. full coupled model:
        all states are predicted from all current states.

Main question:
    Can a coupled low-dimensional state model better reconstruct patch-formation trajectories
    than an independent-state baseline?

Input:
    results/v2_8_patch_formation_dynamics/state_variables_fixed/tables/
        patch_formation_state_timeseries_fixed.csv

Output:
    results/v2_8_patch_formation_dynamics/state_space_simulation/
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


INPUT = Path(
    "results/v2_8_patch_formation_dynamics/state_variables_fixed/tables/"
    "patch_formation_state_timeseries_fixed.csv"
)

OUTDIR = Path("results/v2_8_patch_formation_dynamics/state_space_simulation")
TABLE_DIR = OUTDIR / "tables"
FIG_DIR = OUTDIR / "figures"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


STATES = ["P_state", "D_state", "B_state", "V_state", "F_state"]

RANDOM_SEED = 42
TEST_SIM_FRACTION = 0.30
RIDGE_ALPHA = 1.0

ONSET_FRACTION = 0.30
ONSET_MIN_THRESHOLD = 0.05
ONSET_PERSIST_K = 3


def log(msg: str):
    print(f"[v2.8D] {msg}", flush=True)


def safe_r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 3:
        return np.nan

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot < 1e-12:
        return np.nan

    return float(1.0 - ss_res / ss_tot)


def safe_rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return np.nan

    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def safe_mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return np.nan

    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def make_transition_pairs(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (system, seed, threshold, sim_id), g in df.groupby(
        ["system", "seed", "threshold", "sim_id"]
    ):
        g = g.sort_values("relative_time").copy()

        for s in STATES:
            g[s] = pd.to_numeric(g[s], errors="coerce")
            g[f"{s}_next"] = g[s].shift(-1)

        g["next_relative_time"] = g["relative_time"].shift(-1)

        keep = [
            "system",
            "seed",
            "threshold",
            "sim_id",
            "relative_time",
            "next_relative_time",
        ]
        keep += STATES
        keep += [f"{s}_next" for s in STATES]

        part = g[keep].dropna().copy()

        part = part[
            (part["relative_time"] >= -80)
            & (part["relative_time"] <= -1)
            & (part["next_relative_time"] <= 0)
        ].copy()

        rows.append(part)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def make_sim_key(df: pd.DataFrame) -> pd.Series:
    return (
        df["system"].astype(str)
        + "_"
        + df["seed"].astype(str)
        + "_"
        + df["threshold"].astype(str)
        + "_"
        + df["sim_id"].astype(str)
    )


def split_train_test_by_sim(pairs: pd.DataFrame):
    sim_keys = pairs[["system", "seed", "threshold", "sim_id"]].drop_duplicates().copy()
    sim_keys["key"] = make_sim_key(sim_keys)

    keys = sim_keys["key"].to_numpy()
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(keys)

    n_test = max(1, int(len(keys) * TEST_SIM_FRACTION))
    test_keys = set(keys[:n_test])
    train_keys = set(keys[n_test:])

    pairs = pairs.copy()
    pairs["key"] = make_sim_key(pairs)

    train = pairs[pairs["key"].isin(train_keys)].copy()
    test = pairs[pairs["key"].isin(test_keys)].copy()

    return train, test, train_keys, test_keys


def fit_full_model(train: pd.DataFrame):
    X = train[STATES].to_numpy(dtype=float)
    Y = train[[f"{s}_next" for s in STATES]].to_numpy(dtype=float)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=RIDGE_ALPHA)),
        ]
    )

    model.fit(X, Y)
    return model


def fit_diagonal_model(train: pd.DataFrame):
    models = {}

    for s in STATES:
        X = train[[s]].to_numpy(dtype=float)
        y = train[f"{s}_next"].to_numpy(dtype=float)

        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=RIDGE_ALPHA)),
            ]
        )

        model.fit(X, y)
        models[s] = model

    return models


def predict_full(model, X):
    pred = model.predict(X)
    return np.clip(pred, 0.0, 1.0)


def predict_diagonal(models, X):
    preds = []

    for i, s in enumerate(STATES):
        x_s = X[:, [i]]
        p_s = models[s].predict(x_s)
        preds.append(p_s)

    pred = np.vstack(preds).T
    return np.clip(pred, 0.0, 1.0)


def evaluate_one_step(test: pd.DataFrame, full_model, diag_models, system: str):
    rows = []

    X = test[STATES].to_numpy(dtype=float)
    Y = test[[f"{s}_next" for s in STATES]].to_numpy(dtype=float)

    pred_map = {
        "diagonal_baseline": predict_diagonal(diag_models, X),
        "full_coupled": predict_full(full_model, X),
    }

    for model_name, pred in pred_map.items():
        for i, s in enumerate(STATES):
            rows.append({
                "system": system,
                "model": model_name,
                "evaluation": "one_step",
                "state": s,
                "r2": safe_r2(Y[:, i], pred[:, i]),
                "rmse": safe_rmse(Y[:, i], pred[:, i]),
                "mae": safe_mae(Y[:, i], pred[:, i]),
                "n": int(np.isfinite(Y[:, i]).sum()),
            })

        rows.append({
            "system": system,
            "model": model_name,
            "evaluation": "one_step",
            "state": "ALL",
            "r2": safe_r2(Y.ravel(), pred.ravel()),
            "rmse": safe_rmse(Y.ravel(), pred.ravel()),
            "mae": safe_mae(Y.ravel(), pred.ravel()),
            "n": int(np.isfinite(Y.ravel()).sum()),
        })

    return pd.DataFrame(rows)


def simulate_trajectory_full(model, x0, n_steps):
    xs = [np.asarray(x0, dtype=float)]

    for _ in range(n_steps - 1):
        current = xs[-1].reshape(1, -1)
        nxt = model.predict(current)[0]
        nxt = np.clip(nxt, 0.0, 1.0)
        xs.append(nxt)

    return np.vstack(xs)


def simulate_trajectory_diagonal(models, x0, n_steps):
    xs = [np.asarray(x0, dtype=float)]

    for _ in range(n_steps - 1):
        current = xs[-1]
        nxt = []

        for i, s in enumerate(STATES):
            pred_s = models[s].predict(np.array([[current[i]]]))[0]
            nxt.append(pred_s)

        nxt = np.asarray(nxt, dtype=float)
        nxt = np.clip(nxt, 0.0, 1.0)
        xs.append(nxt)

    return np.vstack(xs)


def recursive_simulation(df_system: pd.DataFrame, test_keys, full_model, diag_models, system: str):
    rows = []
    perf_rows = []

    df_system = df_system.copy()
    df_system["key"] = make_sim_key(df_system)

    test_df = df_system[df_system["key"].isin(test_keys)].copy()
    test_df = test_df[
        (test_df["relative_time"] >= -80)
        & (test_df["relative_time"] <= 0)
    ].copy()

    model_map = {
        "diagonal_baseline": diag_models,
        "full_coupled": full_model,
    }

    for key, g in test_df.groupby("key"):
        g = g.sort_values("relative_time").copy()

        if len(g) < 10:
            continue

        times = g["relative_time"].to_numpy(dtype=float)
        obs = g[STATES].to_numpy(dtype=float)

        x0 = obs[0]
        n_steps = len(g)

        for model_name, model_obj in model_map.items():
            if model_name == "diagonal_baseline":
                sim = simulate_trajectory_diagonal(model_obj, x0, n_steps)
            else:
                sim = simulate_trajectory_full(model_obj, x0, n_steps)

            for i, s in enumerate(STATES):
                perf_rows.append({
                    "system": system,
                    "model": model_name,
                    "evaluation": "recursive",
                    "sim_key": key,
                    "state": s,
                    "r2": safe_r2(obs[:, i], sim[:, i]),
                    "rmse": safe_rmse(obs[:, i], sim[:, i]),
                    "mae": safe_mae(obs[:, i], sim[:, i]),
                    "n": int(len(times)),
                })

            perf_rows.append({
                "system": system,
                "model": model_name,
                "evaluation": "recursive",
                "sim_key": key,
                "state": "ALL",
                "r2": safe_r2(obs.ravel(), sim.ravel()),
                "rmse": safe_rmse(obs.ravel(), sim.ravel()),
                "mae": safe_mae(obs.ravel(), sim.ravel()),
                "n": int(len(times) * len(STATES)),
            })

            for t_idx, t in enumerate(times):
                row_base = {
                    "system": system,
                    "model": model_name,
                    "sim_key": key,
                    "relative_time": t,
                }

                for i, s in enumerate(STATES):
                    row_base[f"obs_{s}"] = obs[t_idx, i]
                    row_base[f"sim_{s}"] = sim[t_idx, i]

                rows.append(row_base)

    sim_df = pd.DataFrame(rows)
    perf_df = pd.DataFrame(perf_rows)

    return sim_df, perf_df


def first_sustained_onset(times, x, threshold):
    times = np.asarray(times, dtype=float)
    x = np.asarray(x, dtype=float)

    mask = np.isfinite(times) & np.isfinite(x)
    times = times[mask]
    x = x[mask]

    if len(x) < ONSET_PERSIST_K:
        return np.nan

    active = x >= threshold

    for i in range(0, len(active) - ONSET_PERSIST_K + 1):
        if active[i:i + ONSET_PERSIST_K].all():
            return float(times[i])

    return np.nan


def onset_and_order_consistency(sim_df: pd.DataFrame):
    rows = []

    if sim_df.empty:
        return pd.DataFrame()

    order_pairs = [
        ("P_state", "D_state"),
        ("P_state", "B_state"),
        ("D_state", "V_state"),
        ("B_state", "V_state"),
        ("P_state", "V_state"),
    ]

    for (system, model, sim_key), g in sim_df.groupby(["system", "model", "sim_key"]):
        g = g.sort_values("relative_time").copy()
        times = g["relative_time"].to_numpy(dtype=float)

        obs_onsets = {}
        sim_onsets = {}

        for s in STATES:
            obs = g[f"obs_{s}"].to_numpy(dtype=float)
            sim = g[f"sim_{s}"].to_numpy(dtype=float)

            obs_peak = np.nanmax(obs) if np.isfinite(obs).any() else np.nan
            threshold = max(ONSET_MIN_THRESHOLD, ONSET_FRACTION * obs_peak) if np.isfinite(obs_peak) else np.nan

            if not np.isfinite(threshold):
                obs_onset = np.nan
                sim_onset = np.nan
            else:
                obs_onset = first_sustained_onset(times, obs, threshold)
                sim_onset = first_sustained_onset(times, sim, threshold)

            obs_onsets[s] = obs_onset
            sim_onsets[s] = sim_onset

            rows.append({
                "system": system,
                "model": model,
                "sim_key": sim_key,
                "type": "state_onset",
                "state": s,
                "obs_onset": obs_onset,
                "sim_onset": sim_onset,
                "onset_error": sim_onset - obs_onset if np.isfinite(obs_onset) and np.isfinite(sim_onset) else np.nan,
                "consistent": np.nan,
            })

        for a, b in order_pairs:
            obs_consistent = (
                np.isfinite(obs_onsets.get(a, np.nan))
                and np.isfinite(obs_onsets.get(b, np.nan))
                and obs_onsets[a] <= obs_onsets[b]
            )

            sim_consistent = (
                np.isfinite(sim_onsets.get(a, np.nan))
                and np.isfinite(sim_onsets.get(b, np.nan))
                and sim_onsets[a] <= sim_onsets[b]
            )

            rows.append({
                "system": system,
                "model": model,
                "sim_key": sim_key,
                "type": "order_consistency",
                "state": f"{a}_before_{b}",
                "obs_onset": np.nan,
                "sim_onset": np.nan,
                "onset_error": np.nan,
                "consistent": int(obs_consistent == sim_consistent),
            })

    return pd.DataFrame(rows)


def summarize_recursive_performance(recursive_perf: pd.DataFrame):
    rows = []

    if recursive_perf.empty:
        return pd.DataFrame()

    for (system, model, state), g in recursive_perf.groupby(["system", "model", "state"]):
        rows.append({
            "system": system,
            "model": model,
            "evaluation": "recursive",
            "state": state,
            "r2_mean": float(g["r2"].mean()),
            "r2_median": float(g["r2"].median()),
            "rmse_mean": float(g["rmse"].mean()),
            "rmse_median": float(g["rmse"].median()),
            "mae_mean": float(g["mae"].mean()),
            "mae_median": float(g["mae"].median()),
            "n_sims": int(g["sim_key"].nunique()),
        })

    return pd.DataFrame(rows)


def summarize_onset_consistency(onset_df: pd.DataFrame):
    rows = []

    if onset_df.empty:
        return pd.DataFrame()

    onset_part = onset_df[onset_df["type"] == "state_onset"].copy()
    order_part = onset_df[onset_df["type"] == "order_consistency"].copy()

    for (system, model, state), g in onset_part.groupby(["system", "model", "state"]):
        valid = g.dropna(subset=["onset_error"])

        rows.append({
            "system": system,
            "model": model,
            "summary_type": "state_onset_error",
            "state_or_order": state,
            "mean_abs_onset_error": float(valid["onset_error"].abs().mean()) if len(valid) else np.nan,
            "median_abs_onset_error": float(valid["onset_error"].abs().median()) if len(valid) else np.nan,
            "n_valid": int(len(valid)),
        })

    for (system, model, state), g in order_part.groupby(["system", "model", "state"]):
        valid = g.dropna(subset=["consistent"])
        rows.append({
            "system": system,
            "model": model,
            "summary_type": "order_consistency",
            "state_or_order": state,
            "mean_abs_onset_error": np.nan,
            "median_abs_onset_error": np.nan,
            "n_valid": int(len(valid)),
            "consistency_rate": float(valid["consistent"].mean()) if len(valid) else np.nan,
        })

    return pd.DataFrame(rows)


def transition_matrix_table(full_model, diag_models, system: str):
    rows = []

    ridge = full_model.named_steps["ridge"]
    full_coef = ridge.coef_

    for target_idx, target in enumerate(STATES):
        for source_idx, source in enumerate(STATES):
            rows.append({
                "system": system,
                "model": "full_coupled",
                "target_next": target,
                "source_current": source,
                "coefficient": float(full_coef[target_idx, source_idx]),
            })

    for target in STATES:
        ridge_d = diag_models[target].named_steps["ridge"]
        rows.append({
            "system": system,
            "model": "diagonal_baseline",
            "target_next": target,
            "source_current": target,
            "coefficient": float(ridge_d.coef_[0]),
        })

    return pd.DataFrame(rows)


def plot_performance_comparison(one_step_perf: pd.DataFrame, recursive_summary: pd.DataFrame):
    combined_rows = []

    one_all = one_step_perf[one_step_perf["state"] == "ALL"].copy()
    for _, r in one_all.iterrows():
        combined_rows.append({
            "system": r["system"],
            "model": r["model"],
            "evaluation": "one_step",
            "r2": r["r2"],
            "rmse": r["rmse"],
        })

    rec_all = recursive_summary[recursive_summary["state"] == "ALL"].copy()
    for _, r in rec_all.iterrows():
        combined_rows.append({
            "system": r["system"],
            "model": r["model"],
            "evaluation": "recursive",
            "r2": r["r2_mean"],
            "rmse": r["rmse_mean"],
        })

    plot_df = pd.DataFrame(combined_rows)

    for system, g_sys in plot_df.groupby("system"):
        for metric in ["r2", "rmse"]:
            g = g_sys.copy()
            labels = [f"{r.evaluation}\n{r.model}" for _, r in g.iterrows()]
            values = g[metric].to_numpy(dtype=float)

            plt.figure(figsize=(8, 5))
            plt.bar(np.arange(len(values)), values)
            plt.axhline(0, linestyle="--", linewidth=1)
            plt.xticks(np.arange(len(values)), labels, rotation=25, ha="right")
            plt.ylabel(metric.upper())
            plt.title(f"State-space model comparison: {system}, {metric.upper()}")
            plt.tight_layout()

            plt.savefig(FIG_DIR / f"state_space_model_comparison_{system}_{metric}.png", dpi=300)
            plt.close()


def plot_recursive_trajectories(sim_df: pd.DataFrame):
    if sim_df.empty:
        return

    for (system, model), g in sim_df.groupby(["system", "model"]):
        agg_rows = []

        for t, s in g.groupby("relative_time"):
            row = {"relative_time": t}

            for state in STATES:
                row[f"obs_{state}"] = float(s[f"obs_{state}"].mean())
                row[f"sim_{state}"] = float(s[f"sim_{state}"].mean())

            agg_rows.append(row)

        agg = pd.DataFrame(agg_rows).sort_values("relative_time")

        plt.figure(figsize=(10, 6))

        for state in STATES:
            plt.plot(agg["relative_time"], agg[f"obs_{state}"], linewidth=2, label=f"obs {state}")
            plt.plot(agg["relative_time"], agg[f"sim_{state}"], linestyle="--", linewidth=2, label=f"sim {state}")

        plt.axvline(0, linestyle="--", linewidth=1)
        plt.ylim(-0.05, 1.05)
        plt.xlabel("Relative time to transition")
        plt.ylabel("State activation")
        plt.title(f"Recursive state-space simulation: {system}, {model}")
        plt.legend(fontsize=7, ncol=2)
        plt.tight_layout()

        plt.savefig(FIG_DIR / f"recursive_trajectory_{system}_{model}.png", dpi=300)
        plt.close()


def plot_transition_matrix(matrix_df: pd.DataFrame):
    for system, g_sys in matrix_df.groupby("system"):
        g = g_sys[g_sys["model"] == "full_coupled"].copy()

        if g.empty:
            continue

        mat = np.zeros((len(STATES), len(STATES)), dtype=float)

        for _, r in g.iterrows():
            i = STATES.index(r["target_next"])
            j = STATES.index(r["source_current"])
            mat[i, j] = r["coefficient"]

        plt.figure(figsize=(6, 5))
        plt.imshow(mat, aspect="auto")
        plt.colorbar(label="Coefficient")
        plt.xticks(np.arange(len(STATES)), STATES, rotation=45, ha="right")
        plt.yticks(np.arange(len(STATES)), [f"{s}_next" for s in STATES])
        plt.title(f"Full coupled transition matrix: {system}")
        plt.tight_layout()

        plt.savefig(FIG_DIR / f"transition_matrix_full_coupled_{system}.png", dpi=300)
        plt.close()


def main():
    if not INPUT.exists():
        raise SystemExit(f"Missing input file: {INPUT}")

    df = pd.read_csv(INPUT)

    required = ["system", "seed", "threshold", "sim_id", "relative_time"] + STATES
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise SystemExit(f"Missing columns: {missing}")

    for c in ["sim_id", "relative_time"] + STATES:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[
        (df["relative_time"] >= -80)
        & (df["relative_time"] <= 0)
    ].copy()

    all_one_step = []
    all_recursive = []
    all_recursive_summary = []
    all_matrix = []
    all_onset = []
    all_onset_summary = []

    for system, df_sys in df.groupby("system"):
        log(f"processing system={system}")

        pairs = make_transition_pairs(df_sys)

        if pairs.empty:
            log(f"skip {system}: no transition pairs")
            continue

        train, test, train_keys, test_keys = split_train_test_by_sim(pairs)

        full_model = fit_full_model(train)
        diag_models = fit_diagonal_model(train)

        one_step_perf = evaluate_one_step(test, full_model, diag_models, system)
        sim_df, recursive_perf = recursive_simulation(df_sys, test_keys, full_model, diag_models, system)
        recursive_summary = summarize_recursive_performance(recursive_perf)
        matrix_df = transition_matrix_table(full_model, diag_models, system)
        onset_df = onset_and_order_consistency(sim_df)
        onset_summary = summarize_onset_consistency(onset_df)

        all_one_step.append(one_step_perf)
        all_recursive.append(sim_df)
        all_recursive_summary.append(recursive_summary)
        all_matrix.append(matrix_df)
        all_onset.append(onset_df)
        all_onset_summary.append(onset_summary)

    one_step_all = pd.concat(all_one_step, ignore_index=True) if all_one_step else pd.DataFrame()
    recursive_all = pd.concat(all_recursive, ignore_index=True) if all_recursive else pd.DataFrame()
    recursive_summary_all = pd.concat(all_recursive_summary, ignore_index=True) if all_recursive_summary else pd.DataFrame()
    matrix_all = pd.concat(all_matrix, ignore_index=True) if all_matrix else pd.DataFrame()
    onset_all = pd.concat(all_onset, ignore_index=True) if all_onset else pd.DataFrame()
    onset_summary_all = pd.concat(all_onset_summary, ignore_index=True) if all_onset_summary else pd.DataFrame()

    one_step_all.to_csv(TABLE_DIR / "state_space_one_step_performance.csv", index=False, encoding="utf-8-sig")
    recursive_all.to_csv(TABLE_DIR / "state_space_recursive_simulations.csv", index=False, encoding="utf-8-sig")
    recursive_summary_all.to_csv(TABLE_DIR / "state_space_recursive_performance_summary.csv", index=False, encoding="utf-8-sig")
    matrix_all.to_csv(TABLE_DIR / "state_space_transition_matrices.csv", index=False, encoding="utf-8-sig")
    onset_all.to_csv(TABLE_DIR / "state_space_onset_by_simulation.csv", index=False, encoding="utf-8-sig")
    onset_summary_all.to_csv(TABLE_DIR / "state_space_onset_consistency_summary.csv", index=False, encoding="utf-8-sig")

    plot_performance_comparison(one_step_all, recursive_summary_all)
    plot_recursive_trajectories(recursive_all)
    plot_transition_matrix(matrix_all)

    log("state-space simulation finished")
    log(f"output directory: {OUTDIR}")

    print("\n=== One-step performance ===")
    print(one_step_all.to_string(index=False))

    print("\n=== Recursive performance summary ===")
    print(recursive_summary_all.to_string(index=False))

    print("\n=== Onset and order consistency summary ===")
    print(onset_summary_all.to_string(index=False))

    print("\n=== Output files ===")
    for p in sorted(OUTDIR.rglob("*")):
        print(p)


if __name__ == "__main__":
    main()
