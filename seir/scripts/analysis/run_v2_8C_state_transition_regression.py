# -*- coding: utf-8 -*-
"""
v2.8C State Transition Regression

Goal:
    Build interpretable low-dimensional state transition models for patch formation dynamics.

Main models:
    D(t+h) ~ P(t) + D(t)
    B(t+h) ~ P(t) + B(t)
    V(t+h) ~ P(t) + D(t) + B(t) + V(t)
    F(t+h) ~ P(t) + D(t) + B(t) + V(t) + F(t)

Interpretation:
    If P(t) positively explains D(t+h) or B(t+h),
    prepatch organization is associated with later restructuring.

    If D(t) / B(t) positively explain V(t+h),
    dynamic restructuring and boundary complexity are associated with visible patch manifestation.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


INPUT = Path(
    "results/v2_8_patch_formation_dynamics/state_variables_fixed/tables/"
    "patch_formation_state_timeseries_fixed.csv"
)

OUTDIR = Path("results/v2_8_patch_formation_dynamics/state_transition_regression")
TABLE_DIR = OUTDIR / "tables"
FIG_DIR = OUTDIR / "figures"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


HORIZONS = [1, 5, 10]

MODELS = [
    {
        "model_name": "D_next_from_P_D",
        "target": "D_state",
        "features": ["P_state", "D_state"],
    },
    {
        "model_name": "B_next_from_P_B",
        "target": "B_state",
        "features": ["P_state", "B_state"],
    },
    {
        "model_name": "V_next_from_P_D_B_V",
        "target": "V_state",
        "features": ["P_state", "D_state", "B_state", "V_state"],
    },
    {
        "model_name": "F_next_from_P_D_B_V_F",
        "target": "F_state",
        "features": ["P_state", "D_state", "B_state", "V_state", "F_state"],
    },
]

RANDOM_SEED = 42
TEST_SIM_FRACTION = 0.30


def log(msg: str):
    print(f"[v2.8C] {msg}", flush=True)


def make_transition_dataset(df: pd.DataFrame, features, target, horizon: int) -> pd.DataFrame:
    rows = []

    for (system, seed, threshold, sim_id), g in df.groupby(
        ["system", "seed", "threshold", "sim_id"]
    ):
        g = g.sort_values("relative_time").copy()

        for c in features + [target]:
            g[c] = pd.to_numeric(g[c], errors="coerce")

        g["target_future"] = g[target].shift(-horizon)
        g["target_future_time"] = g["relative_time"].shift(-horizon)

        keep = ["system", "seed", "threshold", "sim_id", "relative_time", "target_future_time"]
        keep += features + ["target_future"]

        part = g[keep].copy()
        part = part.dropna(subset=features + ["target_future"])

        # Only use pre-transition prediction pairs.
        part = part[
            (part["relative_time"] >= -80)
            & (part["relative_time"] <= -1)
            & (part["target_future_time"] <= 0)
        ].copy()

        rows.append(part)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def split_by_sim(data: pd.DataFrame):
    sim_keys = data[["system", "seed", "threshold", "sim_id"]].drop_duplicates().copy()
    sim_keys["key"] = (
        sim_keys["system"].astype(str)
        + "_"
        + sim_keys["seed"].astype(str)
        + "_"
        + sim_keys["threshold"].astype(str)
        + "_"
        + sim_keys["sim_id"].astype(str)
    )

    rng = np.random.default_rng(RANDOM_SEED)
    keys = sim_keys["key"].to_numpy()
    rng.shuffle(keys)

    n_test = max(1, int(len(keys) * TEST_SIM_FRACTION))
    test_keys = set(keys[:n_test])

    data = data.copy()
    data["key"] = (
        data["system"].astype(str)
        + "_"
        + data["seed"].astype(str)
        + "_"
        + data["threshold"].astype(str)
        + "_"
        + data["sim_id"].astype(str)
    )

    train = data[~data["key"].isin(test_keys)].copy()
    test = data[data["key"].isin(test_keys)].copy()

    return train, test


def fit_one_model(data: pd.DataFrame, features, target_name: str):
    train, test = split_by_sim(data)

    X_train = train[features].to_numpy(dtype=float)
    y_train = train["target_future"].to_numpy(dtype=float)

    X_test = test[features].to_numpy(dtype=float)
    y_test = test["target_future"].to_numpy(dtype=float)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )

    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    ridge = model.named_steps["ridge"]

    coef = ridge.coef_
    intercept = ridge.intercept_

    perf = {
        "target": target_name,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "r2_train": float(r2_score(y_train, pred_train)),
        "r2_test": float(r2_score(y_test, pred_test)),
        "mae_test": float(mean_absolute_error(y_test, pred_test)),
        "rmse_test": float(np.sqrt(mean_squared_error(y_test, pred_test))),
        "intercept": float(intercept),
    }

    coef_rows = []
    for f, c in zip(features, coef):
        coef_rows.append({
            "feature": f,
            "standardized_coefficient": float(c),
        })

    pred_df = test.copy()
    pred_df["prediction"] = pred_test
    pred_df["target_name"] = target_name

    return perf, pd.DataFrame(coef_rows), pred_df


def run_models(df: pd.DataFrame):
    perf_rows = []
    coef_rows = []
    pred_rows = []

    for system, df_sys in df.groupby("system"):
        log(f"system={system}")

        for horizon in HORIZONS:
            for spec in MODELS:
                model_name = spec["model_name"]
                target = spec["target"]
                features = spec["features"]

                data = make_transition_dataset(df_sys, features, target, horizon)

                if data.empty or len(data) < 50:
                    log(f"skip {system} {model_name} h={horizon}: insufficient data")
                    continue

                perf, coef_df, pred_df = fit_one_model(data, features, target)

                perf["system"] = system
                perf["horizon"] = horizon
                perf["model_name"] = model_name
                perf["features"] = ",".join(features)

                coef_df["system"] = system
                coef_df["horizon"] = horizon
                coef_df["model_name"] = model_name
                coef_df["target"] = target

                pred_df["system"] = system
                pred_df["horizon"] = horizon
                pred_df["model_name"] = model_name

                perf_rows.append(perf)
                coef_rows.append(coef_df)
                pred_rows.append(pred_df)

    perf_df = pd.DataFrame(perf_rows)
    coef_all = pd.concat(coef_rows, ignore_index=True) if coef_rows else pd.DataFrame()
    pred_all = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()

    return perf_df, coef_all, pred_all


def plot_coefficients(coef_df: pd.DataFrame):
    if coef_df.empty:
        return

    for system, g_sys in coef_df.groupby("system"):
        for horizon, g_h in g_sys.groupby("horizon"):
            labels = []
            values = []

            for _, r in g_h.iterrows():
                labels.append(f"{r.model_name}\n{r.feature}")
                values.append(r.standardized_coefficient)

            plt.figure(figsize=(12, 6))
            plt.bar(np.arange(len(values)), values)
            plt.axhline(0, linestyle="--", linewidth=1)
            plt.xticks(np.arange(len(values)), labels, rotation=45, ha="right", fontsize=8)
            plt.ylabel("Standardized coefficient")
            plt.title(f"State transition coefficients: {system}, horizon={horizon}")
            plt.tight_layout()

            out = FIG_DIR / f"state_transition_coefficients_{system}_h{horizon}.png"
            plt.savefig(out, dpi=300)
            plt.close()


def plot_performance(perf_df: pd.DataFrame):
    if perf_df.empty:
        return

    for system, g_sys in perf_df.groupby("system"):
        for horizon, g_h in g_sys.groupby("horizon"):
            labels = g_h["model_name"].tolist()
            values = g_h["r2_test"].to_numpy(dtype=float)

            plt.figure(figsize=(9, 5))
            plt.bar(np.arange(len(values)), values)
            plt.axhline(0, linestyle="--", linewidth=1)
            plt.xticks(np.arange(len(values)), labels, rotation=30, ha="right")
            plt.ylabel("Test R2")
            plt.title(f"State transition regression performance: {system}, horizon={horizon}")
            plt.tight_layout()

            out = FIG_DIR / f"state_transition_performance_{system}_h{horizon}.png"
            plt.savefig(out, dpi=300)
            plt.close()


def make_interpretation_table(coef_df: pd.DataFrame, perf_df: pd.DataFrame):
    rows = []

    key_effects = [
        ("D_next_from_P_D", "P_state", "P_to_D"),
        ("B_next_from_P_B", "P_state", "P_to_B"),
        ("V_next_from_P_D_B_V", "P_state", "P_to_V"),
        ("V_next_from_P_D_B_V", "D_state", "D_to_V"),
        ("V_next_from_P_D_B_V", "B_state", "B_to_V"),
        ("F_next_from_P_D_B_V_F", "P_state", "P_to_F"),
        ("F_next_from_P_D_B_V_F", "D_state", "D_to_F"),
        ("F_next_from_P_D_B_V_F", "B_state", "B_to_F"),
        ("F_next_from_P_D_B_V_F", "V_state", "V_to_F"),
    ]

    for system in sorted(coef_df["system"].unique()):
        for horizon in sorted(coef_df["horizon"].unique()):
            for model_name, feature, effect_name in key_effects:
                c = coef_df[
                    (coef_df["system"] == system)
                    & (coef_df["horizon"] == horizon)
                    & (coef_df["model_name"] == model_name)
                    & (coef_df["feature"] == feature)
                ]

                if c.empty:
                    continue

                p = perf_df[
                    (perf_df["system"] == system)
                    & (perf_df["horizon"] == horizon)
                    & (perf_df["model_name"] == model_name)
                ]

                coef_value = float(c["standardized_coefficient"].iloc[0])
                r2_test = float(p["r2_test"].iloc[0]) if not p.empty else np.nan

                if coef_value > 0.05:
                    direction = "positive"
                elif coef_value < -0.05:
                    direction = "negative"
                else:
                    direction = "weak_or_near_zero"

                rows.append({
                    "system": system,
                    "horizon": horizon,
                    "effect": effect_name,
                    "model_name": model_name,
                    "feature": feature,
                    "standardized_coefficient": coef_value,
                    "direction": direction,
                    "test_r2": r2_test,
                })

    return pd.DataFrame(rows)


def main():
    if not INPUT.exists():
        raise SystemExit(f"Missing input file: {INPUT}")

    df = pd.read_csv(INPUT)

    required = [
        "system",
        "seed",
        "threshold",
        "sim_id",
        "relative_time",
        "P_state",
        "D_state",
        "B_state",
        "V_state",
        "F_state",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns: {missing}")

    for c in ["sim_id", "relative_time", "P_state", "D_state", "B_state", "V_state", "F_state"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[(df["relative_time"] >= -80) & (df["relative_time"] <= 0)].copy()

    perf_df, coef_df, pred_df = run_models(df)

    interpretation_df = make_interpretation_table(coef_df, perf_df)

    perf_df.to_csv(TABLE_DIR / "state_transition_performance.csv", index=False, encoding="utf-8-sig")
    coef_df.to_csv(TABLE_DIR / "state_transition_coefficients.csv", index=False, encoding="utf-8-sig")
    pred_df.to_csv(TABLE_DIR / "state_transition_predictions.csv", index=False, encoding="utf-8-sig")
    interpretation_df.to_csv(TABLE_DIR / "state_transition_interpretation.csv", index=False, encoding="utf-8-sig")

    plot_coefficients(coef_df)
    plot_performance(perf_df)

    log("state transition regression finished")
    log(f"output directory: {OUTDIR}")

    print("\n=== State transition performance ===")
    print(perf_df.to_string(index=False))

    print("\n=== Key interpretation effects ===")
    print(interpretation_df.to_string(index=False))

    print("\n=== Output files ===")
    for p in sorted(OUTDIR.rglob("*")):
        print(p)


if __name__ == "__main__":
    main()
