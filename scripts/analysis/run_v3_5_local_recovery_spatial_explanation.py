"""Run the v3.5 direct-recovery and spatial-explanation analysis.

Primary detection: right-censored local recovery hazards.
Secondary interpretation: patch context after controlling inundation stress.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.real_data.recovery_spatial_models import (
    MODEL_FEATURES,
    RecoveryModelConfig,
    add_increment_metrics,
    benjamini_hochberg,
    block_recovery_field,
    cross_site_model_comparison,
    merge_recovery_and_patch,
    multiscale_field_consistency,
    out_of_fold_stress_predictions,
    patch_residual_associations,
)
from src.real_data.tidal_marsh import (
    PATCH_DYNAMIC_FEATURES,
    PATCH_STATIC_FEATURES,
    SITES,
    aggregate_recovery_units,
    compare_published_recovery_rates,
    load_tidal_marsh_site,
    published_recovery_rates,
    summarize_patch_context,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw/tidal_marsh"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/v3_5_local_recovery_spatial_explanation"),
    )
    parser.add_argument(
        "--block-widths-m",
        type=float,
        nargs="+",
        default=[32.0, 64.0, 128.0],
    )
    parser.add_argument("--primary-width-m", type=float, default=64.0)
    parser.add_argument("--patch-resolution-m", type=float, default=1.0)
    parser.add_argument("--min-losses", type=int, default=50)
    parser.add_argument("--min-recoveries", type=int, default=10)
    parser.add_argument("--poisson-alpha", type=float, default=0.1)
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument("--seed", type=int, default=20260822)
    return parser.parse_args()


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def make_summary_figure(
    direct_replication: pd.DataFrame,
    primary_fields: dict[str, pd.DataFrame],
    path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    colors = {"Hellegat": "#b23a48", "Paulina": "#277da1"}
    for site_name, group in direct_replication.groupby("site"):
        ax = axes[0, 0] if site_name == "Hellegat" else axes[0, 1]
        ax.plot(
            group["inundation_percent"],
            group["published_hazard_per_day"],
            color="0.7",
            linewidth=2.5,
            label="published",
        )
        ax.scatter(
            group["inundation_percent"],
            group["hazard_per_day"],
            s=22,
            color=colors[site_name],
            label="Python reproduction",
            zorder=3,
        )
        ax.set_title(f"{site_name}: direct recovery")
        ax.set_xlabel("Inundation (%)")
        ax.set_ylabel("Recovery hazard (day$^{-1}$)")
        ax.legend(frameon=False)

    for ax, site_name in zip(axes[1], ["Hellegat", "Paulina"]):
        field = primary_fields[site_name]
        limit = float(np.nanmax(np.abs(field["log_hazard_residual"])))
        limit = max(limit, 0.1)
        scatter = ax.scatter(
            field["center_x_m"],
            field["center_y_m"],
            c=field["log_hazard_residual"],
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            marker="s",
            s=60,
            linewidths=0,
        )
        ax.invert_yaxis()
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{site_name}: stress-adjusted local recovery")
        ax.set_xlabel("East-west distance (m)")
        ax.set_ylabel("North-south distance (m)")
        fig.colorbar(scatter, ax=ax, label="log observed / stress-expected hazard")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def make_patch_explanation_figure(
    primary_fields: dict[str, pd.DataFrame],
    associations: pd.DataFrame,
    path: Path,
) -> None:
    """Map patch context beside the stress-adjusted recovery field."""
    metrics = [
        (
            "log_hazard_residual",
            "Stress-adjusted recovery",
            "log observed / stress-expected hazard",
            "RdBu_r",
        ),
        ("coverage_mean", "Mean vegetation cover", "cover fraction", "viridis"),
        (
            "largest_component_fraction_mean",
            "Mean largest-patch fraction",
            "largest-patch fraction",
            "magma",
        ),
    ]
    combined = pd.concat(primary_fields.values(), ignore_index=True)
    limits: dict[str, tuple[float, float]] = {}
    residual_limit = float(
        np.nanquantile(np.abs(combined["log_hazard_residual"]), 0.98)
    )
    limits["log_hazard_residual"] = (-max(residual_limit, 0.1), max(residual_limit, 0.1))
    for feature in ["coverage_mean", "largest_component_fraction_mean"]:
        limits[feature] = (
            float(np.nanmin(combined[feature])),
            float(np.nanmax(combined[feature])),
        )

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    site_order = ["Hellegat", "Paulina"]
    for row, site_name in enumerate(site_order):
        field = primary_fields[site_name]
        for column, (feature, title, colorbar_label, cmap) in enumerate(metrics):
            ax = axes[row, column]
            vmin, vmax = limits[feature]
            plotted = ax.scatter(
                field["center_x_m"],
                field["center_y_m"],
                c=field[feature],
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                marker="s",
                s=62,
                linewidths=0,
            )
            suffix = ""
            if feature != "log_hazard_residual":
                association = associations.loc[
                    (associations["site"] == site_name)
                    & (associations["feature"] == feature)
                ]
                if len(association) == 1:
                    record = association.iloc[0]
                    suffix = (
                        f"\nresidual association: ρ={record['spearman_rho']:.2f}, "
                        f"q={record['fdr_q']:.3f}"
                    )
            ax.set_title(f"{site_name}: {title}{suffix}")
            ax.set_xlabel("East-west distance (m)")
            if column == 0:
                ax.set_ylabel("North-south distance (m)")
            ax.invert_yaxis()
            ax.set_aspect("equal", adjustable="box")
            if row == 1:
                fig.colorbar(
                    plotted,
                    ax=axes[:, column],
                    orientation="horizontal",
                    shrink=0.72,
                    pad=0.08,
                    label=colorbar_label,
                )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def build_direction_summary(associations: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for (width, feature), group in associations.groupby(
        ["block_width_m", "feature"]
    ):
        by_site = group.set_index("site")
        if not set(SITES).issubset(by_site.index):
            continue
        h = by_site.loc["Hellegat"]
        p = by_site.loc["Paulina"]
        records.append(
            {
                "block_width_m": width,
                "feature": feature,
                "feature_group": h["feature_group"],
                "hellegat_rho": h["spearman_rho"],
                "hellegat_q": h["fdr_q"],
                "paulina_rho": p["spearman_rho"],
                "paulina_q": p["fdr_q"],
                "same_direction": bool(h["spearman_rho"] * p["spearman_rho"] > 0),
                "both_fdr_below_0p05": bool(h["fdr_q"] < 0.05 and p["fdr_q"] < 0.05),
            }
        )
    return pd.DataFrame.from_records(records)


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without pandas' optional tabulate dependency."""

    def format_value(value: object) -> str:
        if pd.isna(value):
            return "NA"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6g}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    columns = [str(column) for column in frame.columns]
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    rows.extend(
        "| " + " | ".join(format_value(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(rows)


def write_report(
    output_root: Path,
    replication_summary: pd.DataFrame,
    supported_summary: pd.DataFrame,
    model_metrics: pd.DataFrame,
    direction_summary: pd.DataFrame,
    consistency: pd.DataFrame,
    decision: dict[str, object],
    args: argparse.Namespace,
) -> None:
    primary_metrics = model_metrics.loc[
        model_metrics["block_width_m"] == args.primary_width_m
    ]
    primary_directions = direction_summary.loc[
        direction_summary["block_width_m"] == args.primary_width_m
    ]
    lines = [
        "# v3.5 local recovery and spatial patch explanation",
        "",
        "## Locked positioning",
        "",
        "Direct right-censored recovery hazard is the primary resilience/CSD measurement. "
        "Vegetation-patch variables are secondary spatial explanations only; they do not "
        "define a positive CSD label and are not combined into PWSI.",
        "",
        "## Direct recovery reproduction",
        "",
        markdown_table(replication_summary),
        "",
        "## Supported local units",
        "",
        f"Eligibility: at least {args.min_losses} losses and {args.min_recoveries} recoveries.",
        "",
        markdown_table(supported_summary),
        "",
        f"## Cross-site validation at the primary {args.primary_width_m:g} m scale",
        "",
        "Negative percentages are improvements over the stress-only model; positive "
        "percentages are worse.",
        "",
        markdown_table(primary_metrics),
        "",
        "## Stress-adjusted patch associations",
        "",
        "Associations use spatial-block held-out stress predictions, block-level "
        "permutations, and Benjamini-Hochberg correction within site and scale.",
        "",
        markdown_table(primary_directions),
        "",
        "## Multiscale consistency of local recovery fields",
        "",
        markdown_table(consistency),
        "",
        "## Decision",
        "",
        f"- Direct recovery replication: **{decision['direct_recovery_replication']}**",
        f"- Local recovery data sufficiency: **{decision['local_recovery_data']}**",
        f"- Independent patch increment: **{decision['independent_patch_increment']}**",
        f"- Patch role: **{decision['patch_role']}**",
        "",
        "The ecological route proceeds through local-to-landscape direct recovery. "
        "Patch geometry may describe where slow recovery is spatially organized, but "
        "it is promoted beyond context only if it adds held-out cross-site information "
        "after inundation control.",
    ]
    (output_root / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.primary_width_m not in args.block_widths_m:
        raise ValueError("primary width must be included in block widths")
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    model_config = RecoveryModelConfig(alpha=args.poisson_alpha)

    replication_rows: list[dict[str, object]] = []
    direct_comparisons: list[pd.DataFrame] = []
    supported_rows: list[dict[str, object]] = []
    merged_by_scale: dict[float, list[pd.DataFrame]] = {
        float(width): [] for width in args.block_widths_m
    }

    for site_name in SITES:
        print(f"Loading and aggregating {site_name}...")
        data = load_tidal_marsh_site(args.data_root, site_name)
        units_by_scale, direct, counts = aggregate_recovery_units(
            data, args.block_widths_m
        )
        comparison, replication = compare_published_recovery_rates(
            direct, published_recovery_rates(data)
        )
        direct_comparisons.append(comparison)
        replication_rows.append({"site": site_name, **replication, **counts})
        save_csv(direct, output_root / f"direct_recovery_{site_name.lower()}.csv")

        for width in args.block_widths_m:
            width = float(width)
            print(f"  patch context and local units: {width:g} m")
            units = units_by_scale[width]
            patch = summarize_patch_context(
                data,
                block_width_m=width,
                patch_resolution_m=args.patch_resolution_m,
            )
            merged = merge_recovery_and_patch(
                units,
                patch,
                min_losses=args.min_losses,
                min_recoveries=args.min_recoveries,
            )
            merged_by_scale[width].append(merged)
            supported_rows.append(
                {
                    "site": site_name,
                    "block_width_m": width,
                    "supported_units": len(merged),
                    "represented_blocks": merged["block_id"].nunique(),
                    "recoveries": int(merged["n_recovered"].sum()),
                    "exposure_years": float(merged["exposure_years"].sum()),
                }
            )
            suffix = f"{width:g}m"
            save_csv(units, output_root / f"local_recovery_units_{site_name.lower()}_{suffix}.csv")
            save_csv(patch, output_root / f"patch_context_{site_name.lower()}_{suffix}.csv")
            save_csv(merged, output_root / f"model_units_{site_name.lower()}_{suffix}.csv")

        del data

    replication_summary = pd.DataFrame.from_records(replication_rows)
    direct_replication = pd.concat(direct_comparisons, ignore_index=True)
    supported_summary = pd.DataFrame.from_records(supported_rows)
    save_csv(replication_summary, output_root / "direct_recovery_replication_summary.csv")
    save_csv(direct_replication, output_root / "direct_recovery_replication_detail.csv")
    save_csv(supported_summary, output_root / "supported_local_units.csv")

    all_model_metrics: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []
    all_associations: list[pd.DataFrame] = []
    all_fields: list[pd.DataFrame] = []
    consistency_frames: list[pd.DataFrame] = []
    primary_fields: dict[str, pd.DataFrame] = {}

    for width, site_frames in merged_by_scale.items():
        combined = pd.concat(site_frames, ignore_index=True)
        metrics, predictions = cross_site_model_comparison(combined, model_config)
        metrics = add_increment_metrics(metrics)
        metrics.insert(0, "block_width_m", width)
        predictions.insert(0, "analysis_block_width_m", width)
        all_model_metrics.append(metrics)
        all_predictions.append(predictions)

        for site_name, site_frame in combined.groupby("site", sort=False):
            site_frame = site_frame.reset_index(drop=True)
            oof = out_of_fold_stress_predictions(site_frame, model_config)
            field = block_recovery_field(site_frame, oof)
            all_fields.append(field)
            association = patch_residual_associations(
                field,
                n_permutations=args.permutations,
                seed=args.seed,
            )
            all_associations.append(association)
            if width == args.primary_width_m:
                primary_fields[site_name] = field

    model_metrics = pd.concat(all_model_metrics, ignore_index=True)
    model_predictions = pd.concat(all_predictions, ignore_index=True)
    block_fields = pd.concat(all_fields, ignore_index=True)
    associations = pd.concat(all_associations, ignore_index=True)
    associations["fdr_q"] = associations.groupby(
        ["site", "block_width_m"]
    )["permutation_p"].transform(lambda values: benjamini_hochberg(values.to_numpy()))
    direction_summary = build_direction_summary(associations)

    for site_name in SITES:
        site_fields = {
            width: block_fields.loc[
                (block_fields["site"] == site_name)
                & (block_fields["block_width_m"] == width)
            ].copy()
            for width in sorted(merged_by_scale)
        }
        consistency_frames.append(multiscale_field_consistency(site_fields))
    consistency = pd.concat(consistency_frames, ignore_index=True)

    save_csv(model_metrics, output_root / "cross_site_model_metrics.csv")
    save_csv(model_predictions, output_root / "cross_site_model_predictions.csv")
    save_csv(block_fields, output_root / "stress_adjusted_block_recovery_fields.csv")
    save_csv(associations, output_root / "patch_residual_associations.csv")
    save_csv(direction_summary, output_root / "patch_direction_consistency.csv")
    save_csv(consistency, output_root / "multiscale_field_consistency.csv")

    make_summary_figure(
        direct_replication,
        primary_fields,
        output_root / "summary_figure.png",
    )
    make_patch_explanation_figure(
        primary_fields,
        associations.loc[
            associations["block_width_m"] == args.primary_width_m
        ],
        output_root / f"patch_spatial_explanation_{args.primary_width_m:g}m.png",
    )

    direct_pass = bool(
        (replication_summary["max_relative_replication_error"] < 1e-3).all()
        and set(replication_summary["n_bins"]) == {12, 35}
    )
    primary_support = supported_summary.loc[
        supported_summary["block_width_m"] == args.primary_width_m
    ]
    local_pass = bool(
        len(primary_support) == 2
        and (primary_support["supported_units"] >= 100).all()
        and (primary_support["represented_blocks"] >= 20).all()
    )
    primary_metrics = model_metrics.loc[
        model_metrics["block_width_m"] == args.primary_width_m
    ]
    all_patch = primary_metrics.loc[
        primary_metrics["model"] == "stress_plus_all_patch"
    ]
    patch_increment = bool(
        len(all_patch) == 2
        and (all_patch["poisson_deviance"] > 0).all()
        and (all_patch["deviance_change_vs_stress_percent"] < -5.0).all()
    )
    decision = {
        "direct_recovery_replication": "PASS" if direct_pass else "FAIL",
        "local_recovery_data": "PASS" if local_pass else "FAIL",
        "independent_patch_increment": "PASS" if patch_increment else "NOT ESTABLISHED",
        "patch_role": "secondary spatial explanation",
        "primary_block_width_m": args.primary_width_m,
        "patch_resolution_m": args.patch_resolution_m,
        "model_feature_groups": MODEL_FEATURES,
    }
    (output_root / "decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_report(
        output_root,
        replication_summary,
        supported_summary,
        model_metrics,
        direction_summary,
        consistency,
        decision,
        args,
    )
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    print(f"Results written to {output_root.resolve()}")


if __name__ == "__main__":
    main()
