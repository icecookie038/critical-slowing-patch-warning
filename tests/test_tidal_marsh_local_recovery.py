from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.real_data.recovery_spatial_models import (
    RecoveryModelConfig,
    benjamini_hochberg,
    block_recovery_field,
    cross_site_model_comparison,
    permutation_spearman,
)
from src.real_data.tidal_marsh import (
    SITES,
    TidalMarshData,
    TidalMarshSite,
    aggregate_recovery_units,
    analysis_bin_edges,
    filter_supported_units,
    iter_recovery_events,
    snapshot_patch_context,
    summarize_patch_context,
)
from src.real_data.time_safe_survival import (
    CURRENT_PATCH_FEATURES,
    HISTORY_PATCH_FEATURES,
    ORIGIN_PATCH_FEATURES,
    block_bootstrap_model_differences,
    build_discrete_recovery_risk_sets,
    cross_site_survival_comparison,
    leakage_audit,
    restrict_to_common_stress_support,
)


class RecoveryEventTests(unittest.TestCase):
    def test_published_bin_counts(self) -> None:
        self.assertEqual(len(analysis_bin_edges(SITES["Hellegat"])) - 1, 35)
        self.assertEqual(len(analysis_bin_edges(SITES["Paulina"])) - 1, 12)

    def test_irregular_recovery_and_censoring(self) -> None:
        sequence = np.zeros((4, 1, 4), dtype=bool)
        sequence[:, 0, 0] = [1, 0, 0, 1]
        sequence[:, 0, 1] = [1, 0, 0, 0]
        sequence[:, 0, 2] = [0, 1, 0, 1]
        sequence[:, 0, 3] = [0, 0, 1, 0]  # final loss has no follow-up
        batches = list(iter_recovery_events(sequence, [2000, 2002, 2005, 2009]))
        self.assertEqual(len(batches), 2)
        np.testing.assert_array_equal(batches[0].duration_years, [7.0, 7.0])
        np.testing.assert_array_equal(batches[0].recovered, [True, False])
        np.testing.assert_array_equal(batches[1].duration_years, [4.0])
        np.testing.assert_array_equal(batches[1].recovered, [True])
        self.assertFalse(any(3 in batch.cols for batch in batches))

    def test_aggregated_censored_hazard(self) -> None:
        site = TidalMarshSite(
            name="Test",
            years=(2000, 2002, 2005, 2009),
            bin_min=0.10,
            bin_max=0.14,
            bin_step=0.01,
            pixel_size_m=1.0,
        )
        sequence = np.zeros((4, 1, 3), dtype=bool)
        sequence[:, 0, 0] = [1, 0, 0, 1]
        sequence[:, 0, 1] = [1, 0, 0, 0]
        sequence[:, 0, 2] = [0, 1, 0, 1]
        data = TidalMarshData(
            site=site,
            vegetation=sequence,
            inundation=np.full((1, 3), 0.105, dtype=float),
            bundle_root=None,  # type: ignore[arg-type]
        )
        units, direct, counts = aggregate_recovery_units(data, [4.0])
        self.assertEqual(counts["events_in_analysis_gradient"], 3)
        self.assertEqual(counts["recoveries_in_analysis_gradient"], 2)
        self.assertEqual(int(direct.loc[0, "n_losses"]), 3)
        self.assertEqual(int(direct.loc[0, "n_recovered"]), 2)
        expected = 2.0 / ((7.0 + 7.0 + 4.0) * 365.0)
        self.assertAlmostEqual(float(direct.loc[0, "hazard_per_day"]), expected)
        self.assertEqual(len(units[4.0]), 1)


class PatchContextTests(unittest.TestCase):
    def test_patch_context_is_descriptive(self) -> None:
        site = TidalMarshSite(
            name="Test",
            years=(0, 2, 4),
            bin_min=0.10,
            bin_max=0.14,
            bin_step=0.01,
            pixel_size_m=1.0,
        )
        sequence = np.zeros((3, 4, 4), dtype=bool)
        sequence[0, :2, :2] = True
        sequence[1, :3, :2] = True
        sequence[2, :3, :3] = True
        data = TidalMarshData(
            site=site,
            vegetation=sequence,
            inundation=np.full((4, 4), 0.105),
            bundle_root=None,  # type: ignore[arg-type]
        )
        context = summarize_patch_context(
            data,
            block_width_m=4.0,
            patch_resolution_m=1.0,
            min_valid_cells=1,
        )
        self.assertEqual(len(context), 1)
        self.assertGreater(float(context.loc[0, "turnover_rate"]), 0)
        self.assertGreater(float(context.loc[0, "coverage_mean"]), 0)
        self.assertLessEqual(
            float(context.loc[0, "largest_component_fraction_mean"]), 1.0
        )

    def test_snapshot_context_does_not_use_future_maps(self) -> None:
        site = TidalMarshSite(
            name="Test",
            years=(0, 2, 4, 7),
            bin_min=0.10,
            bin_max=0.14,
            bin_step=0.01,
            pixel_size_m=1.0,
        )
        sequence = np.zeros((4, 4, 4), dtype=bool)
        sequence[0, :2, :2] = True
        sequence[1, :3, :2] = True
        sequence[2, :3, :3] = True
        changed = sequence.copy()
        changed[3] = True
        base = TidalMarshData(site, sequence, np.full((4, 4), 0.105), None)  # type: ignore[arg-type]
        future_changed = TidalMarshData(  # type: ignore[arg-type]
            site, changed, np.full((4, 4), 0.105), None
        )
        first = snapshot_patch_context(base, 4.0, 1.0, min_valid_cells=1)
        second = snapshot_patch_context(
            future_changed, 4.0, 1.0, min_valid_cells=1
        )
        columns = [
            "coverage",
            "component_density",
            "largest_component_fraction",
            "edge_density",
            "turnover_rate",
        ]
        pd.testing.assert_frame_equal(
            first.loc[first["snapshot_index"] < 3, columns].reset_index(drop=True),
            second.loc[second["snapshot_index"] < 3, columns].reset_index(drop=True),
        )

    def test_supported_unit_filter(self) -> None:
        frame = pd.DataFrame(
            {
                "n_losses": [49, 50, 60],
                "n_recovered": [20, 9, 10],
                "hazard_per_year": [0.2, 0.2, 0.2],
            }
        )
        supported = filter_supported_units(frame)
        self.assertEqual(supported.index.tolist(), [0])
        self.assertEqual(int(supported.loc[0, "n_losses"]), 60)


class SpatialModelTests(unittest.TestCase):
    @staticmethod
    def synthetic_frame() -> pd.DataFrame:
        rng = np.random.default_rng(7)
        rows = []
        for site_index, site in enumerate(["Hellegat", "Paulina"]):
            for block in range(12):
                patch = rng.normal()
                for inundation in [0.32, 0.35, 0.38, 0.41]:
                    exposure = 200.0 + 10.0 * block
                    rate = np.exp(-2.0 - 3.0 * inundation + 0.15 * patch)
                    rows.append(
                        {
                            "site": site,
                            "block_width_m": 64.0,
                            "block_id": block,
                            "block_row": block // 4,
                            "block_col": block % 4,
                            "center_y_m": float(block // 4 * 64 + 32),
                            "center_x_m": float(block % 4 * 64 + 32),
                            "inundation_fraction": inundation,
                            "n_losses": 100,
                            "n_recovered": max(10, int(rate * exposure)),
                            "exposure_years": exposure,
                            "hazard_per_year": rate * (1.0 + 0.02 * site_index),
                            "patch_resolution_m": 1.0,
                            "valid_patch_cells": 1000,
                            "coverage_mean": patch,
                            "component_density_mean": patch * 2,
                            "largest_component_fraction_mean": patch * 0.5,
                            "edge_density_mean": patch * -0.2,
                            "turnover_rate": patch * 0.1,
                            "coverage_change_rate": patch * 0.2,
                            "fragmentation_change_rate": patch * 0.3,
                            "edge_change_rate": patch * 0.4,
                            "largest_component_change_rate": patch * 0.5,
                        }
                    )
        return pd.DataFrame(rows)

    def test_cross_site_models_return_finite_metrics(self) -> None:
        metrics, predictions = cross_site_model_comparison(
            self.synthetic_frame(), RecoveryModelConfig(alpha=0.1)
        )
        self.assertEqual(len(metrics), 6)
        self.assertTrue(np.isfinite(metrics["weighted_log_rmse"]).all())
        self.assertTrue((predictions["predicted_hazard_per_year"] > 0).all())

    def test_block_field_uses_exposure_aggregation(self) -> None:
        frame = self.synthetic_frame().query("site == 'Hellegat'").reset_index(drop=True)
        observed_unit_hazard = (
            frame["n_recovered"] / frame["exposure_years"]
        ).to_numpy()
        field = block_recovery_field(frame, observed_unit_hazard)
        self.assertEqual(len(field), 12)
        self.assertLess(float(field["log_hazard_residual"].abs().max()), 1e-12)

    def test_permutation_and_fdr(self) -> None:
        rho, p_value = permutation_spearman(
            np.arange(10), np.arange(10), n_permutations=99, seed=1
        )
        self.assertAlmostEqual(rho, 1.0)
        self.assertGreaterEqual(p_value, 0.01)
        adjusted = benjamini_hochberg([0.01, 0.04, 0.03])
        np.testing.assert_allclose(adjusted, [0.03, 0.04, 0.04])


class TimeSafeSurvivalTests(unittest.TestCase):
    def test_irregular_event_risk_sets_and_leakage_order(self) -> None:
        site = TidalMarshSite(
            name="Test",
            years=(2000, 2002, 2005, 2009),
            bin_min=0.10,
            bin_max=0.14,
            bin_step=0.01,
            pixel_size_m=1.0,
        )
        sequence = np.zeros((4, 2, 3), dtype=bool)
        sequence[:, 0, 0] = [1, 0, 0, 1]
        sequence[:, 0, 1] = [1, 0, 0, 0]
        sequence[:, 0, 2] = [0, 1, 0, 1]
        data = TidalMarshData(  # type: ignore[arg-type]
            site, sequence, np.full((2, 3), 0.105), None
        )
        risk, summary = build_discrete_recovery_risk_sets(
            data,
            block_width_m=4.0,
            patch_resolution_m=1.0,
            min_valid_patch_cells=1,
        )
        self.assertEqual(summary["raw_disturbance_events"], 3)
        self.assertEqual(summary["retained_disturbance_events"], 3)
        self.assertEqual(summary["retained_recoveries"], 2)
        self.assertEqual(summary["retained_person_intervals"], 5)
        self.assertEqual(sorted(risk["interval_years"].unique()), [3.0, 4.0])
        self.assertTrue(all(leakage_audit(risk).values()))

    @staticmethod
    def synthetic_survival_frame() -> pd.DataFrame:
        generator = np.random.default_rng(71)
        records = []
        row_id = 0
        for site_index, site in enumerate(["Hellegat", "Paulina"]):
            for block in range(8):
                for inundation in np.linspace(0.31, 0.41, 8):
                    elapsed = float((block % 3) * 2)
                    interval = float(1 + block % 2)
                    patch = generator.normal()
                    eta = -1.8 - 3.0 * inundation - 0.08 * elapsed + 0.1 * patch
                    probability = 1.0 - np.exp(-np.exp(eta) * interval)
                    trials = 80
                    events = int(generator.binomial(trials, probability))
                    record = {
                        "risk_row_id": row_id,
                        "site": site,
                        "block_id": block,
                        "origin_index": 1,
                        "risk_start_index": 1,
                        "risk_end_index": 2,
                        "interval_years": interval,
                        "inundation_fraction": inundation,
                        "elapsed_years_start": elapsed,
                        "log_elapsed_years_start": np.log1p(elapsed),
                        "n_at_risk": trials,
                        "n_recovered": events,
                    }
                    for name in ORIGIN_PATCH_FEATURES:
                        record[name] = patch + 0.01 * site_index
                    for name in CURRENT_PATCH_FEATURES:
                        record[name] = patch
                    for name in HISTORY_PATCH_FEATURES:
                        record[name] = abs(patch) * 0.1
                    records.append(record)
                    row_id += 1
        return pd.DataFrame.from_records(records)

    def test_cross_site_survival_models_and_block_bootstrap(self) -> None:
        frame, support = restrict_to_common_stress_support(
            self.synthetic_survival_frame()
        )
        self.assertGreater(support[1], support[0])
        metrics, predictions, coefficients = cross_site_survival_comparison(frame)
        self.assertEqual(len(metrics), 6)
        self.assertTrue(np.isfinite(metrics["interval_brier"]).all())
        self.assertTrue(
            predictions["predicted_interval_probability"].between(0, 1).all()
        )
        self.assertFalse(coefficients.empty)
        bootstrap = block_bootstrap_model_differences(
            predictions, n_bootstrap=9, seed=4
        )
        self.assertEqual(len(bootstrap), 4)
        self.assertTrue(np.isfinite(bootstrap["brier_delta_median"]).all())


if __name__ == "__main__":
    unittest.main()
