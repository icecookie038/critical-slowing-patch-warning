from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.real_data.published_csd_benchmarks import (
    blocked_permutation_slope,
    clements_threshold_signal,
    estimate_collapse_index,
    r_acf_lag1,
    rindi_grid_metrics,
)


class TemporalBenchmarkTests(unittest.TestCase):
    def test_r_acf_uses_r_normalization(self) -> None:
        values = np.array([1.0, 2.0, 4.0, 3.0])
        centered = values - values.mean()
        expected = np.dot(centered[:-1], centered[1:]) / np.dot(
            centered, centered
        )
        self.assertAlmostEqual(r_acf_lag1(values), expected)

    def test_threshold_requires_excursion(self) -> None:
        indicators = pd.DataFrame(
            {
                "day": np.arange(8),
                "cv": [np.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0],
            }
        )
        signal, day = clements_threshold_signal(indicators, ["cv"], sigma=1.0)
        self.assertTrue(signal)
        self.assertEqual(day, 6.0)

    def test_collapse_index_uses_mean_ratio(self) -> None:
        populations = np.array(
            [
                [10.0, 9.0, 8.0, 3.0],
                [12.0, 11.0, 10.0, 4.0],
            ]
        )
        self.assertEqual(estimate_collapse_index(populations), 3)


class SpatialBenchmarkTests(unittest.TestCase):
    def test_patch_metrics_find_largest_component(self) -> None:
        score = np.zeros((5, 6), dtype=float)
        score[1:3, 1:4] = 2.0
        score[4, 5] = 2.0
        metrics = rindi_grid_metrics(score, turf_threshold=2)
        self.assertAlmostEqual(metrics["turf_fraction"], 7.0 / 30.0)
        self.assertAlmostEqual(
            metrics["largest_turf_patch_fraction"], 6.0 / 30.0
        )
        self.assertAlmostEqual(metrics["component_density"], 2.0 / 30.0)

    def test_blocked_permutation_preserves_plot_pairing(self) -> None:
        rows = []
        for year in (2014, 2015):
            for plot, removal in enumerate((0.0, 25.0, 50.0, 75.0), start=1):
                rows.append(
                    {
                        "year": year,
                        "plot": plot,
                        "removal_percent": removal,
                        "metric": removal + (year - 2014) * 3.0,
                    }
                )
        frame = pd.DataFrame(rows)
        original = frame.copy(deep=True)
        slope, p_value = blocked_permutation_slope(
            frame, "metric", n_permutations=99, seed=4
        )
        pd.testing.assert_frame_equal(frame, original)
        self.assertAlmostEqual(slope, 1.0)
        self.assertGreaterEqual(p_value, 0.01)
        self.assertLessEqual(p_value, 0.10)


if __name__ == "__main__":
    unittest.main()
