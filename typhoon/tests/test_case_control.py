import numpy as np
import pandas as pd

from src.tcg_case_control import _assign_matched_pairs, aggregate_cases


def test_matching_assigns_one_case_of_each_outcome_per_pair():
    cases = pd.DataFrame(
        {
            "sid": ["d1", "d2", "n1", "n2"],
            "name": ["d1", "d2", "n1", "n2"],
            "outcome": [1, 1, 0, 0],
            "endpoint_doy": [200, 250, 205, 245],
            "median_lat": [10, 20, 11, 19],
            "median_lon": [130, 140, 131, 139],
        }
    )
    matched, _ = _assign_matched_pairs(cases)
    assert matched.groupby("pair_id")["outcome"].sum().eq(1).all()
    assert matched.groupby("pair_id").size().eq(2).all()


def test_aggregate_cases_preserves_case_count_and_finite_slopes():
    rows = []
    for sid, outcome in [("d", 1), ("n", 0)]:
        for hour in range(-72, 0, 3):
            row = {
                "sid": sid,
                "name": sid,
                "outcome": outcome,
                "pair_id": 1,
                "endpoint_time": "2020-01-04T00:00:00+00:00",
                "endpoint_type": "test",
                "center_lat": 10.0,
                "center_lon": 130.0,
                "hours_to_endpoint": hour,
                "valid_pixel_fraction": 1.0,
            }
            for index, feature in enumerate(STATIC_FEATURES_FOR_TEST):
                row[feature] = index + hour / 100.0
            rows.append(row)
    result = aggregate_cases(pd.DataFrame(rows))
    assert len(result) == 2
    assert np.isfinite(result.filter(like="__slope_per_day").to_numpy()).all()


STATIC_FEATURES_FOR_TEST = [
    "cold_cloud_fraction",
    "very_cold_fraction",
    "radial_concentration",
    "azimuthal_symmetry",
    "largest_patch_fraction",
    "patch_density_per_10k_px",
    "cold_centroid_offset_km",
    "mean_cold_radius_km",
    "mean_bt_k",
    "min_bt_k",
]
