import pandas as pd

from src.tcg_lead_time import aggregate_cases_at_cutoff


def _frames():
    rows = []
    for sid, outcome in (("d", 1), ("n", 0)):
        for hour in range(-72, 0, 3):
            rows.append(
                {
                    "sid": sid,
                    "name": sid,
                    "outcome": outcome,
                    "pair_id": 1,
                    "endpoint_time": "2020-08-04T00:00:00Z",
                    "observation_time": pd.Timestamp("2020-08-04T00:00:00Z")
                    + pd.Timedelta(hours=hour),
                    "hours_to_endpoint": hour,
                    "center_lat": 15.0,
                    "center_lon": 130.0,
                    "cold_cloud_fraction": outcome + hour / 1000,
                    "very_cold_fraction": outcome + hour / 1000,
                    "radial_concentration": outcome + hour / 1000,
                    "azimuthal_symmetry": outcome + hour / 1000,
                    "largest_patch_fraction": outcome + hour / 1000,
                    "cold_centroid_offset_km": outcome + hour / 1000,
                }
            )
    return pd.DataFrame(rows)


def test_cutoff_aggregation_cannot_see_future_frames():
    original = _frames()
    changed = original.copy()
    changed.loc[changed["hours_to_endpoint"].gt(-24), "cold_cloud_fraction"] = 999
    before = aggregate_cases_at_cutoff(original, -24, 12)
    after = aggregate_cases_at_cutoff(changed, -24, 12)
    assert before.equals(after)
    assert set(before["n_trailing_frames"]) == {4}
