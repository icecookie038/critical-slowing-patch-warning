import numpy as np
import pandas as pd

from src.tcg_environment import (
    _canonical_slice,
    _month_request_records,
    aggregate_environment_cases,
    compute_environment_features,
)


def test_month_requests_do_not_cross_product_calendar_months(tmp_path):
    manifest = pd.DataFrame(
        {
            "sid": ["case"] * 4,
            "name": ["CASE"] * 4,
            "outcome": [1] * 4,
            "observation_time": [
                "2020-07-31T18:00:00Z",
                "2020-07-31T21:00:00Z",
                "2020-08-01T00:00:00Z",
                "2020-08-01T03:00:00Z",
            ],
            "center_lat": [15.0] * 4,
            "center_lon": [135.0] * 4,
        }
    )
    cfg = {
        "area_margin_deg": 8.0,
        "era5_raw_dir": str(tmp_path / "raw"),
        "era5_request_dir": str(tmp_path / "requests"),
        "cds_dataset": "reanalysis-era5-pressure-levels",
        "variables": [
            "u_component_of_wind",
            "v_component_of_wind",
            "relative_humidity",
            "vorticity",
        ],
        "pressure_levels_hpa": [200, 600, 700, 850],
    }
    records = _month_request_records(manifest, cfg)
    assert len(records) == 2
    assert records[0]["payload"]["request"]["day"] == ["31"]
    assert records[1]["payload"]["request"]["day"] == ["01"]
    assert records[0]["payload"]["request"]["area"] == [
        23.0,
        127.0,
        7.0,
        143.0,
    ]


def test_environment_features_recover_constant_known_fields():
    axis = np.linspace(-4.0, 4.0, 33)
    lons, lats = np.meshgrid(130.0 + axis, 15.0 + axis)
    levels = np.array([200, 600, 700, 850])
    shape = (len(levels), *lons.shape)
    u = np.zeros(shape)
    v = np.zeros(shape)
    rh = np.zeros(shape)
    vo = np.zeros(shape)
    u[0] = 10.0
    u[3] = 4.0
    v[0] = 2.0
    v[3] = -6.0
    rh[1:] = 70.0
    vo[3] = 2.0e-5
    result = compute_environment_features(
        {"u": u, "v": v, "r": rh, "vo": vo},
        levels,
        lons,
        lats,
        130.0,
        15.0,
        [300, 500],
    )
    assert np.isclose(
        result["vorticity_850_mean_s1__500km"], 2.0e-5
    )
    assert np.isclose(result["rh_600_850_mean_pct__500km"], 70.0)
    assert np.isclose(result["shear_200_850_ms__500km"], 10.0)


def test_environment_aggregation_preserves_case_and_feature_values():
    rows = []
    for sid, outcome in (("d", 1), ("n", 0)):
        for hour in range(-72, 0, 3):
            row = {
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
            }
            for radius in (300, 500):
                row[f"vorticity_850_mean_s1__{radius}km"] = (
                    1e-5 + hour * 1e-8
                )
                row[f"rh_600_850_mean_pct__{radius}km"] = (
                    70.0 + hour / 100.0
                )
                row[f"shear_200_850_ms__{radius}km"] = (
                    10.0 - hour / 100.0
                )
            for feature in (
                "cold_cloud_fraction",
                "very_cold_fraction",
                "radial_concentration",
                "azimuthal_symmetry",
                "largest_patch_fraction",
                "cold_centroid_offset_km",
            ):
                row[feature] = outcome + hour / 1000.0
            rows.append(row)
    cfg = {"analysis_radii_km": [300, 500]}
    cases = aggregate_environment_cases(pd.DataFrame(rows), cfg)
    assert len(cases) == 2
    assert np.isfinite(
        cases.filter(like="__slope_per_day").to_numpy()
    ).all()
    assert "shear_200_850_ms__500km__late_median" in cases.columns


def test_canonical_slice_accepts_era5_coordinate_and_variable_names():
    import xarray as xr

    times = pd.date_range("2020-08-01", periods=2, freq="3h")
    levels = np.array([200, 600, 700, 850])
    latitudes = np.linspace(17.0, 13.0, 5)
    longitudes = np.linspace(128.0, 132.0, 5)
    shape = (2, 4, 5, 5)
    dataset = xr.Dataset(
        {
            "u": (("valid_time", "pressure_level", "latitude", "longitude"), np.ones(shape)),
            "v": (("valid_time", "pressure_level", "latitude", "longitude"), np.ones(shape) * 2),
            "r": (("valid_time", "pressure_level", "latitude", "longitude"), np.ones(shape) * 70),
            "vo": (("valid_time", "pressure_level", "latitude", "longitude"), np.ones(shape) * 1e-5),
        },
        coords={
            "valid_time": times,
            "pressure_level": levels,
            "latitude": latitudes,
            "longitude": longitudes,
        },
    )
    cfg = {
        "pressure_levels_hpa": [200, 600, 700, 850],
        "max_time_offset_minutes": 90,
    }
    fields, selected_levels, lons, lats, matched = _canonical_slice(
        dataset, pd.Timestamp("2020-08-01T03:00:00Z"), cfg
    )
    assert fields["u"].shape == (4, 5, 5)
    assert np.array_equal(selected_levels, levels)
    assert lons.shape == lats.shape == (5, 5)
    assert matched == pd.Timestamp("2020-08-01T03:00:00Z")
