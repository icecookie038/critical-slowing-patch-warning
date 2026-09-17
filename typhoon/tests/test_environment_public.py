import numpy as np
import pandas as pd

from src.tcg_environment_public import (
    era5_object_key,
    select_anchor_manifest,
    spherical_relative_vorticity,
)


def test_public_era5_object_key_is_daily_pressure_level_analysis():
    key = era5_object_key("u", pd.Timestamp("2019-07-14T12:00:00Z"))
    assert key == (
        "e5.oper.an.pl/201907/"
        "e5.oper.an.pl.128_131_u.ll025uv.2019071400_2019071423.nc"
    )


def test_anchor_selection_requires_one_row_per_case_and_hour():
    rows = []
    for sid in ("a", "b"):
        for hour in (-72, -48, -24, -3, 0):
            rows.append(
                {
                    "sid": sid,
                    "hours_to_endpoint": hour,
                    "observation_time": pd.Timestamp("2020-01-04T00:00:00Z")
                    + pd.Timedelta(hours=hour),
                }
            )
    selected = select_anchor_manifest(pd.DataFrame(rows), [-72, -48, -24, -3])
    assert len(selected) == 8
    assert set(selected["hours_to_endpoint"]) == {-72, -48, -24, -3}


def test_spherical_vorticity_recovers_known_zonal_gradient():
    radius = 6_371_000.0
    target = 2.0e-5
    latitudes = np.linspace(-2.0, 2.0, 17)
    longitudes = np.linspace(128.0, 132.0, 17)
    phi = np.deg2rad(latitudes)[:, None]
    lam = np.deg2rad(longitudes)[None, :]
    u = np.zeros((len(latitudes), len(longitudes)))
    v = radius * target * np.cos(phi) * lam
    result = spherical_relative_vorticity(u, v, latitudes, longitudes)
    assert np.allclose(result[2:-2, 2:-2], target, rtol=1e-5, atol=1e-10)
