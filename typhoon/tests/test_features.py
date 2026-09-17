import numpy as np

from src.tcg_pilot import compute_cloud_features


def test_inner_cold_blob_has_high_concentration():
    axis = np.linspace(-4.0, 4.0, 161)
    lons, lats = np.meshgrid(130.0 + axis, 15.0 + axis)
    radius_deg = np.hypot(lons - 130.0, lats - 15.0)
    bt = np.full_like(lons, 285.0)
    bt[radius_deg < 1.0] = 220.0
    cfg = {
        "outer_radius_km": 600.0,
        "inner_radius_km": 200.0,
        "cold_threshold_k": 235.0,
        "very_cold_threshold_k": 210.0,
        "min_patch_pixels": 9,
    }
    result = compute_cloud_features(bt, lons, lats, 130.0, 15.0, cfg)
    assert result["radial_concentration"] > 0.95
    assert result["largest_patch_fraction"] > 0.95
    assert result["cold_centroid_offset_km"] < 5.0

