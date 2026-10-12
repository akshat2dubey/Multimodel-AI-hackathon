"""4. Hectares must come from the raster transform, never from an assumption.

A hard-coded 0.01 ha (10 m x 10 m) silently rescales every area in the report
whenever the grid is not exactly 10 m — and on a geographic CRS the pixel size
is in degrees, so the error is not even a constant factor.
"""

from __future__ import annotations

import math

import pytest
from rasterio.crs import CRS
from rasterio.transform import from_origin

import damage_assessment
import inference


def test_projected_10m_and_20m_grids_give_expected_hectares():
    tr20 = from_origin(500_000, 3_000_000, 20.0, 20.0)
    tr10 = from_origin(500_000, 3_000_000, 10.0, 10.0)
    utm = CRS.from_epsg(32645)

    assert damage_assessment.pixel_area_ha(tr20, utm) == pytest.approx(0.04, rel=1e-12), (
        "a 20 m x 20 m pixel is 400 m^2 = 0.04 ha; got "
        f"{damage_assessment.pixel_area_ha(tr20, utm)}"
    )
    assert damage_assessment.pixel_area_ha(tr10, utm) == pytest.approx(0.01, rel=1e-12), (
        "a 10 m x 10 m pixel is 100 m^2 = 0.01 ha; got "
        f"{damage_assessment.pixel_area_ha(tr10, utm)}"
    )


def test_geographic_grid_converts_degrees_to_metres_at_its_latitude():
    # 0.00025 deg per pixel, top edge at 28.40 N.
    tr_deg = from_origin(85.60, 28.40, 0.00025, 0.00025)

    lat = 28.40
    m_per_deg = 111_320.0
    px_w_m = 0.00025 * m_per_deg * math.cos(math.radians(lat))
    px_h_m = 0.00025 * m_per_deg
    expected = (px_w_m * px_h_m) / 10_000.0

    got = damage_assessment.pixel_area_ha(tr_deg, CRS.from_epsg(4326))
    assert got == pytest.approx(expected, rel=1e-6), (
        f"a geographic grid must be converted at the scene latitude: expected "
        f"{expected:.8f} ha per pixel, got {got:.8f}. Squaring degrees as if they "
        "were metres is the bug this guards against."
    )
    # 0.00025 deg is ~28 m in latitude, so a pixel is ~680 m^2 = 0.068 ha. The
    # naive "square the degrees" path would produce ~6e-12 ha, so this bound is
    # what actually catches a missing conversion.
    assert 0.01 < got < 0.1, (
        f"{got} ha/pixel is implausible for a 0.00025 deg grid (~0.068 ha); the "
        "degrees were probably not converted to metres at the scene latitude"
    )


def test_change_detection_uses_the_transform_not_a_hardcoded_10m_assumption():
    import numpy as np

    before = np.zeros((10, 10), dtype="float32")
    after = np.zeros((10, 10), dtype="float32")
    after[0:5, :] = 1.0  # 50 pixels of new water

    tr20 = from_origin(500_000, 3_000_000, 20.0, 20.0)
    stats = inference.compute_change_detection(
        before, after, transform=tr20, crs=CRS.from_epsg(32645)
    )

    assert stats["flood_extent_ha"] == pytest.approx(50 * 0.04, rel=1e-9), (
        f"50 pixels of 20 m must be 2.0 ha, got {stats['flood_extent_ha']} ha. "
        "A hard-coded 0.01 ha/pixel would report 0.5 ha."
    )


def test_degenerate_transform_is_rejected():
    with pytest.raises(ValueError):
        damage_assessment.pixel_area_ha(from_origin(0, 0, 0.0, 0.0))
