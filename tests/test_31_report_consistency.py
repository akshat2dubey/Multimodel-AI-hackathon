"""T39 — the report and the dashboard must not disagree about the same flood.

The 2026-10-13 real run produced 1,928.5 ha from inference's pixel area and
1,934.0 ha in the situation report, because the report estimated a pixel's area
at the scene-centre latitude itself. One measurement had two values, in exactly
the two artifacts a judge compares. The report now prefers the figure the run
recorded in change_detection.json, and falls back to its own estimate when that
sidecar is absent or older than the mask it claims to describe.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

import generate_report  # noqa: E402

FLOOD_PIXELS = 40


@pytest.fixture
def mask(tmp_path):
    """A 20x20 degree-grid mask with a known number of flooded pixels."""
    data = np.zeros((20, 20), dtype="float32")
    data.ravel()[:FLOOD_PIXELS] = 1.0
    path = tmp_path / "flood_mask.tif"
    with rasterio.open(
        path, "w", driver="GTiff", height=20, width=20, count=1,
        dtype="float32", crs="EPSG:4326",
        transform=from_bounds(85.4, 28.0, 86.0, 28.6, 20, 20),
    ) as dst:
        dst.write(data, 1)
    return path


def _write_sidecar(mask_path: Path, area_ha: float, pixels: int, *, age_s: float = 0.0):
    sidecar = mask_path.parent / "change_detection.json"
    sidecar.write_text(json.dumps({"flood_pixels": pixels, "flood_area_ha": area_ha}),
                       encoding="utf-8")
    now = time.time()
    os.utime(sidecar, (now - age_s, now - age_s))
    return sidecar


def test_the_recorded_flood_area_wins_over_the_local_estimate(mask):
    _write_sidecar(mask, 123.4, FLOOD_PIXELS)
    stats = generate_report.load_flood_mask_stats(mask)

    assert stats["flood_pixels"] == FLOOD_PIXELS
    assert stats["flood_area_ha"] == 123.4
    assert stats["flood_area_source"] == "change_detection.json"


def test_a_sidecar_written_moments_before_the_mask_is_still_accepted(mask):
    """Inference writes the two files in one pass, and the order varies."""
    _write_sidecar(mask, 123.4, FLOOD_PIXELS, age_s=2.0)
    stats = generate_report.load_flood_mask_stats(mask)

    assert stats["flood_area_source"] == "change_detection.json"
    assert stats["flood_area_ha"] == 123.4


def test_a_stale_sidecar_cannot_speak_for_a_newer_mask(mask):
    _write_sidecar(mask, 123.4, FLOOD_PIXELS, age_s=3600)
    stats = generate_report.load_flood_mask_stats(mask)

    assert stats["flood_area_source"] == "computed"
    assert stats["flood_area_ha"] != 123.4, "an older run's figure was reported as this one's"


def test_a_sidecar_about_a_different_mask_is_ignored(mask):
    """A pixel-count mismatch means the two artifacts describe different runs."""
    _write_sidecar(mask, 123.4, FLOOD_PIXELS + 1)
    stats = generate_report.load_flood_mask_stats(mask)

    assert stats["flood_area_source"] == "computed"


def test_a_mask_without_a_sidecar_still_reports_an_area(mask):
    stats = generate_report.load_flood_mask_stats(mask)

    assert stats["flood_area_source"] == "computed"
    assert stats["flood_area_ha"] > 0
