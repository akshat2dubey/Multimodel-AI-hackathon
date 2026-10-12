"""10. Damage assessment must scale to a full-scene feature set (slow).

5 000 OSM footprints on a 3000x3000 mask is a modest real workload on this
corridor (the pre-event snapshot has 3 755 buildings). Rasterising each footprint
against the WHOLE grid costs ~9 MB and one rasterize call per feature; clipping
to the feature's own pixel window is what keeps this in seconds instead of
minutes.

Marked `slow`: run with `-m "not slow"` to skip it.
"""

from __future__ import annotations

import time

import geopandas as gpd
import numpy as np
import pytest
from rasterio.crs import CRS
from rasterio.transform import from_origin
from shapely.geometry import box

import damage_assessment as da

BUDGET_S = 20.0
N_FEATURES = 5_000
GRID = 3_000
PIXEL_M = 20.0


@pytest.mark.slow
def test_5000_features_on_a_3000x3000_mask_finish_within_budget():
    transform = from_origin(400_000, 3_000_000, PIXEL_M, PIXEL_M)
    crs = CRS.from_epsg(32645)

    rng = np.random.default_rng(20260826)
    mask = (rng.random((GRID, GRID)) < 0.02).astype("float32")

    cols = rng.integers(1, GRID - 2, N_FEATURES)
    rows = rng.integers(1, GRID - 2, N_FEATURES)
    geoms = []
    for r, c in zip(rows, cols):
        x = 400_000 + c * PIXEL_M
        y = 3_000_000 - r * PIXEL_M
        geoms.append(box(x, y - PIXEL_M, x + PIXEL_M, y))
    buildings = gpd.GeoDataFrame(
        {"id": [f"n/{i}" for i in range(N_FEATURES)]},
        geometry=geoms,
        crs=crs,
    )

    started = time.perf_counter()
    result = da.assess_buildings(buildings, mask, transform, crs)
    elapsed = time.perf_counter() - started

    assert len(result) == N_FEATURES, (
        f"expected {N_FEATURES} assessed buildings, got {len(result)}"
    )
    assert elapsed < BUDGET_S, (
        f"assessing {N_FEATURES} features on a {GRID}x{GRID} mask took "
        f"{elapsed:.1f}s (budget {BUDGET_S:.0f}s). Rasterise each feature inside "
        "its own pixel window (rasterio.windows) instead of against the full "
        "grid, and never allocate an array the size of the scene per feature."
    )
