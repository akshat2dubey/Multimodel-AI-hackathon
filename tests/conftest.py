"""Shared fixtures for the Jalnetra logic-test suite.

These are LOGIC TESTS ON TOY DATA. Every input is generated into `tmp_path`;
there is no network access, no Copernicus credential, no GPU and no model
download. A green run says the decision logic behaves as documented — it says
nothing about whether a flood map is scientifically correct.

Project rules this suite enforces on itself:
  * nothing is written into the repository's results/ or data/;
  * everything is deterministic (fixed seeds, fixed geometry).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
DASHBOARD = ROOT / "dashboard"

# Make the pipeline importable the same way the CLI scripts do.
for _p in (str(ROOT), str(SCRIPTS), str(DASHBOARD)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Timestamp taken before any test runs, used to prove results/ and data/ were
# never touched during the session.
SESSION_START = time.time()


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def scripts_dir() -> Path:
    return SCRIPTS


@pytest.fixture(scope="session")
def dashboard_dir() -> Path:
    return DASHBOARD


@pytest.fixture(scope="session")
def session_start() -> float:
    return SESSION_START


@pytest.fixture
def toy_scene(tmp_path: Path) -> dict:
    """Write the toy raster + OSM feature files into tmp_path.

    Returns a dict of paths plus the transform/CRS and the raw masks, so tests
    can reason about both the files and the arrays.
    """
    import toydata  # tests/ is on sys.path (pytest rootdir insertion)

    transform = toydata.toy_transform()
    crs = toydata.toy_crs()

    scene = tmp_path / "scene"
    before, after = toydata.river_and_patch()
    # Delivered flood mask: new water only (the same rule inference.py applies).
    mask_path = toydata.write_mask_tif(
        scene / "flood_mask.tif", toydata.cutoff_flood_mask(), transform=transform, crs=crs
    )

    osm = scene / "osm"
    toydata.write_geojson(osm / "roads.geojson", toydata.toy_road_features())
    toydata.write_geojson(osm / "buildings.geojson", toydata.toy_buildings_features())
    toydata.write_geojson(osm / "hospitals.geojson", toydata.toy_hospitals_features())
    toydata.write_geojson(osm / "bridges.geojson", toydata.toy_bridges_features())

    return {
        "root": scene,
        "mask_path": mask_path,
        "transform": transform,
        "crs": crs,
        "before": before,
        "after": after,
        "osm": osm,
        "roads": osm / "roads.geojson",
        "buildings": osm / "buildings.geojson",
        "hospitals": osm / "hospitals.geojson",
        "bridges": osm / "bridges.geojson",
    }
