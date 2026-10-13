"""Toy-data builders for the Jalnetra logic tests.

Everything here is fabricated, tiny and deterministic: a 200x200 raster on a
0.0002 deg grid (~20 m at this latitude), a handful of OSM-style features, and
helpers to write them out as GeoTIFF / GeoJSON. The whole point is that a test
can state "this building sits in the new flood and that one sits in the river"
and then check the classifier agrees.

Nothing in this module reads or writes the repository's results/ or data/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# --- the toy scene ---------------------------------------------------------
# 200 x 200 pixels, 0.0002 deg per pixel, top-left at (85.60 E, 28.40 N).
# Chosen to mirror the real case study's geography without any real data.
PIXEL_DEG = 0.0002
ORIGIN_LON = 85.60
ORIGIN_LAT = 28.40
SIZE = 200
CRS_EPSG = 4326

# Scene features, in pixel rows/cols (row 0 = north edge).
RIVER_ROWS = (100, 110)      # permanent water: present before AND after
PATCH_ROWS = (60, 90)        # flash-flood patch: only in the after scene
PATCH_COLS = (30, 60)

# Roads run west->east along RIVER_LAT-free latitude 28.38.
ROAD_LAT = 28.380
ROAD_LONS = (85.6010, 85.6133, 85.6256, 85.6380)  # J0..J3, ~1.2 km apart
FLOOD_SEGMENT_COLS = (125, 194)   # flood covers J2->J3 only
FLOOD_SEGMENT_ROWS = (95, 106)


def toy_transform():
    """Affine transform of the toy raster (top-left origin, north-up)."""
    from rasterio.transform import from_origin

    return from_origin(ORIGIN_LON, ORIGIN_LAT, PIXEL_DEG, PIXEL_DEG)


def toy_crs():
    from rasterio.crs import CRS

    return CRS.from_epsg(CRS_EPSG)


def cell_box(row0: int, row1: int, col0: int, col1: int):
    """Shapely box (lon/lat) covering the given pixel block."""
    from shapely.geometry import box

    west = ORIGIN_LON + col0 * PIXEL_DEG
    east = ORIGIN_LON + col1 * PIXEL_DEG
    north = ORIGIN_LAT - row0 * PIXEL_DEG
    south = ORIGIN_LAT - row1 * PIXEL_DEG
    return box(west, south, east, north)


def cell_point(row: int, col: int):
    """Centre of a pixel as a shapely Point."""
    from shapely.geometry import Point

    return Point(
        ORIGIN_LON + (col + 0.5) * PIXEL_DEG,
        ORIGIN_LAT - (row + 0.5) * PIXEL_DEG,
    )


# --- raster helpers --------------------------------------------------------

def _write_tif(path: Path, arrays, transform, crs, nodata=-999.0):
    import rasterio

    path.parent.mkdir(parents=True, exist_ok=True)
    first = np.asarray(arrays[0])
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=first.shape[0],
        width=first.shape[1],
        count=len(arrays),
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress="lzw",
    ) as dst:
        for i, arr in enumerate(arrays, start=1):
            dst.write(np.asarray(arr, dtype="float32"), i)
    return path


def write_mask_tif(path: Path, mask: np.ndarray, valid: np.ndarray | None = None, transform=None, crs=None):
    """Write a flood mask, optionally with a second (validity) band."""
    arrays = [np.asarray(mask, dtype="float32")]
    if valid is not None:
        arrays.append(np.asarray(valid, dtype="float32"))
    return _write_tif(
        path, arrays,
        transform if transform is not None else toy_transform(),
        crs if crs is not None else toy_crs(),
    )


def river_and_patch(size: int = SIZE):
    """Binary before/after water masks: river + (after only) a flood patch.

    Returns (before, after) as float32 arrays. `before` has the river; `after`
    has the river plus the patch, so the *change* is exactly the patch.
    """
    before = np.zeros((size, size), dtype="float32")
    after = np.zeros((size, size), dtype="float32")
    r0, r1 = RIVER_ROWS
    before[r0:r1, :] = 1.0
    after[r0:r1, :] = 1.0
    p_r0, p_r1 = PATCH_ROWS
    p_c0, p_c1 = PATCH_COLS
    after[p_r0:p_r1, p_c0:p_c1] = 1.0
    return before, after


def cutoff_flood_mask(size: int = SIZE) -> np.ndarray:
    """Mask that floods only the final road segment (J2 -> J3)."""
    mask = np.zeros((size, size), dtype="float32")
    r0, r1 = FLOOD_SEGMENT_ROWS
    c0, c1 = FLOOD_SEGMENT_COLS
    mask[r0:r1, c0:c1] = 1.0
    return mask


# --- vector helpers --------------------------------------------------------

def write_geojson(path: Path, features: list[dict], properties: list[dict] | None = None):
    """Write a GeoJSON FeatureCollection from shapely geometries.

    `features` is a list of dicts: {"geometry": <shapely>, **props}. Written in
    EPSG:4326, matching what scripts/fetch_osm.py produces.
    """
    import geopandas as gpd

    path.parent.mkdir(parents=True, exist_ok=True)
    geoms = [f["geometry"] for f in features]
    props = [{k: v for k, v in f.items() if k != "geometry"} for f in features]
    if not geoms:
        path.write_text(json.dumps({"type": "FeatureCollection", "features": []}), encoding="utf-8")
        return path
    gdf = gpd.GeoDataFrame(props, geometry=geoms, crs=f"EPSG:{CRS_EPSG}")
    gdf.to_file(path, driver="GeoJSON")
    return path


def empty_geojson(path: Path):
    """A structurally valid but featureless GeoJSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"type": "FeatureCollection", "features": []}), encoding="utf-8")
    return path


# A valid empty ZIP (end-of-central-directory record only).
_EMPTY_ZIP = b"PK\x05\x06" + b"\x00" * 18


def placeholder_safe_archive(data_dir: Path, label: str) -> Path:
    """Stand in for the SAFE archive a download stage would have produced.

    run_pipeline now refuses to run without the archive each label actually
    downloaded — preprocessing is a required stage, because analysing whatever
    rasters happen to be on disk is how a dashboard ends up showing numbers the
    current code would never produce. Tests that stub run() and only inspect the
    argv therefore need the file to exist; its contents are never read.
    """
    path = Path(data_dir) / "sentinel1" / label / "S1D_TEST_SAFE.SAFE.zip"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_EMPTY_ZIP)
    return path


def toy_road_features():
    """One OSM way (J0..J3) as a single LineString, plus its four junctions."""
    from shapely.geometry import LineString

    line = LineString([(lon, ROAD_LAT) for lon in ROAD_LONS])
    return [{"id": "way/1", "highway": "primary", "name": "Toy Highway", "geometry": line}]


def toy_buildings_features():
    """Settlements along the way + one that no road ever reached."""
    pts = []
    for i, lon in enumerate(ROAD_LONS):
        pts.append({
            "id": f"node/{i}",
            "name": f"Village{i}",
            "building": "yes",
            "geometry": _point(lon, ROAD_LAT),
        })
    # 2.17 km from the nearest junction -> beyond the 2 km snapping radius, so it
    # was already isolated before the flood.
    pts.append({
        "id": "node/iso",
        "name": "Nowhere",
        "building": "yes",
        "geometry": _point(85.6395, 28.3995),
    })
    return pts


def toy_hospitals_features():
    return [{
        "id": "node/hosp",
        "name": "Toy Hospital",
        "amenity": "hospital",
        "geometry": _point(ROAD_LONS[0], ROAD_LAT),
    }]


def toy_bridges_features():
    """One bridge over the permanent river, one over the new flood patch."""
    from shapely.geometry import LineString

    river_row = (RIVER_ROWS[0] + RIVER_ROWS[1]) // 2
    patch_row = (PATCH_ROWS[0] + PATCH_ROWS[1]) // 2
    return [
        {
            "id": "way/bridge-river",
            "man_made": "bridge",
            "geometry": LineString([
                _lonlat(river_row - 8, 150), _lonlat(river_row + 8, 150),
            ]),
        },
        {
            "id": "way/bridge-patch",
            "man_made": "bridge",
            "geometry": LineString([
                _lonlat(patch_row - 8, 45), _lonlat(patch_row + 8, 45),
            ]),
        },
    ]


def toy_damage_buildings_features(valid_block=None):
    """Three buildings: in the new flood, on the permanent river, in a cloud gap."""
    vrow0, vrow1, vcol0, vcol1 = valid_block or (0, 0, 0, 0)
    return [
        {"id": "b/flood", "name": "In the new flood",
         "geometry": cell_box(PATCH_ROWS[0] + 5, PATCH_ROWS[0] + 15,
                              PATCH_COLS[0] + 5, PATCH_COLS[0] + 15)},
        {"id": "b/river", "name": "On the permanent river",
         "geometry": cell_box(RIVER_ROWS[0] + 1, RIVER_ROWS[0] + 6, 150, 156)},
        {"id": "b/cloud", "name": "Under a cloud gap",
         "geometry": cell_box(vrow0 + 1, vrow0 + 5, vcol0 + 1, vcol0 + 5)},
    ]


def toy_damage_roads_features():
    from shapely.geometry import LineString

    mid = (PATCH_ROWS[0] + PATCH_ROWS[1]) // 2
    return [{
        "id": "way/road",
        "highway": "primary",
        "geometry": LineString([_lonlat(mid, 20), _lonlat(mid, 70)]),
    }]


def _point(lon, lat):
    from shapely.geometry import Point

    return Point(lon, lat)


def _lonlat(row, col):
    return (
        ORIGIN_LON + (col + 0.5) * PIXEL_DEG,
        ORIGIN_LAT - (row + 0.5) * PIXEL_DEG,
    )
