#!/usr/bin/env python3
"""
Offline demo/pre-flight dataset generator  —  *** SYNTHETIC DATA ***

Why this exists
---------------
The real pipeline needs Copernicus Data Space credentials (Sentinel-1/2 + DEM)
and a pre-event OSM snapshot. Until those land, nothing downstream can be
exercised: damage assessment, cut-off analysis, flood-path tracing, the
situation report and the dashboard all stay empty.

This script fabricates a *geometrically realistic but entirely synthetic*
scene over the Bhote Koshi - Trishuli corridor:

  data_demo/dem/dem.tif                        valley DEM (WorldDEM-30 stand-in)
  data_demo/sentinel1/before/*_VV.tif *_VH.tif pre-event radar (dB)
  data_demo/sentinel1/after/*_VV.tif  *_VH.tif post-event radar (dB)
  data_demo/sentinel2/before|after/s2_*_B0x.tif optical bands + SCL
  data_demo/osm/*.geojson                      buildings/roads/bridges/hospitals/settlements
  results_demo/flood_mask.tif                  newly flooded only (reference mask)
  results_demo/water_after.tif                 all post-event water (soft probability)
  results_demo/flood_mask_before.tif           pre-event water = permanent-water mask
  results_demo/valid_mask.tif                  pixels the scene actually observed
  results_demo/change_detection.json           change-detection statistics

Neither directory is data/ or results/. Those hold the real downloads and the
real pipeline output, and this script refuses to write into them (or to overwrite
any input file it did not create itself) unless --force is passed.

THE OUTPUT IS NOT REAL EARTH OBSERVATION DATA. It must never be presented as
measurements of the August 2026 Nepal event, and none of it belongs in the
hackathon submission as evidence. Its only jobs are:

  1. prove every stage of the pipeline runs and the artifacts chain together;
  2. let the dashboard/report be developed and demoed before data arrives;
  3. serve as a regression fixture when the real Sentinel-1 pair lands.

Attribution for the *real* data sources is unchanged and still required.
"""

from __future__ import annotations

import argparse

import sys
import json
import math
import random
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy.ndimage import distance_transform_edt, gaussian_filter

# The same speckle rule the real inference applies, so flood_mask.tif from the
# fixture and from a real run are the same kind of object.
from shapely import affinity
from shapely.geometry import LineString, Point, box, mapping

# The project root must be importable so config.presets resolves whether this is
# run as `python demo/make_demo_data.py` or imported by run_offline.py.
_ROOT = Path(__file__).resolve().parent.parent
for _path in (_ROOT / "scripts", _ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from mask_postprocess import MIN_COMPONENT_PIXELS, remove_small_components  # noqa: E402

from config.presets import (  # noqa: E402  (path set up above)
    CENTER_LAT,
    CENTER_LON,
    DATA_DEMO_NAME,
    DEMO_NOTICE_FILENAME,
    FIXTURE_EXTENT as DEFAULT_EXTENT_DEG,
    EVENT_DATE,
    OSM_SNAPSHOT_DATE,
    RESULTS_DEMO_NAME,
    RESULTS_NAME,
    demo_paths_error,
)

if hasattr(sys.stdout, "reconfigure"):  # never crash on a cp1252 Windows console
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ----------------------------------------------------------------------------
# Scene definition
# ----------------------------------------------------------------------------

# Case-study geometry lives in config/presets.py — this module fabricates a
# synthetic scene for it, and must not fork the coordinates.
DEFAULT_CENTER = (CENTER_LAT, CENTER_LON)  # Bhote Koshi -> Trishuli corridor
DEFAULT_EXTENT = DEFAULT_EXTENT_DEG        # degrees, square scene
DEFAULT_SIZE = 640               # pixels per side
DEFAULT_SEED = 20260826          # deterministic: same scene every run

# Written into a generated data directory to record that this script owns its
# contents. Its absence is what stops a demo run from clobbering real downloads.
GENERATOR_MARKER = ".generated_by_make_demo_data"

# Villages, ordered upstream (Tibet border) -> downstream (Trishuli/Dhading).
# `progress` is position along the river, 0 = source, 1 = outlet.
VILLAGES = [
    ("Kodari", 0.03, 1200),
    ("Tatopani", 0.11, 2400),
    ("Larcha", 0.18, 900),
    ("Jhirpu", 0.26, 1400),
    ("Barhabise", 0.35, 9800),
    ("Ramche", 0.45, 2100),
    ("Thuloparsel", 0.54, 3300),
    ("Balephi", 0.64, 4200),
    ("Jalbire", 0.73, 5100),
    ("Sunkhani", 0.82, 2600),
    ("Maneswanra", 0.91, 3600),
]

# Where the highway switches banks = where a bridge must exist.
BRIDGE_PROGRESS = [0.22, 0.55, 0.78]

# Hospitals: reference towns for the cut-off analysis, downstream and central.
HOSPITALS = [("Barhabise Hospital", 0.35), ("Jalbire Health Post", 0.73),
             ("Maneswanra Primary Health Centre", 0.91)]

HIGHWAY_NAME = "Araniko Highway"
HIGHWAY_REF = "H03"


# ----------------------------------------------------------------------------
# Grid + river geometry
# ----------------------------------------------------------------------------

def make_grid(center_lat, center_lon, extent, size):
    """Return (transform, lon_grid, lat_grid)."""
    west = center_lon - extent / 2
    north = center_lat + extent / 2
    res = extent / size
    transform = from_origin(west, north, res, res)
    rows, cols = np.mgrid[0:size, 0:size]
    lons = west + (cols + 0.5) * res
    lats = north - (rows + 0.5) * res
    return transform, lons, lats


def river_centerline(size, n=400):
    """Sinuous centreline from the NE (glacier) to the SW (Trishuli).

    Returns (rows, cols, progress) each of length n.
    """
    t = np.linspace(0.0, 1.0, n)
    rows = 0.04 * size + t * 0.92 * size + 0.055 * size * np.sin(5.0 * math.pi * t)
    cols = 0.94 * size - t * 0.86 * size + 0.06 * size * np.sin(3.4 * math.pi * t + 1.1)
    rows = np.clip(rows, 1, size - 2)
    cols = np.clip(cols, 1, size - 2)
    return rows, cols, t


def centerline_index_raster(rows, cols, size):
    """Raster holding the centreline sample index nearest each centreline pixel."""
    idx = np.full((size, size), -1, dtype=np.int32)
    idx[np.round(rows).astype(int), np.round(cols).astype(int)] = np.arange(len(rows))
    return idx


def distance_and_progress(rows, cols, size):
    """Per-pixel distance to the river (px) and position along it (0..1)."""
    idx = centerline_index_raster(rows, cols, size)
    d_riv, (ind_r, ind_c) = distance_transform_edt(idx < 0, return_indices=True)
    nearest_idx = idx[ind_r, ind_c]
    nearest_idx = np.clip(nearest_idx, 0, len(rows) - 1)
    progress = nearest_idx / (len(rows) - 1)
    return d_riv.astype(np.float64), progress


def soft_extent(distance_px, half_width_px, edge_px=1.3):
    """Logistic falloff -> flood probability in [0, 1]."""
    return 1.0 / (1.0 + np.exp(-(half_width_px - distance_px) / edge_px))


def blob(size, row, col, radius, edge):
    """A circular off-river flood patch (dam-break / valley impoundment)."""
    rows, cols = np.mgrid[0:size, 0:size]
    d = np.hypot(rows - row, cols - col)
    return 1.0 / (1.0 + np.exp(-(radius - d) / edge))


# ----------------------------------------------------------------------------
# Raster writers
# ----------------------------------------------------------------------------

def write_raster(path: Path, array: np.ndarray, transform, nodata=None,
                 dtype="float32", cmap_description=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": array.shape[0],
        "width": array.shape[1],
        "count": 1,
        "dtype": dtype,
        "crs": "EPSG:4326",
        "transform": transform,
        "compress": "lzw",
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(dtype), 1)
        if cmap_description:
            dst.update_tags(DESCRIPTION=cmap_description)
    return path


# ----------------------------------------------------------------------------
# Terrain
# ----------------------------------------------------------------------------

def build_dem(size, d_riv, progress, transform, rng):
    """Valley DEM: steep Himalayan walls, monotonic descent to the outlet."""
    res_deg = transform.a
    lat_mid = transform.f
    m_per_deg = 111320.0
    m_per_px = res_deg * m_per_deg
    d_riv_m = d_riv * m_per_px

    elev = 360.0 + 5400.0 * (1.0 - progress) ** 1.5
    elev = elev + 1500.0 * (1.0 - np.exp(-d_riv_m / 900.0))
    # a couple of ridge lines so the flow routing has something to do
    elev = elev + 260.0 * np.sin(2.0 * math.pi * progress * 3.0)
    elev = elev + gaussian_filter(rng.normal(0, 1, (size, size)), sigma=6) * 90.0
    return np.clip(elev, 120.0, 8200.0)


# ----------------------------------------------------------------------------
# Sentinel-1 (radar, dB)
# ----------------------------------------------------------------------------

def build_sentinel1(size, flood_prob, dem, rng):
    """Backscatter in dB. Open water is specular -> very low VV/VH."""
    slope = np.hypot(*np.gradient(dem))
    slope_n = np.clip(slope / (np.percentile(slope, 99) + 1e-6), 0, 1)

    land_vv = -7.5 - 4.0 * slope_n + rng.normal(0, 1.1, (size, size))
    land_vh = -13.5 - 3.0 * slope_n + rng.normal(0, 1.2, (size, size))
    water = flood_prob > 0.5
    vv = np.where(water, -19.0 + rng.normal(0, 1.0, (size, size)), land_vv)
    vh = np.where(water, -25.0 + rng.normal(0, 1.1, (size, size)), land_vh)
    return vv.astype("float32"), vh.astype("float32"), flood_prob.astype("float32")


# ----------------------------------------------------------------------------
# Sentinel-2 (optical reflectance + scene classification)
# ----------------------------------------------------------------------------

def build_sentinel2(size, flood_prob, dem, rng, cloud_cover):
    water = flood_prob > 0.5
    veg = (dem < np.percentile(dem, 55)) & ~water
    bare = ~water & ~veg

    def band(w, v, b, noise=0.012):
        out = np.where(water, w, np.where(veg, v, b))
        return np.clip(out + rng.normal(0, noise, (size, size)), 0.0, 1.0)

    bands = {
        "B02": band(0.020, 0.045, 0.095),
        "B03": band(0.022, 0.075, 0.115),
        "B04": band(0.018, 0.060, 0.150),
        "B08": band(0.015, 0.330, 0.240),
        "B11": band(0.008, 0.230, 0.290),
        "B12": band(0.006, 0.140, 0.250),
    }

    scl = np.where(water, 6, np.where(veg, 4, 5)).astype("uint8")  # 6=water 4=veg 5=bare
    if cloud_cover > 0:
        n_clouds = max(1, int(round(cloud_cover / 12)))
        cloud = np.zeros((size, size), dtype=bool)
        for _ in range(n_clouds):
            r = rng.uniform(0.08, 0.22) * size
            row = rng.uniform(0, size)
            col = rng.uniform(0, size)
            rows, cols = np.mgrid[0:size, 0:size]
            cloud |= np.hypot(rows - row, cols - col) < r
        scl[cloud] = 9  # 9 = cloud high probability
        for k in bands:
            bands[k] = np.where(cloud, 0.45, bands[k]).astype("float32")
    return bands, scl


# ----------------------------------------------------------------------------
# OSM-like vector layers
# ----------------------------------------------------------------------------

def river_lonlat(rows, cols, transform):
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    return np.asarray(xs), np.asarray(ys)


def unit_normal(rows, cols):
    """Per-sample unit normal of the centreline, in pixel space."""
    d_r = np.gradient(rows)
    d_c = np.gradient(cols)
    norm = np.hypot(d_r, d_c) + 1e-9
    return -d_c / norm, d_r / norm  # (-dc, dr) / |.|


def bank_sign(progress):
    s = 1.0
    for cp in BRIDGE_PROGRESS:
        if progress > cp:
            s = -s
    return s


def build_highway(rows, cols, progress, transform, rng, offset_px=14.0):
    """Araniko Highway: hugs one bank, switches banks (over a bridge) 3 times."""
    nr, nc = unit_normal(rows, cols)
    offs = np.array([bank_sign(p) * offset_px for p in progress])
    meander = 3.0 * np.sin(6.0 * math.pi * progress) + rng.normal(0, 0.8, len(progress))
    hr = rows + nr * offs + nr * 0.0 + meander * 0.0
    hc = cols + nc * offs + meander

    hxs, hys = river_lonlat(hr, hc, transform)
    coords = list(zip(hxs.tolist(), hys.tolist()))

    # split into road segments so the network graph gets real junctions
    n_seg = 6
    per = max(2, len(coords) // n_seg - 1)
    features = []
    for i in range(0, len(coords) - 1, per):
        chunk = coords[i:i + per + 1]
        if len(chunk) < 2:
            continue
        mid = chunk[len(chunk) // 2]
        features.append({
            "type": "Feature",
            "properties": {
                "id": f"hw-{len(features)}",
                "highway": "primary" if i < len(coords) // 2 else "secondary",
                "name": HIGHWAY_NAME,
                "ref": HIGHWAY_REF,
                "surface": "asphalt",
                "oneway": "no",
                "settlement": mid[0],
            },
            "geometry": mapping(LineString(chunk)),
        })
    return features, (hr, hc)


def village_pixel_pos(village_progress, rows, cols):
    i = int(round(village_progress * (len(rows) - 1)))
    return rows[i], cols[i], i


def build_buildings(size, rows, cols, transform, rng):
    """Building footprint clusters per village.

    Each village gets a 'riverside' row of footprints (placed inside the
    post-event flood envelope so the damage classes are populated) plus the
    bulk of the settlement further up the bank.
    """
    nr, nc = unit_normal(rows, cols)
    features = []
    counter = 0

    for name, progress, population in VILLAGES:
        vrow, vcol, i = village_pixel_pos(progress, rows, cols)
        n = int(np.clip(population / 420, 6, 26))

        # riverside band: 20% of footprints, 1.5-4.5 px off the centreline
        n_river = max(2, int(round(n * 0.22)))
        placements = []
        for k in range(n_river):
            side = 1.0 if k % 2 == 0 else -1.0
            offset = rng.uniform(1.5, 4.5) * side
            placements.append((vrow + nr[i] * offset + rng.normal(0, 1.0),
                               vcol + nc[i] * offset + rng.normal(0, 1.0)))
        # settled bank: the rest, 13-24 px out (clear of the flood envelope)
        for k in range(n - n_river):
            offset = rng.uniform(13.0, 24.0) * (1.0 if rng.random() < 0.7 else -1.0)
            placements.append((vrow + nr[i] * offset + rng.normal(0, 4.0),
                               vcol + nc[i] * offset + rng.normal(0, 4.0)))

        for prow, pcol in placements:
            prow = float(np.clip(prow, 2, size - 3))
            pcol = float(np.clip(pcol, 2, size - 3))
            # Footprints are deliberately 1-2.5 px across (~90-220 m) rather than
            # true house size: a real 10 m building is sub-pixel at this grid and
            # would exercise only the centroid fallback, not the area-overlap path.
            w_px = rng.uniform(1.1, 2.4)
            h_px = rng.uniform(1.0, 2.2)
            poly = box(pcol - w_px / 2, prow - h_px / 2, pcol + w_px / 2, prow + h_px / 2)
            poly = affinity.rotate(poly, rng.uniform(0, 180), origin="center")
            xs, ys = rasterio.transform.xy(transform, poly.exterior.xy[1], poly.exterior.xy[0])
            ring = list(zip(np.asarray(xs).tolist(), np.asarray(ys).tolist()))
            counter += 1
            features.append({
                "type": "Feature",
                "properties": {
                    "id": counter,
                    "building": rng.choice(["yes", "house", "residential"]),
                    "name": name,
                    "addr:place": name,
                    "village": name,
                    "population": population,
                },
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            })
    return features


def build_spurs(rows, cols, progress, transform, highway_rc, rng):
    """Track roads linking each village to the highway."""
    hr, hc = highway_rc
    features = []
    for idx, (name, vp, _) in enumerate(VILLAGES):
        i = int(round(vp * (len(rows) - 1)))
        # village reference point ~18 px off the centreline (same side as before)
        nr, nc = unit_normal(rows, cols)
        vr = rows[i] + nr[i] * 18.0
        vc = cols[i] + nc[i] * 18.0
        # nearest highway vertex
        d = np.hypot(hr - vr, hc - vc)
        j = int(np.argmin(d))
        mid_r, mid_c = (vr + hr[j]) / 2 + rng.normal(0, 1.5), (vc + hc[j]) / 2 + rng.normal(0, 1.5)
        pr = np.array([vr, mid_r, hr[j]])
        pc = np.array([vc, mid_c, hc[j]])
        xs, ys = river_lonlat(pr, pc, transform)
        features.append({
            "type": "Feature",
            "properties": {
                "id": f"spur-{idx}",
                "highway": "unclassified" if vp > 0.3 else "track",
                "name": f"{name} access road",
                "surface": "gravel",
                "settlement": name,
            },
            "geometry": mapping(LineString(list(zip(np.asarray(xs).tolist(),
                                                    np.asarray(ys).tolist())))),
        })
    return features


def build_bridges(rows, cols, transform):
    features = []
    for k, p in enumerate(BRIDGE_PROGRESS):
        i = int(round(p * (len(rows) - 1)))
        xs, ys = river_lonlat(np.array([rows[i]]), np.array([cols[i]]), transform)
        features.append({
            "type": "Feature",
            "properties": {
                "id": f"bridge-{k}",
                "bridge": "yes",
                "highway": "primary",
                "name": f"{HIGHWAY_NAME} bridge {k + 1}",
            },
            "geometry": mapping(Point(float(xs[0]), float(ys[0]))),
        })
    return features


def build_hospitals(rows, cols, transform):
    features = []
    for k, (name, p) in enumerate(HOSPITALS):
        i = int(round(p * (len(rows) - 1)))
        xs, ys = river_lonlat(np.array([rows[i]]), np.array([cols[i]]), transform)
        features.append({
            "type": "Feature",
            "properties": {
                "id": f"hospital-{k}",
                "amenity": "hospital",
                "name": name,
                "healthcare": "hospital",
            },
            "geometry": mapping(Point(float(xs[0]), float(ys[0]))),
        })
    return features


def build_settlements(rows, cols, transform):
    features = []
    for k, (name, p, population) in enumerate(VILLAGES):
        i = int(round(p * (len(rows) - 1)))
        xs, ys = river_lonlat(np.array([rows[i]]), np.array([cols[i]]), transform)
        features.append({
            "type": "Feature",
            "properties": {
                "id": f"place-{k}",
                "place": "village" if population < 5000 else "town",
                "name": name,
                "population": population,
            },
            "geometry": mapping(Point(float(xs[0]), float(ys[0]))),
        })
    return features


def write_geojson(path: Path, features, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "type": "FeatureCollection",
        "metadata": metadata,
        "features": features,
    }, indent=2, default=str), encoding="utf-8")
    return path


# ----------------------------------------------------------------------------
# Clobber protection
# ----------------------------------------------------------------------------

def unmarked_contents(directory: Path) -> list[Path]:
    """Files in `directory` that this script did not write.

    A directory is considered ours only when it carries GENERATOR_MARKER, which
    is dropped after a successful run. Anything else — the 258 MB Copernicus DEM
    already in data/, a real Sentinel scene, an OSM snapshot — counts as foreign
    evidence, and overwriting it would destroy the only real input in the repo.
    """
    if not directory.is_dir():
        return []
    if (directory / GENERATOR_MARKER).is_file():
        return []
    return sorted(p for p in directory.rglob("*") if p.is_file())


def refuse_if_occupied(directory: Path, flag: str, force: bool) -> bool:
    """Return True when it is safe to write into `directory`."""
    if force:
        return True
    foreign = unmarked_contents(directory)
    if not foreign:
        return True
    shown = "\n".join(f"    {p}" for p in foreign[:8])
    more = f"\n    ... and {len(foreign) - 8} more" if len(foreign) > 8 else ""
    print(
        f"\n  ✋ refusing to write into {directory}: it already holds "
        f"{len(foreign)} file(s) this script did not generate.\n"
        f"{shown}{more}\n\n"
        f"  Those may be real downloads. Overwriting them would destroy evidence\n"
        f"  that cannot be re-downloaded without a Copernicus account.\n\n"
        f"  Pick one:\n"
        f"    {flag} <empty-dir>   build the demo somewhere else\n"
        f"    --force          overwrite anyway (only if you are certain)\n",
        file=sys.stderr,
    )
    return False


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate SYNTHETIC offline demo data (not real EO data)")
    parser.add_argument("--center-lat", type=float, default=DEFAULT_CENTER[0])
    parser.add_argument("--center-lon", type=float, default=DEFAULT_CENTER[1])
    parser.add_argument("--extent", type=float, default=DEFAULT_EXTENT)
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--root", type=Path, default=None,
                        help="Project root (default: repository root)")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Where the synthetic inputs go. Default: "
                             f"<root>/{DATA_DEMO_NAME}, NOT data/ — that tree holds "
                             "real downloads, and this script will not overwrite "
                             "files it did not create")
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="Where the synthetic results go. Default: "
                             f"<root>/{RESULTS_DEMO_NAME} (kept OUT of results/ so a "
                             "demo run can never be mistaken for a real one)")
    parser.add_argument("--force", action="store_true",
                        help="Allow writing into a non-empty directory this script "
                             "did not generate, or into results/")
    parser.add_argument("--after-cloud-cover", type=float, default=38.0,
                        help="Simulated cloud cover %% in the post-event S2 scene")
    args = parser.parse_args()

    root = args.root or Path(__file__).resolve().parent.parent
    # Demo inputs go to data_demo/, NOT data/: data/ holds the real Copernicus DEM
    # and the real satellite/OSM downloads, and this script's output is fabricated.
    data = args.data_dir or (root / DATA_DEMO_NAME)
    # Demo output goes to results_demo/, NOT results/. Keeping them apart is what
    # lets the dashboard tell a fabricated fixture from a real run.
    results = args.results_dir or (root / RESULTS_DEMO_NAME)

    unsafe_reason = demo_paths_error(data, results)
    if unsafe_reason:
        print(f"\n  ✋ {unsafe_reason}", file=sys.stderr)
        return 1

    if results.name == RESULTS_NAME and not args.force:
        print(
            f"\n  ✋ refusing to write into {results}: that is the real results "
            f"directory.\n     A demo must not be able to overwrite a real run's "
            f"artifacts.\n     Use --results-dir {RESULTS_DEMO_NAME} (the default) "
            f"or pass --force.\n",
            file=sys.stderr,
        )
        return 1
    if not refuse_if_occupied(data, "--data-dir", args.force):
        return 1

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    size = args.size
    transform, lons, lats = make_grid(
        args.center_lat, args.center_lon, args.extent, size)
    res = transform.a

    print("=" * 72)
    print(" SYNTHETIC DEMO DATA  —  not real Earth observation data")
    print("=" * 72)
    print(f"  Centre:  {args.center_lat}, {args.center_lon}")
    print(f"  Scene:   {size}x{size} px  ({args.extent} deg  ~ {res * 111320:.0f} m/px)")
    print(f"  Seed:    {args.seed}")

    # --- river geometry -----------------------------------------------------
    rows, cols, progress_cl = river_centerline(size)
    d_riv, progress = distance_and_progress(rows, cols, size)

    half_before = 1.1 + 0.7 * progress                 # px
    half_after = 2.6 + 4.6 * progress                  # px
    prob_before = soft_extent(d_riv, half_before)
    prob_after = soft_extent(d_riv, half_after)

    # three dam-break / impoundment patches upstream of Barhabise
    for p, radius in ((0.30, 16.0), (0.42, 12.0), (0.66, 10.0)):
        i = int(round(p * (len(rows) - 1)))
        prob_after = np.maximum(
            prob_after, blob(size, rows[i] + 6, cols[i] + 9, radius, 2.2))
    prob_after = np.clip(prob_after, 0.0, 1.0)

    print(f"\n  Pre-event water:  {(prob_before > 0.5).mean() * 100:.2f}% of scene")
    print(f"  Post-event flood: {(prob_after > 0.5).mean() * 100:.2f}% of scene")

    # --- DEM ----------------------------------------------------------------
    dem = build_dem(size, d_riv, progress, transform, rng)
    write_raster(data / "dem" / "dem.tif", dem, transform, nodata=-9999.0,
                 cmap_description="SYNTHETIC DEM (WorldDEM-30 stand-in)")
    (data / "dem").mkdir(parents=True, exist_ok=True)
    (data / "dem" / "PRODUCT_INFO.json").write_text(json.dumps({
        "product_type": "SYNTHETIC DEM — WorldDEM-30 stand-in",
        "synthetic": True,
        "resolution": f"{res * 111320:.1f} m (EPSG:4326)",
        "note": "Fabricated for pipeline testing. Replace with real Copernicus "
                "WorldDEM-30 before submission.",
    }, indent=2), encoding="utf-8")
    print(f"\n  DEM:            {data / 'dem' / 'dem.tif'}  "
          f"({dem.min():.0f}-{dem.max():.0f} m)")

    # --- Sentinel-1 ---------------------------------------------------------
    for label, prob in (("before", prob_before), ("after", prob_after)):
        vv, vh, _ = build_sentinel1(size, prob, dem, rng)
        st = transform
        write_raster(data / "sentinel1" / label / f"s1_{label}_VV.tif", vv, st,
                     nodata=-9999.0)
        write_raster(data / "sentinel1" / label / f"s1_{label}_VH.tif", vh, st,
                     nodata=-9999.0)
    print(f"  Sentinel-1:     2 scenes x VV+VH (dB)")

    # --- Sentinel-2 ---------------------------------------------------------
    after_scl = None
    for label, prob, cc in (("before", prob_before, 14.0),
                            ("after", prob_after, args.after_cloud_cover)):
        bands, scl = build_sentinel2(size, prob, dem, rng, cc)
        if label == "after":
            after_scl = scl
        for bname, arr in bands.items():
            write_raster(data / "sentinel2" / label / f"s2_{label}_{bname}.tif",
                         arr, transform, dtype="float32")
        write_raster(data / "sentinel2" / label / f"s2_{label}_SCL.tif", scl,
                     transform, dtype="uint8")
    print(f"  Sentinel-2:     2 scenes x 6 bands + SCL "
          f"(after-scene cloud cover {args.after_cloud_cover:.0f}%)")

    # --- OSM ----------------------------------------------------------------
    osm = data / "osm"
    features, highway_rc = build_highway(rows, cols, progress_cl, transform, rng)
    roads = features + build_spurs(rows, cols, progress_cl, transform, highway_rc, rng)
    buildings = build_buildings(size, rows, cols, transform, rng)
    bridges = build_bridges(rows, cols, transform)
    hospitals = build_hospitals(rows, cols, transform)
    settlements = build_settlements(rows, cols, transform)

    meta = {
        "source": "SYNTHETIC — structurally mimics an ohsome/OSM pre-event snapshot",
        "synthetic": True,
        "snapshot_date": OSM_SNAPSHOT_DATE,
        "event_date": EVENT_DATE,
        "attribution": "© OpenStreetMap contributors. (synthetic stand-in)",
    }
    write_geojson(osm / "buildings.geojson", buildings, meta)
    write_geojson(osm / "roads.geojson", roads, meta)
    write_geojson(osm / "bridges.geojson", bridges, meta)
    write_geojson(osm / "hospitals.geojson", hospitals, meta)
    write_geojson(osm / "settlements.geojson", settlements, meta)
    (osm / "meta.json").write_text(json.dumps(
        {**meta, "counts": {
            "buildings": len(buildings), "roads": len(roads),
            "bridges": len(bridges), "hospitals": len(hospitals),
            "settlements": len(settlements)}}, indent=2), encoding="utf-8")
    print(f"  OSM:            {len(buildings)} buildings, {len(roads)} road segments, "
          f"{len(bridges)} bridges, {len(hospitals)} hospitals")

    # --- reference flood masks + change statistics --------------------------
    # Mirror of the artifact contract scripts/inference.py writes, so this fixture
    # is a faithful stand-in for a real run:
    #   flood_mask.tif        newly flooded only (after AND NOT before), speckle removed
    #   water_after.tif       all post-event water, permanent water included
    #   flood_mask_before.tif pre-event water = the permanent-water mask
    #   valid_mask.tif        pixels the scene actually observed (no cloud/shadow)
    results.mkdir(parents=True, exist_ok=True)

    before_bin = prob_before > 0.5
    after_bin = prob_after > 0.5
    newly = after_bin & ~before_bin
    # Same speckle rule as inference.py: isolated dark pixels are not a flood.
    delivered_bin, removed_blobs = remove_small_components(newly, MIN_COMPONENT_PIXELS)
    # Deliver a probability that thresholds to exactly the delivered binary mask.
    flood_prob = np.where(delivered_bin > 0.5, prob_after, 0.0).astype("float32")

    write_raster(results / "flood_mask_before.tif", prob_before.astype("float32"),
                 transform, nodata=-9999.0,
                 cmap_description="SYNTHETIC pre-event water (permanent water)")
    write_raster(results / "water_after.tif", prob_after.astype("float32"),
                 transform, nodata=-9999.0,
                 cmap_description="SYNTHETIC post-event water (permanent included)")
    write_raster(results / "flood_mask.tif", flood_prob,
                 transform, nodata=-9999.0,
                 cmap_description="SYNTHETIC newly-flooded probability (reference mask)")

    if after_scl is not None:
        # SCL 0 = no data, 1 = saturated/defective, 3 = cloud shadow, 8/9/10 = cloud.
        valid = ~np.isin(after_scl, [0, 1, 3, 8, 9, 10])
        write_raster(results / "valid_mask.tif", valid.astype("float32"),
                     transform, nodata=None,
                     cmap_description="SYNTHETIC observed-pixel mask (1 = observed)")

    # area per pixel in metres, from the geographic transform at scene centre
    m_per_deg = 111320.0
    px_h_m = abs(transform.e) * m_per_deg
    px_w_m = abs(transform.a) * m_per_deg * math.cos(math.radians(args.center_lat))
    pixel_ha = (px_h_m * px_w_m) / 10000.0
    total_ha = size * size * pixel_ha

    delivered_px = int(delivered_bin.sum())
    change = {
        "flood_extent_pixels": delivered_px,
        "flood_extent_ha": round(delivered_px * pixel_ha, 2),
        "persistent_water_pixels": int((after_bin & before_bin).sum()),
        "persistent_water_ha": round((after_bin & before_bin).sum() * pixel_ha, 2),
        "total_water_after_pixels": int(after_bin.sum()),
        "total_water_after_ha": round(after_bin.sum() * pixel_ha, 2),
        "disappeared_pixels": int((~after_bin & before_bin).sum()),
        "flood_percentage": round(delivered_px / max(after_bin.sum(), 1) * 100, 1),
        # What the delivered flood_mask.tif actually is.
        "flood_extent_raw_pixels": int(newly.sum()),
        "small_components_removed_pixels": int(newly.sum()) - delivered_px,
        "small_components_removed": removed_blobs,
        "min_component_pixels": int(MIN_COMPONENT_PIXELS),
        "newly_flooded_pixels": delivered_px,
        "newly_flooded_ha": round(delivered_px * pixel_ha, 2),
        "permanent_water_pixels": int((after_bin & before_bin).sum()),
        "water_after_pixels": int(after_bin.sum()),
        "delivered_mask": "flood_mask.tif",
        "water_after_mask": "water_after.tif",
        "permanent_water_mask": "flood_mask_before.tif",
        "valid_mask": "valid_mask.tif",
        "method": "synthetic_fixture",
        # keys consumed by dashboard/app.py + index.html
        "flood_pixels": delivered_px,
        "total_pixels": int(size * size),
        "flood_area_ha": round(delivered_px * pixel_ha, 2),
        "total_area_ha": round(total_ha, 1),
        "resolution_m": round(px_w_m, 2),
        "mean_flood_probability": round(float(flood_prob[delivered_bin > 0.5].mean()), 3),
        "shape": [size, size],
        "crs": "EPSG:4326",
        "data_source": "SYNTHETIC demo dataset (demo/make_demo_data.py)",
        "synthetic": True,
    }
    if after_scl is not None:
        valid = ~np.isin(after_scl, [0, 1, 3, 8, 9, 10])
        change["valid_pixels"] = int(valid.sum())
        change["invalid_pixels"] = int(valid.size - valid.sum())
        change["valid_percentage"] = round(valid.sum() / max(valid.size, 1) * 100, 1)
    (results / "change_detection.json").write_text(json.dumps(change, indent=2), encoding="utf-8")
    print(f"  Flood mask:     {results / 'flood_mask.tif'}  (newly flooded only)")
    print(f"  Water after:    {results / 'water_after.tif'}")
    print(f"  Newly flooded:  {change['flood_extent_ha']:,} ha "
          f"({change['flood_extent_pixels']:,} px; "
          f"{change['small_components_removed_pixels']:,} px dropped as speckle)")

    (results / DEMO_NOTICE_FILENAME).write_text(
        "*** SYNTHETIC DEMO DATA — DO NOT SUBMIT AS EVIDENCE ***\n\n"
        f"Everything in {data.name}/ written by demo/make_demo_data.py, and every\n"
        f"result in {results.name}/, is fabricated geometry. It exists so that\n"
        "damage_assessment.py, cutoff_analysis.py, flood_path.py,\n"
        "generate_report.py and the dashboard can be developed, demoed and\n"
        "regression-tested before the real Copernicus Data Space downloads land.\n\n"
        "These results are deliberately NOT written to results/: the dashboard only\n"
        "reads demo data when it is pointed at it, and shows a banner when it does.\n\n"
        "Replace with real inputs before any submission:\n"
        "  data/sentinel1/  <- Copernicus Sentinel-1 GRD, same orbit track\n"
        "  data/sentinel2/  <- Copernicus Sentinel-2 L2A\n"
        "  data/dem/        <- Copernicus WorldDEM-30\n"
        f"  data/osm/        <- OSM snapshot dated on/before {OSM_SNAPSHOT_DATE}\n"
        "  results/flood_mask.tif <- output of scripts/inference.py\n",
        encoding="utf-8",
    )

    # Mark the input tree as generated: a later run (e.g. `run_offline.py
    # --skip-data-generation` then a re-generate) may overwrite its own output,
    # while a real download dropped in here is still protected by --force.
    (data / GENERATOR_MARKER).write_text(
        "This directory was generated by demo/make_demo_data.py.\n"
        "Its contents are SYNTHETIC and may be overwritten by the next run.\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print(f" Synthetic inputs written to  {data}")
    print(f" Synthetic results written to {results}")
    print(" Next:  python demo/run_offline.py")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
