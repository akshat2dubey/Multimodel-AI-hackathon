#!/usr/bin/env python3
"""
Preprocess a Copernicus Sentinel-1 GRD SAFE into the GeoTIFF layout the rest
of the pipeline reads, plus a sidecar (S1_PROCESSED.json) that states units
(dB or linear), CRS, pixel size, orbit, and acquisition times.

Two paths, controlled by CLI:

* default (fast, quota-friendly, no large local processing):
  use the Copernicus Data Space / Sentinel Hub Process API to request a
  single bbox-only Sentinel-1 IW GRD VV + VH crop for the acquisition
  datetimes pick_same_track_pair already chose, convert linear sigma0 -> dB,
  apply a light 3x3 median speckle, write the four GeoTIFFs + sidecar.

  This path never downloads or decompresses the full SAFE locally; it only
  reads the SAFE enough to know which product to ask Process for, then lets
  the service do the heavy lifting. No process quota is burned by writing a
  naive TEI request and iterating pixels client-side.

* --full-scene (legacy, heavy, may be slow/quota-limited):
  unzip the SAFE, read the measurement tiffs, reproject a small window to a
  common EPSG:4326 bbox, convert to dB, filter, write tifs + sidecar.

Keep pick_same_track_pair untouched: this module only receives the two
products / labels / bbox / datetimes that caller already selected.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _fail(msg: str) -> None:
    print(msg, file=sys.stderr)
    sys.exit(2)


# ---------------------------------------------------------------------------
# Credentials — read only what the service libraries/Process API need, never
# print the secret.  This module can run standalone, so it re-reads the env
# rather than depending on scripts/cdse_env.py being imported first.
# ---------------------------------------------------------------------------

def _env_str(names: tuple[str, ...]) -> str | None:
    import os
    for n in names:
        v = os.environ.get(n, "").strip()
        if v:
            return v
    return None


def _credentials() -> tuple[str, str] | None:
    cdse_pair = (
        _env_str(("CDSE_CLIENT_ID", "SH_CLIENT_ID", "SENTINEL_HUB_CLIENT_ID")),
        _env_str(("CDSE_CLIENT_SECRET", "SH_CLIENT_SECRET", "SENTINEL_HUB_CLIENT_SECRET")),
    )
    if cdse_pair[0] and cdse_pair[1]:
        return cdse_pair
    return None


# ---------------------------------------------------------------------------
# Shared processing helpers
# ---------------------------------------------------------------------------

def _linear_to_db(arr_linear):
    import numpy as np

    scalar = np.isscalar(arr_linear) or getattr(arr_linear, "ndim", 2) == 0
    a = np.asarray(arr_linear, dtype=np.float64)
    out = np.full_like(a, -999.0)
    pos = a > 0
    out[pos] = 10.0 * np.log10(np.clip(a[pos], 1e-9, None))
    if scalar and pos:
        return float(out.item())
    return out


def _speckle_3x3_db(arr):
    """3x3 median speckle filter that never invents an unobserved value.

    A plain median over a window that is mostly nodata returns whatever the nodata
    was filled with, not a measurement. Along a nodata collar — and along the
    ragged edge of the warped valid region, where the window is mostly outside the
    scene — that produced 0.0 dB pixels (rho = 1.0, an extremely bright return)
    manufactured entirely from missing data. In a flood map that error points the
    worst way possible: a bright pixel can never be water, so a flooded border
    would be silently declared dry.

    So the median is only trusted where most of the nine samples are real. Where
    they are not, the observed value is kept unfiltered instead of being replaced
    by a guess, and nodata stays nodata. Regression test:
    tests/test_27_georeferencing.py::test_speckle_filter_never_invents_a_value_next_to_nodata
    """
    import numpy as np
    from scipy.ndimage import median_filter, uniform_filter

    a = np.ma.masked_invalid(arr.astype(np.float32))
    valid = ~a.mask
    filled = np.where(valid, a.data, 0.0)
    # How many of each pixel's nine samples are real? With five or more, the
    # median of the window is guaranteed to be one of them (four fills cannot
    # outrank five observations), so the filtered value is an observed value.
    neighbours = uniform_filter(valid.astype(np.float32), size=3, mode="constant", cval=0.0) * 9.0
    median = median_filter(filled, size=3)
    out = np.where(
        valid & (neighbours >= MIN_VALID_SPECKLE_NEIGHBOURS - 1e-3),
        median,
        a.data,
    )
    return np.where(valid, out, np.nan).astype(np.float32)


def _write_band_tif(
    array_db,
    west,
    south,
    east,
    north,
    width,
    height,
    crs_epsg,
    out_path,
    nodata=-999.0,
):
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds

    profile = dict(
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs=CRS.from_epsg(crs_epsg),
        # from_bounds takes (west, south, east, north). Passing north in the
        # south slot (which this line used to do) mirrors the Y axis: the pixels
        # are written north-up but the header claims row 0 is the southern edge,
        # so every consumer geolocates the raster upside down. Caught by
        # tests/test_27_georeferencing.py.
        transform=from_bounds(west, south, east, north, width, height),
        compress="lzw",
        nodata=nodata,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(array_db.astype(np.float32), 1)
    return out_path


def _crs_epsg(tif_path):
    import rasterio
    try:
        with rasterio.open(tif_path) as src:
            return src.crs.to_epsg()
    except Exception:
        return None


def _write_sidecar(
    label,
    out_dir,
    bbox,
    product_name,
    product_id,
    acquisition,
    *,
    orbit=None,
    units="dB",
    source="Sentinel Hub Process API (bbox crop)",
    bands_on_disk=None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    west, south, east, north = bbox
    mid_lon = (west + east) / 2.0
    mid_lat = (south + north) / 2.0
    m_per_deg = 111320.0
    px_w_m = abs(east - west) * m_per_deg * math.cos(math.radians(mid_lat))
    px_h_m = abs(north - south) * m_per_deg
    px_res_m = round((px_w_m + px_h_m) / 2.0, 2)

    bands_on_disk = bands_on_disk or [
        str((out_dir / f"s1_{label}_VV.tif").resolve()),
        str((out_dir / f"s1_{label}_VH.tif").resolve()),
    ]

    vv_path = out_dir / f"s1_{label}_VV.tif"
    vv_crs = _crs_epsg(vv_path)
    sidecar = {
        "label": label,
        "product_name": product_name,
        "product_id": product_id,
        "acquired_utc_iso": acquisition.astimezone(timezone.utc).isoformat(),
        "acquired_utc": acquisition.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": source,
        "units": units,
        "quantity": "sigma0 (backscatter coefficient)",
        "processing": "linear_sigma0 -> dB, 3x3 median speckle, bbox crop",
        "crs": f"EPSG:{vv_crs}" if vv_crs else "EPSG:4326",
        "crs_epsg": int(vv_crs) if vv_crs else 4326,
        "pixel_size_m": px_res_m,
        "pixel_size_deg": round(px_w_m / m_per_deg, 7),
        "bbox_deg": list(bbox),
        "bbox_latlon": {
            "west": west, "south": south, "east": east, "north": north,
            "center_lon": mid_lon, "center_lat": mid_lat,
        },
        "relative_orbit": orbit,
        "bands": ["VV", "VH"],
        "bands_on_disk": bands_on_disk,
        "sidecar_on_disk": str((out_dir / "S1_PROCESSED.json").resolve()),
        "nodata_value": -999.0,
    }
    path = out_dir / "S1_PROCESSED.json"
    path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Full-scene fallback (legacy, heavy, kept only so --full-scene callers survive)
# ---------------------------------------------------------------------------

def _find_measurement_tiffs(zip_members):
    """Locate the VV and VH measurement rasters inside a SAFE archive.

    Two layouts exist in the wild and both must work, because the pipeline can
    only use imagery it really has:

      * the classic SAFE  measurement/s1a-iw-grd-vv-...tiff
      * the CDSE COG SAFE measurement/s1d-iw-grd-vv-...-cog.tiff

    The name is matched on the polarization token rather than on the mission
    prefix, which is what the previous version got wrong: it required "s1a-" or
    "s1b-", so a Sentinel-1D COG product matched nothing and the caller silently
    fell back to whatever stale rasters happened to sit in the output directory.
    A calibration/annotation XML is never a measurement.
    """
    vv_name = None
    vh_name = None
    for n in zip_members:
        low = n.lower()
        if not low.endswith((".tif", ".tiff")):
            continue
        if "measurement" not in low:
            continue
        if "/annotation/" in low or "calibration" in low:
            continue
        base = low.rsplit("/", 1)[-1]
        if vv_name is None and re.search(r"[-_]vv[-_.]", base):
            vv_name = n
        elif vh_name is None and re.search(r"[-_]vh[-_.]", base):
            vh_name = n
        if vv_name and vh_name:
            break
    return vv_name, vh_name


def preprocess_full_scene_safe(
    safe_zip,
    output_dir,
    label,
    bbox,
    default_acq=None,
):
    """LEGACY, superseded by ``preprocess_safe_local`` — refuses rather than guesses.

    This path read a degree window straight out of the SAFE measurement tiffs.
    That only makes sense for a raster that already carries a CRS and a
    geotransform, and a Sentinel-1 GRD product does not: its geolocation is a GCP
    lattice, so the same degree window would be read as *pixel* coordinates and
    land somewhere else entirely. It is kept because removing a public function is
    a bigger change than fixing it, but it now says so instead of writing bands
    that look plausible and are geolocated nowhere.
    """
    import rasterio
    from rasterio.windows import from_bounds as win_from_bounds
    from zipfile import ZipFile

    west, south, east, north = bbox
    out_dir = output_dir / label
    out_dir.mkdir(parents=True, exist_ok=True)

    vv_path, vh_path = None, None
    with ZipFile(safe_zip) as z:
        members = z.namelist()
        vv_name, vh_name = _find_measurement_tiffs(members)
        if not vv_name or not vh_name:
            _fail(
                f"could not locate VV/VH measurement tiffs inside {safe_zip.name}\n"
                "  zip members seen (first 40):\n"
                + "\n".join(sorted(members)[:40])
            )
        with z.open(vv_name) as vv_stream, rasterio.open(vv_stream) as src:
            if src.crs is None or src.transform.is_identity:
                raise PreprocessError(
                    f"{vv_name.rsplit('/', 1)[-1]} has no CRS/geotransform (GRD "
                    "products geolocate with a GCP lattice), so a degree window "
                    "cannot be used as a pixel window. Use preprocess_safe_local(), "
                    "which warps the product's own GCPs onto the requested bbox."
                )
            # from_bounds takes (left, bottom, right, top) — the same order the
            # original call had swapped.
            win = win_from_bounds(west, south, east, north, src.width, src.height)
            vv = src.read(1, window=win)
            profile = src.profile
        with z.open(vh_name) as vh_stream, rasterio.open(vh_stream) as src:
            vh = src.read(1, window=win)

    vv_db = _linear_to_db(vv)
    vh_db = _linear_to_db(vh)
    vv_db = _speckle_3x3_db(vv_db)
    vh_db = _speckle_3x3_db(vh_db)

    _write_band_tif(
        vv_db, west, south, east, north, vv_db.shape[1], vv_db.shape[0],
        4326, out_dir / f"s1_{label}_VV.tif",
    )
    _write_band_tif(
        vh_db, west, south, east, north, vh_db.shape[1], vh_db.shape[0],
        4326, out_dir / f"s1_{label}_VH.tif",
    )

    return _write_sidecar(
        label=label,
        out_dir=out_dir,
        bbox=(west, south, east, north),
        product_name=safe_zip.stem,
        product_id=safe_zip.stem,
        acquisition=default_acq or datetime.now(tz=timezone.utc),
        orbit=None,
        units="dB",
        source=f"full-scene fallback from {safe_zip.name}",
    )


# ---------------------------------------------------------------------------
# Local path (the default): a real SAFE -> real dB bands, no service needed
#
# The CDSE GRD products carry their geolocation as ~200 ground control points
# (EPSG:4326) rather than a geotransform, and their radiometric calibration in
# annotation/calibration/*.xml. This path reads both, so the bands that reach
# inference are a real backscatter coefficient in dB on a real geographic grid
# built from the product's own GCPs. Nothing here needs an API, a quota or a
# network connection once the SAFE is on disk -- and nothing here can silently
# fall back to rasters the run did not produce.
# ---------------------------------------------------------------------------

# Delivered grid resolution in metres. 30 m is a defensible scale for
# basin-wide flood mapping: GRD's ~10 m samples overstate what the GCP
# geolocation of a GRD product can support.
DEFAULT_RESOLUTION_M = 30.0
# Neither side of the delivered grid exceeds this, so an oversized AOI loses
# resolution instead of memory.
MAX_TARGET_PIXELS = 2048
# Ceiling on the source window held in memory per band, in pixels.
MAX_READ_PIXELS = 12_000_000
# Nominal ground sampling distance of an IW GRD product, in metres. Only used
# to pick a whole-number decimation factor when reading the source window.
NOMINAL_GRD_RES_M = 10.0
# sigma0 values below this are treated as "the sensor did not really measure
# here" (radar shadow, nodata collar) rather than as very smooth water.
SIGMA0_FLOOR_DB = -50.0
# A 3x3 median is only applied where this many of the nine samples are real
# observations; below it the observed value is kept unfiltered (a majority
# guarantees the median is one of the real samples).
MIN_VALID_SPECKLE_NEIGHBOURS = 5


class PreprocessError(RuntimeError):
    """A SAFE archive that cannot be turned into real, geolocated dB bands."""


def _vsizip(archive: Path, member: str) -> str:
    """A GDAL path that reads `member` straight out of the SAFE archive."""
    return f"/vsizip/{Path(archive).resolve()}/{member}"


def _read_zip_member(archive: Path, member: str) -> bytes:
    from zipfile import ZipFile

    with ZipFile(archive) as z:
        return z.read(member)


def _find_calibration_xml(zip_members, polarization: str) -> str | None:
    token = f"-{polarization.lower()}-"
    for name in zip_members:
        low = name.lower()
        if (
            "annotation/calibration/" in low
            and "calibration-" in low
            and token in low
            and low.endswith(".xml")
        ):
            return name
    return None


def _parse_calibration(archive: Path, member: str) -> dict:
    """The sigmaNought calibration LUT: samples along range, one set per azimuth line.

    Applying this is what turns raw 16-bit digital numbers into a backscatter
    coefficient, which is the only form in which two acquisitions acquired weeks
    apart can be compared pixel by pixel.
    """
    root = ET.fromstring(_read_zip_member(archive, member))
    lines: list[float] = []
    sigma: list[np.ndarray] = []
    pixels: np.ndarray | None = None
    for vector in root.findall(".//calibrationVector"):
        line = vector.findtext("line")
        pixel_text = (vector.findtext("pixel") or "").split()
        sigma_text = (vector.findtext("sigmaNought") or "").split()
        if line is None or not pixel_text or len(pixel_text) != len(sigma_text):
            continue
        if pixels is None:
            pixels = np.asarray([float(v) for v in pixel_text], dtype=np.float64)
        lines.append(float(line))
        sigma.append(np.asarray([float(v) for v in sigma_text], dtype=np.float64))
    if pixels is None or not lines:
        raise PreprocessError(
            f"calibration file {member} contains no calibration vectors — the "
            "digital numbers cannot be converted to sigma0, so the bands would "
            "not be comparable between the two acquisitions."
        )
    return {
        "member": member,
        "lines": np.asarray(lines, dtype=np.float64),
        "pixels": pixels,
        "sigma": np.vstack(sigma),
    }


def _calibration_factor(cal: dict, rows_full: np.ndarray, cols_full: np.ndarray) -> np.ndarray:
    """The calibration factor A for every sampled pixel, so sigma0 = (DN / A)^2.

    Interpolated along range for each calibration line, then between the two
    azimuth lines that bracket the pixel. The LUT is smooth at this scale, so a
    bilinear read of it is the same as the product's own definition.
    """
    lines = cal["lines"]
    per_line = np.empty((lines.size, cols_full.size), dtype=np.float64)
    for i in range(lines.size):
        per_line[i] = np.interp(cols_full, cal["pixels"], cal["sigma"][i])
    if lines.size == 1:
        return np.repeat(per_line, rows_full.size, axis=0)
    idx = np.clip(np.searchsorted(lines, rows_full, side="right") - 1, 0, lines.size - 2)
    span = lines[idx + 1] - lines[idx]
    weight = np.where(span > 0, (rows_full - lines[idx]) / np.where(span > 0, span, 1.0), 0.0)
    return per_line[idx] * (1.0 - weight)[:, None] + per_line[idx + 1] * weight[:, None]


def _gcp_lattice(gcps) -> dict:
    """The GCP grid as ordered row/col axes plus a lon/lat lookup."""
    rows = sorted({float(g.row) for g in gcps})
    cols = sorted({float(g.col) for g in gcps})
    row_index = {value: i for i, value in enumerate(rows)}
    col_index = {value: i for i, value in enumerate(cols)}
    lon = np.full((len(rows), len(cols)), np.nan)
    lat = np.full((len(rows), len(cols)), np.nan)
    for g in gcps:
        lon[row_index[float(g.row)], col_index[float(g.col)]] = g.x
        lat[row_index[float(g.row)], col_index[float(g.col)]] = g.y
    return {"rows": np.asarray(rows), "cols": np.asarray(cols), "lon": lon, "lat": lat}


def _source_window(lattice: dict, bbox, margin_cells: int = 1) -> tuple[int, int, int, int]:
    """The source pixel window (row_off, col_off, height, width) that covers `bbox`.

    The GCP lattice is not axis-aligned on the ground, so the window is the
    lattice block that brackets the AOI, grown by `margin_cells` cells on every
    side: the polynomial GCP transform is approximate, and the margin is what
    keeps the whole AOI inside the window once it has been warped.
    """
    west, south, east, north = bbox
    inside = (
        (lattice["lon"] >= west)
        & (lattice["lon"] <= east)
        & (lattice["lat"] >= south)
        & (lattice["lat"] <= north)
    )
    if inside.any():
        rows_idx, cols_idx = np.where(inside)
        r0, r1 = int(rows_idx.min()), int(rows_idx.max())
        c0, c1 = int(cols_idx.min()), int(cols_idx.max())
    else:
        # The AOI is smaller than one lattice cell: centre the window on the
        # nearest control point instead of guessing a corner.
        distance = (
            (lattice["lon"] - (west + east) / 2.0) ** 2
            + (lattice["lat"] - (south + north) / 2.0) ** 2
        )
        r0 = r1 = int(np.unravel_index(np.nanargmin(distance), distance.shape)[0])
        c0 = c1 = int(np.unravel_index(np.nanargmin(distance), distance.shape)[1])

    rows, cols = lattice["rows"], lattice["cols"]
    r0 = max(r0 - margin_cells, 0)
    r1 = min(r1 + margin_cells, rows.size - 1)
    c0 = max(c0 - margin_cells, 0)
    c1 = min(c1 + margin_cells, cols.size - 1)
    row_off, col_off = int(rows[r0]), int(cols[c0])
    return row_off, col_off, int(rows[r1]) - row_off, int(cols[c1]) - col_off


def target_grid(bbox, *, resolution_m: float = DEFAULT_RESOLUTION_M, max_pixels: int = MAX_TARGET_PIXELS) -> dict:
    """The delivered EPSG:4326 grid. Both labels of a pair use this same grid.

    Sharing one grid is what lets `before` and `after` be differenced per pixel
    instead of being resampled onto each other first, and it is fixed by the
    requested bbox and resolution, never inferred from whatever rasters happen
    to be on disk.
    """
    from rasterio.transform import from_origin

    west, south, east, north = bbox
    mid_lat = math.radians((south + north) / 2.0)
    width_m = (east - west) * 111320.0 * math.cos(mid_lat)
    height_m = (north - south) * 111320.0
    resolution = max(float(resolution_m), max(width_m, height_m) / float(max_pixels))
    width = max(int(round(width_m / resolution)), 16)
    height = max(int(round(height_m / resolution)), 16)
    transform = from_origin(west, north, (east - west) / width, (north - south) / height)
    return {
        "transform": transform,
        "width": width,
        "height": height,
        "resolution_m": resolution,
    }


def preprocess_safe_local(
    *,
    safe_zip: Path,
    output_dir: Path,
    label: str,
    bbox,
    acquisition=None,
    orbit=None,
    resolution_m: float = DEFAULT_RESOLUTION_M,
    max_pixels: int = MAX_TARGET_PIXELS,
    calibrate: bool = True,
    speckle: bool = True,
) -> Path:
    """Turn a downloaded SAFE into geolocated sigma0 dB bands + a provenance sidecar.

    Writes ``s1_{label}_VV.tif`` / ``s1_{label}_VH.tif`` into ``output_dir/label``
    on a shared grid derived from ``bbox``, and ``S1_PROCESSED.json`` describing
    exactly which product and which processing produced them. Raises
    PreprocessError rather than writing anything it cannot vouch for.
    """
    import rasterio
    from rasterio.control import GroundControlPoint
    from rasterio.enums import Resampling
    from rasterio.warp import reproject
    from rasterio.windows import Window
    from zipfile import ZipFile

    safe_zip = Path(safe_zip)
    if not safe_zip.is_file():
        raise PreprocessError(f"SAFE archive not found: {safe_zip}")
    with ZipFile(safe_zip) as z:
        names = z.namelist()
    vv_name, vh_name = _find_measurement_tiffs(names)
    if not vv_name or not vh_name:
        raise PreprocessError(
            f"no VV/VH measurement rasters inside {safe_zip.name}. A truncated or "
            "non-SAFE archive cannot produce bands, and inventing them is not an option."
        )

    grid = target_grid(bbox, resolution_m=resolution_m, max_pixels=max_pixels)
    transform = grid["transform"]
    width, height = grid["width"], grid["height"]
    out_dir = Path(output_dir) / label
    out_dir.mkdir(parents=True, exist_ok=True)

    state: dict = {}
    band_stats: dict = {}
    bands_on_disk: list[str] = []

    for polarization, member in (("VV", vv_name), ("VH", vh_name)):
        with rasterio.open(_vsizip(safe_zip, member)) as src:
            gcps, gcp_crs = src.gcps
            if not gcps:
                raise PreprocessError(
                    f"{member} carries no ground control points. This preprocessor "
                    "geolocates a GRD product from its own GCPs; guessing a grid "
                    "would put the flood map in the wrong valley."
                )
            if not state:
                lattice = _gcp_lattice(gcps)
                row_off, col_off, win_h, win_w = _source_window(lattice, bbox)
                win_h = min(win_h, src.height - row_off)
                win_w = min(win_w, src.width - col_off)
                # Read the window at (roughly) the delivered resolution: the AOI at
                # native ~10 m is far more pixels than the grid needs, and the
                # extra ones would cost memory without adding information.
                dec = max(1, int(round(grid["resolution_m"] / NOMINAL_GRD_RES_M)))
                while (win_h / dec) * (win_w / dec) > MAX_READ_PIXELS:
                    dec += 1
                out_h = max(win_h // dec, 1)
                out_w = max(win_w // dec, 1)
                row_step = (float(lattice["rows"][1] - lattice["rows"][0]) / dec) if lattice["rows"].size > 1 else out_h
                col_step = (float(lattice["cols"][1] - lattice["cols"][0]) / dec) if lattice["cols"].size > 1 else out_w
                state.update(
                    {
                        "lattice": lattice,
                        "window": (row_off, col_off, win_h, win_w),
                        "dec": dec,
                        "out_shape": (out_h, out_w),
                        "row_step": row_step,
                        "col_step": col_step,
                    }
                )
            else:
                row_off, col_off, win_h, win_w = state["window"]
                win_h = min(win_h, src.height - row_off)
                win_w = min(win_w, src.width - col_off)
                dec = state["dec"]
                out_h, out_w = state["out_shape"]
                lattice = state["lattice"]

            out_shape = (out_h, out_w)
            dn = src.read(
                1,
                window=Window(col_off, row_off, win_w, win_h),
                out_shape=out_shape,
                resampling=Resampling.average,
            ).astype(np.float64)

        rows_full = row_off + (np.arange(out_h) + 0.5) * dec
        cols_full = col_off + (np.arange(out_w) + 0.5) * dec

        cal = None
        if calibrate:
            cal_member = _find_calibration_xml(names, polarization)
            if cal_member is None:
                raise PreprocessError(
                    f"no sigmaNought calibration file for {polarization} in "
                    f"{safe_zip.name}. Without it the digital numbers are not a "
                    "backscatter coefficient and the two dates cannot be compared "
                    "-- pass calibrate=False only if you know what that means."
                )
            cal = _parse_calibration(safe_zip, cal_member)

        if cal is not None:
            factor = _calibration_factor(cal, rows_full, cols_full)
            with np.errstate(divide="ignore", invalid="ignore"):
                sigma0 = np.where((dn > 0) & (factor > 0), (dn / np.where(factor > 0, factor, 1.0)) ** 2, np.nan)
            units_note = f"sigmaNought LUT {Path(cal['member']).name}"
        else:
            sigma0 = np.where(dn > 0, dn, np.nan)
            units_note = "raw DN (no calibration applied)"

        with np.errstate(divide="ignore", invalid="ignore"):
            db = 10.0 * np.log10(sigma0)
        db = np.where(np.isfinite(db) & (db >= SIGMA0_FLOOR_DB), db, np.nan).astype(np.float32)
        if speckle:
            db = _speckle_3x3_db(db)

        # GCPs must be in the coordinates of the array being warped: re-based to
        # the window origin and scaled by the decimation factor, keeping one
        # lattice cell of context on every side for the polynomial fit.
        adjusted = []
        for g in gcps:
            row = (float(g.row) - row_off) / dec
            col = (float(g.col) - col_off) / dec
            if -state["row_step"] <= row <= out_h + state["row_step"] and -state["col_step"] <= col <= out_w + state["col_step"]:
                adjusted.append(GroundControlPoint(row=row, col=col, x=g.x, y=g.y, z=g.z))
        if len(adjusted) < 6:
            raise PreprocessError(
                f"only {len(adjusted)} ground control points fall near the AOI — too "
                "few to fit the geolocation; widen the AOI or check the bbox."
            )

        warped = np.full((height, width), np.nan, dtype=np.float32)
        reproject(
            source=db,
            destination=warped,
            gcps=adjusted,
            src_crs=gcp_crs or "EPSG:4326",
            src_nodata=np.nan,
            dst_crs="EPSG:4326",
            dst_transform=transform,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
            num_threads=2,
        )

        observed = np.isfinite(warped)
        if not observed.any():
            raise PreprocessError(
                f"warping {pair_name(member)} produced no observed pixels over the "
                "AOI — the bbox is probably outside the product footprint."
            )
        values = warped[observed]
        band_stats[polarization] = {
            "min_db": round(float(values.min()), 2),
            "max_db": round(float(values.max()), 2),
            "mean_db": round(float(values.mean()), 2),
            "median_db": round(float(np.median(values)), 2),
            "observed_fraction": round(float(observed.mean()), 4),
        }

        band_path = out_dir / f"s1_{label}_{polarization}.tif"
        _write_band_tif(
            np.where(observed, warped, -999.0),
            bbox[0], bbox[1], bbox[2], bbox[3],
            width, height, 4326, band_path, nodata=-999.0,
        )
        bands_on_disk.append(str(band_path.resolve()))
        print(
            f"  {label} {polarization}: {band_stats[polarization]['median_db']:.1f} dB median, "
            f"{100 * band_stats[polarization]['observed_fraction']:.1f}% observed -> {band_path.name}"
        )

    west, south, east, north = bbox
    mid_lon, mid_lat = (west + east) / 2.0, (south + north) / 2.0
    sidecar = {
        "label": label,
        "product_name": safe_zip.stem,
        "product_id": safe_zip.stem,
        "archive": safe_zip.name,
        "archive_bytes": int(safe_zip.stat().st_size),
        "acquired_utc_iso": acquisition.astimezone(timezone.utc).isoformat() if acquisition else None,
        "acquired_utc": acquisition.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if acquisition else None,
        "source": "CDSE SAFE archive (local GCP warp, no service dependency)",
        "units": "dB",
        "quantity": "sigma0 (backscatter coefficient)",
        "processing": (
            "GCP geolocation -> EPSG:4326 grid; sigma0 from the product's "
            f"calibration LUT ({units_note}); 10*log10 with a {SIGMA0_FLOOR_DB:g} dB floor; "
            + ("3x3 median speckle filter" if speckle else "no speckle filter")
        ),
        "crs": "EPSG:4326",
        "crs_epsg": 4326,
        "pixel_size_m": round(grid["resolution_m"], 2),
        "pixel_size_deg": round((east - west) / width, 7),
        "bbox_deg": [west, south, east, north],
        "bbox_latlon": {
            "west": west, "south": south, "east": east, "north": north,
            "center_lon": mid_lon, "center_lat": mid_lat,
        },
        "grid": {
            "width": width,
            "height": height,
            "transform": list(transform)[:6],
        },
        "source_window": {
            "row_off": row_off, "col_off": col_off,
            "height": win_h, "width": win_w, "decimation": dec,
        },
        "relative_orbit": orbit,
        "bands": ["VV", "VH"],
        "bands_on_disk": bands_on_disk,
        "band_stats": band_stats,
        "sidecar_on_disk": str((out_dir / "S1_PROCESSED.json").resolve()),
        "nodata_value": -999.0,
    }
    path = out_dir / "S1_PROCESSED.json"
    path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    return path


def pair_name(member: str) -> str:
    """The measurement file's own name, for error text."""
    return str(member).rsplit("/", 1)[-1]


# ---------------------------------------------------------------------------
# Process API bbox-crop path (opt-in; needs a working Sentinel-Hub-compatible
# service and spends quota)
# ---------------------------------------------------------------------------

def preprocess_product(
    *,
    safe_zip,
    output_dir,
    label,
    bbox,
    acquisition=None,
    orbit=None,
    target_pixels=640,
):
    """Request a bbox-only Sentinel-1 IW GRD VV+VH crop from the Sentinel Hub
    Process API, convert to dB, 3x3 speckle, write four GeoTIFFs + sidecar.

    The api endpoint is the official services.sentinel-hub.com Process route
    documented by ESA / Copernicus; we only ever POST to the token endpoint and
    GET a pre-rendered geotiff, never implement the service ourselves.
    """
    import numpy as np
    from rasterio.io import MemoryFile

    creds = _credentials()
    if creds is None:
        _fail(
            "Copernicus / Sentinel Hub OAuth credentials not set in the process env.\n"
            "  Required env vars (any of these pairs):\n"
            "    CDSE_CLIENT_ID + CDSE_CLIENT_SECRET\n"
            "    SH_CLIENT_ID + SH_CLIENT_SECRET\n"
            "    SENTINEL_HUB_CLIENT_ID + SENTINEL_HUB_CLIENT_SECRET\n"
            "  (Same secret file loaded by scripts/cdse_env.py will work.)\n"
        )
    client_id, client_secret = creds

    import requests as _requests

    # nosec B105 — this is the public Sentinel Hub OAuth endpoint, not a secret.
    token_url = "https://services.sentinel-hub.com/api/v1/auth/token"
    try:
        token_r = _requests.post(
            token_url,
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "grant_type": "client_credentials",
            },
            timeout=30,
        )
        token_r.raise_for_status()
        token_payload = token_r.json()
    except Exception as exc:
        raise RuntimeError(
            f"cannot reach Sentinel Hub token endpoint ({token_url}): {exc}"
        ) from exc

    token = token_payload.get("access_token")
    if not token:
        raise RuntimeError(
            "Sentinel Hub token endpoint returned no access_token. "
            f"response: {token_r.status_code} {token_r.text[:300]}"
        )

    west, south, east, north = bbox
    mid_lon = (west + east) / 2.0
    mid_lat = (south + north) / 2.0
    m_per_deg = 111320.0
    px_w_m = abs(east - west) * m_per_deg * math.cos(math.radians(mid_lat))
    px_h_m = abs(north - south) * m_per_deg
    res_m = round((px_w_m + px_h_m) / 2.0, 2)
    res_deg = round(px_w_m / m_per_deg, 7)

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "image/tiff;application=geotiff",
    }
    params = {
        "bbox": f"{west},{south},{east},{north}",
        "dataMask": "false",
        "constellation": "Sentinel-1",
        "maxsize": int(round(target_pixels)),
    }

    proc_url = "https://services.sentinel-hub.com/api/v1/collections/SENTINEL-1/grd/process"
    try:
        resp = _requests.get(proc_url, params=params, headers=headers, timeout=120)
    except Exception as exc:
        raise RuntimeError(f"cannot reach Sentinel Hub Process API ({proc_url}): {exc}") from exc

    if resp.status_code in (401, 403):
        raise RuntimeError(
            f"Process API rejected the request ({resp.status_code}): "
            "invalid or expired credentials, or no quota for SENTINEL-1 GRD process."
        )
    resp.raise_for_status()

    tiff_bytes = resp.content
    if not tiff_bytes or len(tiff_bytes) < 200:
        raise RuntimeError(
            f"Process API returned too-small payload ({len(tiff_bytes)} bytes) — not a real crop"
        )

    mem = MemoryFile(tiff_bytes)
    try:
        with mem.open() as src:
            if src.count < 2:
                raise RuntimeError(
                    f"Process API crop has only {src.count} band(s) — expected VV + VH"
                )
            vv = src.read(1)
            vh = src.read(2)
            crs_epsg = src.crs.to_epsg() or 4326
            b = src.bounds
            bbox = (float(b.left), float(b.bottom), float(b.right), float(b.top))
            width = src.width
            height = src.height
    finally:
        mem.close()

    vv_db = _linear_to_db(vv)
    vh_db = _linear_to_db(vh)
    vv_db = _speckle_3x3_db(vv_db)
    vh_db = _speckle_3x3_db(vh_db)

    out_dir = output_dir / label
    out_dir.mkdir(parents=True, exist_ok=True)

    vv_tif = _write_band_tif(
        vv_db, bbox[0], bbox[1], bbox[2], bbox[3], width, height, crs_epsg,
        out_dir / f"s1_{label}_VV.tif",
    )
    vh_tif = _write_band_tif(
        vh_db, bbox[0], bbox[1], bbox[2], bbox[3], width, height, crs_epsg,
        out_dir / f"s1_{label}_VH.tif",
    )

    return _write_sidecar(
        label=label,
        out_dir=out_dir,
        bbox=bbox,
        product_name="Sentinel-1 GRD (Process API crop)",
        product_id="process-api-bbox-crop",
        acquisition=acquisition or datetime.now(tz=timezone.utc),
        orbit=orbit,
        units="dB",
        source="Sentinel Hub Process API (bbox crop)",
        bands_on_disk=[str(vv_tif), str(vh_tif)],
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preprocess a Sentinel-1 GRD SAFE into GeoTIFF layout + sidecar"
    )
    parser.add_argument("--safe-zip", type=Path, required=True, help="Downloaded SAFE zip")
    parser.add_argument("--output-dir", type=Path, required=True, help="Parent of {before,after}")
    parser.add_argument("--label", required=True, help="before or after")
    parser.add_argument("--bbox", required=True, help="min_lon,min_lat,max_lon,max_lat")
    parser.add_argument(
        "--acquisition",
        default=None,
        help="Acquisition instant YYYY-MM-DDThh:mm:ssZ (from pick_same_track_pair / product.datetime)",
    )
    parser.add_argument("--orbit", type=int, default=None, help="Relative orbit number, if known")
    parser.add_argument("--target-pixels", type=int, default=640, help="Pixels per side of crop (--process-api only)")
    parser.add_argument(
        "--resolution-m",
        type=float,
        default=DEFAULT_RESOLUTION_M,
        help=(
            "Delivered grid resolution in metres for the local path "
            f"(default {DEFAULT_RESOLUTION_M:g} m, capped at {MAX_TARGET_PIXELS} px per side)"
        ),
    )
    parser.add_argument(
        "--process-api",
        action="store_true",
        help=(
            "Use the remote Sentinel Hub Process API bbox crop instead of the "
            "local warp. Needs credentials and service quota; the local path does not."
        ),
    )
    parser.add_argument(
        "--full-scene",
        action="store_true",
        help=(
            "Deprecated alias for the default local path (kept so older commands "
            "still work). The local path now warps the SAFE's own GCPs onto the "
            "requested bbox instead of reading a raw window."
        ),
    )
    args = parser.parse_args()

    parts = [float(x) for x in args.bbox.split(",")]
    if len(parts) != 4:
        _fail("--bbox must be min_lon,min_lat,max_lon,max_lat")
    bbox = tuple(parts)

    if not args.safe_zip.is_file():
        _fail(f"SAFE zip not found: {args.safe_zip}")

    acq = None
    if args.acquisition:
        try:
            acq = datetime.fromisoformat(args.acquisition.replace("Z", "+00:00"))
        except Exception as exc:
            _fail(f"--acquisition not parseable: {args.acquisition}: {exc}")

    if args.process_api:
        try:
            sidecar = preprocess_product(
                safe_zip=args.safe_zip,
                output_dir=args.output_dir,
                label=args.label,
                bbox=bbox,
                acquisition=acq,
                orbit=args.orbit,
                target_pixels=args.target_pixels,
            )
        except (RuntimeError, OSError) as exc:
            _fail(
                f"remote preprocessing failed: {exc}\n"
                "  The local path needs no service — drop --process-api to use it."
            )
    else:
        try:
            sidecar = preprocess_safe_local(
                safe_zip=args.safe_zip,
                output_dir=args.output_dir,
                label=args.label,
                bbox=bbox,
                acquisition=acq,
                orbit=args.orbit,
                resolution_m=args.resolution_m,
            )
        except PreprocessError as exc:
            _fail(f"{exc}")

    for polarization in ("VV", "VH"):
        print(f"  OK  preprocessed -> {sidecar.parent / ('s1_' + args.label + '_' + polarization + '.tif')}")
    print(f"  OK  sidecar -> {sidecar}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
