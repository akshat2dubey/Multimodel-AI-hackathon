"""27. Geospatial contract of the Sentinel-1 preprocessing bridge.

The brief's Candidate A: ``_write_band_tif`` took its bounds as
``(west, south, east, north)`` but called Rasterio's ``from_bounds`` — whose
parameters are also ``(west, south, east, north)`` — as
``from_bounds(west, north, east, south)``. The two middle arguments were swapped,
so every GeoTIFF the bridge wrote carried a mirrored Y transform: the pixels were
north-up while the header said the array starts at the southern edge. Anything
that georeferences those files (the dashboard, QGIS, the damage overlay) placed
the flood map upside down, and nothing in the pipeline noticed because inference
only compares arrays.

These tests are offline and synthetic: a tiny stand-in SAFE archive whose
geolocation (3x3 GCP lattice) and radiometry (a constant sigmaNought LUT) are
known exactly, so the expected dB value of every output pixel can be computed by
hand.
"""

from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.control import GroundControlPoint
from rasterio.transform import from_bounds

import preprocess_s1

# ~2 km x 2 km AOI, so an 80x80 stand-in scene is ~27 m per pixel.
WEST, SOUTH, EAST, NORTH = 85.40, 28.00, 85.42, 28.02
SIZE = 80
LATTICE = (0, 40, 79)          # rows and columns carrying a ground control point
UNIFORM_DN = 400               # sigma0 = (400/1000)^2 = 0.16 -> -7.96 dB
MARKER_DN = 4000               # the bright block used to check geolocation
# 12x12 source pixels. Bigger than the decimated grid's 3x3 median window on
# purpose (a 3x3-pixel block would be filtered away as speckle before the
# geolocation could be checked at all), and small enough that it stays a minority
# of the scene, so a percentile over the whole scene still describes the uniform
# part of it.
MARKER_ROWS = (28, 40)
MARKER_COLS = (28, 40)
BORDER = 4                     # outermost source ring is nodata (DN = 0)
CAL_FACTOR = 1000.0            # constant calibration LUT for both polarizations
NODATA = -999.0


def _gcps():
    """A 3x3 lattice spanning the AOI, north at row 0 and east at the last column."""
    rows = cols = LATTICE
    lats = np.linspace(NORTH, SOUTH, len(rows))
    lons = np.linspace(WEST, EAST, len(cols))
    points = []
    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            points.append(
                GroundControlPoint(row=float(row), col=float(col),
                                   x=float(lons[j]), y=float(lats[i]), z=0.0)
            )
    return points


def _scene(marker: bool) -> np.ndarray:
    dn = np.full((SIZE, SIZE), UNIFORM_DN, dtype=np.uint16)
    if marker:
        dn[MARKER_ROWS[0]:MARKER_ROWS[1], MARKER_COLS[0]:MARKER_COLS[1]] = MARKER_DN
    dn[:BORDER, :] = 0
    dn[-BORDER:, :] = 0
    dn[:, :BORDER] = 0
    dn[:, -BORDER:] = 0
    return dn


def _calibration_xml(polarization: str) -> str:
    vectors = "".join(
        f"""
    <calibrationVector>
      <azimuthTime>2026-08-24T00:18:44.500000</azimuthTime>
      <line>{line}</line>
      <pixel count="2">0 {SIZE - 1}</pixel>
      <sigmaNought count="2">{CAL_FACTOR} {CAL_FACTOR}</sigmaNought>
      <betaNought count="2">{CAL_FACTOR} {CAL_FACTOR}</betaNought>
      <gamma count="2">{CAL_FACTOR} {CAL_FACTOR}</gamma>
      <dn count="2">{CAL_FACTOR} {CAL_FACTOR}</dn>
    </calibrationVector>"""
        for line in LATTICE
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<calibration>
  <adsHeader>
    <missionId>S1X</missionId>
    <polarisation>{polarization.upper()}</polarisation>
  </adsHeader>
  <calibrationVectorList count="3">{vectors}
  </calibrationVectorList>
</calibration>
"""


def _write_measurement(path: Path, array: np.ndarray) -> None:
    with rasterio.open(
        path, "w", driver="GTiff", height=array.shape[0], width=array.shape[1],
        count=1, dtype="uint16", nodata=0,
    ) as dst:
        dst.write(array, 1)
        # A CDSE GRD COG carries no CRS/transform at all: its geolocation is the
        # GCP lattice, which is exactly what the preprocessor must use.
        dst.gcps = (_gcps(), CRS.from_epsg(4326))


@pytest.fixture()
def stand_in_safe(tmp_path: Path) -> dict:
    """A SAFE-shaped archive with a known GCP lattice and a known calibration LUT."""
    archives = {}
    for label, marker in (("before", False), ("after", True)):
        archive = tmp_path / f"{label}.SAFE.zip"
        name = f"S1X_TEST_{label.upper()}.SAFE"
        with zipfile.ZipFile(archive, "w") as zf:
            for polarization in ("vv", "vh"):
                member = f"{name}/measurement/s1x-iw-grd-{polarization}-20260824t001844-001-cog.tiff"
                tmp_tif = tmp_path / f"{label}_{polarization}.tiff"
                _write_measurement(tmp_tif, _scene(marker))
                zf.write(tmp_tif, member)
                zf.writestr(
                    f"{name}/annotation/calibration/calibration-s1x-iw-grd-{polarization}-20260824t001844-001-cog.xml",
                    _calibration_xml(polarization),
                )
        archives[label] = archive
    return archives


def _expected_db(dn: float) -> float:
    return round(10.0 * np.log10((dn / CAL_FACTOR) ** 2), 2)


# ─── NoData must never be turned into a measurement ─────────────────────────


def test_speckle_filter_never_invents_a_value_next_to_nodata():
    """The 3x3 median must only ever return observed values.

    Repro of a defect the fixture above surfaced: the filter filled masked pixels
    with 0.0 and then took a plain median over the 3x3 window. Along every nodata
    boundary five or more of the nine samples were that fill value, so the median
    — and therefore the delivered pixel — became 0.0 dB, i.e. rho = 1.0, a very
    bright return invented entirely out of missing data. In a flood map that is
    the worst possible direction of error to make silently.
    """
    observed = -12.0
    db = np.full((11, 11), np.nan, dtype=np.float32)
    db[3:7, 3:7] = observed                 # a valid 4x4 island
    db[9, 1:10] = observed                  # a 1-px-wide valid strip
    #        ^ six of the nine neighbours of a pixel in that strip are nodata,
    #          which is the case the old fill-with-0.0 median got wrong.

    filtered = preprocess_s1._speckle_3x3_db(db)
    was_valid = np.isfinite(db)

    assert np.isfinite(filtered[was_valid]).all(), (
        "an observed pixel was turned into nodata by the filter"
    )
    assert np.allclose(filtered[was_valid], observed, atol=1e-5), (
        "the filter returned values that were never observed: "
        f"{sorted({round(float(v), 3) for v in filtered[was_valid]})}"
    )


# ─── Candidate A: the written transform ─────────────────────────────────────


def test_write_band_tif_round_trips_bounds_and_is_north_up(tmp_path: Path):
    """A raster written for (west, south, east, north) must describe that box.

    Fails against the swapped ``from_bounds(west, north, east, south)`` call: the
    bounds come back mirrored, the Y pixel size goes positive and row 0 is placed
    at the southern edge.
    """
    out = tmp_path / "band.tif"
    array = np.arange(20 * 30, dtype=np.float32).reshape(20, 30)

    preprocess_s1._write_band_tif(
        array, WEST, SOUTH, EAST, NORTH, 30, 20, 4326, out, nodata=NODATA
    )

    with rasterio.open(out) as src:
        bounds = src.bounds
        transform = src.transform
        assert bounds.left == pytest.approx(WEST, abs=1e-9)
        assert bounds.right == pytest.approx(EAST, abs=1e-9)
        assert bounds.bottom == pytest.approx(SOUTH, abs=1e-9), (
            f"south bound is {bounds.bottom} but {SOUTH} was requested — the "
            "south/north arguments are swapped"
        )
        assert bounds.top == pytest.approx(NORTH, abs=1e-9)
        assert transform.e < 0, (
            f"Y pixel size is {transform.e:+.3e}: a north-up raster must have a "
            "negative Y step, otherwise the array is stored upside down"
        )

        # Row 0 must sit at the northern edge, which is what "north-up" means.
        col, row = 0, 0
        lon, lat = src.xy(row, col, offset="center")
        assert lat > (NORTH + SOUTH) / 2, (
            f"the first row maps to latitude {lat}, below the AOI centre — the "
            "image would be displayed mirrored"
        )
        assert lon < (WEST + EAST) / 2
        assert src.crs.to_epsg() == 4326
        assert src.read(1)[0, 0] == array[0, 0]


# ─── P0.2: the local SAFE -> dB bands contract ──────────────────────────────


def test_local_preprocess_calibrates_geolocates_and_preserves_nodata(
    stand_in_safe: dict, tmp_path: Path
):
    # The "after" fixture carries the bright marker block, which is what makes the
    # geolocation checkable; the "before" fixture is uniform.
    sidecar_path = preprocess_s1.preprocess_safe_local(
        safe_zip=stand_in_safe["after"],
        output_dir=tmp_path / "sentinel1",
        label="after",
        bbox=(WEST, SOUTH, EAST, NORTH),
        acquisition=datetime(2026, 8, 24, 0, 18, 44, tzinfo=timezone.utc),
        orbit=19,
    )
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))

    vv_path = tmp_path / "sentinel1" / "after" / "s1_after_VV.tif"
    with rasterio.open(vv_path) as src:
        vv = src.read(1)
        bounds = src.bounds
        assert bounds.left == pytest.approx(WEST, abs=1e-6)
        assert bounds.right == pytest.approx(EAST, abs=1e-6)
        assert bounds.bottom == pytest.approx(SOUTH, abs=1e-6)
        assert bounds.top == pytest.approx(NORTH, abs=1e-6)
        assert src.transform.e < 0, "the delivered band must be north-up"
        assert src.crs.to_epsg() == 4326
        assert src.nodata == NODATA

    # The uniform part of the scene must calibrate to the value the LUT implies.
    observed = vv != NODATA
    expected = _expected_db(UNIFORM_DN)
    assert observed.sum() > 100, "almost all of the scene came back as nodata"
    # Everything that is not part of the bright marker must sit at exactly the
    # calibrated level. (A mis-calibrated *bright* value cannot hide here: the
    # marker is excluded by the threshold and its own position is checked below.)
    marker_mask = vv >= expected + 1.0
    background = observed & ~marker_mask
    assert background.sum() > 100, "almost the whole scene was classified as marker"
    assert np.median(vv[background]) == pytest.approx(expected, abs=0.2)
    deviation = np.abs(vv[background] - expected)
    assert np.percentile(deviation, 98) < 0.3, (
        "the constant-DN scene did not calibrate to one value: "
        f"98th-percentile deviation {np.percentile(deviation, 98):.3f} dB"
    )

    # The nodata ring must stay unknown, never become a plausible observation.
    assert (vv == NODATA).any(), "nodata pixels were converted into measurements"

    # A geolocated scene: the bright block must land where its GCPs place it.
    assert vv[observed].max() == pytest.approx(_expected_db(MARKER_DN), abs=0.5)
    found = np.argwhere(marker_mask)
    assert len(found) >= 4, "the bright marker block vanished"
    centre_row, centre_col = found.mean(axis=0)
    mid_row = (MARKER_ROWS[0] + MARKER_ROWS[1]) / 2
    mid_col = (MARKER_COLS[0] + MARKER_COLS[1]) / 2
    expected_lat = NORTH - (NORTH - SOUTH) * mid_row / LATTICE[-1]
    expected_lon = WEST + (EAST - WEST) * mid_col / LATTICE[-1]
    expected_row = (NORTH - expected_lat) / src_transform_step(vv_path, "lat")
    expected_col = (expected_lon - WEST) / src_transform_step(vv_path, "lon")
    assert abs(centre_row - expected_row) <= 3, (
        f"marker row {centre_row:.1f} is not where the GCPs put it "
        f"({expected_row:.1f}) — the geolocation is wrong"
    )
    assert abs(centre_col - expected_col) <= 3, (
        f"marker column {centre_col:.1f} is not where the GCPs put it "
        f"({expected_col:.1f})"
    )

    # The sidecar is the provenance a later stage is allowed to trust.
    assert sidecar["units"] == "dB"
    assert sidecar["bands"] == ["VV", "VH"]
    assert sidecar["relative_orbit"] == 19
    assert sidecar["archive"] == stand_in_safe["after"].name
    assert [round(v, 6) for v in sidecar["bbox_deg"]] == [WEST, SOUTH, EAST, NORTH]
    assert sidecar["grid"]["width"] and sidecar["grid"]["height"]
    assert sidecar["band_stats"]["VV"]["observed_fraction"] < 1.0
    assert sidecar["band_stats"]["VV"]["median_db"] == pytest.approx(
        _expected_db(UNIFORM_DN), abs=0.2
    )


def src_transform_step(path: Path, axis: str) -> float:
    """The magnitude of one delivered grid step, in degrees."""
    with rasterio.open(path) as src:
        return abs(src.transform.a) if axis == "lon" else abs(src.transform.e)


def test_before_and_after_land_on_one_grid_with_the_expected_delta(
    stand_in_safe: dict, tmp_path: Path
):
    """Both labels share one grid, so a pixel difference is meaningful."""
    for label in ("before", "after"):
        preprocess_s1.preprocess_safe_local(
            safe_zip=stand_in_safe[label],
            output_dir=tmp_path / "sentinel1",
            label=label,
            bbox=(WEST, SOUTH, EAST, NORTH),
            acquisition=datetime(2026, 8, 24, 0, 18, 44, tzinfo=timezone.utc),
        )

    with rasterio.open(tmp_path / "sentinel1" / "before" / "s1_before_VV.tif") as b, \
         rasterio.open(tmp_path / "sentinel1" / "after" / "s1_after_VV.tif") as a:
        assert b.shape == a.shape
        assert b.transform == a.transform
        assert b.crs == a.crs
        before, after = b.read(1), a.read(1)

    valid = (before != NODATA) & (after != NODATA)
    delta = (after - before)[valid]
    # The after scene differs only inside the marker block, so outside it the two
    # dates must be identical to within the median filter's reach.
    assert np.median(np.abs(delta)) == pytest.approx(0.0, abs=0.2)
    assert delta.max() == pytest.approx(
        _expected_db(MARKER_DN) - _expected_db(UNIFORM_DN), abs=0.7
    )


# ─── Candidate B: the CLI's reporting stage ─────────────────────────────────


def test_cli_reports_existing_paths_after_preprocessing(
    stand_in_safe: dict, tmp_path: Path, monkeypatch
):
    """main() must survive its own success message.

    The reviewed version built the printed path as
    ``sidecar.parent / 's1_' + args.label + '_VV.tif'``, which is a Path divided
    by a str concatenation — a TypeError raised *after* the expensive work.
    """
    monkeypatch.setattr(
        preprocess_s1.sys, "argv",
        [
            "preprocess_s1.py",
            "--safe-zip", str(stand_in_safe["after"]),
            "--output-dir", str(tmp_path / "sentinel1"),
            "--label", "after",
            "--bbox", f"{WEST},{SOUTH},{EAST},{NORTH}",
            "--acquisition", "2026-08-24T00:18:44Z",
            "--orbit", "19",
        ],
    )

    assert preprocess_s1.main() == 0

    for name in ("s1_after_VV.tif", "s1_after_VH.tif", "S1_PROCESSED.json"):
        assert (tmp_path / "sentinel1" / "after" / name).is_file(), (
            f"the CLI reported success without writing {name}"
        )
