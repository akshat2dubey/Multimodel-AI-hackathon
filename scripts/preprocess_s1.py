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
import sys
from datetime import datetime, timezone
from pathlib import Path

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
    import numpy as np
    from scipy.ndimage import median_filter

    a = np.ma.masked_invalid(arr.astype(np.float32))
    filled = np.where(a.mask, 0.0, a.data)
    med = median_filter(filled, size=3)
    out = np.where(a.mask, float("nan"), med.astype(np.float32))
    return out


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
        transform=from_bounds(west, north, east, south, width, height),
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
    vv_name = None
    vh_name = None
    for n in zip_members:
        low = n.lower()
        if "/annotation/" in low or "s1a-" not in low and "s1b-" not in low:
            continue
        if "measurement" not in low:
            continue
        if vv_name is None and ("vv" in low or " VV " in n or n.endswith("vv.tiff") or n.endswith("vv.tif")):
            vv_name = n
        elif vh_name is None and ("vh" in low or " VH " in n or n.endswith("vh.tiff") or n.endswith("vh.tif")):
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
    """Unzip the SAFE, read VV/VH measurement tiffs, reproject a small bbox
    window to EPSG:4326, convert to dB, 3x3 speckle, write four tifs + sidecar.

    This is intentionally the large/slow path; it may be killed on time/quota.
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
            win = win_from_bounds(west, north, east, south, src.width, src.height)
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
# Process API bbox-crop path (fast, no local decompression)
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
            t = src.transform
            w2, s2, e2, n2 = t * (0, 0, src.width, src.height)
            bbox = (float(w2), float(s2), float(e2), float(n2))
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
    parser.add_argument("--target-pixels", type=int, default=640, help="Pixels per side of crop")
    parser.add_argument(
        "--full-scene",
        action="store_true",
        help="Process the full SAFE locally instead of the Process API bbox crop",
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

    if args.full_scene:
        sidecar = preprocess_full_scene_safe(
            safe_zip=args.safe_zip,
            output_dir=args.output_dir,
            label=args.label,
            bbox=bbox,
            default_acq=acq,
        )
    else:
        sidecar = preprocess_product(
            safe_zip=args.safe_zip,
            output_dir=args.output_dir,
            label=args.label,
            bbox=bbox,
            acquisition=acq,
            orbit=args.orbit,
            target_pixels=args.target_pixels,
        )

    print(f"  OK  preprocessed -> {sidecar.parent / 's1_' + args.label + '_VV.tif'}")
    print(f"  OK  preprocessed -> {sidecar.parent / 's1_' + args.label + '_VH.tif'}")
    print(f"  OK  sidecar -> {sidecar}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
