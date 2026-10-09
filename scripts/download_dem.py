#!/usr/bin/env python3
"""
Download Copernicus DEM (GLO-30 / WorldDEM-30) elevation tiles for flood path tracing.

The DEM drives the BONUS feature: given an upstream point, trace the flood path down
the valley using steepest-descent flow routing and list the settlements along it.

Source: the public Copernicus DEM bucket on AWS Open Data.
    https://copernicus-dem-30m.s3.amazonaws.com/
This route is anonymous — no Copernicus account is required, unlike Sentinel-1/2.

Tiles are 1 degree of latitude by 1 degree of longitude at 30m posting, named by their
south-west corner (N28_00_E078_00 covers 28N-29N, 78E-79E). Any tile intersecting the
area of interest is downloaded and mosaicked into a single GeoTIFF.

Usage:
    python download_dem.py \
        --center-lat 28.3 --center-lon 85.7 \
        --output ./data/dem

Attribution (required in any published output):
    "Produced using Copernicus WorldDEM-30 - DLR e.V. 2010-2014 and
     (c) Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by
     the European Union and ESA; all rights reserved."

Exit codes:
    0  a mosaicked DEM GeoTIFF was written to <output>/dem.tif
    1  no tile could be fetched, or the mosaic could not be written
"""

import argparse
import json
import math
import sys
import urllib.error
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):  # never crash on a cp1252 Windows console
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    import rasterio
    from rasterio.merge import merge
except ImportError:
    print(
        "❌ rasterio is required to mosaic DEM tiles.\n"
        "   Install it:  pip install rasterio",
        file=sys.stderr,
    )
    sys.exit(1)

BUCKET = "https://copernicus-dem-30m.s3.amazonaws.com"

ATTRIBUTION = (
    "Produced using Copernicus WorldDEM-30 - DLR e.V. 2010-2014 and "
    "(c) Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by "
    "the European Union and ESA; all rights reserved."
)

# A real 1-degree GLO-30 tile is ~30-45 MB. Anything smaller is an S3 error document.
MIN_TILE_BYTES = 1_000_000


def tile_names(min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> list[str]:
    """
    Return the GLO-30 tile keys covering the bounding box.

    Tiles are addressed by their south-west corner, floored to whole degrees.
    A box spanning an integer boundary needs an extra tile on the far side, so
    the range is taken over floor(min) .. floor(max) inclusive.
    """
    lat0 = math.floor(min_lat)
    lat1 = math.floor(max_lat)
    lon0 = math.floor(min_lon)
    lon1 = math.floor(max_lon)

    names = []
    for lat in range(lat0, lat1 + 1):
        for lon in range(lon0, lon1 + 1):
            ns = "N" if lat >= 0 else "S"
            ew = "E" if lon >= 0 else "W"
            names.append(
                f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM"
            )
    return names


def fetch_tile(tile: str, dest_dir: Path) -> Path:
    """
    Download one DEM tile, verifying a real raster landed.

    Raises RuntimeError when the object is missing or is an error document.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{tile}.tif"

    # Key format verified against the bucket listing: the tile name is used
    # verbatim as both the prefix and the file name, _COG included.
    url = f"{BUCKET}/{tile}/{tile}.tif"
    print(f"  GET {url}")

    try:
        with urllib.request.urlopen(url, timeout=120) as resp:
            data = resp.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} — tile {tile} does not exist in the bucket") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"network error fetching {tile}: {exc.reason}") from exc

    if len(data) < MIN_TILE_BYTES:
        raise RuntimeError(
            f"tile {tile} is only {len(data):,} bytes (< {MIN_TILE_BYTES:,}) — "
            "this is an error document, not a DEM"
        )
    if data[:2] != b"II" and data[:2] != b"MM":
        raise RuntimeError(f"tile {tile} is not a TIFF (bad magic bytes)")

    dest.write_bytes(data)
    print(f"    OK  {tile}.tif  ({len(data) / 1_048_576:.1f} MB)")
    return dest


def mosaic(tiles: list[Path], output: Path) -> Path:
    """Merge downloaded tiles into one GeoTIFF covering the full area."""
    sources = [rasterio.open(t) for t in tiles]
    try:
        array, transform = merge(sources)
        profile = sources[0].profile.copy()
    finally:
        for src in sources:
            src.close()

    profile.update(
        height=array.shape[1],
        width=array.shape[2],
        transform=transform,
        count=1,
        dtype=array.dtype,
        compress="deflate",
    )
    profile.pop("photometric", None)

    output.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output, "w", **profile) as dst:
        dst.write(array[0], 1)

    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Copernicus DEM tiles")
    parser.add_argument("--center-lat", type=float, required=True)
    parser.add_argument("--center-lon", type=float, required=True)
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--half-extent", type=float, default=0.3, help="Half extent in degrees")
    parser.add_argument("--bbox", help="Bounding box: min_lon,min_lat,max_lon,max_lat")
    args = parser.parse_args()

    if args.bbox:
        parts = [float(x) for x in args.bbox.split(",")]
        if len(parts) != 4:
            print("❌ --bbox must be min_lon,min_lat,max_lon,max_lat", file=sys.stderr)
            return 1
        min_lon, min_lat, max_lon, max_lat = parts
    else:
        min_lon = args.center_lon - args.half_extent
        min_lat = args.center_lat - args.half_extent
        max_lon = args.center_lon + args.half_extent
        max_lat = args.center_lat + args.half_extent

    output_dir = Path(args.output)
    tiles = tile_names(min_lon, min_lat, max_lon, max_lat)

    print(f"\n{'=' * 70}")
    print("Copernicus DEM Download")
    print(f"{'=' * 70}")
    print(f"  Bounding box: [{min_lon:.3f}, {min_lat:.3f}, {max_lon:.3f}, {max_lat:.3f}]")
    print(f"  Tiles:        {len(tiles)} ({', '.join(t[:34] for t in tiles)})")
    print(f"  Source:       AWS Open Data (anonymous, no account required)")
    print()

    tile_dir = output_dir / "tiles"
    fetched: list[Path] = []
    failed: list[str] = []

    for tile in tiles:
        try:
            fetched.append(fetch_tile(tile, tile_dir))
        except RuntimeError as exc:
            print(f"  ❌ {exc}", file=sys.stderr)
            failed.append(tile)

    if not fetched:
        print(
            "❌ No DEM tiles could be downloaded. Flood path tracing cannot run.\n"
            "   Check network access to copernicus-dem-30m.s3.amazonaws.com",
            file=sys.stderr,
        )
        return 1

    if failed:
        print(
            f"  ⚠ {len(failed)} of {len(tiles)} tiles failed — mosaicking what we have; "
            "the DEM will not cover the whole area.",
            file=sys.stderr,
        )

    print("\n🧩 Mosaicking tiles...")
    dem_path = output_dir / "dem.tif"
    try:
        mosaic(fetched, dem_path)
    except Exception as exc:
        print(f"❌ Could not write mosaic: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    size = dem_path.stat().st_size
    if size < MIN_TILE_BYTES:
        print(
            f"❌ Mosaic is only {size:,} bytes — not a usable DEM. Nothing written.",
            file=sys.stderr,
        )
        dem_path.unlink(missing_ok=True)
        return 1

    with rasterio.open(dem_path) as src:
        bounds = src.bounds
        resolution = src.res[0]

    # Sidecar consumed by flood_path.dem_is_synthetic(): synthetic must be False
    # so the tracer labels this as real Copernicus data rather than the demo fixture.
    (output_dir / "PRODUCT_INFO.json").write_text(
        json.dumps(
            {
                "product_type": "Copernicus DEM GLO-30 (WorldDEM-30)",
                "resolution_m": round(resolution, 6),
                "tiles": [t.name for t in fetched],
                "bounds": [bounds.left, bounds.bottom, bounds.right, bounds.top],
                "source": BUCKET,
                "attribution": ATTRIBUTION,
                "synthetic": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\n✅ DEM written: {dem_path}  ({size / 1_048_576:.1f} MB)")
    print(f"   Resolution: {resolution:.4f}° (~30 m)")
    print(f"   Bounds:     {bounds.left:.3f}, {bounds.bottom:.3f} -> "
          f"{bounds.right:.3f}, {bounds.top:.3f}")
    print("   Use: flood path tracing (bonus feature)")
    print(f"\n   Attribution required: {ATTRIBUTION}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
