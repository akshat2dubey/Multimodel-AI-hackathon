#!/usr/bin/env python3
"""
Download Sentinel-2 L2A (Level-2A, surface reflectance) products via Copernicus Data Space.

Sentinel-2 provides 10-20m resolution optical imagery in 13 spectral bands.
Cloud cover is a major issue in monsoon season — use only when sky is clear.
Sentinel-1 radar remains the primary source for this event; S2 is supplementary.

Authentication (OAuth2 client credentials):
    1. Register free at https://dataspace.copernicus.eu/ and create an app
       under "Identity & Access Management" -> "create new OAuth client".
    2. Export the credentials:
           set CDSE_CLIENT_ID=<client-id>
           set CDSE_CLIENT_SECRET=<client-secret>

Usage:
    python download_sentinel2.py \
        --center-lat 28.3 --center-lon 85.7 \
        --before-date 2026-08-14 --after-date 2026-08-28 \
        --output ./data/sentinel2 \
        --max-cloud 30

Sentinel-2 bands used for flood mapping:
  B02 (Blue, 10m)  — water absorbs blue light → good for water index (NDWI)
  B03 (Green, 10m) — enhanced for water
  B04 (Red, 10m)   — water absorbs red → NDWI
  B08 (NIR, 10m)   — water has very low NIR reflectance → key for water detection
  B11 (SWIR, 20m)  — helps distinguish water from shadow/clouds
  B12 (SWIR, 20m)  — additional SWIR band

Exit codes:
    0  a product was downloaded for each requested date
    1  credentials missing, search failed, no cloud-free scene, or download failed
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):  # never crash on a cp1252 Windows console
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    from cdse import CDSEClient, CDSEError, Product
except ImportError:
    print(
        "❌ The 'cdse-client' package is required for Sentinel-2 downloads.\n"
        "   Install it:  pip install cdse-client\n"
        "   (Note: this is NOT the 'cdse' package, which does not exist on PyPI.)",
        file=sys.stderr,
    )
    sys.exit(1)

COLLECTION = "sentinel-2-l2a"

# An L2A SAFE archive is hundreds of MB. Anything smaller is an error page or stub.
MIN_PRODUCT_BYTES = 1_000_000


def build_client() -> CDSEClient:
    try:
        return CDSEClient()
    except CDSEError as exc:
        print(
            f"❌ Could not authenticate with Copernicus Data Space: {exc}\n"
            "   Sentinel-2 downloads require a free account.\n"
            "     1. Register: https://dataspace.copernicus.eu/\n"
            "     2. Create an OAuth client under Identity & Access Management\n"
            "     3. set CDSE_CLIENT_ID=... and set CDSE_CLIENT_SECRET=...\n"
            "   Sentinel-2 is optional for this pipeline — Sentinel-1 radar is the\n"
            "   primary source and works through monsoon cloud.",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as exc:
        print(f"❌ Could not initialise CDSE client: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)


def search_scenes(
    client: CDSEClient,
    bbox: tuple[float, float, float, float],
    target_date: str,
    max_cloud_cover: float,
    margin_days: int,
) -> list[Product]:
    """
    Search Sentinel-2 L2A scenes near target_date, cloud-filtered and sorted.

    The CDSE client applies cloud_cover_max server-side and returns products
    sorted by acquisition date; we re-sort by real cloud cover.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    target = datetime.strptime(target_date, "%Y-%m-%d")
    start = (target - timedelta(days=margin_days)).strftime("%Y-%m-%d")
    end = (target + timedelta(days=margin_days)).strftime("%Y-%m-%d")

    print(f"  Searching {COLLECTION} near {target_date} (max cloud: {max_cloud_cover}%)")
    print(f"  Window: {start} .. {end}")
    print(f"  Area: [{min_lon:.3f}, {min_lat:.3f}] -> [{max_lon:.3f}, {max_lat:.3f}]")

    try:
        products = client.search(
            bbox=[min_lon, min_lat, max_lon, max_lat],
            start_date=start,
            end_date=end,
            collection=COLLECTION,
            cloud_cover_max=max_cloud_cover,
            limit=20,
        )
    except CDSEError as exc:
        print(f"❌ CDSE search failed: {exc}", file=sys.stderr)
        sys.exit(1)

    products = [p for p in products if _cloud(p) <= max_cloud_cover]
    products.sort(key=lambda p: (_cloud(p), p.datetime or datetime.min))
    return products


def _cloud(product: Product) -> float:
    value = product.cloud_cover
    return float(value) if value is not None else 100.0


def download_product(client: CDSEClient, product: Product, output_dir: Path) -> Path:
    """Download one product and verify a real, non-trivial file landed."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Downloading {product.name} ...")
    try:
        path = client.download(product, output_dir=str(output_dir))
    except CDSEError as exc:
        raise RuntimeError(f"CDSE download failed for {product.name}: {exc}") from exc

    path = Path(path)
    if not path.exists():
        raise RuntimeError(f"CDSE reported success but {path} does not exist")

    size = path.stat().st_size
    if size < MIN_PRODUCT_BYTES:
        raise RuntimeError(
            f"downloaded file is only {size:,} bytes (< {MIN_PRODUCT_BYTES:,}) — "
            "this is not a real L2A product"
        )

    print(f"    OK  {path.name}  ({size / 1_048_576:.1f} MB)")

    sidecar = {
        "product_name": product.name,
        "product_id": product.id,
        "collection": product.collection,
        "acquired": product.datetime.isoformat() if product.datetime else None,
        "cloud_cover": _cloud(product),
        "bbox": product.bbox,
        "downloaded_file": path.name,
        "downloaded_bytes": size,
        "source": "Copernicus Data Space Ecosystem",
        "bands": ["B02", "B03", "B04", "B08", "B11", "B12"],
    }
    (output_dir / "PRODUCT_INFO.json").write_text(
        json.dumps(sidecar, indent=2), encoding="utf-8"
    )
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Sentinel-2 L2A products")
    parser.add_argument("--center-lat", type=float, required=True)
    parser.add_argument("--center-lon", type=float, required=True)
    parser.add_argument("--before-date", required=True, help="Before-event date (YYYY-MM-DD)")
    parser.add_argument("--after-date", required=True, help="After-event date (YYYY-MM-DD)")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--bbox", help="Bounding box: min_lon,min_lat,max_lon,max_lat")
    parser.add_argument(
        "--max-cloud", type=float, default=30.0, help="Maximum cloud cover (percent)"
    )
    parser.add_argument("--margin-days", type=int, default=7, help="Search window each side")
    args = parser.parse_args()

    center_lat, center_lon = args.center_lat, args.center_lon
    half_extent = 0.3

    if args.bbox:
        parts = [float(x) for x in args.bbox.split(",")]
        if len(parts) != 4:
            print("❌ --bbox must be min_lon,min_lat,max_lon,max_lat", file=sys.stderr)
            return 1
        min_lon, min_lat, max_lon, max_lat = parts
    else:
        min_lon = center_lon - half_extent
        min_lat = center_lat - half_extent
        max_lon = center_lon + half_extent
        max_lat = center_lat + half_extent

    bbox = (min_lon, min_lat, max_lon, max_lat)
    output_dir = Path(args.output)

    print(f"\n{'=' * 70}")
    print("Sentinel-2 Data Download")
    print(f"{'=' * 70}")
    print(f"  Area center:  ({center_lat:.4f}, {center_lon:.4f})")
    print(f"  Bounding box: [{min_lon:.3f}, {min_lat:.3f}, {max_lon:.3f}, {max_lat:.3f}]")
    print(f"  Note: optical imagery may be blocked by monsoon cloud.")
    print()

    client = build_client()

    failures: list[str] = []
    for label, target_date in (("before", args.before_date), ("after", args.after_date)):
        print("─" * 70)
        print(f"📥 Searching for {label}-event Sentinel-2 ...")

        candidates = search_scenes(
            client, bbox, target_date, args.max_cloud, args.margin_days
        )

        if not candidates:
            print(
                f"  ❌ No {label}-event scene with cloud cover <= {args.max_cloud}% "
                "in the search window.",
                file=sys.stderr,
            )
            failures.append(label)
            continue

        best = candidates[0]
        print(f"  Selected: {best.name}")
        print(f"    Cloud cover: {_cloud(best):.1f}%   Acquired: {best.datetime}")

        try:
            download_product(client, best, output_dir / label)
        except RuntimeError as exc:
            print(f"  ❌ {exc}", file=sys.stderr)
            failures.append(label)

    if failures:
        print(
            f"\n❌ Sentinel-2 download FAILED for: {', '.join(failures)}\n"
            f"   Nothing usable was written to {output_dir}.\n"
            "   This is usually cloud cover, not a bug — Sentinel-1 radar is the\n"
            "   primary source for this event and does not need clear skies.",
            file=sys.stderr,
        )
        return 1

    print(f"\n✅ Sentinel-2 download complete → {output_dir}")
    print(f"   Before: {output_dir / 'before'}")
    print(f"   After:  {output_dir / 'after'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
