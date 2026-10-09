#!/usr/bin/env python3
"""
Download Sentinel-1 GRD (Ground Range Detected) products via Copernicus Data Space Ecosystem.

CRITICAL: For change detection, we need images from the SAME orbit track
(same relative orbit number, same time of day, ~12 days apart).
Different tracks view terrain from different angles — cannot compare pixel-by-pixel.

Authentication (OAuth2 client credentials):
    1. Register free at https://dataspace.copernicus.eu/ and create an app
       under "Identity & Access Management" -> "create new OAuth client".
    2. Export the credentials:
           set CDSE_CLIENT_ID=<client-id>
           set CDSE_CLIENT_SECRET=<client-secret>
       (PowerShell: $env:CDSE_CLIENT_ID="..."; $env:CDSE_CLIENT_SECRET="...")

Usage:
    python download_sentinel.py \
        --center-lat 28.3 --center-lon 85.7 \
        --before-date 2026-08-24 --after-date 2026-08-31 \
        --output ./data/sentinel1

Exit codes:
    0  both before/after GRD products downloaded (non-trivial file size)
    1  search failed, no matching product, download failed, or credentials missing
"""

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
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
        "❌ The 'cdse-client' package is required for Sentinel-1 downloads.\n"
        "   Install it:  pip install cdse-client\n"
        "   (Note: this is NOT the 'cdse' package, which does not exist on PyPI.)",
        file=sys.stderr,
    )
    sys.exit(1)

COLLECTION = "sentinel-1-grd"

# Sentinel-1 GRD products are SAFE archives of several hundred MB. Anything
# smaller is an error page, a truncated transfer, or a stub — not real data.
MIN_PRODUCT_BYTES = 1_000_000


def _relative_orbit(product: Product) -> int | None:
    """Extract the relative orbit number from a CDSE product, if present."""
    for key in ("sat:relative_orbit", "relativeOrbit", "relative_orbit"):
        value = product.properties.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return None


def _acq_time(product: Product) -> datetime | None:
    return product.datetime


def build_client() -> CDSEClient:
    """Build an authenticated CDSE client, or exit non-zero with instructions."""
    try:
        return CDSEClient()
    except CDSEError as exc:
        print(
            f"❌ Could not authenticate with Copernicus Data Space: {exc}\n"
            "   Sentinel-1 downloads require a free account.\n"
            "     1. Register: https://dataspace.copernicus.eu/\n"
            "     2. Create an OAuth client under Identity & Access Management\n"
            "     3. set CDSE_CLIENT_ID=... and set CDSE_CLIENT_SECRET=...\n"
            "   The pipeline cannot produce a flood mask without real SAR data.",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as exc:  # AuthenticationError subclasses CDSEError, but be safe
        print(f"❌ Could not initialise CDSE client: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)


def search_products(
    client: CDSEClient,
    bbox: tuple[float, float, float, float],
    target_date: str,
    margin_days: int,
    label: str,
) -> list[Product]:
    """Search Sentinel-1 GRD products covering bbox near target_date."""
    min_lon, min_lat, max_lon, max_lat = bbox
    target_dt = datetime.strptime(target_date, "%Y-%m-%d").replace(        tzinfo=timezone.utc)
    start = (target_dt - timedelta(days=margin_days)).strftime("%Y-%m-%d")
    end = (target_dt + timedelta(days=margin_days)).strftime("%Y-%m-%d")

    print(f"  Searching {COLLECTION} near {target_date} (window {start}..{end})")
    print(f"  Area: [{min_lon:.3f}, {min_lat:.3f}] -> [{max_lon:.3f}, {max_lat:.3f}]")

    try:
        products = client.search(
            bbox=[min_lon, min_lat, max_lon, max_lat],
            start_date=start,
            end_date=end,
            collection=COLLECTION,
            limit=30,
        )
    except CDSEError as exc:
        print(f"❌ CDSE search failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"  Found {len(products)} Sentinel-1 products for {label}-event window")
    return products


def pick_same_track_pair(
    products: list[Product], target_date: str
) -> tuple[Product, Product, int | None, str | None]:
    """
    Pick a before/after pair from the same relative orbit.

    Returns (before, after, orbit, warning). Raises ValueError when no valid
    pair exists — the caller must surface that, not invent one.
    """
    # cdse returns timezone-aware datetimes (UTC); the target must be
    # expressed in the same frame so the comparisons below are valid.
    target_dt = datetime.strptime(target_date, "%Y-%m-%d").replace(
        tzinfo=timezone.utc
    )

    # An orbit may be None when cdse does not report one; keep those
    # separate so we can fall back to a same-day pairing as a last resort.
    by_orbit: dict[int | None, list[Product]] = {}
    for p in products:
        orbit = _relative_orbit(p)
        by_orbit.setdefault(orbit, []).append(p)

    def _acq(p: Product) -> datetime:
        """Return the acquisition instant for a product, normalised to UTC.

        cdse may return either timezone-aware (UTC) or naive datetimes across
        versions and collections, so every comparison below is done in a
        single, explicit frame instead of assuming one or the other.
        """
        dt = p.datetime
        if dt is None:
            raise ValueError(f"product {p.name!r} has no acquisition datetime")
        if dt.tzinfo is None:
            return dt.replace(        tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    def bracket(candidates: list[Product], need_second: bool) -> tuple[Product, Product | None] | None:
        """Pick the best single product, plus optionally the best opposite-side product.

        ``need_second`` exists so that the main pass and the fall-back pass below
        can share one selection rule: ``need_second=False`` returns only the best
        product on one side of the event; ``need_second=True`` also returns the
        best product on the opposite side, with a different acquisition day when
        the same-day opposite is unavailable.
        """
        # Closest product to the event whose acquisition is on the requested side
        # of the event date. ``need_second=False`` -> before side (strictly before
        # the event UTC day), ``need_second=True`` -> after side (on or after).
        side_candidates = (
            [p for p in candidates if _acq(p) < target_dt] if not need_second else [p for p in candidates if _acq(p) >= target_dt]
        )

        # Need at least one candidate on the requested side.
        if not side_candidates:
            return None

        best = min(side_candidates, key=lambda p: abs((_acq(p) - target_dt).total_seconds()))
        if not need_second:
            return best, None

        # For the second side, choose the closest acquisition on the opposite side,
        # but never the same acquisition we already picked for the first side.
        opposite = [p for p in candidates if _acq(p) >= target_dt and p is not best]
        if not opposite:
            # Same-day opposite-side acquisition does not exist in this orbit.
            return None
        second = min(opposite, key=lambda p: abs((_acq(p) - target_dt).total_seconds()))
        return best, second

    # Prefer a same-relative-orbit, same-day pair: pixel-level comparison requires
    # the two scenes to view the terrain from the same geometry.
    for orbit, candidates in sorted(by_orbit.items()):
        if orbit is None:
            continue
        pair = bracket(candidates, need_second=True)
        if pair is not None:
            before, after = pair
            return before, after, orbit, None

    # No same-orbit, same-day pair exists. Fall back to a closest-before /
    # closest-after pair across all orbits (same calendar day is not guaranteed).
    all_candidates = [p for ps in by_orbit.values() for p in ps]
    first = bracket(all_candidates, need_second=False)
    if first is None:
        raise ValueError(
            "no Sentinel-1 acquisition before the event was found in this window. "
            "Widen --margin-days, or the area was not imaged by Sentinel-1 that fortnight."
        )
    before, _ = first
    second = bracket(all_candidates, need_second=True)
    if second is None or second[1] is None:
        raise ValueError(
            "no Sentinel-1 acquisition on or after the event was found in this window. "
            "Widen --margin-days, or the area was not imaged by Sentinel-1 that fortnight."
        )
    after = second[1]
    orbit = _relative_orbit(before)
    warning = (
        "Relative orbit not reported — same-track pairing could not be confirmed"
        if orbit is None
        else None
    )
    return before, after, orbit, warning


def download_product(client: CDSEClient, product: Product, output_dir: Path) -> Path:
    """
    Download one product and verify a real file landed.

    Returns the downloaded path. Raises RuntimeError on a zero/truncated write.
    """
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
            "this is not a real GRD product"
        )

    print(f"    OK  {path.name}  ({size / 1_048_576:.1f} MB)")

    # Metadata about a download that actually happened — not a stand-in for one.
    sidecar = {
        "product_name": product.name,
        "product_id": product.id,
        "collection": product.collection,
        "acquired": product.datetime.isoformat() if product.datetime else None,
        "bbox": product.bbox,
        "downloaded_file": path.name,
        "downloaded_bytes": size,
        "source": "Copernicus Data Space Ecosystem",
    }
    (output_dir / "PRODUCT_INFO.json").write_text(
        json.dumps(sidecar, indent=2), encoding="utf-8"
    )
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Sentinel-1 GRD products")
    parser.add_argument("--center-lat", type=float, required=True)
    parser.add_argument("--center-lon", type=float, required=True)
    parser.add_argument("--before-date", required=True, help="Date of event (YYYY-MM-DD)")
    parser.add_argument("--after-date", required=True, help="Date after event (YYYY-MM-DD)")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--bbox", help="Bounding box: min_lon,min_lat,max_lon,max_lat")
    parser.add_argument(
        "--margin-days", type=int, default=7, help="Search window each side of target date"
    )
    args = parser.parse_args()

    center_lat, center_lon = args.center_lat, args.center_lon
    half_extent = 0.3  # ~30km

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
    print("Sentinel-1 Data Download")
    print(f"{'=' * 70}")
    print(f"  Area center:  ({center_lat:.4f}, {center_lon:.4f})")
    print(f"  Bounding box: [{min_lon:.3f}, {min_lat:.3f}, {max_lon:.3f}, {max_lat:.3f}]")
    print(f"  Event date:   {args.before_date}")
    print()

    client = build_client()

    # One search spanning both dates, so the pair comes from a single
    # observation set and same-track pairing is meaningful.
    span_start = min(args.before_date, args.after_date)
    products = search_products(client, bbox, span_start, args.margin_days * 2, "event")

    if not products:
        print(
            "❌ No Sentinel-1 GRD products found for this area and date range.\n"
            "   Widen --margin-days, or check that the area is covered by an\n"
            "   ascending/descending IW pass in that fortnight.",
            file=sys.stderr,
        )
        return 1

    try:
        before_prod, after_prod, orbit, warning = pick_same_track_pair(
            products, args.before_date
        )
    except ValueError as exc:
        print(f"❌ Cannot form a before/after pair: {exc}", file=sys.stderr)
        return 1

    if orbit is not None:
        print(f"\n  Same-track pair selected: relative orbit {orbit}")
    if warning:
        print(f"  ⚠ {warning}")

    failures: list[str] = []
    for label, product in (("before", before_prod), ("after", after_prod)):
        print("─" * 70)
        print(f"📥 Downloading {label}-event product ({product.datetime})")
        try:
            download_product(client, product, output_dir / label)
        except RuntimeError as exc:
            print(f"  ❌ {exc}", file=sys.stderr)
            failures.append(label)

    if failures:
        print(
            f"\n❌ Sentinel-1 download FAILED for: {', '.join(failures)}\n"
            f"   Nothing usable was written to {output_dir}.\n"
            "   Do not run inference — a missing scene must not be masked over.",
            file=sys.stderr,
        )
        return 1

    print(f"\n✅ Sentinel-1 download complete → {output_dir}")
    print(f"   Before: {output_dir / 'before'}")
    print(f"   After:  {output_dir / 'after'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
