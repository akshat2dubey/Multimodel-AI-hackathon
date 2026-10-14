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
        --bbox <min_lon,min_lat,max_lon,max_lat> \
        --before-date <YYYY-MM-DD> --after-date <YYYY-MM-DD> \
        --output ./data/sentinel1

    --bbox is the PRIMARY area-of-interest input. --center-lat/--center-lon
    (plus an optional --extent) is only a fallback for when no box is known. The
    case-study values live in config/presets.py and, for a resolved run, in
    results/run_meta.json.

Exit codes:
    0  both before/after GRD products downloaded (non-trivial file size)
    1  search failed, no matching product, download failed, or credentials missing
"""

import argparse
import hashlib
import json
import sys
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Project root on sys.path so `config.*` resolves whether this is run as
# `python scripts/download_sentinel.py` or from anywhere else.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.presets import DEFAULT_EXTENT  # noqa: E402
from config.run_meta import resolve_bbox  # noqa: E402

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


CATALOGUE_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"


def catalogue_expectation(client: CDSEClient, product_name: str) -> dict:
    """What the catalogue says this product should be: byte size and MD5.

    The product name is looked up exactly (``Name eq``), which is the same query
    the CDSE downloader itself uses, so no extra catalogue load is generated.

    Returns an empty dict when the lookup fails: a missing expectation must not
    block a download that is otherwise verifiable, since the archive's own zip
    central directory is checked either way.
    """
    name = product_name if product_name.endswith((".SAFE", ".SEN3", ".nc")) else f"{product_name}.SAFE"
    try:
        session = client.downloader.session
        response = session.get(
            CATALOGUE_URL,
            params={
                "$filter": f"Name eq '{name}'",
                "$select": "Id,Name,ContentLength,Checksum",
            },
            timeout=60,
        )
        response.raise_for_status()
        values = response.json().get("value") or []
    except Exception as exc:  # pylint: disable=broad-except
        print(
            f"  ⚠ could not read the catalogue's size/checksum for {name}: {exc}",
            file=sys.stderr,
        )
        return {}
    if not values:
        return {}
    entry = values[0]
    md5 = next(
        (c.get("Value") for c in entry.get("Checksum", []) if c.get("Algorithm") == "MD5"),
        None,
    )
    return {
        "id": entry.get("Id"),
        "name": entry.get("Name", name),
        "bytes": int(entry["ContentLength"]) if entry.get("ContentLength") is not None else None,
        "md5": md5.lower() if isinstance(md5, str) else None,
    }


def verify_safe_archive(path, *, expected_bytes=None, expected_md5=None) -> tuple[bool, str]:
    """Is this file actually the complete product it claims to be?

    Why this exists: the downloader skips a file that already exists, so a
    transfer that ended early used to be trusted forever. A 622 MB prefix of a
    1.26 GB product passed the old "bigger than 1 MB" check, was kept by
    skip_existing, and every later run preprocessed it as if it were the scene.

    Two layers, cheapest first:
      * the zip central directory — a truncated archive cannot be opened, so
        this catches a partial transfer with one seek;
      * the catalogue's own ContentLength and MD5 — the strong check, and the
        one that catches corruption rather than just truncation.
    """
    p = Path(path)
    if not p.is_file():
        return False, "file is missing"
    size = p.stat().st_size
    if size < MIN_PRODUCT_BYTES:
        return False, f"only {size:,} bytes (< {MIN_PRODUCT_BYTES:,}) — not a real product"
    if expected_bytes is not None and size != int(expected_bytes):
        return False, f"size {size:,} != catalogue {int(expected_bytes):,} (incomplete transfer)"
    try:
        with zipfile.ZipFile(p) as archive:
            if not archive.namelist():
                return False, "archive has no members"
    except zipfile.BadZipFile as exc:
        return False, f"not a readable SAFE archive ({exc})"
    except OSError as exc:
        return False, f"unreadable ({exc})"
    if expected_md5:
        digest = hashlib.md5(usedforsecurity=False)
        with p.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != str(expected_md5).lower():
            return False, "MD5 does not match the catalogue's checksum"
    return True, f"verified ({size:,} bytes)"


def download_product(client: CDSEClient, product: Product, output_dir: Path) -> Path:
    """
    Download one product and prove a complete one landed.

    An existing file is verified before it is reused, and the download is
    verified (zip structure, catalogue size, catalogue MD5) before it is kept —
    the only outcomes are "a complete product" or a RuntimeError.

    Preprocessing is deliberately NOT done here: it is an explicit step of
    run_pipeline.py, so a download can never quietly decide which pixels the
    analysis is allowed to see. (Taking the label and the bbox out of this
    function is also what fixes the crash in main(): it called this with three
    arguments while the signature demanded five, so the download step raised a
    TypeError before it could ever fetch anything.)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    expected = catalogue_expectation(client, product.name)
    if expected:
        print(
            f"  Catalogue says: {expected['bytes']:,} bytes, md5 {expected['md5']}"
            if expected.get("bytes")
            else f"  Catalogue entry found for {product.name}"
        )

    path = output_dir / f"{product.name}.zip"
    if path.is_file():
        ok, detail = verify_safe_archive(
            path, expected_bytes=expected.get("bytes"), expected_md5=expected.get("md5")
        )
        if ok:
            print(f"    OK  {path.name} already on disk and {detail}")
            _write_product_info(output_dir, product, path, verified=expected, reused=True)
            return path
        print(
            f"  ⚠ the file already on disk is not usable ({detail}) — re-downloading "
            "instead of trusting it",
            file=sys.stderr,
        )
        path.unlink(missing_ok=True)

    print(f"  Downloading {product.name} ...")
    try:
        downloaded = client.download(product, output_dir=str(output_dir), skip_existing=False)
    except CDSEError as exc:
        raise RuntimeError(f"CDSE download failed for {product.name}: {exc}") from exc

    downloaded = Path(downloaded)
    if not downloaded.exists():
        raise RuntimeError(f"CDSE reported success but {downloaded} does not exist")

    ok, detail = verify_safe_archive(
        downloaded, expected_bytes=expected.get("bytes"), expected_md5=expected.get("md5")
    )
    if not ok:
        downloaded.unlink(missing_ok=True)
        raise RuntimeError(
            f"the downloaded {downloaded.name} failed verification ({detail}) and was "
            "deleted. A partial product must never reach the analysis: re-run to retry "
            "the transfer."
        )

    size = downloaded.stat().st_size
    print(f"    OK  {downloaded.name}  ({size / 1_048_576:.1f} MB) — {detail}")
    _write_product_info(output_dir, product, downloaded, verified=expected, reused=False)
    return downloaded


def _write_product_info(output_dir: Path, product: Product, path: Path, *, verified: dict, reused: bool) -> None:
    """Record what was downloaded, and what it was checked against."""
    sidecar = {
        "product_name": product.name,
        "product_id": product.id,
        "collection": product.collection,
        "acquired": product.datetime.isoformat() if product.datetime else None,
        "bbox": product.bbox,
        "downloaded_file": path.name,
        "downloaded_bytes": path.stat().st_size,
        "relative_orbit": _relative_orbit(product),
        "reused_existing_file": reused,
        "verified_against_catalogue": bool(verified),
        "catalogue_bytes": verified.get("bytes"),
        "catalogue_md5": verified.get("md5"),
        "source": "Copernicus Data Space Ecosystem",
    }
    (output_dir / "PRODUCT_INFO.json").write_text(
        json.dumps(sidecar, indent=2), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Sentinel-1 GRD products")
    parser.add_argument(
        "--bbox",
        help=(
            "PRIMARY area of interest: min_lon,min_lat,max_lon,max_lat. "
            "Falls back to --center-lat/--center-lon/--extent when omitted."
        ),
    )
    parser.add_argument("--center-lat", type=float, default=None,
                        help="Fallback AOI centre latitude (prefer --bbox)")
    parser.add_argument("--center-lon", type=float, default=None,
                        help="Fallback AOI centre longitude (prefer --bbox)")
    parser.add_argument("--extent", type=float, default=DEFAULT_EXTENT,
                        help=f"Fallback half-width in degrees (default {DEFAULT_EXTENT})")
    parser.add_argument("--before-date", required=True, help="Date of event (YYYY-MM-DD)")
    parser.add_argument("--after-date", required=True, help="Date after event (YYYY-MM-DD)")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--margin-days", type=int, default=7, help="Search window each side of target date"
    )
    args = parser.parse_args()

    try:
        bbox = resolve_bbox(args.bbox, args.center_lat, args.center_lon, args.extent)
    except ValueError as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 1
    min_lon, min_lat, max_lon, max_lat = bbox
    # The box's midpoint: logged below and used for the "did we get the right
    # scene" sanity checks, so a stale centre cannot disagree with the AOI.
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2
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
