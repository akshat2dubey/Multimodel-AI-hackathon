#!/usr/bin/env python3
"""
Fetch pre-event OpenStreetMap data via the ohsome API.

The ohsome API (https://api.ohsome.org/v1) provides access to OSM history data
with time-based queries. We need a snapshot BEFORE the flood event; the event
and snapshot dates are arguments (--event-date / --snapshot-date), and a
run_pipeline.py run resolves them into results/run_meta.json.

Data to fetch (all as of snapshot date):
  - Buildings (osm_type=building)
  - Roads (osm_type=way, highway=*)
  - Bridges (osm_type=bridge OR man_made=bridge)
  - Hospitals (amenity=hospital)
  - Settlements/villages (place=village, place=hamlet)

Usage:
    python fetch_osm.py \
        --bbox <min_lon,min_lat,max_lon,max_lat>   # PRIMARY AOI input \
        --snapshot-date <YYYY-MM-DD>               # pre-event snapshot \
        --event-date <YYYY-MM-DD>                  # the flood event \
        --output ./data/osm

    --bbox is the PRIMARY area-of-interest input; --center-lat/--center-lon with
    an optional --extent is only a fallback.

⚠️ CRITICAL RULE: the snapshot must be BEFORE the event date. OSM edits made
after the event are prohibited as input (checked below and centrally in
config/run_meta.py).
"""

import argparse

import sys
import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

# Project root on sys.path so `config.*` resolves whether this is run as
# `python scripts/fetch_osm.py` or from anywhere else.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.presets import DEFAULT_EXTENT, EVENT_DATE  # noqa: E402
from config.run_meta import resolve_bbox  # noqa: E402

# Every outbound fetch goes through the HTTPS-only wrapper (T36).
from net_safe import urlopen  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):  # never crash on a cp1252 Windows console
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# Overpass API endpoint with historical (attic) data.
# [date:"..."] selects the OSM state AS OF that timestamp — the pre-event snapshot
# required by the hackathon rules (no post-event edits may be used as input).
# The event date itself is an argument (--event-date), never a literal here.
#
# Why not ohsome for geometry? Verified live 2026-10-09:
#   - /elements/geometry returns HTTP 403 (Apache-level block on the public host)
#   - aggregation endpoints (e.g. /elements/count) work and are used below to
#     cross-check our extract against the official OSH statistics.
OVERPASS_API = "https://overpass-api.de/api/interpreter"
OHSOME_API = "https://api.ohsome.org/v1"          # used only for aggregate counts
# Identifies this tool to Overpass/ohsome, as the tile/API etiquette asks.
USER_AGENT = "Jalnetra/0.1 (Multimodal AI Hackathon 2026, Track B)"


def overpass_ql(
    query: str,
    timeout: int = 180,
    retries: int = 3,
) -> dict[str, Any] | None:
    """POST an Overpass QL query (with attic date) and return parsed JSON.

    The main endpoint occasionally returns transient 504s under attic load;
    backing off and retrying a couple of times has proven reliable for the
    small per-category queries we run here.
    """
    import time
    data = urllib.parse.urlencode({"data": query}).encode("utf-8")
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                OVERPASS_API, data=data,
                headers={"Accept": "application/json", "User-Agent": USER_AGENT},
            )
            with urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8")
                ct = resp.headers.get("Content-Type", "")
                if "json" not in ct:
                    raise ValueError(f"server returned {ct} (likely rate-limit page)")
                return json.loads(body)
        except urllib.error.HTTPError as e:
            print(f"  ⚠ Overpass HTTP {e.code} (attempt {attempt}/{retries})")
            if e.code == 429:
                print("  ℹ️ Rate limited — backing off")
            time.sleep(10 * attempt)
        except (urllib.error.URLError, ValueError, json.JSONDecodeError) as e:
            print(f"  ⚠ Overpass error ({attempt}/{retries}): {e}")
            time.sleep(10 * attempt)
    return None


def run_attic_query(
    selector: str,
    bbox: tuple[float, float, float, float],
    snapshot_date: str,
    timeout: int = 180,
) -> list[dict]:
    """Run one attic Overpass query and return the raw OSM elements list."""
    min_lon, min_lat, max_lon, max_lat = bbox
    q = (
        f'[out:json][timeout:{timeout}]'
        f'[date:"{snapshot_date}T00:00:00Z"];'
        f'(node{selector}({min_lat},{min_lon},{max_lat},{max_lon});'
        f'way{selector}({min_lat},{min_lon},{max_lat},{max_lon}););'
        f'out geom tags;'
    )
    payload = overpass_ql(q, timeout=timeout + 30)
    if payload is None:
        return []
    return payload.get("elements", [])


def elements_to_geojson(elements: list[dict]) -> dict[str, Any]:
    """Convert raw Overpass elements (with out geom) to a proper GeoJSON FeatureCollection."""
    feats = []
    for el in elements:
        el_type = el.get("type")
        lat, lon = el.get("lat"), el.get("lon")
        coords: Any = None
        if el_type == "node" and lat is not None:
            coords = (lon, lat)
        elif el_type == "way":
            geometry = el.get("geometry") or []
            line = [(p.get("lon"), p.get("lat")) for p in geometry if p.get("lon") is not None]
            if len(line) >= 2:
                coords = line
        if coords is None:
            continue
        feats.append({
            "type": "Feature",
            "id": f"{el_type}/{el.get('id')}",
            "properties": {
                "osm_id": el.get("id"),
                "osm_type": el_type,
                **(el.get("tags") or {}),
            },
            "geometry": (
                {"type": "Point", "coordinates": coords}
                if el_type == "node"
                else {"type": "LineString", "coordinates": coords}
            ),
        })
    return {"type": "FeatureCollection", "features": feats}


def ohsome_count(
    label: str,
    filter_expr: str,
    bbox: tuple[float, float, float, float],
    snapshot_date: str,
) -> int | None:
    """One 'count' statistic against the ohsome aggregation API (works publicly).

    Returns the element count at snapshot_date, or None if the API is down.
    Used to cross-check the Overpass extract against the official OSH statistics.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    # ohsome v1 conventions verified live: aggregation endpoints take 'bboxes'
    # (not 'bbox') and GDAL-free free-form filters like building=* — the
    # JSON-array filter strings ohsome docs used before v1.10 get HTTP 400.
    params = {
        "bboxes": f"{min_lon},{min_lat},{max_lon},{max_lat}",
        "time": snapshot_date,
        "filter": filter_expr,
    }
    url = f"{OHSOME_API}/elements/count?{urllib.parse.urlencode(params)}"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": USER_AGENT})
        with urlopen(req, timeout=90) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        value = int(payload["result"][0]["value"])
        print(f"  ℹ️ ohsome official count [{label}]: {value:,}")
        return value
    except Exception as e:
        print(f"  ⚠ ohsome count unavailable for {label}: {e}")
        return None


# Kept for compatibility with anything that imported the old name.
def ohsome_query(endpoint: str, params: dict[str, Any], timeout: int = 60) -> dict[str, Any]:
    """Legacy helper — aggregation queries only (geometry was never possible here)."""
    url = f"{OHSOME_API}/{endpoint}"
    full_url = f"{url}?{urllib.parse.urlencode(params, doseq=True)}"
    print(f"  GET {full_url[:120]}{'...' if len(full_url) > 120 else ''}")
    try:
        req = urllib.request.Request(full_url, headers={"Accept": "application/json", "User-Agent": USER_AGENT})
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError) as e:
        return {"error": str(e)}


def fetch_osm_features(
    center_lat: float,
    center_lon: float,
    snapshot_date: str,
    bbox: tuple[float, float, float, float],
    output_dir: Path,
    timeout: int = 120,
    event_date: str | None = None,
) -> dict[str, Path]:
    """
    Fetch all required OSM feature types for the study area.

    Returns dict mapping feature type → path to saved GeoJSON file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    event_date = event_date or EVENT_DATE

    results: dict[str, Path] = {}
    counts: dict[str, int | None] = {}

    # Each category = one attic Overpass query (nodes+ways as of snapshot_date)
    # plus one official ohsome count cross-check. Categories are independent:
    # a failure in one writes an empty file and never aborts the rest.
    categories: list[tuple[str, str, str, Path]] = [
        # (label, overpass selector, ohsome filter, output file)
        ("buildings", "[building]", "building=*", output_dir / "buildings.geojson"),
        ("roads",
         '[highway~"motorway|trunk|primary|secondary|tertiary|unclassified|residential|living_street|service|track|path|footway|bridleway"]',
         "highway in (motorway,trunk,primary,secondary,tertiary,unclassified,residential,living_street,service,track,path,footway,bridleway)",
         output_dir / "roads.geojson"),
        ("bridges", "[bridge=yes]", "bridge=yes", output_dir / "bridges.geojson"),
        ("hospitals", '[amenity~"hospital|clinic|doctors"]', "amenity in (hospital,clinic,doctors)", output_dir / "hospitals.geojson"),
        ("settlements", '[place~"village|hamlet|town|city"]', "place in (village,hamlet,town,city)", output_dir / "settlements.geojson"),
    ]

    for label, selector, ohsome_filter, out_path in categories:
        print("\n" + "\u2500" * 50)
        print(f"📥 Fetching {label} (as of {snapshot_date})...")
        elements = run_attic_query(selector, bbox, snapshot_date, timeout=timeout)
        geojson = elements_to_geojson(elements)
        out_path.write_text(json.dumps(geojson, indent=2), encoding="utf-8")
        counts[label] = len(geojson["features"])
        results[label] = out_path
        print(f"  ✅ {counts[label]} {label} features saved → {out_path}")
        # Optional cross-check against the official ohsome aggregate (best effort).
        ohsome_count(label, ohsome_filter, bbox, snapshot_date)
        if not elements:
            print(f"  ⚠ 0 elements for {label} — check connectivity / retry later")

    # Cross-check aggregate: official ohsome count for ALL tagged elements.
    print("\n" + "─" * 50)
    print("📦 ohsome aggregate cross-check...")
    total_ohsome = ohsome_count("all tagged elements", "type=node or type=way", bbox, snapshot_date)

    # Write a combined metadata file
    meta = {
        "snapshot_date": snapshot_date,
        "event_date": event_date,
        "area_center": [center_lat, center_lon],
        "bbox": list(bbox),
        "feature_counts": counts,
        "ohsome_total_check": total_ohsome,
        "api": "Overpass attic (geometry) + ohsome v1 (aggregate counts)",
        "overpass_url": OVERPASS_API,
        "ohsome_url": OHSOME_API,
        "attribution": "© OpenStreetMap contributors (ODbL)",
        "note": (
            f"Pre-event OSM snapshot — data edits after {event_date} are NOT "
            "included"
        ),
    }
    meta_path = output_dir / "osm_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\n✅ OSM metadata → {meta_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Fetch pre-event OSM data via ohsome API")
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
    parser.add_argument("--snapshot-date", required=True, help="OSM snapshot date (YYYY-MM-DD)")
    parser.add_argument("--event-date", default=EVENT_DATE,
                        help="Flood event date (YYYY-MM-DD); the snapshot must precede it")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()

    try:
        bbox = resolve_bbox(args.bbox, args.center_lat, args.center_lon, args.extent)
    except ValueError as exc:
        print(f"❌ {exc}", file=sys.stderr)
        sys.exit(1)
    min_lon, min_lat, max_lon, max_lat = bbox
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2
    output_dir = Path(args.output)

    # Validate snapshot date is before the event
    snapshot_dt = datetime.strptime(args.snapshot_date, "%Y-%m-%d")
    event_dt = datetime.strptime(args.event_date, "%Y-%m-%d")
    if snapshot_dt >= event_dt:
        print(f"❌ Snapshot date {args.snapshot_date} is NOT before the event date ({args.event_date})!")
        print(f"   Using OSM data after the event as INPUT is prohibited and leads to disqualification.")
        sys.exit(1)

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║     🌍  OpenStreetMap Pre-Event Data Fetch                  ║
╠══════════════════════════════════════════════════════════════╣
║     Center:        ({center_lat:.4f}, {center_lon:.4f})          ║
║     Bounding box:  [{min_lon:.3f}, {min_lat:.3f},             ║
║                      {max_lon:.3f}, {max_lat:.3f}]             ║
║     Snapshot date: {args.snapshot_date}  (must be < {args.event_date})    ║
║     API:           {OHSOME_API}                 ║
╚══════════════════════════════════════════════════════════════╝
""")

    results = fetch_osm_features(
        center_lat=center_lat,
        center_lon=center_lon,
        snapshot_date=args.snapshot_date,
        bbox=(min_lon, min_lat, max_lon, max_lat),
        output_dir=output_dir,
        event_date=args.event_date,
    )

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║     ✅ OSM data saved to: {output_dir}              ║
╚══════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
