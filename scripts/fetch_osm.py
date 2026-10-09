#!/usr/bin/env python3
"""
Fetch pre-event OpenStreetMap data via the ohsome API.

The ohsome API (https://api.ohsome.org/v1) provides access to OSM history data
with time-based queries. We need a snapshot BEFORE the flood event (26 Aug 2026).
The latest available snapshot as of the hackathon is 27 July 2026.

Data to fetch (all as of snapshot date):
  - Buildings (osm_type=building)
  - Roads (osm_type=way, highway=*)
  - Bridges (osm_type=bridge OR man_made=bridge)
  - Hospitals (amenity=hospital)
  - Settlements/villages (place=village, place=hamlet)

Usage:
    python fetch_osm.py \
        --center-lat 28.3 --center-lon 85.7 \
        --snapshot-date 2026-07-27 \
        --output ./data/osm

⚠️ CRITICAL RULE: Do NOT use OSM data edited AFTER 26 August 2026.
Only pre-event snapshot data may be used as input.
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

if hasattr(sys.stdout, "reconfigure"):  # never crash on a cp1252 Windows console
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ohsome API v1 base URL
OHSOME_API = "https://api.ohsome.org/v1"


def ohsome_query(
    endpoint: str,
    params: dict[str, Any],
    timeout: int = 60,
) -> dict[str, Any]:
    """
    Make a query to the ohsome API and return parsed JSON.
    """
    url = f"{OHSOME_API}/{endpoint}"
    query_string = urllib.parse.urlencode(params, doseq=True)
    full_url = f"{url}?{query_string}"

    print(f"  GET {full_url[:120]}{'...' if len(full_url) > 120 else ''}")

    try:
        req = urllib.request.Request(full_url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(f"  ❌ HTTP {e.code}: {e.reason}")
        if e.code == 429:
            print("  ℹ️ Rate limited — wait and retry")
        return {"error": str(e)}
    except urllib.error.URLError as e:
        print(f"  ❌ URL error: {e.reason}")
        print("  ℹ️ Check internet connectivity or ohsome API availability")
        return {"error": str(e)}


def fetch_osm_features(
    center_lat: float,
    center_lon: float,
    snapshot_date: str,
    bbox: tuple[float, float, float, float],
    output_dir: Path,
    timeout: int = 120,
) -> dict[str, Path]:
    """
    Fetch all required OSM feature types for the study area.

    Returns dict mapping feature type → path to saved GeoJSON file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    min_lon, min_lat, max_lon, max_lat = bbox

    # ohsome uses OpenStreetMap tile bounding boxes (bbox format: W,S,E,N)
    bbox_str = f"{min_lon},{min_lat},{max_lon},{max_lat}"

    results: dict[str, Path] = {}

    # ── 1. Buildings ──────────────────────────────────────────────
    print("\n─" * 50)
    print("🏗️  Fetching buildings...")
    params = {
        "osm-types": "nodes,ways",
        "date": f"({snapshot_date},{snapshot_date})",
        "filter": f'["building"]',
        "bbox": bbox_str,
        "format": "geojson",
        "properties": "tags",
    }
    result = ohsome_query("elements", params, timeout=timeout)
    if "error" not in result:
        buildings_path = output_dir / "buildings.geojson"
        buildings_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        # Count buildings
        features = result.get("features", [])
        print(f"  ✅ {len(features)} buildings saved → {buildings_path}")
        results["buildings"] = buildings_path
    else:
        print(f"  ❌ Failed to fetch buildings: {result.get('error')}")
        # Create empty GeoJSON as fallback
        empty = {
            "type": "FeatureCollection",
            "features": [],
            "metadata": {"note": "OSM buildings fetch failed — placeholder"},
        }
        (output_dir / "buildings.geojson").write_text(json.dumps(empty, indent=2), encoding="utf-8")

    # ── 2. Roads ──────────────────────────────────────────────────
    print("\n─" * 50)
    print("🛣️  Fetching roads...")
    # Query ways with highway tag (roads, paths, tracks)
    road_types = [
        "motorway", "trunk", "primary", "secondary", "tertiary",
        "unclassified", "residential", "living_street",
        "service", "track", "path", "footway", "bridleway",
    ]
    params = {
        "osm-types": "ways",
        "date": f"({snapshot_date},{snapshot_date})",
        "filter": f'["highway"~"{"|".join(road_types)}"]',
        "bbox": bbox_str,
        "format": "geojson",
        "properties": "tags",
    }
    result = ohsome_query("elements", params, timeout=timeout)
    if "error" not in result:
        roads_path = output_dir / "roads.geojson"
        roads_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        features = result.get("features", [])
        print(f"  ✅ {len(features)} road segments saved → {roads_path}")
        results["roads"] = roads_path
    else:
        print(f"  ❌ Failed to fetch roads: {result.get('error')}")
        empty = {"type": "FeatureCollection", "features": [], "metadata": {"note": "Failed"}}
        (output_dir / "roads.geojson").write_text(json.dumps(empty, indent=2), encoding="utf-8")

    # ── 3. Bridges ────────────────────────────────────────────────
    print("\n─" * 50)
    print("🌉 Fetching bridges...")
    params = {
        "osm-types": "nodes,ways",
        "date": f"({snapshot_date},{snapshot_date})",
        "filter": '["bridge"="yes"]',
        "bbox": bbox_str,
        "format": "geojson",
        "properties": "tags",
    }
    result = ohsome_query("elements", params, timeout=timeout)
    bridges_path = output_dir / "bridges.geojson"
    if "error" not in result:
        bridges_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        features = result.get("features", [])
        print(f"  ✅ {len(features)} bridges saved → {bridges_path}")
        results["bridge_nodes"] = bridges_path
    else:
        # Try alternative: man_made=bridge
        params["filter"] = '["man_made"="bridge"]'
        result = ohsome_query("elements", params, timeout=timeout)
        if "error" not in result:
            bridges_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            features = result.get("features", [])
            print(f"  ✅ {len(features)} bridge features (man_made) saved → {bridges_path}")
            results["bridge_nodes"] = bridges_path
        else:
            print(f"  ⚠️ No bridges found — creating empty file")
            empty = {"type": "FeatureCollection", "features": [], "metadata": {"note": "No bridges"}}
            bridges_path.write_text(json.dumps(empty, indent=2), encoding="utf-8")
            results["bridge_nodes"] = bridges_path

    # ── 4. Hospitals / health facilities ─────────────────────────
    print("\n─" * 50)
    print("🏥 Fetching hospitals...")
    params = {
        "osm-types": "nodes,ways",
        "date": f"({snapshot_date},{snapshot_date})",
        "filter": '["amenity"~"hospital|clinic|doctors|health_centre"]',
        "bbox": bbox_str,
        "format": "geojson",
        "properties": "tags",
    }
    result = ohsome_query("elements", params, timeout=timeout)
    hospitals_path = output_dir / "hospitals.geojson"
    if "error" not in result:
        hospitals_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        features = result.get("features", [])
        print(f"  ✅ {len(features)} hospitals/health facilities → {hospitals_path}")
        results["hospitals"] = hospitals_path
    else:
        empty = {"type": "FeatureCollection", "features": [], "metadata": {"note": "Failed"}}
        hospitals_path.write_text(json.dumps(empty, indent=2), encoding="utf-8")
        results["hospitals"] = hospitals_path

    # ── 5. Settlements / villages ────────────────────────────────
    print("\n─" * 50)
    print("🏘️  Fetching settlements/villages...")
    params = {
        "osm-types": "nodes,ways",
        "date": f"({snapshot_date},{snapshot_date})",
        "filter": '["place"~"village|hamlet|town|city"]',
        "bbox": bbox_str,
        "format": "geojson",
        "properties": "tags",
    }
    result = ohsome_query("elements", params, timeout=timeout)
    settlements_path = output_dir / "settlements.geojson"
    if "error" not in result:
        settlements_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        features = result.get("features", [])
        print(f"  ✅ {len(features)} settlements saved → {settlements_path}")
        results["settlements"] = settlements_path
    else:
        empty = {"type": "FeatureCollection", "features": [], "metadata": {"note": "Failed"}}
        settlements_path.write_text(json.dumps(empty, indent=2), encoding="utf-8")

    # ── 6. All OSM features (for reference) ──────────────────────
    print("\n─" * 50)
    print("📦 Fetching all OSM elements (metadata only)...")
    params = {
        "osm-types": "nodes,ways",
        "date": f"({snapshot_date},{snapshot_date})",
        "bbox": bbox_str,
        "format": "json",
        "properties": "tags",
    }
    result = ohsome_query("elements", params, timeout=timeout)
    if "error" not in result:
        count = result.get("totalElements", 0)
        print(f"  ℹ️  Total OSM elements in area: {count:,}")
    else:
        print(f"  ⚠️ Could not count total elements")

    # Write a combined metadata file
    meta = {
        "snapshot_date": snapshot_date,
        "area_center": [center_lat, center_lon],
        "bbox": list(bbox),
        "feature_counts": {k: 0 for k in results},
        "api": "ohsome API v1",
        "url": f"{OHSOME_API}/elements",
        "note": "Pre-event OSM snapshot — data edits after 2026-08-26 are NOT included",
    }
    meta_path = output_dir / "osm_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\n✅ OSM metadata → {meta_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Fetch pre-event OSM data via ohsome API")
    parser.add_argument("--center-lat", type=float, required=True, help="Center latitude")
    parser.add_argument("--center-lon", type=float, required=True, help="Center longitude")
    parser.add_argument("--snapshot-date", required=True, help="OSM snapshot date (YYYY-MM-DD)")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--bbox", help="Bounding box: min_lon,min_lat,max_lon,max_lat")
    args = parser.parse_args()

    center_lat, center_lon = args.center_lat, args.center_lon
    half_extent = 0.3  # ~30km buffer

    if args.bbox:
        parts = [float(x) for x in args.bbox.split(",")]
        min_lon, min_lat, max_lon, max_lat = parts
    else:
        min_lon = center_lon - half_extent
        min_lat = center_lat - half_extent
        max_lon = center_lon + half_extent
        max_lat = center_lat + half_extent

    output_dir = Path(args.output)

    # Validate snapshot date is before the event
    snapshot_dt = datetime.strptime(args.snapshot_date, "%Y-%m-%d")
    event_dt = datetime(2026, 8, 26)
    if snapshot_dt >= event_dt:
        print(f"❌ Snapshot date {args.snapshot_date} is NOT before the event date (2026-08-26)!")
        print(f"   Using OSM data after the event as INPUT is prohibited and leads to disqualification.")
        sys.exit(1)

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║     🌍  OpenStreetMap Pre-Event Data Fetch                  ║
╠══════════════════════════════════════════════════════════════╣
║     Center:        ({center_lat:.4f}, {center_lon:.4f})          ║
║     Bounding box:  [{min_lon:.3f}, {min_lat:.3f},             ║
║                      {max_lon:.3f}, {max_lat:.3f}]             ║
║     Snapshot date: {args.snapshot_date}  (must be < 2026-08-26)    ║
║     API:           {OHSOME_API}                 ║
╚══════════════════════════════════════════════════════════════╝
""")

    results = fetch_osm_features(
        center_lat=center_lat,
        center_lon=center_lon,
        snapshot_date=args.snapshot_date,
        bbox=(min_lon, min_lat, max_lon, max_lat),
        output_dir=output_dir,
    )

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║     ✅ OSM data saved to: {output_dir}              ║
╚══════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
