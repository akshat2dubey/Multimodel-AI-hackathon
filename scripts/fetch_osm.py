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
import hashlib
import http.client
import os
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

from config.presets import DEFAULT_EXTENT, EVENT_DATE, MAX_BBOX_DEG2  # noqa: E402
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
# Endpoints are tried in order, and a failure moves to the next one. Verified
# live 2026-10-13: overpass-api.de answers the building extract for this AOI
# with HTTP 504 while overpass.kumi.systems returns 3745 elements (3738 ways)
# for the SAME attic query, and ohsome's official count for `building=*` in the
# same bbox is 3755. A single endpoint is therefore not just a reliability
# risk, it is the difference between a real building layer and an empty one.
OVERPASS_ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
)
# Kept for compatibility with anything that imported the old name.
OVERPASS_API = OVERPASS_ENDPOINTS[0]
OHSOME_API = "https://api.ohsome.org/v1"          # used only for aggregate counts
OHSOME_API_V2 = "https://api.heigit.org/ohsome-api/v2"

# ohsome filter syntax is `type:node`, NOT `type=node`. The wrong form does not
# error — it returns 0 — which silently disabled this cross-check until it was
# verified live on 2026-10-13 (type=node or type=way → 0, type:node or
# type:way → 8124).
OHSOME_ALL_TAGGED_FILTER = "type:node or type:way"
# Identifies this tool to Overpass/ohsome, as the tile/API etiquette asks.
USER_AGENT = "Jalnetra/0.1 (Multimodal AI Hackathon 2026, Track B)"


def _endpoint_host(endpoint: str) -> str:
    """Just the host, for a log line that stays short."""
    return endpoint.split("//")[-1].split("/")[0]


def overpass_ql(
    query: str,
    timeout: int = 180,
    rounds: int = 1,
) -> tuple[dict[str, Any] | None, str | None]:
    """POST an Overpass QL query and return (parsed JSON, endpoint used).

    Every mirror is tried across `rounds` passes before giving up, and the
    failure case returns (None, None) — distinct from a successful query that
    found nothing. That distinction is the point: a busy server that answers
    HTTP 504 is not evidence that the study area has no buildings, and treating
    it as such is how this pipeline reported 0 affected buildings for a corridor
    that has 3755 of them.

    `rounds` defaults to 1 — one attempt per mirror — because the retry loop is
    wall-clock bounded only by the client timeout: with more rounds, a slow or
    rate-limited Overpass can hold the whole pipeline for tens of minutes before
    it admits failure. The caller stops asking after the first category that
    exhausts every endpoint (see fetch_osm_features).
    """
    import time
    data = urllib.parse.urlencode({"data": query}).encode("utf-8")
    endpoints = OVERPASS_ENDPOINTS
    total_attempts = max(1, rounds) * len(endpoints)
    for attempt in range(1, total_attempts + 1):
        endpoint = endpoints[(attempt - 1) % len(endpoints)]
        host = _endpoint_host(endpoint)
        try:
            req = urllib.request.Request(
                endpoint, data=data,
                headers={"Accept": "application/json", "User-Agent": USER_AGENT},
            )
            with urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8")
                ct = resp.headers.get("Content-Type", "")
                if "json" not in ct:
                    raise ValueError(f"server returned {ct} (likely rate-limit page)")
                return json.loads(body), endpoint
        except urllib.error.HTTPError as e:
            print(f"  ⚠ Overpass HTTP {e.code} from {host} (attempt {attempt}/{total_attempts})")
            if e.code == 429:
                print("  ℹ️ Rate limited — backing off")
        except (urllib.error.URLError, http.client.HTTPException, ValueError, json.JSONDecodeError, OSError) as e:
            print(f"  ⚠ Overpass error from {host} (attempt {attempt}/{total_attempts}): {e}")
        time.sleep(min(5 * attempt, 20))
    return None, None


def run_attic_query(
    selector: str,
    bbox: tuple[float, float, float, float],
    snapshot_date: str,
    timeout: int = 180,
) -> list[dict] | None:
    """Run one attic Overpass query.

    Returns the raw element list — possibly empty, which is a real answer ("the
    snapshot has none of these here") — or None when every endpoint failed, which
    is not an answer at all and must never be written out as an empty layer.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    q = (
        f'[out:json][timeout:{timeout}]'
        f'[date:"{snapshot_date}T00:00:00Z"];'
        f'(node{selector}({min_lat},{min_lon},{max_lat},{max_lon});'
        f'way{selector}({min_lat},{min_lon},{max_lat},{max_lon}););'
        f'out geom tags;'
    )
    payload, endpoint = overpass_ql(q, timeout=timeout + 30)
    if payload is None:
        return None
    print(f"  ℹ️ answered by {_endpoint_host(endpoint or OVERPASS_API)}")
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


def ohsome_v2_features(
    filter_expr: str,
    bbox: tuple[float, float, float, float],
    snapshot_date: str,
) -> dict[str, Any] | None:
    """Fetch a dated GeoParquet extract when the local ohsome key is available."""
    key = os.environ.get("OHSOME_API_KEY", "").strip()
    secret_file = _ROOT / ".secrets" / "ohsome.env"
    if not key and secret_file.is_file():
        for line in secret_file.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("OHSOME_API_KEY="):
                key = line.strip().split("=", 1)[1].strip()
                break
    if not key:
        return None

    body = json.dumps({"aoi": bbox, "time": snapshot_date, "filter": filter_expr}).encode()
    request = urllib.request.Request(
        f"{OHSOME_API_V2}/extraction/features.parquet",
        data=body,
        headers={"Authorization": key, "Content-Type": "application/json", "User-Agent": USER_AGENT},
    )
    try:
        with urlopen(request, timeout=240) as response:
            import geopandas as gpd
            from io import BytesIO

            frame = gpd.read_parquet(
                BytesIO(response.read()),
                to_pandas_kwargs={"maps_as_pydicts": "strict"},
            )
        features = []
        for index, row in frame.iterrows():
            geom = frame.geometry.loc[index]
            if geom is None or geom.is_empty:
                continue
            tags = row.get("tags") or {}
            osm_type, osm_id = row["osm_type"], int(row["osm_id"])
            features.append({
                "type": "Feature",
                "id": f"{osm_type}/{osm_id}",
                "properties": {"osm_id": osm_id, "osm_type": osm_type, **tags},
                "geometry": geom.__geo_interface__,
            })
        return {"type": "FeatureCollection", "features": features}
    except Exception as exc:
        if isinstance(exc, urllib.error.HTTPError) and exc.code == 403:
            print("  ⚠ ohsome rejected the API key (HTTP 403); check .secrets/ohsome.env")
        else:
            print(f"  ⚠ ohsome API v2 geometry extract failed: {type(exc).__name__}: {exc}")
        return None


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


# Layers the rest of the pipeline cannot be honest without: the damage assessment
# and the cut-off analysis describe buildings and roads. If one of these cannot
# be fetched, the run must stop rather than report "0 buildings affected", which
# is indistinguishable from a real count of zero.
REQUIRED_CATEGORIES = ("buildings", "roads")


def _feature_count(path: Path) -> int:
    """Features in an existing GeoJSON file, or 0 when it is missing/unreadable."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    return len(payload.get("features") or [])


def fetch_osm_features(
    center_lat: float,
    center_lon: float,
    snapshot_date: str,
    bbox: tuple[float, float, float, float],
    output_dir: Path,
    timeout: int = 120,
    event_date: str | None = None,
) -> tuple[dict[str, Path], list[str]]:
    """
    Fetch all required OSM feature types for the study area.

    Returns (paths by feature type, required categories that could not be
    fetched). A non-empty second element means the run must not continue.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    event_date = event_date or EVENT_DATE
    width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    if width <= 0 or height <= 0 or width * height > MAX_BBOX_DEG2:
        raise ValueError(f"OSM bbox must be positive and no larger than {MAX_BBOX_DEG2} deg²")

    results: dict[str, Path] = {}
    counts: dict[str, int | None] = {}
    statuses: dict[str, str] = {}
    ohsome_counts: dict[str, int | None] = {}
    failed_required: list[str] = []
    api_used = "Overpass attic (geometry) + ohsome v1 (aggregate counts)"

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

    server_unavailable: str | None = None

    for label, selector, ohsome_filter, out_path in categories:
        print("\n" + "\u2500" * 50)
        print(f"📥 Fetching {label} (as of {snapshot_date})...")

        cache_key = hashlib.sha256(json.dumps(
            {"bbox": bbox, "snapshot_date": snapshot_date, "filter": ohsome_filter},
            sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        cache_path = output_dir.parent / "cache" / "osm" / f"{cache_key}-{label}.geojson"
        cached_geojson = None
        if cache_path.is_file():
            try:
                candidate = json.loads(cache_path.read_text(encoding="utf-8"))
                if candidate.get("type") == "FeatureCollection" and isinstance(candidate.get("features"), list):
                    cached_geojson = candidate
                    api_used = "local OSM cache (AOI + snapshot keyed)"
            except (OSError, json.JSONDecodeError):
                pass

        skipped: str | None = None
        if cached_geojson is not None:
            geojson = cached_geojson
            elements = geojson["features"]
        elif server_unavailable:
            # Circuit breaker. Once a category has failed at every endpoint, the
            # server is down or rate-limiting us; sweeping the remaining
            # categories would multiply a multi-minute stall by their number,
            # and the run stops anyway because a required layer is missing. The
            # skip is recorded, not hidden.
            elements, skipped = None, f"not attempted ({server_unavailable})"
        else:
            geojson = ohsome_v2_features(ohsome_filter, bbox, snapshot_date)
            if geojson is not None:
                elements = geojson["features"]
                api_used = "ohsome API v2 extraction + ohsome v1 aggregate counts"
            else:
                elements = run_attic_query(selector, bbox, snapshot_date, timeout=timeout)
                geojson = elements_to_geojson(elements) if elements is not None else None
                if elements is None:
                    server_unavailable = "every Overpass endpoint failed"

        # Asked before anything is written: the official OSM statistic is what
        # tells an empty extract apart from a failed one.
        official = ohsome_count(label, ohsome_filter, bbox, snapshot_date)
        ohsome_counts[label] = official
        results[label] = out_path

        failure_reason = skipped
        if failure_reason is None:
            if elements is None:
                failure_reason = server_unavailable or "every Overpass endpoint failed"
            elif not elements and official:
                failure_reason = (
                    f"extract is empty but ohsome reports {official:,} in this bbox"
                )

        if failure_reason:
            # NEVER write an empty layer over a failed fetch. An all-zero
            # FeatureCollection is indistinguishable from a real "nothing here"
            # and downstream becomes "0 buildings affected" — a fabricated
            # measurement of the disaster, in the direction that hides people.
            previous = _feature_count(out_path) if out_path.is_file() else 0
            if previous > 0:
                counts[label] = previous
                statuses[label] = "failed_kept_previous"
                print(f"  ⚠ {label} fetch FAILED ({failure_reason}) — keeping the "
                      f"existing {out_path.name} with {previous} features rather "
                      "than overwriting it with nothing")
            else:
                counts[label] = None
                statuses[label] = "skipped" if skipped else "failed"
                print(f"  ❌ {label} fetch FAILED ({failure_reason}) — no layer written")
            if label in REQUIRED_CATEGORIES:
                failed_required.append(label)
            continue

        out_path.write_text(json.dumps(geojson, indent=2), encoding="utf-8")
        if cached_geojson is None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(geojson, indent=2), encoding="utf-8")
        counts[label] = len(geojson["features"])
        statuses[label] = "ok"
        print(f"  ✅ {counts[label]} {label} features saved → {out_path}")
        if not elements:
            print(f"  ⚠ 0 elements for {label} and ohsome agrees — the snapshot "
                  "has none in this bbox")

    # Cross-check aggregate: official ohsome count for ALL tagged elements.
    print("\n" + "─" * 50)
    print("📦 ohsome aggregate cross-check...")
    total_ohsome = ohsome_count("all tagged elements", OHSOME_ALL_TAGGED_FILTER, bbox, snapshot_date)

    # Write a combined metadata file
    meta = {
        "snapshot_date": snapshot_date,
        "event_date": event_date,
        "area_center": [center_lat, center_lon],
        "bbox": list(bbox),
        "feature_counts": counts,
        "fetch_status": statuses,
        "ohsome_counts": ohsome_counts,
        "required_categories": list(REQUIRED_CATEGORIES),
        "failed_required_categories": failed_required,
        "ohsome_total_check": total_ohsome,
        "api": api_used,
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

    return results, failed_required


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

    results, failed_required = fetch_osm_features(
        center_lat=center_lat,
        center_lon=center_lon,
        snapshot_date=args.snapshot_date,
        bbox=(min_lon, min_lat, max_lon, max_lat),
        output_dir=output_dir,
        event_date=args.event_date,
    )

    if failed_required:
        print(
            "\n❌ OSM fetch failed for required layer(s): "
            + ", ".join(failed_required),
            file=sys.stderr,
        )
        print(
            "   Refusing to continue: a damage assessment without these layers "
            "would report 0 affected features,\n   which reads as a real "
            "finding of no impact. See "
            f"{output_dir / 'osm_metadata.json'} for the per-layer status.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║     ✅ OSM data saved to: {output_dir}              ║
╚══════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
