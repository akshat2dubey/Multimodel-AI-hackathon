#!/usr/bin/env python3
"""
Jalnetra Dashboard — Flask Backend

Serves a web dashboard for visualizing flood mapping results:
  - Interactive map with flood overlay (Leaflet + MapLibre)
  - OSM buildings/roads/bridges with damage coloring
  - Cut-off settlement markers
  - Situation report viewer
  - Flood path (bonus)

Run:  cd dashboard && python app.py
Open: http://localhost:5000
"""

import hmac
import json
import math
import os
import subprocess
import sys
from pathlib import Path

# The dashboard is normally started as `cd dashboard && python app.py`, so the
# project root is NOT on sys.path. Add it before importing config.presets — the
# single source of truth for paths and case-study constants shared with
# scripts/run_pipeline.py.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS

from config.presets import (  # noqa: E402  (path set up above)
    AREA as CASE_STUDY_AREA,
    CENTER_LAT,
    CENTER_LON,
    DATA_DIR,
    DEFAULT_PRESET,
    EVENT_DATE,
    REPORT_DIR,
    RESULTS_DIR,
    is_demo_dir,
)
from config.run_meta import load_run_meta  # noqa: E402

app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)

# results/ is the one directory the whole pipeline writes to; importing it from
# config keeps the dashboard and run_pipeline.py from drifting apart.
#
# JALNETRA_RESULTS_DIR points the dashboard at a different directory (e.g.
# results_demo) without touching code — a relative value resolves against the
# project root, so `JALNETRA_RESULTS_DIR=results_demo` is the documented demo
# command. Either way the directory must be inside the project.
_env_results = os.environ.get("JALNETRA_RESULTS_DIR", "").strip()
if _env_results:
    RESULTS = Path(_env_results)
    if not RESULTS.is_absolute():
        RESULTS = (ROOT / RESULTS).resolve()
else:
    RESULTS = RESULTS_DIR
DATA = DATA_DIR

# POST /api/run-pipeline spawns a long download+compute job, so it must not be
# callable by anyone who can reach the port. Without a configured token the
# endpoint refuses every request (fail closed).
PIPELINE_TOKEN = os.environ.get("JALNETRA_PIPELINE_TOKEN", "")
PIPELINE_TOKEN_HEADER = "X-Pipeline-Token"


def json_safe(value):
    """Recursively strip values that are not legal JSON (NaN, inf, numpy scalars).

    json.dumps writes a bare NaN token by default, which browsers refuse to
    parse. One NaN anywhere in a payload would otherwise blank the whole
    dashboard, so sanitise on the way out.
    """
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float):
        return None if (math.isnan(value) or math.isinf(value)) else value
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    try:
        item = value.item()          # numpy scalar -> python scalar
        if isinstance(item, float) and (math.isnan(item) or math.isinf(item)):
            return None
        return item
    except Exception:
        return str(value)


def serving_demo_data() -> bool:
    """True when the directory being served is the synthetic fixture.

    Decided by the directory the operator pointed at (results_demo, or a
    directory carrying the demo notice), never by trusting a flag inside the
    data — a banner that depends on the data it is warning about is not a
    warning.
    """
    return is_demo_dir(RESULTS)


REPORT_FILENAME = "situation_report.html"


def situation_report_path() -> Path | None:
    """The report belonging to the results directory currently being served.

    A run writes its report next to the numbers it states: results/situation_
    report.html for a real run, results_demo/situation_report.html for the
    synthetic fixture. REPORT_DIR (report/) holds the published handout copy and
    is only a fallback — otherwise pointing the dashboard at results_demo would
    show a real report above synthetic layers, or a real run would inherit the
    demo's report.
    """
    for base in (RESULTS, REPORT_DIR):
        candidate = base / REPORT_FILENAME
        if candidate.is_file():
            return candidate
    return None


def load_geojson(path_name: str) -> dict | None:
    """Load a GeoJSON file from results or data directory."""
    for base in [RESULTS, DATA]:
        path = base / path_name
        if path.exists():
            try:
                return json_safe(json.loads(path.read_text(encoding="utf-8")))
            except Exception as e:
                print(f"  ⚠ Could not parse {path}: {e}")
    return None


def current_run_meta() -> dict:
    """The metadata of the run whose artifacts this dashboard is serving.

    ``<results>/run_meta.json`` is written by run_pipeline.py (or run_offline.py)
    and names the area, the bbox and the dates that run actually used. Serving
    those instead of the config defaults is what keeps the header and the map
    centre describing the same run as the numbers below them. Empty when no run
    has been executed into this directory yet.
    """
    return load_run_meta(RESULTS) or {}


def load_flood_mask_info() -> dict | None:
    """Get flood mask statistics from change_detection.json or compute from GeoTIFF."""
    info_path = RESULTS / "change_detection.json"
    if info_path.exists():
        return json.loads(info_path.read_text(encoding="utf-8"))

    # Try to compute basic stats
    try:
        import rasterio
        mask_path = RESULTS / "flood_mask.tif"
        if mask_path.exists():
            with rasterio.open(mask_path) as src:
                mask = src.read(1)
                return {
                    "flood_pixels": int((mask > 0.5).sum()),
                    "total_pixels": mask.size,
                    "flood_percentage": round((mask > 0.5).sum() / mask.size * 100, 2),
                    "shape": list(mask.shape),
                }
    except ImportError:
        pass
    except Exception:
        pass
    return None


@app.route("/")
def index():
    """Serve the main dashboard page."""
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    """API endpoint: overall pipeline status and summary stats."""
    flood_mask_info = load_flood_mask_info() or {}
    damage = load_geojson("damage_assessment.geojson") or {}
    cutoffs = load_geojson("cutoff_settlements.geojson") or {}

    damage_features = damage.get("features", [])
    buildings = [f for f in damage_features if f["properties"]["feature_type"] == "building"]
    roads = [f for f in damage_features if f["properties"]["feature_type"] == "road"]

    cutoff_features = cutoffs.get("cutoff_settlements", [])
    summary = cutoffs.get("summary", {})

    demo = serving_demo_data()
    run_meta = current_run_meta()
    return jsonify({
        "status": "complete" if flood_mask_info else "in_progress",
        # The served run's own metadata, falling back to the configured preset
        # before any run has written into this directory.
        "area": run_meta.get("area") or CASE_STUDY_AREA,
        "event_date": run_meta.get("event_date") or EVENT_DATE,
        "before_date": run_meta.get("before_date"),
        "after_date": run_meta.get("after_date"),
        "osm_snapshot_date": run_meta.get("osm_snapshot_date"),
        # The map centres on the run's own AOI, not on a coordinate pair
        # copy-pasted into the template.
        "center_lat": run_meta.get("center_lat", CENTER_LAT),
        "center_lon": run_meta.get("center_lon", CENTER_LON),
        "bbox": run_meta.get("bbox"),
        "aoi_confirmed": bool(run_meta.get("aoi_confirmed", False)),
        "bbox_source": run_meta.get("bbox_source"),
        # True whenever the current results came from the synthetic fixture, so
        # the UI can label them instead of presenting them as real measurements.
        "synthetic": bool(flood_mask_info.get("synthetic")) or demo,
        # Which directory is being served, and whether it is the demo fixture.
        # The dashboard banner is driven by this, so a demo run is never
        # presented as a real one.
        "results_dir": Path(RESULTS).name,
        "demo_data": demo,
        "data_source": flood_mask_info.get("data_source", ""),
        "flood_stats": flood_mask_info,
        "damage_summary": {
            "total_buildings": len(buildings),
            "affected_buildings": sum(
                1 for b in buildings
                if b["properties"]["damage_level"] in ("PARTIAL", "SEVERELY_DAMAGED", "DESTROYED")
            ),
            "destroyed_buildings": sum(
                1 for b in buildings if b["properties"]["damage_level"] == "DESTROYED"
            ),
            "total_roads": len(roads),
            # Feature-level: whole OSM ways whose majority is under water.
            "cut_roads": sum(
                1 for r in roads if r["properties"]["damage_level"] == "SEGMENT_CUT"
            ),
            # Network-level: individual road-graph edges severed by flooding.
            # This is the number that explains the cut-off settlements, and it is
            # usually much larger than the feature-level count above.
            "flooded_road_segments": summary.get("flooded_road_segments", 0),
            "affected_roads": sum(
                1 for r in roads
                if r["properties"]["damage_level"] in ("SEGMENT_CUT", "SEGMENT_AFFECTED")
            ),
            "total_bridges": sum(
                1 for f in damage_features if f["properties"]["feature_type"] == "bridge"
            ),
            "damaged_bridges": sum(
                1 for f in damage_features
                if f["properties"]["feature_type"] == "bridge"
                and f["properties"]["damage_level"] == "BRIDGE_DAMAGED"
            ),
        },
        "cutoff_summary": {
            # cutoff_count counts structures; village_count counts distinct places.
            "cutoff_count": len(cutoff_features),
            "village_count": len(_group_cutoffs_by_village(cutoff_features)),
            "total_settlements": summary.get("total_settlements", 0),
            "cutoff_percentage": summary.get("cutoff_percentage", 0),
            "flooded_road_segments": summary.get("flooded_road_segments", 0),
        },
    })


def _vectorize_flood_mask(mask_path: Path, cache_path: Path) -> dict:
    """Vectorise the flood raster once, then reuse the cached GeoJSON.

    Vectorising 400k pixels takes ~5 s and yields ~7 MB of coordinates, which
    makes the map unusable on every page load. The result only changes when the
    mask changes, so cache it and simplify contours to roughly one pixel.
    """
    import rasterio
    from rasterio.features import shapes
    from shapely.geometry import shape as shapely_shape, mapping

    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        transform = src.transform
        # Binarise first: shapes() groups polygons by pixel VALUE, so feeding it a
        # soft probability raster splits the outline into thousands of tiny
        # polygons. Grouping on 0/1 gives one clean contour per flood body.
        binary = (mask > 0.5).astype("uint8")
        mean_prob = float(mask[binary > 0].mean()) if binary.any() else 0.0
        # one pixel of tolerance keeps the outline faithful but drops most vertices
        tolerance = abs(transform.a) * 0.75
        feats = []
        for geom, value in shapes(binary, transform=transform):
            if int(value) != 1:
                continue
            try:
                simplified = shapely_shape(geom).simplify(tolerance, preserve_topology=True)
            except Exception:
                simplified = shapely_shape(geom)
            coords = mapping(simplified)
            if not coords.get("coordinates"):
                continue
            feats.append({
                "type": "Feature",
                "properties": {
                    "flood": True,
                    "probability": round(mean_prob, 3),
                    "source": "flood_mask.tif (threshold 0.5)",
                },
                "geometry": coords,
            })

    result = {"type": "FeatureCollection", "features": feats}
    try:
        cache_path.write_text(json.dumps(result), encoding="utf-8")
    except Exception as e:
        print(f"  could not cache flood mask vectors: {e}")
    return result


@app.route("/api/flood-mask")
def api_flood_mask():
    """Serve flood mask as GeoJSON for display on map."""
    flood_path = RESULTS / "flood_mask.tif"
    cache_path = RESULTS / "flood_mask.geojson"

    # Prefer a cache that is newer than the raster it came from.
    if cache_path.exists():
        if not flood_path.exists() or cache_path.stat().st_mtime >= flood_path.stat().st_mtime:
            try:
                return jsonify(json.loads(cache_path.read_text(encoding="utf-8")))
            except Exception as e:
                print(f"  stale flood-mask cache, regenerating: {e}")

    result = {"type": "FeatureCollection", "features": []}

    if flood_path.exists():
        try:
            result = _vectorize_flood_mask(flood_path, cache_path)
        except ImportError:
            print("  rasterio not installed - flood mask available as GeoTIFF only")
        except Exception as e:
            print(f"  Could not vectorize flood mask: {e}")

    # If no vector data, return an explicit empty collection rather than a fake
    # bounding box, so an empty map is never mistaken for a real flood extent.
    if not result["features"]:
        return jsonify({
            "type": "FeatureCollection",
            "features": [],
            "note": "No flood mask available yet — run the pipeline first.",
        })

    return jsonify(result)


@app.route("/api/damage")
def api_damage():
    """Serve damage assessment data."""
    return jsonify(load_geojson("damage_assessment.geojson") or {"features": []})


def _group_cutoffs_by_village(cutoff_list: list) -> list:
    """Collapse per-building cut-off records into one row per village.

    The analysis walks building footprints, so answer #3 ("who is cut off?") gets
    more useful when the dashboard names villages rather than listing 68 separate
    structures.
    """
    villages: dict = {}
    for c in cutoff_list:
        key = c.get("settlement_name") or c.get("settlement_id") or "Unknown"
        entry = villages.setdefault(key, {
            "name": key,
            "structures_cut_off": 0,
            "nearest_reachable_town_name": None,
            "min_distance_km": None,
            "coordinates": c.get("coordinates"),
        })
        entry["structures_cut_off"] += 1
        dist = c.get("nearest_reachable_distance_km")
        if dist is not None and (entry["min_distance_km"] is None or dist < entry["min_distance_km"]):
            entry["min_distance_km"] = dist
            entry["nearest_reachable_town_name"] = (
                c.get("nearest_reachable_town_name")
                or c.get("nearest_reachable_town"))
            entry["coordinates"] = c.get("coordinates")
    return sorted(villages.values(),
                  key=lambda v: (-v["structures_cut_off"], v["name"]))


@app.route("/api/cutoffs")
def api_cutoffs():
    """Serve cut-off settlement data."""
    cutoffs = load_geojson("cutoff_settlements.geojson") or {}
    cutoff_list = cutoffs.get("cutoff_settlements", [])
    villages = _group_cutoffs_by_village(cutoff_list)
    return jsonify({
        "cutoff_settlements": cutoff_list,
        "cutoff_villages": villages,
        "village_count": len(villages),
        "summary": cutoffs.get("summary", {}),
        "flooded_roads": cutoffs.get("flooded_roads", []),
        "note": cutoffs.get("note"),
    })


@app.route("/api/flood-path")
def api_flood_path():
    """Serve flood path (bonus feature) data."""
    return jsonify(load_geojson("flood_path.geojson") or {"features": []})


@app.route("/api/situation-report")
def api_situation_report():
    """Serve the situation report HTML.

    An absent report is an empty state, not an error: the dashboard must render
    gracefully before the pipeline has ever run. Returns 200 with an explicit
    `available: false` flag so the UI can show a "not generated yet" message
    without treating a 404 as a crash.
    """
    report_path = situation_report_path()
    if report_path is not None:
        return send_from_directory(report_path.parent, REPORT_FILENAME)
    return jsonify({
        "available": False,
        "type": "situation-report",
        "html": "",
        "note": "Situation report not generated yet — run the pipeline first.",
    })


@app.route("/report")
def report_page():
    """Render the situation report in a browser-friendly wrapper."""
    report_path = situation_report_path()
    if report_path is not None:
        # Read the report and embed it
        report_html = report_path.read_text(encoding="utf-8")
        # Extract the body content
        body_start = report_html.find("<div class=\"page\">")
        body_end = report_html.find("</div>\n</body>")
        if body_start > 0 and body_end > body_start:
            body_content = report_html[body_start:body_end + 6]
        else:
            body_content = report_html
        return render_template("report_wrapper.html", report_body=body_content)
    return render_template("report_placeholder.html")


@app.route("/demo")
def demo_mode():
    """Run in demo mode with sample data (no real satellite data needed)."""
    return render_template("demo.html")


@app.route("/api/run-pipeline", methods=["POST"])
def api_run_pipeline():
    """Trigger the full pipeline (for demo/UI trigger).

    Runs the same case-study preset the CLI documents (``--preset``), so the
    dashboard cannot drift from the pipeline: the area, bbox and dates are
    resolved by run_pipeline.py into results/run_meta.json, not restated here.

    Gated by a shared token (header ``X-Pipeline-Token``) so a public port cannot
    be made to start a multi-minute download job. Fails closed: with no
    JALNETRA_PIPELINE_TOKEN configured, every request is rejected.
    """
    supplied = request.headers.get(PIPELINE_TOKEN_HEADER, "")
    if not PIPELINE_TOKEN or not hmac.compare_digest(supplied, PIPELINE_TOKEN):
        return jsonify({
            "success": False,
            "error": "Forbidden: missing or invalid pipeline token.",
        }), 403

    try:
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "run_pipeline.py"),
             "--preset", DEFAULT_PRESET],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )
        return jsonify({
            "success": result.returncode == 0,
            "stdout": result.stdout[-2000:],
            "stderr": result.stderr[-1000:],
        })
    except subprocess.TimeoutExpired:
        return jsonify({"success": False, "error": "Pipeline timed out (>5 min)"}), 504
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    if port == 0:
        # Windows ships with PORT=0 in some environments, which makes Flask bind
        # a random port and the printed URL useless. Fall back to 5000.
        port = 5000
    # Debug (and with it the interactive Werkzeug debugger and the auto-reloader)
    # is OFF unless explicitly asked for: it must never be on for a demo a judge
    # can reach. FLASK_DEBUG=1 opts in.
    debug = os.environ.get("FLASK_DEBUG", "0").lower() in ("1", "true", "yes", "on")
    # Bind loopback by default. Set HOST=0.0.0.0 explicitly for a LAN demo.
    host = os.environ.get("HOST", "127.0.0.1")
    print(f'\
Jalnetra Dashboard — Open: http://{host}:{port}  (debug={debug})\
Features: flood map, OSM damage, cut-offs, report, flood path\
')
    app.run(host=host, port=port, debug=debug)
