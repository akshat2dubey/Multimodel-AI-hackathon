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
import re
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

# The dashboard is normally started as `cd dashboard && python app.py`, so the
# project root is NOT on sys.path. Add it before importing config.presets — the
# single source of truth for paths and case-study constants shared with
# scripts/run_pipeline.py.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flask import Flask, jsonify, render_template, request, send_from_directory

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
# No CORS: the dashboard page and its API are served from the same origin, so the
# browser never needs cross-origin permission. `CORS(app)` reflected any Origin
# back as Access-Control-Allow-Origin, which let any page you happened to visit
# read every result while the dashboard was running.

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

# State-changing routes spawn a long download+compute job, so they must not be
# callable by anyone who can reach the port. Without a configured token these
# endpoints refuse every request (fail closed).
PIPELINE_TOKEN = os.environ.get("JALNETRA_PIPELINE_TOKEN", "")
PIPELINE_TOKEN_HEADER = "X-Pipeline-Token"


# ─── Background pipeline job ────────────────────────────────────────────────
# POST /api/runs starts scripts/run_pipeline.py as a BACKGROUND process and
# returns 202 immediately: a real run downloads imagery and can take minutes, so
# blocking the request would freeze the page a judge is watching. The UI polls
# GET /api/run-status instead.
#
# Only one job may run at a time — two runs would fight over the same results/
# directory. The state is mirrored into results/run_status.json so the status
# survives a page reload and a crash leaves evidence behind.
RUN_STATUS_FILENAME = "run_status.json"
MAX_ARG_STRING = 80
LOG_TAIL_LINES = 40
# The orchestrator's steps, in order. Used only to turn a log line into a step
# label and a rough progress fraction — it is not a scheduler.
PIPELINE_STEPS = [
    "download_sentinel.py",
    "download_sentinel2.py",
    "fetch_osm.py",
    "download_dem.py",
    "inference.py",
    "damage_assessment.py",
    "cutoff_analysis.py",
    "flood_path.py",
    "generate_report.py",
]
STEP_LABELS = {
    "download_sentinel.py": "downloading Sentinel-1 radar",
    "download_sentinel2.py": "downloading Sentinel-2 optical",
    "fetch_osm.py": "fetching the pre-event OSM snapshot",
    "download_dem.py": "downloading the Copernicus DEM",
    "inference.py": "computing the flood mask",
    "damage_assessment.py": "assessing damage to roads and buildings",
    "cutoff_analysis.py": "finding cut-off settlements",
    "flood_path.py": "tracing the flood path",
    "generate_report.py": "generating the situation report",
}

_job_lock = threading.Lock()
_active_job: dict | None = None          # JSON-safe description of the run
_active_proc: "subprocess.Popen | None" = None   # the process itself


# Credential shapes that must never reach a log tail. The tail is rendered in the
# browser, so a signed URL or an Authorization header printed by a download step
# would otherwise be published to whoever is looking at the dashboard.
_SECRET_PATTERNS = (
    re.compile(
        r"(?i)\b(client_secret|access_token|authorization|password|api[_-]?key|"
        r"token)\b(\s*[:=]?\s*)(\"?)([^\s\"',;]+)"
    ),
    re.compile(r"\b[A-Za-z0-9_\-]{32,}\b"),
)


def redact(text: str) -> str:
    """Mask credentials before a line is stored or served."""
    out = str(text)
    out = _SECRET_PATTERNS[0].sub(
        lambda m: f"{m.group(1)}{m.group(2)}{m.group(3)}<redacted>", out
    )
    out = _SECRET_PATTERNS[1].sub("<redacted>", out)
    return out


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _status_path() -> Path:
    return RESULTS / RUN_STATUS_FILENAME


def _write_status(status: dict) -> None:
    """Write the status file atomically so a reader never sees a half-written JSON."""
    try:
        RESULTS.mkdir(parents=True, exist_ok=True)
        path = _status_path()
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(status, indent=2), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        pass  # a status file that cannot be written must not kill the run


def _read_status_file() -> dict:
    path = _status_path()
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    return {"state": "idle", "run_id": None, "step": None, "progress": 0.0,
            "log_tail": [], "started_utc": None}


def _step_from_line(line: str) -> str | None:
    for script in PIPELINE_STEPS:
        if script in line:
            return script
    return None


def _parse_run_request(payload: dict) -> dict:
    """Validate a run request into run_pipeline.py keyword arguments.

    Returns the kwargs dict, or ``(message, field)`` on a bad value. This is a
    deliberately small guard — T17 replaces it with the full allowlist. It exists
    here so no unvalidated value can reach argv or the filesystem.
    """
    def _num(name, low, high):
        if name not in payload or payload[name] in (None, ""):
            return None
        try:
            value = float(payload[name])
        except (TypeError, ValueError):
            return (f"{name} must be a number", name)
        if not (low <= value <= high):
            return (f"{name} must be between {low} and {high}", name)
        return value

    def _date(name):
        if name not in payload or payload[name] in (None, ""):
            return None
        value = str(payload[name]).strip()
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            return (f"{name} must be an ISO date (YYYY-MM-DD)", name)
        return value

    def _text(name, limit=MAX_ARG_STRING):
        if name not in payload or payload[name] in (None, ""):
            return None
        value = str(payload[name]).strip()
        if len(value) > limit:
            return (f"{name} must be at most {limit} characters", name)
        return value

    bbox = payload.get("bbox")
    if isinstance(bbox, str):
        bbox = [part for part in re.split(r"[,\s]+", bbox.strip()) if part]
    if bbox not in (None, "", []):
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return ("bbox must be four numbers: min_lon,min_lat,max_lon,max_lat", "bbox")
        try:
            bbox = [float(v) for v in bbox]
        except (TypeError, ValueError):
            return ("bbox values must be numbers", "bbox")
        west, south, east, north = bbox
        if not (-180 <= west <= 180 and -180 <= east <= 180):
            return ("bbox longitudes must be between -180 and 180", "bbox")
        if not (-90 <= south <= 90 and -90 <= north <= 90):
            return ("bbox latitudes must be between -90 and 90", "bbox")
        if west >= east or south >= north:
            return ("bbox must be min_lon,min_lat,max_lon,max_lat with min<max", "bbox")
    else:
        bbox = None

    params = {"bbox": bbox}
    for name, low, high in (("center_lat", -90, 90), ("center_lon", -180, 180),
                            ("extent", 0.0001, 90)):
        value = _num(name, low, high)
        if isinstance(value, tuple):
            return value
        params[name] = value
    for name in ("event_date", "osm_snapshot", "before_date", "after_date"):
        value = _date(name)
        if isinstance(value, tuple):
            return value
        params[name] = value
    for name in ("area", "preset"):
        value = _text(name)
        if isinstance(value, tuple):
            return value
        params[name] = value

    # Nothing supplied: run the documented case-study preset so the one-click
    # button still works. A judge-entered bbox always takes precedence.
    if not any(params.values()):
        params["preset"] = DEFAULT_PRESET
    return params


def _pipeline_argv(params: dict) -> list[str]:
    """Build the orchestrator's argv.

    An argv LIST, never a shell string: a value typed into the browser must not
    be able to become a command.
    """
    argv = [sys.executable, str(ROOT / "scripts" / "run_pipeline.py")]
    if params.get("preset"):
        argv += ["--preset", params["preset"]]
    if params.get("area"):
        argv += ["--area", params["area"]]
    if params.get("bbox"):
        argv += ["--bbox", ",".join(f"{v:.6f}" for v in params["bbox"])]
    if params.get("center_lat") is not None:
        argv += ["--center-lat", str(params["center_lat"])]
    if params.get("center_lon") is not None:
        argv += ["--center-lon", str(params["center_lon"])]
    if params.get("extent") is not None:
        argv += ["--extent", str(params["extent"])]
    for flag, key in (("--before-date", "before_date"), ("--after-date", "after_date"),
                      ("--event-date", "event_date"), ("--osm-snapshot", "osm_snapshot")):
        if params.get(key):
            argv += [flag, params[key]]
    return argv


def _terminate(proc) -> None:
    """Stop a running job, including the child processes it spawned.

    run_pipeline.py is a thin orchestrator: killing only it would leave a
    download running. On POSIX the job gets its own process group; on Windows it
    gets CREATE_NEW_PROCESS_GROUP and is killed with taskkill /T.
    """
    if proc is None or proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True, check=False)
        else:
            os.killpg(os.getpgid(proc.pid), 15)
    except Exception:  # noqa: BLE001 — cancellation must never raise at the caller
        try:
            proc.terminate()
        except Exception:  # noqa: BLE001
            pass


def _launch_job(argv: list[str], params: dict) -> dict:
    """Start the pipeline in the background and return its JSON-safe status."""
    run_id = str(uuid.uuid4())
    job = {
        "run_id": run_id,
        "state": "running",
        "step": "starting",
        "progress": 0.0,
        "started_utc": _utc_now(),
        "finished_utc": None,
        "exit_code": None,
        "params": {k: v for k, v in params.items() if v not in (None, [], "")},
        "log_tail": [],
    }
    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
    proc = subprocess.Popen(
        argv,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=creationflags,
    )
    global _active_proc
    _active_proc = proc
    _write_status(job)

    def _reader() -> None:
        seen: list[str] = []
        try:
            for raw in proc.stdout:  # type: ignore[union-attr]
                line = redact(str(raw).rstrip("\n"))
                job["log_tail"].append(line)
                del job["log_tail"][:-LOG_TAIL_LINES]
                script = _step_from_line(line)
                if script and script not in seen:
                    seen.append(script)
                    job["step"] = STEP_LABELS.get(script, script)
                    job["progress"] = round(min(0.95, len(seen) / len(PIPELINE_STEPS)), 3)
                with _job_lock:
                    _write_status(job)
        finally:
            code = proc.wait()
            global _active_job, _active_proc
            with _job_lock:
                job["exit_code"] = code
                job["finished_utc"] = _utc_now()
                if job["state"] != "cancelled":
                    if code == 0:
                        job["state"] = "done"
                        job["step"] = "finished"
                        job["progress"] = 1.0
                    else:
                        job["state"] = "failed"
                        job["step"] = "failed"
                        job["error"] = f"the pipeline exited with code {code}"
                _write_status(job)
                _active_job = None
                _active_proc = None

    threading.Thread(target=_reader, name=f"jalnetra-run-{run_id[:8]}", daemon=True).start()
    return job


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


def _token_ok() -> bool:
    """State-changing routes fail closed: no configured token means no access."""
    supplied = request.headers.get(PIPELINE_TOKEN_HEADER, "")
    if not PIPELINE_TOKEN:
        return False
    return hmac.compare_digest(supplied, PIPELINE_TOKEN)


@app.route("/api/runs", methods=["POST"])
@app.route("/api/run-pipeline", methods=["POST"])  # legacy name, same behaviour
def api_start_run():
    """Start the pipeline as a background job and return 202 with a run_id.

    Runs the same orchestrator the CLI documents, so the dashboard cannot drift
    from the pipeline: the area, bbox and dates are resolved by run_pipeline.py
    into results/run_meta.json, not restated here.

    Token-gated and single-job: a public port must not be able to start (or
    stack) multi-minute download jobs. The request returns immediately and the
    UI polls /api/run-status, so the page never freezes during a live demo.
    """
    if not _token_ok():
        return jsonify({
            "success": False,
            "error": "Forbidden: missing or invalid pipeline token.",
        }), 403

    global _active_job
    with _job_lock:
        if _active_job is not None and _active_job.get("state") == "running":
            return jsonify({
                "success": False,
                "error": "A pipeline run is already in progress.",
                "run_id": _active_job.get("run_id"),
            }), 409

        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            return jsonify({"success": False, "error": "invalid input"}), 400

        params = _parse_run_request(payload)
        if isinstance(params, tuple):
            message, field = params
            return jsonify({"success": False, "error": message, "field": field}), 400

        _active_job = _launch_job(_pipeline_argv(params), params)
        return jsonify({
            "success": True,
            "run_id": _active_job["run_id"],
            "state": _active_job["state"],
        }), 202


@app.route("/api/run-status")
def api_run_status():
    """The active job's state, or the last one recorded on disk.

    Safe to poll: the payload carries no filesystem paths and no secrets (the
    log tail is redacted as it is read).
    """
    with _job_lock:
        payload = dict(_active_job) if _active_job is not None else _read_status_file()
    return jsonify(json_safe(payload))


@app.route("/api/cancel", methods=["POST"])
def api_cancel_run():
    """Kill switch: stop the active job (and the children it spawned)."""
    if not _token_ok():
        return jsonify({
            "success": False,
            "error": "Forbidden: missing or invalid pipeline token.",
        }), 403

    global _active_job, _active_proc
    with _job_lock:
        job = _active_job
        proc = _active_proc
        if job is None:
            return jsonify({"success": False,
                            "error": "No pipeline run is active."}), 409
    _terminate(proc)
    with _job_lock:
        job["state"] = "cancelled"
        job["step"] = "cancelled"
        job["finished_utc"] = _utc_now()
        _write_status(job)
        _active_job = None
        _active_proc = None
    return jsonify({"success": True, "run_id": job["run_id"], "state": "cancelled"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    if port == 0:
        # Windows ships with PORT=0 in some environments, which makes Flask bind
        # a random port and the printed URL useless. Fall back to 5000.
        port = 5000
    # Bind loopback by default: the dashboard has no login, so binding 0.0.0.0
    # puts every result on whatever Wi-Fi the demo box is attached to.
    # JALNETRA_HOST=0.0.0.0 opts in explicitly (HOST is the older name, still
    # honoured so existing commands keep working).
    host = os.environ.get("JALNETRA_HOST") or os.environ.get("HOST") or "127.0.0.1"
    # The Werkzeug debugger is a live Python REPL: anyone who can reach a
    # traceback gets code execution on this machine. So debug is enabled only
    # when it is explicitly asked for AND the server is bound to loopback.
    _wants_debug = (
        os.environ.get("JALNETRA_DEBUG") or os.environ.get("FLASK_DEBUG") or "0"
    ).lower() in ("1", "true", "yes", "on")
    debug = _wants_debug and host in ("127.0.0.1", "localhost", "::1")
    print(f'\
Jalnetra Dashboard — Open: http://{host}:{port}  (debug={debug})\
Features: flood map, OSM damage, cut-offs, report, flood path\
')
    app.run(host=host, port=port, debug=debug)
