"""Single source of truth for paths and case-study presets.

Nothing here is secret. The point of this module is that scripts/ and
dashboard/ must not carry the case-study coordinates, the event dates or the
output directories as literals: a judge (or a teammate) who wants to point the
pipeline at another corridor or date changes exactly one file.

The case-study values themselves live in PRESETS below. Once resolved (with any
CLI overrides) they are written to ``<results>/run_meta.json`` by
``config/run_meta.py``, and every stage reads them from there — so a report, a
damage assessment and a cut-off list produced by one run all describe the same
area, bbox and dates.

Layout conventions (do not fork these):
    ROOT/results       — every pipeline artifact the dashboard serves
    ROOT/results_demo  — the synthetic fixture ONLY (never mixed into results/)
    ROOT/data          — downloaded inputs (satellite, OSM, DEM)
    ROOT/data_demo     — generated inputs for the synthetic fixture ONLY
    ROOT/report        — the generated one-page situation report

Demo output lives in its own directory on purpose: a judge opening the
dashboard must never see fabricated numbers and real numbers side by side in
one directory with no way to tell them apart.
"""

from __future__ import annotations

from pathlib import Path

# config/presets.py -> config/ -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_NAME = RESULTS_DIR.name
# Synthetic fixture destination. Kept separate from RESULTS_DIR so a demo run
# cannot overwrite (or be mistaken for) a real run.
RESULTS_DEMO_DIR = PROJECT_ROOT / "results_demo"
RESULTS_DEMO_NAME = RESULTS_DEMO_DIR.name
DATA_DIR = PROJECT_ROOT / "data"
# Generated (synthetic) inputs. Separate from DATA_DIR because data/ holds real
# downloads whose replacement would destroy the only real evidence in the repo.
DATA_DEMO_DIR = PROJECT_ROOT / "data_demo"
DATA_DEMO_NAME = DATA_DEMO_DIR.name
REPORT_DIR = PROJECT_ROOT / "report"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# A results directory is "demo" if it is the demo directory by name, or if it
# carries the notice make_demo_data.py drops next to the mask. The dashboard uses
# this to show a banner; scripts use it to pick a default output dir.
DEMO_NOTICE_FILENAME = "DEMO_DATA_NOTICE.txt"


def is_demo_dir(path) -> bool:
    """True when `path` is serving (or is) synthetic demo data."""
    from pathlib import Path as _Path

    p = _Path(path)
    return p.name == RESULTS_DEMO_NAME or (p / DEMO_NOTICE_FILENAME).is_file()


def demo_notice_blocker(results_dir) -> str | None:
    """Why a REAL run must not write into `results_dir`, or None when it may.

    The notice file is what makes the dashboard warn a viewer that the numbers
    on screen are fabricated. If a real run left it in place, the real flood
    extent would be served under a "SYNTHETIC DEMO DATA" banner — a judge would
    have no way to tell which claim is true — and if the notice were deleted
    while the fixture's own artifacts stayed behind, the opposite error would
    happen. Either way the directory is ambiguous, so the run stops and asks a
    human to resolve it instead of guessing.
    """
    from pathlib import Path as _Path

    notice = _Path(results_dir) / DEMO_NOTICE_FILENAME
    if not notice.is_file():
        return None
    return (
        f"{notice} marks this directory as the SYNTHETIC fixture, so a real run "
        "would be served under a demo warning. Move the fixture aside first "
        f"(e.g. move {notice.parent} to {RESULTS_DEMO_NAME}/), then re-run."
    )

# --- Case-study presets -----------------------------------------------------
# A preset is what a judge flips when they want a different corridor:
#
#     python scripts/run_pipeline.py --preset trishuli
#
# Every value here is a DEFAULT; an explicit CLI flag always wins, and the merged
# result is written to <results>/run_meta.json so every downstream stage reads
# the same numbers (see config/run_meta.py). Nothing in scripts/ or dashboard/
# may restate these as literals — tests/test_14_no_hardcoded_presets.py enforces
# that, docstrings included.

TRISHULI = {
    "area": "Bhote Koshi – Trishuli River Corridor, Nepal",
    # TODO(AOI): the authoritative case-study AOI is the Copernicus EMSR927
    # AOI01 footprint (Syabru Bensi, Rasuwa district). It is being taken from
    # the EMSR927 catalogue and must NOT be guessed — a wrong AOI silently
    # downloads the wrong valley. Put the AOI01 bounds here, in
    # [min_lon, min_lat, max_lon, max_lat] order, and set aoi_confirmed=True.
    # Until then this preset runs on the unverified fallback below and stamps
    # every artifact it produces with aoi_confirmed: false.
    "bbox": None,
    "aoi_confirmed": False,
    # The Sentinel-1 pair actually downloaded for this case study (orbit 19):
    # 2026-08-24 (2 d pre-event) and 2026-09-05 (10 d post-event). Provenance:
    # data/sentinel1/*/PRODUCT_INFO.json and docs/EMSR927_COMPARISON.md. The
    # event itself is 2026-08-26 (GLIDE FF-2026-000162-NPL, EMSR927 activated
    # 26/08/2026 09:53 UTC).
    "before_date": "2026-08-24",
    "after_date": "2026-09-05",
    "event_date": "2026-08-26",
    "osm_snapshot_date": "2026-07-27",
    # Fallback AOI ONLY, used while "bbox" is None. UNVERIFIED: the real event is
    # at Syabru Bensi (~85.2°E), so this 85.7°E center is suspect. It is kept
    # because it is the center the existing OSM snapshot was actually pulled at
    # (data/osm/osm_metadata.json), i.e. it reproduces what is on disk rather
    # than asserting it is correct.
    "center_lat": 28.30,
    "center_lon": 85.70,
    # Half-width of the fallback square, in degrees. 0.30 reproduces the real
    # OSM snapshot's bbox (85.4–86.0 °E / 28.0–28.6 °N).
    "extent": 0.30,
}

PRESETS: dict[str, dict] = {"trishuli": TRISHULI}
DEFAULT_PRESET = "trishuli"

# The default preset's values as module constants, for the callers that predate
# config/run_meta.py. Prefer resolve_run_meta()/load_run_meta() in new code: these
# do not know about a --bbox or about what a particular run actually used.
AREA = TRISHULI["area"]
CENTER_LAT = TRISHULI["center_lat"]
CENTER_LON = TRISHULI["center_lon"]
EVENT_DATE = TRISHULI["event_date"]
BEFORE_DATE = TRISHULI["before_date"]
AFTER_DATE = TRISHULI["after_date"]
OSM_SNAPSHOT_DATE = TRISHULI["osm_snapshot_date"]
# Half-width (degrees) of the synthetic fixture scene produced by
# make_demo_data.py / run_offline.py. This is a fabricated 640x640 grid, so it
# only sets the fixture's pixel size — it is NOT a case-study AOI.
FIXTURE_EXTENT = 0.50
# Half-width (degrees) of the AOI used when a run has no --bbox: the preset's
# fallback. Distinct from FIXTURE_EXTENT on purpose.
DEFAULT_EXTENT = TRISHULI["extent"]


def case_study(preset: str | None = None) -> dict:
    """A preset's values as a plain dict (handy for subprocess argv / templates)."""
    name = preset or DEFAULT_PRESET
    if name not in PRESETS:
        raise KeyError(
            f"unknown preset {name!r}; available: {', '.join(sorted(PRESETS))}"
        )
    return dict(PRESETS[name])
