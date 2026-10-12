"""14. Case-study values live in config/presets.py, not in the code.

A judge (or a teammate) must be able to point the pipeline at another corridor
or date by changing one file or passing a preset. Coordinates, acquisition dates
and the AOI bbox scattered through the scripts is how the "just change the area"
promise quietly stops being true — and a stale snapshot date is a rules
violation.

The scan covers scripts/ and dashboard/ source including docstrings and comments,
so the strings cannot simply be moved into prose. Everything banned here is a
value that must come from a preset (config/presets.py) or from the run's metadata
(results/run_meta.json), never restated in code.
"""

from __future__ import annotations

from pathlib import Path

BANNED_LITERALS = [
    # Case-study centres and AOI edges: --bbox / the preset now own these.
    "85.7", "85.4", "86.0", "28.3",
    # Stale 78.x coordinates (the old, wrong location) and the old hardcoded
    # dashboard map bounds. Both classes were real bugs found here.
    "78.55", "78.85", "78.6", "78.7", "27.5", "29.1", "84.5", "86.9",
    # Acquisition, event and OSM-snapshot dates: run_meta.json now owns these.
    "2026-07-27", "2026-08-14", "2026-08-24", "2026-08-26", "2026-08-28",
    "2026-08-31", "2026-09-05",
]
SCAN_SUFFIXES = {".py", ".html", ".js"}

# dashboard/templates/demo.html is a deliberately fabricated showcase page whose
# entire content (villages, coordinates, headline numbers) is mock data for a
# demo click-through. It is not part of the pipeline and is excluded here on
# purpose; every other template and every Python source IS in scope.
EXCLUDED_FILES = {"demo.html"}


def _scan(scripts_dir: Path, dashboard_dir: Path):
    offenders = []
    for base in (scripts_dir, dashboard_dir):
        for path in sorted(base.rglob("*")):
            if "__pycache__" in path.parts or not path.is_file():
                continue
            if path.suffix.lower() not in SCAN_SUFFIXES:
                continue
            if path.name in EXCLUDED_FILES:
                continue
            for lineno, line in enumerate(
                path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
            ):
                for literal in BANNED_LITERALS:
                    if literal in line:
                        offenders.append(f"{path}:{lineno}: {line.strip()[:110]}")
    return offenders


def test_case_study_values_are_not_hardcoded(scripts_dir, dashboard_dir):
    offenders = _scan(scripts_dir, dashboard_dir)

    assert not offenders, (
        "case-study literals (" + ", ".join(BANNED_LITERALS) + ") must live only in "
        "config/presets.py (or the run's results/run_meta.json). Move them there and "
        "read them from there:\n  " + "\n  ".join(offenders)
    )


def test_presets_expose_the_case_study_without_guessing_the_aoi(repo_root: Path):
    from config import presets

    assert (repo_root / "config" / "presets.py").is_file(), (
        "config/presets.py must exist — it is the documented single place to "
        "change the case study"
    )
    assert presets.DEFAULT_PRESET in presets.PRESETS, (
        f"DEFAULT_PRESET {presets.DEFAULT_PRESET!r} is not in PRESETS "
        f"({sorted(presets.PRESETS)})"
    )

    preset = presets.PRESETS[presets.DEFAULT_PRESET]
    for key in ("area", "before_date", "after_date", "event_date",
                "osm_snapshot_date", "bbox", "aoi_confirmed"):
        assert key in preset, f"the case-study preset is missing {key!r}"

    # The bbox is the PRIMARY AOI input and must not be invented. Either the
    # confirmed EMSR927 AOI01 bounds are in place, or the preset states plainly
    # that its AOI is unconfirmed so every artifact is stamped accordingly.
    bbox = preset["bbox"]
    if bbox is None:
        assert preset["aoi_confirmed"] is False, (
            "a preset with no confirmed bbox must set aoi_confirmed=False, so no "
            "artifact it produces can pass as covering a confirmed AOI"
        )
    else:
        assert len(bbox) == 4 and bbox[0] < bbox[2] and bbox[1] < bbox[3], (
            f"a preset bbox must be [min_lon,min_lat,max_lon,max_lat], got {bbox!r}"
        )

    # The pre-event OSM snapshot is a hard rules requirement whatever the AOI.
    assert preset["osm_snapshot_date"] < preset["event_date"], (
        f"OSM snapshot {preset['osm_snapshot_date']} is not before the event "
        f"{preset['event_date']} — using post-event OSM edits as input is prohibited"
    )
