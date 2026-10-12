"""13. The required data attributions must appear where people read them.

The deliverables list requires the data attributions and the training-dataset
citation to be present. Two places count: the one-page situation report (a
published artifact) and the dashboard's own footer — a reader must not have to
open the README to know whose data the numbers come from.

The report is generated here through the real loaders and the real HTML
renderer, so the check covers what a fresh run produces, not a stale file.
"""

from __future__ import annotations

import json
from pathlib import Path

import generate_report
import toydata

# (label, substring that must be present)
REQUIRED_ATTRIBUTIONS = [
    ("Copernicus Sentinel", "Contains modified Copernicus Sentinel data"),
    ("Copernicus DEM", "Copernicus WorldDEM-30"),
    ("OpenStreetMap", "OpenStreetMap contributors"),
    ("training dataset", "Kuro Siwo"),
]


def _write_inputs(tmp_path: Path):
    """A minimal but real set of report inputs in tmp_path."""
    import inference

    mask_path = toydata.write_mask_tif(tmp_path / "flood_mask.tif", toydata.cutoff_flood_mask())

    before, after = toydata.river_and_patch()
    delivered = inference.derive_change_mask(before, after)
    damage_path = tmp_path / "damage_assessment.geojson"
    damage_path.write_text(json.dumps({
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {
                "feature_type": "building",
                "damage_level": "DESTROYED",
                "damage_overlap_ratio": 1.0,
                "damage_method": "raster_overlap",
                "flood_affected": True,
            },
            "geometry": {"type": "Point", "coordinates": [85.61, 28.39]},
        }],
        "metadata": {"assessment_method": "flood_mask_overlay"},
    }), encoding="utf-8")

    cutoff_path = tmp_path / "cutoff_settlements.geojson"
    cutoff_path.write_text(json.dumps({
        "type": "FeatureCollection",
        "features": [],
        "status": "ok",
        "summary": {
            "total_settlements": 0, "total_hospitals": 1,
            "reachable_settlements": 0, "cutoff_settlements": 0,
            "cutoff_percentage": 0.0, "already_isolated_settlements": 0,
            "flooded_road_segments": 0,
        },
        "cutoff_settlements": [],
    }), encoding="utf-8")

    return mask_path, damage_path, cutoff_path


def test_generated_situation_report_carries_every_required_attribution(tmp_path):
    mask_path, damage_path, cutoff_path = _write_inputs(tmp_path)

    flood_stats = generate_report.load_flood_mask_stats(mask_path)
    damage_stats = generate_report.load_damage_stats(damage_path)
    cutoff_stats = generate_report.load_cutoff_stats(cutoff_path)

    out = tmp_path / "situation_report.html"
    generate_report.generate_html_report(
        "Toy Area", "2026-08-14", "2026-08-28",
        flood_stats, damage_stats, cutoff_stats, out,
        synthetic=True,
    )
    html = out.read_text(encoding="utf-8")

    missing = [label for label, needle in REQUIRED_ATTRIBUTIONS if needle not in html]
    assert not missing, (
        f"the generated situation report is missing required attribution(s): "
        f"{missing}. Required attributions must be in the footer of every "
        "report, not only in the README."
    )


def test_dashboard_index_template_carries_every_required_attribution(dashboard_dir: Path):
    html = (dashboard_dir / "templates" / "index.html").read_text(encoding="utf-8")

    missing = [label for label, needle in REQUIRED_ATTRIBUTIONS if needle not in html]
    assert not missing, (
        f"the dashboard index template is missing required attribution(s): "
        f"{missing}. The footer must name the data sources shown on the map."
    )


def test_checked_in_situation_report_carries_every_required_attribution(repo_root: Path):
    report = repo_root / "report" / "situation_report.html"
    if not report.exists():
        return  # not generated in this checkout; the generator is covered above
    html = report.read_text(encoding="utf-8")

    missing = [label for label, needle in REQUIRED_ATTRIBUTIONS if needle not in html]
    assert not missing, (
        f"the committed situation report is missing required attribution(s): "
        f"{missing} — regenerate it with scripts/generate_report.py"
    )
