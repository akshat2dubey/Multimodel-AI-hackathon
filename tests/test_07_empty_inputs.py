"""7. Empty inputs are a valid outcome, not a crash.

An area with no mapped buildings, bridges or hospitals is normal (OSM coverage
is patchy at altitude). Both stages must exit 0 and write a structurally valid,
all-zero artifact, because a traceback here means the dashboard and the report
get nothing at all — and an operator cannot tell "no data" from "broken".
"""

from __future__ import annotations

import json
import subprocess
import sys

import toydata

TIMEOUT_S = 180


def _run(script, args, cwd):
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True, text=True,
        # The scripts re-configure stdout to UTF-8 and print emoji.
        encoding="utf-8", errors="replace",
        cwd=str(cwd), timeout=TIMEOUT_S,
    )


def _empty_inputs(tmp_path):
    osm = tmp_path / "osm"
    toydata.empty_geojson(osm / "buildings.geojson")
    toydata.empty_geojson(osm / "roads.geojson")
    toydata.empty_geojson(osm / "bridges.geojson")
    toydata.empty_geojson(osm / "hospitals.geojson")
    mask = toydata.write_mask_tif(tmp_path / "flood_mask.tif", toydata.cutoff_flood_mask())
    return osm, mask


def test_damage_assessment_on_empty_features_exits_zero_with_zero_counts(tmp_path, scripts_dir):
    osm, mask = _empty_inputs(tmp_path)
    out = tmp_path / "damage_assessment.geojson"

    proc = _run(
        scripts_dir / "damage_assessment.py",
        ["--flood-mask", str(mask),
         "--osm-buildings", str(osm / "buildings.geojson"),
         "--osm-roads", str(osm / "roads.geojson"),
         "--osm-bridges", str(osm / "bridges.geojson"),
         "--output", str(out)],
        cwd=tmp_path,
    )

    assert proc.returncode == 0, (
        "damage_assessment.py must exit 0 when there are no features to assess, "
        f"got {proc.returncode}\nstdout: {proc.stdout[-800:]}\nstderr: {proc.stderr[-800:]}"
    )
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["type"] == "FeatureCollection", f"expected a FeatureCollection, got {data.get('type')!r}"
    assert data["features"] == [], f"expected zero features, got {len(data['features'])}"
    assert data["metadata"]["assessment_method"] == "flood_mask_overlay", (
        "the empty output must still carry its metadata block"
    )


def test_cutoff_analysis_on_empty_features_exits_zero_with_zero_counts(tmp_path, scripts_dir):
    osm, mask = _empty_inputs(tmp_path)
    out = tmp_path / "cutoff_settlements.geojson"

    proc = _run(
        scripts_dir / "cutoff_analysis.py",
        ["--roads", str(osm / "roads.geojson"),
         "--buildings", str(osm / "buildings.geojson"),
         "--flood-mask", str(mask),
         "--hospitals", str(osm / "hospitals.geojson"),
         "--output", str(out)],
        cwd=tmp_path,
    )

    assert proc.returncode == 0, (
        "cutoff_analysis.py must exit 0 when there is nothing to analyse, got "
        f"{proc.returncode}\nstdout: {proc.stdout[-800:]}\nstderr: {proc.stderr[-800:]}"
    )
    data = json.loads(out.read_text(encoding="utf-8"))
    metadata = data["metadata"]
    summary = metadata["summary"]
    assert summary["total_settlements"] == 0, (
        f"expected 0 settlements, got {summary['total_settlements']}"
    )
    assert summary["cutoff_settlements"] == 0, (
        f"expected 0 cut-off settlements, got {summary['cutoff_settlements']}"
    )
    assert summary["flooded_road_segments"] == 0, (
        f"expected 0 flooded road segments, got {summary['flooded_road_segments']}"
    )
    assert metadata["cutoff_settlements"] == [], (
        f"expected an empty cutoff list, got {len(metadata['cutoff_settlements'])} entries"
    )
