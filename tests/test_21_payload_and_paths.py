"""21. Response size and file safety (T22).

The damage layer is mostly UNAFFECTED footprints, each carrying a copy of every
OSM tag, which made /api/damage ~88 KB on the case study to show 36 interesting
features. The default response is affected-only and drops the raw tag blob;
`?all=1` returns everything, still capped. JSON is gzipped when the client asks,
and no route may serve a file outside the results directory.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest


def _feature(level, feature_type="building", **extra):
    props = {
        "feature_type": feature_type,
        "damage_level": level,
        "damage_overlap_ratio": 0.0,
        "flood_affected": False,
        "osm_tags": {"name": f"tag-{level}-{feature_type}", "building": "residential"},
    }
    props.update(extra)
    return {
        "type": "Feature",
        "properties": props,
        "geometry": {"type": "Point", "coordinates": [85.0, 28.0]},
    }


FIXTURE = {
    "type": "FeatureCollection",
    "metadata": {"generated_by": "fixture", "event_date": "2026-08-26"},
    "features": [
        _feature("DESTROYED", damage_overlap_ratio=0.8, flood_affected=True),
        _feature("UNAFFECTED"),
        _feature("UNKNOWN"),
        _feature("UNAFFECTED", feature_type="road", road_cut=True),
        # Enough bulk that the JSON clears the gzip threshold.
        *[_feature("UNAFFECTED") for _ in range(60)],
    ],
}


@pytest.fixture
def dashboard(tmp_path, monkeypatch):
    import app as app_module

    results = tmp_path / "results"
    results.mkdir()
    (results / "damage_assessment.geojson").write_text(json.dumps(FIXTURE), encoding="utf-8")
    monkeypatch.setattr(app_module, "RESULTS", results)
    monkeypatch.setattr(app_module, "DATA", tmp_path / "data")
    app_module.app.config.update(TESTING=True)
    return app_module


def test_damage_default_returns_only_affected(dashboard):
    payload = dashboard.app.test_client().get("/api/damage").get_json()
    returned = payload["features"]

    assert all(dashboard._is_affected(f) for f in returned), (
        "the default response must only contain features the affected rule keeps"
    )
    # The level is not the whole story: a road whose segment is cut counts as
    # affected even though its damage level says UNAFFECTED.
    plain_unaffected = [
        f for f in returned
        if f["properties"]["damage_level"] == "UNAFFECTED"
        and not f["properties"].get("road_cut")
    ]
    assert not plain_unaffected, (
        "unaffected footprints are most of the file and say nothing on a damage map"
    )
    # DESTROYED, the UNKNOWN one, and the cut road stay.
    assert len(returned) == 3
    assert payload["metadata"]["features_total"] == 64
    assert payload["metadata"]["affected_only"] is True


def test_damage_default_omits_the_raw_tag_blob(dashboard):
    payload = dashboard.app.test_client().get("/api/damage").get_json()

    assert all("osm_tags" not in f["properties"] for f in payload["features"]), (
        "nothing in the UI renders osm_tags, so shipping every OSM tag to the "
        "browser is pure payload"
    )
    assert payload["metadata"]["osm_tags_omitted"] is True


def test_damage_all_returns_everything_with_properties(dashboard):
    payload = dashboard.app.test_client().get("/api/damage?all=1").get_json()

    assert len(payload["features"]) == 64
    assert payload["metadata"]["affected_only"] is False
    assert any("osm_tags" in f["properties"] for f in payload["features"]), (
        "?all=1 is the escape hatch for inspecting the raw layer"
    )


def test_damage_response_is_capped(dashboard, monkeypatch):
    monkeypatch.setattr(dashboard, "MAX_FEATURES", 5)

    payload = dashboard.app.test_client().get("/api/damage?all=1").get_json()

    assert len(payload["features"]) == 5
    assert payload["metadata"]["capped"] is True, (
        "a truncated layer must say so rather than look complete"
    )


def test_json_is_gzipped_when_the_client_asks(dashboard):
    client = dashboard.app.test_client()
    plain = client.get("/api/damage?all=1")
    zipped = client.get("/api/damage?all=1", headers={"Accept-Encoding": "gzip"})

    assert zipped.headers.get("Content-Encoding") == "gzip", (
        "the flood/damage layers are mostly coordinates; gzip should be automatic"
    )
    body = gzip.decompress(zipped.get_data())
    assert json.loads(body)["metadata"]["features_total"] == 64, (
        "the gzipped body must decompress to the same payload"
    )
    assert len(body) < len(plain.get_data()) * 1.05
    assert zipped.headers.get("Vary") == "Accept-Encoding"


def test_no_compression_without_the_client_asking(dashboard):
    response = dashboard.app.test_client().get("/api/damage?all=1")
    assert response.headers.get("Content-Encoding") is None


def test_is_affected_matches_the_documented_rule(dashboard):
    affected = dashboard._is_affected
    assert affected(_feature("DESTROYED")) is True
    assert affected(_feature("UNAFFECTED")) is False
    assert affected(_feature("UNKNOWN")) is True, (
        "UNKNOWN means 'could not assess', which is a finding worth showing"
    )
    assert affected(_feature("UNAFFECTED", road_cut=True)) is True
    assert affected(_feature("UNAFFECTED", damage_overlap_ratio=0.3)) is True


def test_is_inside_rejects_escapes(tmp_path, dashboard):
    inside = dashboard._is_inside
    results = tmp_path / "results"

    assert inside(results / "flood_mask.tif", results) is True
    assert inside(results / ".." / "config" / "presets.py", results) is False, (
        "a relative escape must not count as inside"
    )
    assert inside(Path("/etc/passwd"), results) is False
    assert inside(Path("C:/Windows/system32/config/sam"), results) is False


@pytest.mark.parametrize("path", [
    "/report/../../etc/passwd",
    "/api/situation-report/../../etc/passwd",
    "/api/situation-report?file=../../etc/passwd",
    "/report?name=..%2f..%2fetc%2fpasswd",
])
def test_traversal_attempts_never_serve_a_file(dashboard, path):
    response = dashboard.app.test_client().get(path)

    assert response.status_code in (404, 400, 200), "no 500 and no file contents"
    body = response.get_data(as_text=True)
    assert "root:" not in body and "daemon:" not in body, (
        f"{path} leaked a system file"
    )
