"""The dashboard must not describe mismatched or unconfirmed results as live."""

from __future__ import annotations

import json

import pytest


@pytest.fixture
def provenance_case(tmp_path, monkeypatch):
    import dashboard.app as app

    results = tmp_path / "results"
    data = tmp_path / "data"
    results.mkdir()
    for label, date in (("before", "2026-08-24"), ("after", "2026-09-05")):
        directory = data / "sentinel1" / label
        directory.mkdir(parents=True)
        (directory / "S1_PROCESSED.json").write_text(json.dumps({
            "bbox_deg": [85.1, 28.0, 85.5, 28.1],
            "acquired_utc": date + "T00:00:00Z",
            "relative_orbit": 19,
            "product_name": "test-scene",
        }), encoding="utf-8")

    metadata = {
        "area": "Test Valley", "bbox": [85.1, 28.0, 85.5, 28.1],
        "before_date": "2026-08-24", "after_date": "2026-09-05",
        "event_date": "2026-08-26", "osm_snapshot_date": "2026-07-27",
        "aoi_confirmed": True,
    }
    (results / "run_meta.json").write_text(json.dumps(metadata), encoding="utf-8")
    (results / "manifest.json").write_text(json.dumps({
        "parameters": metadata,
        "repo_commit": "a" * 40,
        "created_utc": "2026-10-14T00:00:00Z",
        "method": "log_ratio_baseline",
        "rules_check": {
            "osm_snapshot_before_event": True,
            "acquisitions_bracket_event": True,
            "no_forbidden_input_sources": True,
            "sentinel_orbit_same_for_pair": True,
        },
    }), encoding="utf-8")
    osm = data / "osm"
    osm.mkdir(parents=True)
    (osm / "osm_metadata.json").write_text(json.dumps({
        "bbox": metadata["bbox"],
        "snapshot_date": metadata["osm_snapshot_date"],
        "failed_required_categories": [],
    }), encoding="utf-8")
    (results / "flood_mask.tif").write_bytes(b"test result")

    monkeypatch.setattr(app, "RESULTS", results)
    monkeypatch.setattr(app, "DATA", data)
    monkeypatch.setattr(app, "_current_repo_commit", lambda: "a" * 40)
    return app, metadata, data, results


def test_matching_confirmed_provenance_is_live(provenance_case):
    app, _, _, _ = provenance_case
    assert app.results_provenance()["state"] == "live"


def test_unconfirmed_aoi_cannot_be_reported_as_live(provenance_case):
    app, metadata, _, results = provenance_case
    metadata["aoi_confirmed"] = False
    (results / "run_meta.json").write_text(json.dumps(metadata), encoding="utf-8")

    provenance = app.results_provenance()
    assert provenance["state"] == "unverified"
    assert "AOI has not been confirmed" in provenance["missing"]


def test_sentinel_bounds_must_match_run_aoi(provenance_case):
    app, _, data, _ = provenance_case
    path = data / "sentinel1" / "after" / "S1_PROCESSED.json"
    sidecar = json.loads(path.read_text(encoding="utf-8"))
    sidecar["bbox_deg"] = [85.4, 28.0, 86.0, 28.6]
    path.write_text(json.dumps(sidecar), encoding="utf-8")

    provenance = app.results_provenance()
    assert provenance["state"] == "unverified"
    assert "after Sentinel bounds do not match run_meta.json" in provenance["missing"]


def test_result_commit_must_match_current_code(provenance_case, monkeypatch):
    app, _, _, _ = provenance_case
    monkeypatch.setattr(app, "_current_repo_commit", lambda: "b" * 40)

    provenance = app.results_provenance()
    assert provenance["state"] == "unverified"
    assert any("different or unverified code revision" in item for item in provenance["missing"])


def test_osm_bounds_must_match_run_aoi(provenance_case):
    app, _, data, _ = provenance_case
    path = data / "osm" / "osm_metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["bbox"] = [85.4, 28.0, 86.0, 28.6]
    path.write_text(json.dumps(metadata), encoding="utf-8")

    provenance = app.results_provenance()
    assert provenance["state"] == "unverified"
    assert "OSM bounds do not match run_meta.json" in provenance["missing"]
