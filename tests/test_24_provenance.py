"""T34 — provenance manifest and the hard data rules.

The rules decide whether a submission is even eligible: only Sentinel-1/2,
Copernicus DEM, a pre-event OSM snapshot and the listed training data may be
inputs. These tests pin the booleans, the refusal behaviour, and the fact that
every manifest entry carries a real hash.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

import manifest  # noqa: E402


def _meta(**overrides) -> dict:
    meta = {
        "run_id": "run-1",
        "area": "toy",
        "bbox": [85.4, 28.0, 85.5, 28.1],
        "event_date": "2026-08-26",
        "before_date": "2026-08-14",
        "after_date": "2026-08-26",
        "osm_snapshot_date": "2026-07-27",
    }
    meta.update(overrides)
    return meta


def test_snapshot_after_the_event_is_a_violation():
    """Post-event OSM edits as input forfeits the run, so the rule must fail."""
    checks = manifest.rules_check(_meta(osm_snapshot_date="2026-09-01"))
    assert checks["osm_snapshot_before_event"] is False
    with pytest.raises(manifest.RuleViolation) as excinfo:
        manifest.enforce_rules(checks)
    assert "osm_snapshot_before_event" in str(excinfo.value)


def test_a_clean_run_passes_the_date_rules():
    checks = manifest.rules_check(_meta())
    assert checks["osm_snapshot_before_event"] is True
    assert checks["acquisitions_bracket_event"] is True
    manifest.enforce_rules(checks)          # must not raise


def test_unknown_rules_are_null_not_true():
    """A rule that cannot be evaluated is recorded as None, never as a pass."""
    checks = manifest.rules_check({})
    assert set(checks.values()) == {None}
    manifest.enforce_rules(checks)          # unknown is not a failure


def test_forbidden_input_source_is_caught():
    good = [Path("data/sentinel1/before/s1_before_VV.tif")]
    bad = [Path("docs/EMSR927_reference.geojson"), Path("data/osm/buildings.geojson")]
    assert manifest.rules_check(_meta(), good)["no_forbidden_input_sources"] is True
    assert manifest.rules_check(_meta(), bad)["no_forbidden_input_sources"] is False


def test_orbit_rule_compares_the_two_sidecars(tmp_path):
    sentinel = tmp_path / "sentinel1"
    for label, orbit in (("before", 19), ("after", 19)):
        folder = sentinel / label
        folder.mkdir(parents=True)
        (folder / "S1_PROCESSED.json").write_text(
            json.dumps({"relative_orbit": orbit}), encoding="utf-8"
        )
    assert manifest.rules_check(_meta(), sentinel_dir=sentinel)["sentinel_orbit_same_for_pair"] is True

    (sentinel / "after" / "S1_PROCESSED.json").write_text(
        json.dumps({"relative_orbit": 42}), encoding="utf-8"
    )
    assert manifest.rules_check(_meta(), sentinel_dir=sentinel)["sentinel_orbit_same_for_pair"] is False


def test_manifest_hashes_every_input_and_output(tmp_path):
    (tmp_path / "in.tif").write_bytes(b"input bytes")
    (tmp_path / "out.geojson").write_text('{"type": "FeatureCollection"}', encoding="utf-8")
    model = tmp_path / "flood_unet.pt"
    model.write_bytes(b"weights")
    (tmp_path / "flood_unet.pt.sha256").write_text(
        manifest.sha256_file(model) + "\n", encoding="utf-8"
    )

    payload = manifest.build_manifest(
        run_meta=_meta(),
        inputs=[tmp_path / "in.tif"],
        outputs=[tmp_path / "out.geojson"],
        repo_root=tmp_path,
        method="unet",
        model=model,
    )

    assert payload["method"] == "unet"
    assert payload["model"]["verified_against_sidecar"] is True
    assert [entry["path"] for entry in payload["inputs"]] == ["in.tif"]
    assert [entry["path"] for entry in payload["outputs"]] == ["out.geojson"]
    assert payload["inputs"][0]["sha256"] == manifest.sha256_file(tmp_path / "in.tif")
    assert payload["inputs"][0]["bytes"] == len(b"input bytes")
    assert payload["versions"]["python"]
    assert set(payload["rules_check"]) >= {"osm_snapshot_before_event"}


def test_manifest_round_trips_to_disk(tmp_path):
    payload = manifest.build_manifest(_meta(), [], [], tmp_path)
    path = manifest.write_manifest(tmp_path / "results", payload)
    assert path.name == "manifest.json"
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["run_id"] == "run-1"
    # The manifest is written for every run, including a run whose later steps
    # failed, so it must always be valid JSON with the rule block present.
    assert "rules_check" in written


def test_a_tampered_model_is_recorded_as_unverified(tmp_path):
    model = tmp_path / "flood_unet.pt"
    model.write_bytes(b"weights")
    (tmp_path / "flood_unet.pt.sha256").write_text("0" * 64 + "\n", encoding="utf-8")
    payload = manifest.build_manifest(_meta(), [], [], tmp_path, model=model)
    assert payload["model"]["verified_against_sidecar"] is False


def test_pipeline_refuses_a_broken_rule_before_downloading(monkeypatch, tmp_path):
    """run_pipeline must exit non-zero (code 3) rather than start a download."""
    import run_pipeline

    calls = {"steps": 0}

    def _no_step(cmd, cwd=None):            # pragma: no cover - must not run
        calls["steps"] += 1
        raise AssertionError("a broken data rule must stop the run before any step")

    monkeypatch.setattr(run_pipeline, "run", _no_step)
    monkeypatch.setattr(
        run_pipeline, "resolve_run_meta",
        lambda **kwargs: _meta(osm_snapshot_date="2026-09-09"),
    )
    # Refused runs must not touch the real results/ directory either.
    monkeypatch.setattr(run_pipeline, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "--preset", "trishuli"])

    with pytest.raises(SystemExit) as excinfo:
        run_pipeline.main()
    assert excinfo.value.code == 3
    assert calls["steps"] == 0
    # Nothing was written at all — not even the metadata for the refused run.
    assert not (tmp_path / "results").exists()
