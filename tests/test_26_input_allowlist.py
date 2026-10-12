"""T17 — input allowlist for the run request.

The request body is the only thing a browser can push toward `argparse` and the
filesystem, so the shape is fixed: known keys only, a bbox small enough to be a
flood analysis, dates in the order the pipeline needs and that actually exist,
a valid area name and a preset that is in the list.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "dashboard"))
sys.path.insert(0, str(REPO_ROOT))

TOKEN_HEADER = {"X-Jalnetra-Token": "unit-test-token"}

VALID = {
    "bbox": "10,20,11,21",
    "area": "toy area",
    "event_date": "2026-08-26",
    "osm_snapshot": "2026-07-27",
    "before_date": "2026-08-14",
    "after_date": "2026-08-26",
}


@pytest.fixture()
def dashboard(tmp_path, monkeypatch):
    import app as app_module

    monkeypatch.setattr(app_module, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(app_module, "PIPELINE_TOKEN", "unit-test-token")
    monkeypatch.setattr(app_module, "_active_job", None)
    monkeypatch.setattr(app_module, "_active_proc", None)
    monkeypatch.setattr(app_module, "_post_times", {})
    app_module.app.config.update(TESTING=True)

    def _stub_launch(argv, params):
        job = {"run_id": "33333333-3333-3333-3333-333333333333", "state": "running",
               "step": "starting", "progress": 0.0, "started_utc": "2026-10-13T00:00:00+00:00",
               "finished_utc": None, "exit_code": None, "params": params, "log_tail": []}
        monkeypatch.setattr(app_module, "_active_job", job)
        return job

    monkeypatch.setattr(app_module, "_launch_job", _stub_launch)
    return app_module


def _post(dashboard, body):
    return dashboard.app.test_client().post("/api/runs", json=body, headers=TOKEN_HEADER)


def test_a_complete_request_is_accepted(dashboard):
    response = _post(dashboard, VALID)
    assert response.status_code == 202, response.get_data(as_text=True)[:200]


@pytest.mark.parametrize(
    "extra",
    [
        {"admin": True},
        {"run_id": "../../etc/passwd"},
        {"shell": "; rm -rf /"},
        {"model": "models/evil.pt"},
    ],
)
def test_unknown_keys_are_refused(dashboard, extra):
    body = dict(VALID, **extra)
    response = _post(dashboard, body)
    assert response.status_code == 400
    assert "unknown field" in response.get_json()["error"]


@pytest.mark.parametrize(
    "body, field",
    [
        (dict(VALID, bbox="10,20,60,70"), "bbox"),             # ~2500 deg²
        (dict(VALID, bbox="not,a,bbox"), "bbox"),
        (dict(VALID, bbox="10,20,9,21"), "bbox"),              # west >= east
        (dict(VALID, bbox="10,20,11,200"), "bbox"),            # latitude range
        (dict(VALID, event_date="2999-01-01"), "event_date"),  # future
        (dict(VALID, event_date="2026-02-31"), "event_date"),  # not a real date
        (dict(VALID, event_date="26-08-2026"), "event_date"),  # not ISO
        (dict(VALID, osm_snapshot="2026-09-01"), "osm_snapshot"),
        (dict(VALID, before_date="2026-09-30"), "before_date"),
        (dict(VALID, area="; rm -rf /"), "area"),
        (dict(VALID, area="x" * 120), "area"),
        (dict(VALID, preset="../../etc/passwd"), "preset"),
        (dict(VALID, preset="no-such-preset"), "preset"),
    ],
)
def test_bad_values_are_refused_with_the_field(dashboard, body, field):
    response = _post(dashboard, body)
    assert response.status_code == 400, f"{field}: {response.get_data(as_text=True)[:200]}"
    assert response.get_json()["field"] == field


def test_injection_strings_never_reach_argv(dashboard, monkeypatch):
    """A value that survives validation must still arrive as a single argv item."""
    captured = {}

    def _capture(argv, params):
        captured["argv"] = argv
        job = {"run_id": "44444444-4444-4444-4444-444444444444", "state": "running",
               "step": "starting", "progress": 0.0, "started_utc": "2026-10-13T00:00:00+00:00",
               "finished_utc": None, "exit_code": None, "params": params, "log_tail": []}
        monkeypatch.setattr(dashboard, "_active_job", job)
        return job

    monkeypatch.setattr(dashboard, "_launch_job", _capture)
    body = dict(VALID, area="toy area (north)")
    assert _post(dashboard, body).status_code == 202

    argv = captured["argv"]
    assert isinstance(argv, list)
    assert "toy area (north)" in argv              # one element, not split
    assert all(isinstance(part, str) for part in argv)
