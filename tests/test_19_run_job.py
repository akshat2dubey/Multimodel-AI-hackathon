"""19. Background pipeline job (T16).

POST /api/runs must return immediately with a run_id while the pipeline runs in
the background, the UI must be able to poll /api/run-status without ever seeing
a filesystem path or a secret, and only one job may run at a time — a second
request while one is running is a 409, not a second download.

Nothing here spawns the real pipeline: `_launch_job` is replaced with a stub, so
the suite stays offline and never writes into results/ or data/.
"""

from __future__ import annotations

import pytest

TOKEN_HEADER = {"X-Pipeline-Token": "unit-test-token"}
VALID_BBOX = [10.0, 20.0, 11.0, 21.0]
FAKE_RUN_ID = "11111111-1111-1111-1111-111111111111"


@pytest.fixture
def dashboard(tmp_path, monkeypatch):
    """The dashboard with an isolated results dir and a known token."""
    import app as app_module

    monkeypatch.setattr(app_module, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(app_module, "PIPELINE_TOKEN", "unit-test-token")
    monkeypatch.setattr(app_module, "_active_job", None)
    monkeypatch.setattr(app_module, "_active_proc", None)
    app_module.app.config.update(TESTING=True)
    return app_module


def _stub_launch(dashboard, monkeypatch, state="running"):
    """Replace the real spawner with one that records a job without running it."""

    def _launch(argv, params):
        job = {
            "run_id": FAKE_RUN_ID,
            "state": state,
            "step": "starting",
            "progress": 0.0,
            "started_utc": "2026-10-12T00:00:00+00:00",
            "finished_utc": None,
            "exit_code": None,
            "params": {k: v for k, v in params.items() if v not in (None, [], "")},
            "log_tail": ["pipeline output line"],
        }
        monkeypatch.setattr(dashboard, "_active_job", job)
        return job

    monkeypatch.setattr(dashboard, "_launch_job", _launch)


def test_post_run_returns_202_with_a_run_id(dashboard, monkeypatch):
    _stub_launch(dashboard, monkeypatch)
    client = dashboard.app.test_client()

    response = client.post(
        "/api/runs",
        json={"bbox": VALID_BBOX, "event_date": "2026-08-26"},
        headers=TOKEN_HEADER,
    )

    assert response.status_code == 202, (
        "POST /api/runs must return 202 immediately — the pipeline runs in the "
        f"background, so the page never blocks. Got {response.status_code}: "
        f"{response.get_data(as_text=True)[:200]}"
    )
    payload = response.get_json()
    assert payload["run_id"] == FAKE_RUN_ID
    assert payload["state"] == "running"


def test_second_run_while_one_is_running_is_a_conflict(dashboard, monkeypatch):
    _stub_launch(dashboard, monkeypatch)
    client = dashboard.app.test_client()

    first = client.post("/api/runs", json={"bbox": VALID_BBOX}, headers=TOKEN_HEADER)
    assert first.status_code == 202

    second = client.post("/api/runs", json={"bbox": VALID_BBOX}, headers=TOKEN_HEADER)
    assert second.status_code == 409, (
        "a second POST while a job is running must be refused with 409 — two "
        "concurrent runs would fight over the same results/ directory"
    )
    assert second.get_json()["run_id"] == FAKE_RUN_ID


def test_run_status_reports_idle_then_the_active_job(dashboard, monkeypatch):
    client = dashboard.app.test_client()

    idle = client.get("/api/run-status")
    assert idle.status_code == 200
    assert idle.get_json()["state"] == "idle", (
        "before any run the status route must answer 200 with state=idle, so the "
        "UI can poll it from the moment the page loads"
    )

    _stub_launch(dashboard, monkeypatch)
    client.post("/api/runs", json={"bbox": VALID_BBOX}, headers=TOKEN_HEADER)

    running = client.get("/api/run-status").get_json()
    assert running["state"] == "running"
    assert running["run_id"] == FAKE_RUN_ID
    assert running["progress"] >= 0.0


def test_run_endpoints_require_the_token(dashboard):
    client = dashboard.app.test_client()

    assert client.post("/api/runs", json={}).status_code == 403
    assert client.post(
        "/api/runs", json={}, headers={"X-Pipeline-Token": "wrong"}
    ).status_code == 403
    assert client.post("/api/cancel").status_code == 403
    # The legacy endpoint keeps the same guard.
    assert client.post("/api/run-pipeline", json={}).status_code == 403


def test_cancel_without_an_active_job_is_a_conflict(dashboard):
    client = dashboard.app.test_client()
    assert client.post("/api/cancel", headers=TOKEN_HEADER).status_code == 409


def test_cancel_marks_the_job_cancelled(dashboard, monkeypatch):
    _stub_launch(dashboard, monkeypatch)
    client = dashboard.app.test_client()
    client.post("/api/runs", json={"bbox": VALID_BBOX}, headers=TOKEN_HEADER)

    response = client.post("/api/cancel", headers=TOKEN_HEADER)

    assert response.status_code == 200
    assert response.get_json()["state"] == "cancelled"
    assert dashboard._active_job is None, (
        "after a cancel the job slot must be free, or no further run could start"
    )


def test_bad_bbox_is_rejected_before_any_process_starts(dashboard, monkeypatch):
    def _must_not_run(argv, params):  # pragma: no cover - must never be called
        raise AssertionError("a request with an invalid bbox must not start a job")

    monkeypatch.setattr(dashboard, "_launch_job", _must_not_run)
    client = dashboard.app.test_client()

    for bad in ([10.0, 20.0, 11.0], "not-a-bbox", [10.0, 20.0, 9.0, 21.0],
                [10.0, 20.0, 11.0, 200.0]):
        response = client.post("/api/runs", json={"bbox": bad}, headers=TOKEN_HEADER)
        assert response.status_code == 400, f"bbox {bad!r} must be rejected"
        assert response.get_json()["field"] == "bbox"


def test_run_status_payload_carries_no_paths_or_secrets(dashboard, monkeypatch):
    _stub_launch(dashboard, monkeypatch)
    client = dashboard.app.test_client()
    client.post("/api/runs", json={"bbox": VALID_BBOX}, headers=TOKEN_HEADER)

    body = client.get("/api/run-status").get_data(as_text=True)

    assert "\\\\" not in body and ":/" not in body, (
        "the status payload is rendered in the browser, so it must not leak a "
        f"local path: {body[:200]}"
    )


def test_redact_masks_credentials_in_log_lines(dashboard):
    masked = dashboard.redact("Authorization: Bearer abcdefghijklmnopqrstuvwxyz012345")
    assert "abcdefghijklmnopqrstuvwxyz012345" not in masked
    masked = dashboard.redact("client_secret=supersecretvalue")
    assert "supersecretvalue" not in masked, (
        "a credential printed by a download step must not reach the dashboard"
    )
