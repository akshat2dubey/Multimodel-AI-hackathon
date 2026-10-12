"""12. Dashboard contract: graceful empty state, safe defaults, gated pipeline.

Three things must hold before the pipeline has ever run:
  * every GET route answers 200 on an empty results directory (a 500 or a 404
    turns the demo into an error page the moment a judge opens it);
  * the server binds loopback with the debugger off by default;
  * POST /api/run-pipeline refuses without the shared token (it starts a
    multi-minute download job; a public port must not be able to trigger it).
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest

GET_ROUTES = [
    "/",
    "/api/status",
    "/api/flood-mask",
    "/api/damage",
    "/api/cutoffs",
    "/api/flood-path",
    "/api/situation-report",
    "/report",
    "/demo",
]


@pytest.fixture
def empty_dashboard(tmp_path, monkeypatch):
    """The dashboard pointed at empty results/data/report directories."""
    import app as app_module

    empty = tmp_path / "empty"
    (empty / "results").mkdir(parents=True)
    (empty / "data").mkdir(parents=True)
    (empty / "report").mkdir(parents=True)
    monkeypatch.setattr(app_module, "RESULTS", empty / "results")
    monkeypatch.setattr(app_module, "DATA", empty / "data")
    monkeypatch.setattr(app_module, "REPORT_DIR", empty / "report")
    app_module.app.config.update(TESTING=True)
    return app_module, empty


@pytest.mark.parametrize("route", GET_ROUTES)
def test_get_routes_return_200_on_empty_results(empty_dashboard, route):
    app_module, _ = empty_dashboard
    client = app_module.app.test_client()

    response = client.get(route)
    assert response.status_code == 200, (
        f"GET {route} returned {response.status_code} with an empty results "
        "directory; the dashboard must render a graceful empty state before the "
        f"pipeline has run.\nbody: {response.get_data(as_text=True)[:300]}"
    )


def test_absent_situation_report_is_an_empty_state_not_an_error(empty_dashboard):
    app_module, _ = empty_dashboard
    client = app_module.app.test_client()

    response = client.get("/api/situation-report")
    assert response.status_code == 200, (
        f"/api/situation-report returned {response.status_code}; a missing report "
        "is an empty state (200 + an explanatory body), not a 404"
    )
    payload = json.loads(response.get_data(as_text=True))
    assert payload.get("available") is False, (
        f"the empty-state payload must say available: false, got {payload}"
    )
    assert payload.get("note"), "the empty state must explain itself to the UI"


def test_run_pipeline_without_token_is_forbidden(empty_dashboard, monkeypatch):
    app_module, _ = empty_dashboard
    monkeypatch.setattr(app_module, "PIPELINE_TOKEN", "")
    client = app_module.app.test_client()

    response = client.post("/api/run-pipeline")
    assert response.status_code == 403, (
        f"POST /api/run-pipeline without a token returned {response.status_code}; "
        "it must be 403. This endpoint spawns the full download+compute pipeline "
        "and must fail closed when no token is configured."
    )


def test_run_pipeline_rejects_a_wrong_token(empty_dashboard, monkeypatch):
    app_module, _ = empty_dashboard
    monkeypatch.setattr(app_module, "PIPELINE_TOKEN", "correct-horse")
    client = app_module.app.test_client()

    response = client.post(
        "/api/run-pipeline", headers={app_module.PIPELINE_TOKEN_HEADER: "battery-staple"}
    )
    assert response.status_code == 403, (
        f"a wrong token must be rejected, got {response.status_code}"
    )


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_server_binds_loopback_with_debug_off_by_default(dashboard_dir, tmp_path):
    port = _free_port()
    env = dict(os.environ)
    env["PORT"] = str(port)
    env.pop("FLASK_DEBUG", None)
    env.pop("HOST", None)

    proc = subprocess.Popen(
        [sys.executable, str(dashboard_dir / "app.py")],
        cwd=str(dashboard_dir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        deadline = time.time() + 30
        ready = False
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=1) as resp:
                    if resp.status == 200:
                        ready = True
                        break
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
                time.sleep(0.3)

        assert ready, (
            f"the dashboard did not answer on http://127.0.0.1:{port}/ within 30s. "
            "It must bind loopback by default."
        )

        # It must NOT be reachable on a non-loopback interface.
        lan_ip = _non_loopback_ipv4()
        if lan_ip:
            try:
                urllib.request.urlopen(f"http://{lan_ip}:{port}/", timeout=2).read()
                reachable_on_lan = True
            except Exception:
                reachable_on_lan = False
            assert not reachable_on_lan, (
                f"the dashboard is reachable on {lan_ip}:{port} — it must bind "
                "127.0.0.1 by default so a demo box is not exposed to the network"
            )
    finally:
        proc.terminate()
        try:
            stdout, _ = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, _ = proc.communicate()

    assert "debug=False" in (stdout or ""), (
        "the startup banner must report debug=False by default; a debug server "
        "exposes the Werkzeug console to anyone who can reach the port. "
        f"banner was:\n{stdout}"
    )


def _non_loopback_ipv4():
    try:
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        return None if ip.startswith("127.") else ip
    except Exception:
        return None
