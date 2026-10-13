"""T38 — the dashboard login gate.

The dashboard serves flood results and can start a multi-minute download job, so
the gate has to hold on three fronts at once: pages redirect to the form, API
callers get a 401 they can act on, and the form itself must not be usable as a
password oracle or an open redirect. These tests drive the real Flask app.

`JALNETRA_AUTH=off` remains the documented no-login mode for a loopback demo and
is covered last, so turning the gate off stays a deliberate choice.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "dashboard"))
sys.path.insert(0, str(REPO_ROOT))

import app as app_module  # noqa: E402

PASSWORD = "correct-horse-battery-staple"

PAGES = ("/", "/report", "/demo")
API_ROUTES = ("/api/status", "/api/damage", "/api/cutoffs")


def test_wsgi_import_defaults_to_authentication(monkeypatch):
    """A WSGI import must fail closed even though main() is never called."""
    monkeypatch.delenv("JALNETRA_AUTH", raising=False)
    monkeypatch.delenv("JALNETRA_DASHBOARD_PASSWORD", raising=False)
    auth = app_module._Auth()
    assert auth.enabled is True
    assert auth.password == ""


def test_missing_wsgi_password_cannot_be_bypassed_with_blank_login(gated, monkeypatch):
    monkeypatch.setattr(gated.AUTH, "password", "")
    client = gated.app.test_client()
    response = client.post(
        "/login", data={"password": ""}, headers={"Origin": "http://localhost"}
    )
    assert "Incorrect password" in response.get_data(as_text=True)
    assert client.get("/").status_code == 302


@pytest.fixture
def gated(monkeypatch, tmp_path):
    """The dashboard with the login gate ON, pointed at empty result dirs."""
    empty = tmp_path / "empty"
    (empty / "results").mkdir(parents=True)
    (empty / "data").mkdir(parents=True)
    (empty / "report").mkdir(parents=True)
    monkeypatch.setattr(app_module, "RESULTS", empty / "results")
    monkeypatch.setattr(app_module, "DATA", empty / "data")
    monkeypatch.setattr(app_module, "REPORT_DIR", empty / "report")
    monkeypatch.setattr(app_module, "LOGIN_REQUIRED", True)
    monkeypatch.setattr(app_module.AUTH, "password", PASSWORD)
    monkeypatch.setattr(app_module.AUTH, "enabled", True)
    monkeypatch.setattr(app_module, "_login_attempts", {})
    app_module.app.config.update(TESTING=True)
    return app_module


def _sign_in(client, password: str = PASSWORD, **kwargs):
    return client.post("/login", data={"password": password}, **kwargs)


def test_pages_redirect_to_the_login_form_and_the_api_answers_401(gated):
    """A browser can follow the redirect; a script gets a status it can branch on."""
    client = gated.app.test_client()

    for route in PAGES:
        response = client.get(route)
        assert response.status_code == 302, f"GET {route} was not gated"
        assert "/login" in response.headers["Location"]

    for route in API_ROUTES:
        response = client.get(route)
        assert response.status_code == 401, f"GET {route} leaked without a session"
        assert response.get_json().get("status") == 401


def test_the_sign_in_page_and_its_assets_need_no_session(gated):
    """The form cannot depend on a session it exists to create."""
    client = gated.app.test_client()
    page = client.get("/login")
    assert page.status_code == 200
    body = page.get_data(as_text=True)
    assert 'name="password"' in body and 'type="password"' in body
    assert client.get("/static/vendor/leaflet/leaflet.css").status_code != 302


def test_a_wrong_password_leaves_the_caller_signed_out(gated):
    client = gated.app.test_client()
    response = _sign_in(client, "not-the-password")

    assert response.status_code == 200
    assert "Incorrect password" in response.get_data(as_text=True)
    assert client.get("/").status_code == 302, "a failed login opened a session"


def test_the_right_password_opens_a_session_for_pages_and_api(gated):
    client = gated.app.test_client()
    response = _sign_in(client)

    assert response.status_code == 302
    assert response.headers["Location"] == "/"
    assert client.get("/").status_code == 200
    assert client.get("/api/status").status_code == 200


def test_a_foreign_origin_cannot_use_the_form_as_a_password_oracle(gated):
    """A correct password submitted cross-origin must not sign that caller in."""
    client = gated.app.test_client()
    response = _sign_in(client, headers={"Origin": "https://evil.example"})

    assert response.status_code == 403
    assert client.get("/").status_code == 302


def test_an_opaque_origin_can_still_sign_in(gated):
    """Regression (found live): `Origin: null` locked the form out entirely.

    An embedded browser view or a sandboxed context sends the opaque origin, and
    rejecting it made the sign-in form impossible to submit — a 403 the user
    could do nothing about. The opaque origin names no site, so it is accepted on
    the cookie-session routes and logged.
    """
    client = gated.app.test_client()
    response = _sign_in(client, headers={"Origin": "null"})

    assert response.status_code == 302
    assert client.get("/").status_code == 200


def test_an_opaque_origin_still_cannot_start_a_pipeline_run(gated):
    """The strict check stays on the routes that change server state.

    Checked *after* signing in, so the 403 comes from the origin rule rather than
    from the login gate (which would answer 401 first).
    """
    client = gated.app.test_client()
    _sign_in(client)
    assert client.get("/").status_code == 200

    response = client.post("/api/run-pipeline", headers={"Origin": "null"})
    assert response.status_code == 403, (
        "an opaque-origin POST must not reach a state-changing route"
    )
    assert "cross-origin" in str(response.get_json().get("error", ""))


def test_the_next_parameter_cannot_send_a_user_off_origin(gated):
    """`//evil.example` is protocol-relative: it must fall back to this origin."""
    client = gated.app.test_client()
    response = client.post(
        "/login?next=//evil.example/steal", data={"password": PASSWORD}
    )
    assert response.status_code == 302
    assert response.headers["Location"] == "/"


def test_repeated_failures_are_rate_limited(gated):
    """Bounded guessing: even the correct password waits once the limit is hit."""
    client = gated.app.test_client()
    for _ in range(gated.LOGIN_MAX_ATTEMPTS):
        client.post("/login", data={"password": "guess"})

    blocked = _sign_in(client)
    assert blocked.status_code == 429
    assert client.get("/").status_code == 302


def test_signing_out_is_post_only_and_ends_the_session(gated):
    client = gated.app.test_client()
    _sign_in(client)
    assert client.get("/").status_code == 200

    assert client.get("/logout").status_code == 405, "a GET link could sign a user out"
    assert client.post("/logout").status_code == 302
    assert client.get("/").status_code == 302


def test_the_session_cookie_is_not_readable_by_scripts(gated):
    """The credential the browser holds must be HttpOnly + SameSite."""
    client = gated.app.test_client()
    response = _sign_in(client)
    cookie = response.headers.get("Set-Cookie", "")
    assert "HttpOnly" in cookie
    assert "SameSite=Lax" in cookie
    assert "Secure" in cookie


def test_secure_cookie_can_be_disabled_for_loopback_http(monkeypatch):
    assert app_module.app.config["SESSION_COOKIE_SECURE"] is True


def test_auth_off_is_the_documented_loopback_mode(monkeypatch, tmp_path):
    """With the gate deliberately off, nothing changes for a local demo."""
    empty = tmp_path / "empty"
    (empty / "results").mkdir(parents=True)
    monkeypatch.setattr(app_module, "RESULTS", empty / "results")
    monkeypatch.setattr(app_module, "LOGIN_REQUIRED", False)
    app_module.app.config.update(TESTING=True)

    client = app_module.app.test_client()
    assert client.get("/").status_code == 200
    assert client.get("/login").status_code == 302   # nothing to sign in to
