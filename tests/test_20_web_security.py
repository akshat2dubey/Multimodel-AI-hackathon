"""20. Web security: headers, CSP nonce, no third-party scripts, escaped OSM text (T21).

The dashboard renders OpenStreetMap tags. Anybody can edit an OSM `name` tag, so
a template that interpolates one straight into `innerHTML` or a map popup turns
vandalism into script execution in whoever opens the dashboard. Escaping is the
fix; the CSP is the backstop; both are checked here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

TEMPLATES = Path(__file__).resolve().parent.parent / "dashboard" / "templates"
STATIC_VENDOR = Path(__file__).resolve().parent.parent / "dashboard" / "static" / "vendor"


@pytest.fixture
def client(monkeypatch):
    import app as app_module

    monkeypatch.setattr(app_module, "LOGIN_REQUIRED", False)
    app_module.app.config.update(TESTING=True)
    return app_module.app.test_client()


def _html_templates():
    return sorted(TEMPLATES.glob("*.html"))


def test_html_responses_carry_the_security_headers(client):
    response = client.get("/")

    csp = response.headers.get("Content-Security-Policy", "")
    assert "default-src 'self'" in csp, f"no usable CSP: {csp!r}"
    assert "frame-ancestors 'none'" in csp, "the dashboard must not be framable"
    assert "script-src 'self' 'nonce-" in csp, (
        "scripts must be limited to this origin plus the page's own nonce"
    )
    assert "basemaps.cartocdn.com" in csp, (
        "the map tiles are the one allowed third-party resource — if the tile host "
        "changed, the CSP must change with it"
    )
    assert response.headers.get("X-Content-Type-Options") == "nosniff"
    assert response.headers.get("X-Frame-Options") == "DENY"
    assert response.headers.get("Referrer-Policy") == "no-referrer"


def test_api_responses_are_not_cacheable(client):
    response = client.get("/api/status")
    assert response.headers.get("Cache-Control") == "no-store", (
        "results change on every run; a cached /api response would show stale "
        "numbers to a judge"
    )


def test_nonce_is_unique_per_response_and_matches_the_page(client):
    first = client.get("/")
    second = client.get("/")

    def nonce_of(response):
        match = re.search(r"'nonce-([^']+)'", response.headers.get("Content-Security-Policy", ""))
        assert match, "the CSP must carry a nonce"
        return match.group(1)

    first_nonce, second_nonce = nonce_of(first), nonce_of(second)
    assert first_nonce != second_nonce, "a reused nonce is a reusable allowance"

    body = first.get_data(as_text=True)
    assert f'nonce="{first_nonce}"' in body, (
        "the page's own inline script must carry the nonce the CSP granted, or "
        "the dashboard will not run at all under the policy"
    )


def test_templates_load_no_third_party_scripts_or_styles():
    """Nothing may be *loaded* from another origin.

    Plain links are allowed and expected: the map tiles carry their required OSM
    and CARTO attribution as <a href> to their licence pages, which loads nothing.
    Only resource-loading tags (script/link/img/iframe) are checked.
    """
    offenders = []
    for template in _html_templates():
        text = template.read_text(encoding="utf-8")
        for tag, _attr, url in re.findall(
            r'<(\w+)\b[^>]*?\b(src|href)\s*=\s*"([^"]+)"', text
        ):
            if tag.lower() == "a":
                continue  # an attribution link, not a resource
            if url.startswith(("http://", "https://", "//")):
                offenders.append(f"{template.name}: <{tag}> {url}")
    assert not offenders, (
        "every asset must be served from this origin so a strict CSP holds and the "
        "demo works offline (tile images are the one allowed exception, and they are "
        "requested by JavaScript, not by a tag):\n  " + "\n  ".join(offenders)
    )


def test_every_inline_script_has_a_nonce():
    """Any script without a nonce would be blocked by our own CSP."""
    problems = []
    for template in _html_templates():
        text = template.read_text(encoding="utf-8")
        for tag in re.findall(r"<script\b[^>]*>", text):
            has_nonce = "nonce=" in tag
            has_src = "src=" in tag
            if not has_src and not has_nonce:
                problems.append(f"{template.name}: {tag}")
    assert not problems, (
        "inline <script> tags must carry nonce=\"{{ csp_nonce }}\" or the CSP blocks "
        "them:\n  " + "\n  ".join(problems)
    )


def test_osm_text_is_escaped_before_it_reaches_html():
    text = (TEMPLATES / "index.html").read_text(encoding="utf-8")

    assert "function escapeHtml(" in text, "the dashboard must define an escaper"

    # These are the OSM-controlled values that reach innerHTML / bindPopup. They
    # must only ever appear wrapped in escapeHtml(...).
    for raw in ("${v.name}", "${nearest}", "${f.properties.name}",
                "${v.nearest_reachable_town_name}"):
        assert raw not in text, (
            f"{raw} is interpolated into HTML without escaping — an OSM name tag "
            "could inject markup"
        )
    assert "escapeHtml(v.name)" in text
    assert "escapeHtml(f.properties.name)" in text


def test_leaflet_is_vendored_and_served_locally(client):
    response = client.get("/static/vendor/leaflet/leaflet.js")

    assert response.status_code == 200, (
        "Leaflet must be served from this origin; the CSP forbids the CDN"
    )
    assert (STATIC_VENDOR / "leaflet" / "leaflet.css").is_file()

    for image in ("marker-icon.png", "marker-shadow.png", "layers.png"):
        path = STATIC_VENDOR / "leaflet" / "images" / image
        assert path.is_file(), f"missing vendored Leaflet asset {image}"
        assert path.read_bytes()[:4] == b"\x89PNG", f"{image} is not a real PNG"
