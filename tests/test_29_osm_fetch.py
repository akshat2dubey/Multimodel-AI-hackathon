"""T37 — an OSM fetch that FAILED must never look like an empty layer.

Reproduced live on 2026-10-13: the primary Overpass endpoint answered the
building query for the run's AOI with HTTP 504, `run_attic_query` turned that
into `[]`, `buildings.geojson` was written as an empty FeatureCollection, and the
run reported "0 / 0 buildings affected" for a corridor whose official OSM count
at the snapshot date is 3755 (ohsome `building=*`) — the same query answered by
overpass.kumi.systems returns 3745 elements.

These tests pin the three defences: mirror fallback, "failed" being distinct from
"empty", and a failed required layer stopping the run instead of being written
out as nothing.
"""

from __future__ import annotations

import io
import json
import sys
import urllib.error
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

import fetch_osm  # noqa: E402

BBOX = (85.4, 28.0, 86.0, 28.6)
SNAPSHOT = "2026-07-27"

# The selectors fetch_osm_features uses, so the stub can answer per category.
# An unrecognised selector returns an empty answer rather than raising, which
# keeps a future edit to a selector from turning into an opaque test error.
SELECTOR = {
    "buildings": "[building]",
    "roads": ('[highway~"motorway|trunk|primary|secondary|tertiary|unclassified|'
              'residential|living_street|service|track|path|footway|bridleway"]'),
    "bridges": "[bridge=yes]",
    "hospitals": '[amenity~"hospital|clinic|doctors"]',
    "settlements": '[place~"village|hamlet|town|city"]',
}


class _Response:
    """Minimal stand-in for the http.client response urlopen() returns."""

    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode("utf-8")
        self.headers = {"Content-Type": "application/json"}

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _http_error(code: int, url: str) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url, code, "server error", {}, io.BytesIO(b""))


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch):
    """Keep the retry/backoff schedule out of the test's running time."""
    monkeypatch.setattr("time.sleep", lambda _seconds: None)


def _ways(n: int) -> list[dict]:
    return [
        {"type": "way", "id": 1000 + i,
         "geometry": [{"lon": 85.5, "lat": 28.1}, {"lon": 85.51, "lat": 28.11}],
         "tags": {"building": "yes"}}
        for i in range(n)
    ]


def _run_fetch(tmp_path, *, failed=(), elements=None, official=None):
    """Run fetch_osm_features with the network and the ohsome count stubbed.

    `failed` names the categories whose fetch fails at every endpoint (None from
    run_attic_query); `elements` gives the element count for the rest. Returns
    (paths, failed_required, selectors actually queried).
    """
    elements = elements or {}
    official = official or {}
    calls: list[str] = []

    def fake_attic(selector, bbox, snapshot_date, timeout=180):
        calls.append(selector)
        for label, sel in SELECTOR.items():
            if sel == selector:
                if label in failed:
                    return None
                return _ways(elements.get(label, 0))
        return []

    def fake_ohsome(label, filter_expr, bbox, snapshot_date):
        return official.get(label)

    original = (fetch_osm.run_attic_query, fetch_osm.ohsome_count, fetch_osm.ohsome_v2_features)
    fetch_osm.run_attic_query, fetch_osm.ohsome_count = fake_attic, fake_ohsome
    fetch_osm.ohsome_v2_features = lambda *_: None
    try:
        paths, failed_required = fetch_osm.fetch_osm_features(
            center_lat=28.3, center_lon=85.7, snapshot_date=SNAPSHOT,
            bbox=BBOX, output_dir=tmp_path,
        )
    finally:
        fetch_osm.run_attic_query, fetch_osm.ohsome_count, fetch_osm.ohsome_v2_features = original
    return paths, failed_required, calls


def _meta(tmp_path) -> dict:
    return json.loads((tmp_path / "osm_metadata.json").read_text(encoding="utf-8"))


def test_a_busy_endpoint_falls_through_to_the_next_mirror(monkeypatch):
    """A 504 from one endpoint must not end the query."""
    asked: list[str] = []

    def fake_urlopen(req, timeout=None):
        asked.append(req.full_url)
        if "overpass-api.de" in req.full_url:
            raise _http_error(504, req.full_url)
        return _Response({"elements": [{"type": "way", "id": 1}]})

    monkeypatch.setattr(fetch_osm, "urlopen", fake_urlopen)
    payload, endpoint = fetch_osm.overpass_ql("[out:json];way[building];out;")

    assert payload is not None, "the mirror's answer was discarded"
    assert "overpass-api.de" in asked[0]
    assert endpoint and "overpass-api.de" not in endpoint
    assert len(payload["elements"]) == 1


def test_every_endpoint_failing_returns_none_not_an_empty_result(monkeypatch):
    """(None, None) is the only honest answer when no endpoint replies."""
    def fake_urlopen(req, timeout=None):
        raise _http_error(504, req.full_url)

    monkeypatch.setattr(fetch_osm, "urlopen", fake_urlopen)
    payload, endpoint = fetch_osm.overpass_ql("[out:json];way[building];out;")
    assert payload is None and endpoint is None


def test_run_attic_query_distinguishes_failure_from_an_empty_answer(monkeypatch):
    monkeypatch.setattr(fetch_osm, "overpass_ql", lambda q, timeout=0: (None, None))
    assert fetch_osm.run_attic_query("[building]", BBOX, SNAPSHOT) is None

    monkeypatch.setattr(fetch_osm, "overpass_ql",
                        lambda q, timeout=0: ({"elements": []}, "https://x/api"))
    assert fetch_osm.run_attic_query("[building]", BBOX, SNAPSHOT) == []


def test_a_failed_building_fetch_writes_no_empty_layer_and_reports_failure(tmp_path):
    """The exact live failure: 504 everywhere, and no prior file on disk.

    The failure is total (no endpoint answered), so the circuit breaker stops
    asking: roads is reported as skipped rather than sweeping the server again,
    which is what held the pipeline for minutes on end.
    """
    _, failed, calls = _run_fetch(
        tmp_path,
        failed={"buildings"},
        elements={"roads": 3},
        official={"buildings": 3755, "roads": 719},
    )

    assert set(failed) == {"buildings", "roads"}
    assert calls == [SELECTOR["buildings"]], "kept querying a dead server"
    assert not (tmp_path / "buildings.geojson").exists(), (
        "a failed fetch must not leave an empty layer for downstream to read as 0"
    )
    meta = _meta(tmp_path)
    assert meta["fetch_status"]["buildings"] == "failed"
    assert meta["feature_counts"]["buildings"] is None
    assert meta["ohsome_counts"]["buildings"] == 3755
    assert meta["failed_required_categories"] == ["buildings", "roads"]
    assert meta["fetch_status"]["roads"] == "skipped"
    assert not (tmp_path / "roads.geojson").exists()


def test_a_failed_fetch_keeps_the_previous_good_layer(tmp_path):
    """A good file from an earlier run must not be replaced by nothing."""
    existing = tmp_path / "buildings.geojson"
    existing.write_text(json.dumps(
        {"type": "FeatureCollection", "features": _ways(5)}
    ), encoding="utf-8")

    _, failed, _ = _run_fetch(
        tmp_path,
        failed={"buildings"},
        elements={"roads": 3},
        official={"buildings": 3755, "roads": 719},
    )

    assert "buildings" in failed
    kept = json.loads(existing.read_text(encoding="utf-8"))
    assert len(kept["features"]) == 5, "the previous building layer was overwritten"
    meta = _meta(tmp_path)
    assert meta["fetch_status"]["buildings"] == "failed_kept_previous"
    assert meta["feature_counts"]["buildings"] == 5


def test_an_extract_that_contradicts_the_official_count_is_a_failure(tmp_path):
    """Empty extract but ohsome says 4 hospitals: the fetch lied, so it failed."""
    _, failed, _ = _run_fetch(
        tmp_path,
        elements={"roads": 3, "buildings": 2, "hospitals": 0},
        official={"buildings": 3755, "roads": 719, "hospitals": 4},
    )

    meta = _meta(tmp_path)
    assert meta["fetch_status"]["hospitals"] == "failed"
    assert meta["feature_counts"]["hospitals"] is None
    assert not (tmp_path / "hospitals.geojson").exists()
    # Not a required layer, so the run may continue without it — but the
    # contradiction is recorded rather than hidden.
    assert "hospitals" not in failed


def test_a_genuinely_empty_optional_layer_is_allowed(tmp_path):
    """No hospitals in the bbox is a real answer and must be written as 0."""
    _, failed, _ = _run_fetch(
        tmp_path,
        elements={"roads": 3, "buildings": 2, "hospitals": 0},
        official={"buildings": 3755, "roads": 719, "hospitals": 0},
    )

    assert failed == []
    payload = json.loads((tmp_path / "hospitals.geojson").read_text(encoding="utf-8"))
    assert payload["features"] == []
    meta = _meta(tmp_path)
    assert meta["fetch_status"]["hospitals"] == "ok"
    assert meta["feature_counts"]["hospitals"] == 0


def test_the_official_count_uses_valid_ohsome_filter_syntax(monkeypatch):
    """`type=node` returns 0 without erroring; only `type:node` is a real check."""
    seen: list[str] = []

    def fake_urlopen(req, timeout=None):
        seen.append(req.full_url)
        return _Response({"result": [{"value": 8124.0}]})

    monkeypatch.setattr(fetch_osm, "urlopen", fake_urlopen)
    value = fetch_osm.ohsome_count("all tagged elements",
                                   fetch_osm.OHSOME_ALL_TAGGED_FILTER,
                                   BBOX, SNAPSHOT)

    assert value == 8124
    assert "type%3Anode" in seen[0] or "type:node" in seen[0]
    assert "type%3Dnode" not in seen[0]
