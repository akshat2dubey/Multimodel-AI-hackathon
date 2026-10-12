"""T20 — rate limit + resource limits.

The dashboard has no login, so anything that starts work must be bounded: a
burst of POSTs, a huge body, a hung job, an oversized cache. These tests exercise
each bound through the Flask test client / the job helpers, with the subprocess
replaced by a dummy so nothing downloads.
"""

import importlib
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "dashboard"))
sys.path.insert(0, str(REPO_ROOT))


FAKE_RUN_ID = "22222222-2222-2222-2222-222222222222"
# A complete request: a custom AOI needs the area name, the OSM snapshot and the
# two acquisition dates as well as the bbox, otherwise it is a 400 and never
# reaches the launch (and so would never spend rate-limit budget).
VALID_BODY = {
    "bbox": "10,20,11,21",
    "area": "toy area",
    "event_date": "2026-08-26",
    "osm_snapshot": "2026-07-27",
    "before_date": "2026-08-14",
    "after_date": "2026-08-26",
}


@pytest.fixture()
def app_module(tmp_path, monkeypatch):
    """A fresh import of dashboard/app.py with limits shrunk and results local."""
    monkeypatch.setenv("JALNETRA_PIPELINE_TOKEN", "test-token")
    monkeypatch.setenv("JALNETRA_RATE_POSTS", "3")
    monkeypatch.setenv("JALNETRA_MAX_BODY", "2048")
    monkeypatch.setenv("JALNETRA_JOB_TIMEOUT", "2")
    monkeypatch.setenv("JALNETRA_CACHE_MAX_MB", "1")
    monkeypatch.setenv("JALNETRA_MIN_FREE_MB", "0")
    for name in ("app",):
        sys.modules.pop(name, None)
    module = importlib.import_module("app")
    module.app.config["TESTING"] = True
    # Limits are the subject here, not the pipeline: keep results/ untouched and
    # never spawn a real job (an accepted POST would otherwise start one and
    # write results/run_status.json).
    monkeypatch.setattr(module, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(module, "_post_times", {})

    def _stub_launch(argv, params):
        job = {"run_id": FAKE_RUN_ID, "state": "running", "step": "starting",
               "progress": 0.0, "started_utc": "2026-10-13T00:00:00+00:00",
               "finished_utc": None, "exit_code": None, "params": params,
               "log_tail": []}
        monkeypatch.setattr(module, "_active_job", job)
        return job

    monkeypatch.setattr(module, "_launch_job", _stub_launch)
    return module


def _post(client, token="test-token", body=None):
    headers = {"X-Jalnetra-Token": token, "Origin": "http://localhost"}
    return client.post("/api/runs", json=VALID_BODY if body is None else body,
                       headers=headers)


def test_rate_limit_returns_429(app_module):
    """A burst past the configured limit is refused with 429, not queued.

    Each iteration frees the job slot first, so the only thing that can refuse a
    request is the rate limiter (the single-job 409 is test_19's subject).
    """
    client = app_module.app.test_client()
    codes = []
    for _ in range(app_module.RATE_LIMIT_POSTS + 2):
        app_module._active_job = None
        codes.append(_post(client).status_code)
    assert codes == [202] * app_module.RATE_LIMIT_POSTS + [429, 429], codes


def test_oversized_body_returns_413(app_module):
    """A body over MAX_CONTENT_LENGTH is refused before it is parsed."""
    client = app_module.app.test_client()
    big = dict(VALID_BODY, pad="x" * 4096)
    response = client.post(
        "/api/runs",
        data=__import__("json").dumps(big),
        content_type="application/json",
        headers={"X-Jalnetra-Token": "test-token", "Origin": "http://localhost"},
    )
    assert response.status_code == 413


def test_job_timeout_kills_and_labels(app_module, monkeypatch):
    """A job that outlives the ceiling is killed and marked `timeout`."""
    monkeypatch.setattr(app_module, "JOB_TIMEOUT_SECONDS", 1.0)
    job = {"run_id": "dummy", "state": "running", "step": "starting"}
    import subprocess

    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        app_module._arm_timeout(proc, job)
        deadline = time.monotonic() + 15
        while proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.2)
        assert proc.poll() is not None, "watchdog never stopped the job"
        assert job["state"] == "timeout"
        assert "exceeded" in job["error"]
    finally:
        if proc.poll() is None:
            proc.kill()


def test_capacity_error_blocks_when_disk_is_low(app_module, monkeypatch):
    """Disk floor measured in bytes: below it the run is refused with a message."""
    monkeypatch.setattr(app_module, "MIN_FREE_BYTES", 10 ** 18)
    message = app_module._capacity_error()
    assert message and "free disk space" in message


def test_cache_pruned_to_cap(app_module, tmp_path, monkeypatch):
    """Oldest cache files are removed until the cache is under the cap."""
    cache = tmp_path / "cache"
    cache.mkdir()
    old, new = cache / "old.bin", cache / "new.bin"
    old.write_bytes(b"x" * 600)
    new.write_bytes(b"y" * 600)
    import os

    os.utime(old, (1, 1))                     # unambiguously older
    monkeypatch.setattr(app_module, "CACHE_DIR", cache)
    monkeypatch.setattr(app_module, "CACHE_MAX_BYTES", 1000)
    removed = app_module._prune_cache()
    assert removed == 1
    assert not old.exists() and new.exists()


def test_capacity_error_never_blocks_when_unmeasurable(app_module, monkeypatch):
    """An unmeasurable disk must not turn into a refusal (fail-open on check)."""
    def boom(_path):
        raise OSError("no such device")

    monkeypatch.setattr(app_module.shutil, "disk_usage", boom)
    assert app_module._capacity_error() is None
