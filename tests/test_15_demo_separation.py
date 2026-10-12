"""15. The synthetic fixture lives in results_demo/ + data_demo/ — nowhere else.

Two failure modes this guards, both silent:

  * a demo run writing into results/, so a judge opening the dashboard sees
    fabricated numbers in the same directory a real run fills;
  * a demo run overwriting data/, which holds real downloads (the 258 MB
    Copernicus DEM among them) that cannot be re-fetched without an account.

The generator therefore refuses to write into a directory it does not own, and
the dashboard must say out loud when it is serving the fixture.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


def _run_make_demo_data(root: Path, *extra: str) -> int:
    """Invoke the generator in-process, with `root` as the project root.

    Small scene (64 px over 0.05 deg) so the test stays sub-second; the geometry
    is irrelevant here, only where the bytes land.
    """
    import make_demo_data

    argv = [
        "make_demo_data.py",
        "--root", str(root),
        "--size", "64",
        "--extent", "0.05",
        *extra,
    ]
    original = sys.argv
    sys.argv = argv
    try:
        return make_demo_data.main() or 0
    finally:
        sys.argv = original


# --- where the generator writes ---------------------------------------------

def test_generator_writes_the_demo_trees_and_never_results_or_data(tmp_path: Path):
    assert _run_make_demo_data(tmp_path) == 0

    demo_results = tmp_path / "results_demo"
    demo_data = tmp_path / "data_demo"
    assert (demo_results / "flood_mask.tif").is_file(), (
        "the fixture mask must be written to results_demo/flood_mask.tif"
    )
    assert (demo_results / "DEMO_DATA_NOTICE.txt").is_file()
    assert (demo_data / "dem" / "dem.tif").is_file()
    assert (demo_data / "osm" / "buildings.geojson").is_file()

    for real in ("results", "data"):
        assert not (tmp_path / real).exists(), (
            f"the generator created {tmp_path / real}. A synthetic run must never "
            "populate the real results/ or data/ trees: the dashboard serves "
            "results/ for real runs, and data/ holds downloads that cannot be "
            "replaced without a Copernicus account."
        )


def test_rerunning_the_generator_may_overwrite_its_own_output(tmp_path: Path):
    assert _run_make_demo_data(tmp_path) == 0
    assert _run_make_demo_data(tmp_path) == 0, (
        "a second run into the same demo directory must succeed — the guard is "
        "about provenance, not about the directory being empty"
    )


def test_demo_directory_names_are_defined_apart_from_the_real_ones(repo_root: Path):
    from config import presets

    assert Path(presets.RESULTS_DEMO_DIR) == repo_root / "results_demo"
    assert Path(presets.DATA_DEMO_DIR) == repo_root / "data_demo"
    assert Path(presets.RESULTS_DIR) == repo_root / "results"
    assert Path(presets.DATA_DIR) == repo_root / "data"
    assert presets.RESULTS_DEMO_NAME != presets.RESULTS_NAME
    assert presets.DATA_DEMO_NAME != presets.DATA_DIR.name


# --- the generator refuses to destroy real inputs ---------------------------

def test_generator_refuses_to_overwrite_files_it_did_not_create(
    tmp_path: Path, repo_root: Path
):
    real_data = tmp_path / "data"
    (real_data / "dem").mkdir(parents=True)
    real_dem = real_data / "dem" / "dem.tif"
    payload = b"REAL Copernicus WorldDEM-30 tile - not reproducible offline"
    real_dem.write_bytes(payload)
    (real_data / "sentinel1" / "after").mkdir(parents=True)
    real_scene = real_data / "sentinel1" / "after" / "s1_after_VV.tif"
    real_scene.write_bytes(b"real Sentinel-1 GRD")

    assert _run_make_demo_data(tmp_path, "--data-dir", str(real_data)) == 1, (
        "generating into a directory that already holds inputs it did not create "
        "must fail, not overwrite them"
    )

    assert real_dem.read_bytes() == payload
    assert real_scene.read_bytes() == b"real Sentinel-1 GRD"
    assert not (real_data / ".generated_by_make_demo_data").exists(), (
        "a refused run must not claim ownership of a directory it did not write"
    )
    assert not (tmp_path / "results_demo" / "flood_mask.tif").exists(), (
        "the guard must run before any output is written"
    )

    # ... and the CLI path (what an operator actually runs) exits non-zero too.
    completed = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "make_demo_data.py"),
         "--root", str(tmp_path), "--data-dir", str(real_data)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    assert completed.returncode == 1, (
        "the CLI must exit non-zero when it refuses; a silent zero would let a "
        f"scripted run continue on stale data. stdout/stderr:\n{completed.stdout}"
        f"\n{completed.stderr}"
    )
    assert "refusing" in (completed.stderr + completed.stdout).lower()
    assert real_dem.read_bytes() == payload


def test_generator_refuses_to_write_into_the_real_results_tree(tmp_path: Path):
    assert _run_make_demo_data(tmp_path, "--results-dir", str(tmp_path / "results")) == 1
    assert not (tmp_path / "results").exists(), (
        "results/ is reserved for real pipeline output; a demo must not be able to "
        "write there at all"
    )


def test_force_is_available_but_explicit(tmp_path: Path):
    real_data = tmp_path / "data"
    (real_data / "dem").mkdir(parents=True)
    (real_data / "dem" / "dem.tif").write_bytes(b"placeholder")

    assert _run_make_demo_data(
        tmp_path, "--data-dir", str(real_data), "--force"
    ) == 0, "--force must be the documented escape hatch for an intentional overwrite"


# --- the offline runner stays inside the demo trees -------------------------

def test_run_offline_keeps_every_stage_out_of_results_and_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, repo_root: Path
):
    import run_offline

    data = tmp_path / "data_demo"
    results = tmp_path / "results_demo"
    (data / "osm").mkdir(parents=True)
    (data / "dem").mkdir(parents=True)
    for name in ("buildings", "roads", "bridges", "hospitals", "settlements"):
        (data / "osm" / f"{name}.geojson").write_text("{}", encoding="utf-8")
    (data / "dem" / "dem.tif").write_bytes(b"stub")
    results.mkdir(parents=True)
    (results / "flood_mask.tif").write_bytes(b"stub")

    monkeypatch.setattr(run_offline, "DATA_DEMO_DIR", data)
    monkeypatch.setattr(run_offline, "RESULTS_DEMO_DIR", results)

    commands: list[list[str]] = []

    def fake_run(cmd, **kwargs):  # noqa: ANN001 - mimics subprocess.run
        assert kwargs.get("cwd") == run_offline.ROOT
        commands.append([str(part) for part in cmd])
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(run_offline.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["run_offline.py"])

    assert run_offline.main() == 0
    assert len(commands) == 5, (
        f"expected the 5 offline stages, captured {len(commands)}: {commands}"
    )

    generator_stage = commands[0]
    assert "--data-dir" in generator_stage and str(data) in generator_stage
    assert "--results-dir" in generator_stage and str(results) in generator_stage

    report_stage = next(c for c in commands if "generate_report.py" in " ".join(c))
    assert str(results / "situation_report.html") in report_stage, (
        "the offline situation report must be written into the demo results "
        "directory: it states that run's numbers, and overwriting the published "
        "report in report/ with synthetic figures is the same class of mistake as "
        f"overwriting the real results/. Got: {report_stage}"
    )

    # No argument of any stage may resolve inside the repository's real trees.
    forbidden = {
        (repo_root / "results").resolve(): "results/",
        (repo_root / "data").resolve(): "data/",
    }
    for cmd in commands:
        for arg in cmd:
            if arg.startswith("-"):
                continue
            try:
                resolved = Path(arg).resolve()
            except OSError:  # pragma: no cover - defensive
                continue
            for real, label in forbidden.items():
                assert resolved != real and real not in resolved.parents, (
                    f"the offline pipeline passes {arg!r} into the real {label} "
                    f"tree ({resolved}). Every offline stage must read and write "
                    "inside data_demo/ and results_demo/."
                )


# --- the dashboard says so out loud -----------------------------------------

def test_dashboard_reports_demo_mode_only_for_the_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import app as app_module
    from config import presets

    app_module.app.config.update(TESTING=True)
    client = app_module.app.test_client()

    demo = tmp_path / presets.RESULTS_DEMO_NAME
    demo.mkdir()
    (demo / presets.DEMO_NOTICE_FILENAME).write_text("synthetic\n", encoding="utf-8")
    monkeypatch.setattr(app_module, "RESULTS", demo)

    payload = json.loads(client.get("/api/status").get_data(as_text=True))
    assert payload["demo_data"] is True, (
        "the dashboard must flag the demo fixture in /api/status so the UI can "
        f"show a banner; got {payload}"
    )
    assert payload["results_dir"] == presets.RESULTS_DEMO_NAME, (
        "the status payload must name the directory being served, so the banner "
        "can tell the operator which one it is"
    )

    real = tmp_path / presets.RESULTS_NAME
    real.mkdir()
    monkeypatch.setattr(app_module, "RESULTS", real)

    payload = json.loads(client.get("/api/status").get_data(as_text=True))
    assert payload["demo_data"] is False, (
        f"a real results/ directory must not be labelled demo: {payload}"
    )


def test_dashboard_serves_the_report_that_belongs_to_the_served_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import app as app_module
    from config import presets

    results = tmp_path / presets.RESULTS_DEMO_NAME
    results.mkdir()
    (results / presets.DEMO_NOTICE_FILENAME).write_text("synthetic\n", encoding="utf-8")
    (results / "situation_report.html").write_text(
        "<html><body><div class=\"page\">DEMO REPORT</div></body></html>", encoding="utf-8"
    )
    handouts = tmp_path / "report"
    handouts.mkdir()
    (handouts / "situation_report.html").write_text(
        "<html><body><div class=\"page\">REAL HANDOUT</div></body></html>", encoding="utf-8"
    )

    app_module.app.config.update(TESTING=True)
    monkeypatch.setattr(app_module, "RESULTS", results)
    monkeypatch.setattr(app_module, "REPORT_DIR", handouts)
    client = app_module.app.test_client()

    body = client.get("/report").get_data(as_text=True)
    assert "DEMO REPORT" in body and "REAL HANDOUT" not in body, (
        "when the dashboard serves results_demo/ it must show the demo's report, "
        "not the published handout copy"
    )

    real = tmp_path / presets.RESULTS_NAME
    real.mkdir()
    monkeypatch.setattr(app_module, "RESULTS", real)
    assert "REAL HANDOUT" in client.get("/report").get_data(as_text=True), (
        "a results directory with no report of its own must fall back to the "
        "report/ copy, so a real run still shows a report"
    )


def test_dashboard_template_carries_a_visible_demo_banner(dashboard_dir: Path):
    html = (dashboard_dir / "templates" / "index.html").read_text(encoding="utf-8")

    assert 'id="demo-banner"' in html, (
        "index.html must contain a demo banner element to render"
    )
    assert "DEMO DATA" in html.upper(), "the banner must say what it is"
    assert 'id="demo-dir"' in html, (
        "the banner must name the directory being served (filled from "
        "status.results_dir)"
    )
    assert "demo_data" in html, (
        "the page must read the demo_data flag from /api/status to decide whether "
        "to show the banner"
    )
    assert "SYNTHETIC" in html, (
        "the empty-state notice must call the data synthetic, not just 'demo'"
    )
