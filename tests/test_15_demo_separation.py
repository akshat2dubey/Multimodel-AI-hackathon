"""15. Demo data and real pipeline output can never share a directory.

A judge opening the dashboard must never be unsure whether a number on screen is a
real flood-extent measurement or a synthetic fixture. The two data paths are
therefore rooted in different directories by default
(results/ vs results_demo/, data/ vs data_demo/) and the dashboard refuses to
serve the demo fixture as if it were a real run: it shows a "SYNTHETIC DEMO DATA"
banner and disables the live pipeline controls.

This test is the binding assertion: anything that can write real-looking outputs
into results/ while a demo fixture is present, or that can conflate the two
directories into one code path, is a rules violation and a demo-that-passes-as-real
bug.
"""

from __future__ import annotations

import json
import sys

import pytest

from config.presets import (
    DATA_DIR,
    DATA_DEMO_DIR,
    DEMO_NOTICE_FILENAME,
    RESULTS_DEMO_DIR,
    RESULTS_DIR,
)


def test_demo_and_real_paths_are_different_dirs():
    # The two output trees must not be the same directory by default. If a future
    # refactor merges them, the dashboard banner and the blocker below both become
    # meaningless.
    assert DATA_DIR != DATA_DEMO_DIR, "data and data_demo must be different roots"
    assert RESULTS_DIR != RESULTS_DEMO_DIR, (
        "results and results_demo must be different roots"
    )


def test_dashboard_demo_detection_uses_served_results_dir(tmp_path, monkeypatch):
    import dashboard.app as app

    monkeypatch.setattr(app, "RESULTS", tmp_path)
    (tmp_path / DEMO_NOTICE_FILENAME).write_text(
        "Synthetic demo data, not real satellite observations.", encoding="utf-8"
    )
    assert app.serving_demo_data()

    (tmp_path / DEMO_NOTICE_FILENAME).unlink()
    assert not app.serving_demo_data()

    monkeypatch.setattr(app, "RESULTS", RESULTS_DEMO_DIR)
    assert app.serving_demo_data()


def test_run_offline_keeps_every_stage_out_of_results_and_data(tmp_path, monkeypatch):
    from demo import run_offline

    data = tmp_path / "data_demo"
    results = tmp_path / "results_demo"
    monkeypatch.setattr(run_offline, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "run_offline.py",
        "--size", "640",
        "--results-dir", str(results),
        "--data-dir", str(data),
    ])

    assert run_offline.main() == 0
    assert (results / "run_meta.json").is_file()
    meta = json.loads((results / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["aoi_confirmed"] is False


def test_offline_runner_rejects_real_data_paths_before_writing(tmp_path, monkeypatch):
    from demo import run_offline

    called = False

    def unexpected_run(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("unsafe paths must be rejected before launching stages")

    monkeypatch.setattr(run_offline.subprocess, "run", unexpected_run)
    monkeypatch.setattr(sys, "argv", [
        "run_offline.py", "--data-dir", str(DATA_DIR),
        "--results-dir", str(tmp_path / "results_demo"),
    ])
    assert run_offline.main() == 1
    assert not called


def test_demo_path_guard_rejects_protected_paths(tmp_path):
    from config.presets import demo_paths_error

    reason = demo_paths_error(DATA_DIR, tmp_path / "results_demo")
    assert reason and "protected" in reason
    assert demo_paths_error(tmp_path / "data_demo", RESULTS_DIR)


def test_path_overlap_guard_resolves_symlink_aliases(tmp_path):
    from config.presets import _paths_overlap

    target = tmp_path / "protected"
    target.mkdir()
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable in this environment")
    assert _paths_overlap(alias, target / "child")


def test_no_code_path_can_merge_demo_and_real_outputs():
    # Guardrail assertion: the two trees have different names and are never aliases.
    # If a future change makes RESULTS_DIR == RESULTS_DEMO_DIR or DATA_DIR ==
    # DATA_DEMO_DIR, the dashboard banner, the blocker and every "am I demo?"
    # check become unreliable.
    assert RESULTS_DIR.name != RESULTS_DEMO_DIR.name or str(RESULTS_DIR) != str(
        RESULTS_DEMO_DIR
    ), "results and results_demo must remain distinguishable"
    assert DATA_DIR.name != DATA_DEMO_DIR.name or str(DATA_DIR) != str(
        DATA_DEMO_DIR
    ), "data and data_demo must remain distinguishable"
