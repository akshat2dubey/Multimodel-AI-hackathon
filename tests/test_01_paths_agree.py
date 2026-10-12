"""1. The orchestrator and the dashboard must resolve the same results/ dir.

An earlier revision had run_pipeline.py writing to scripts/results while the
dashboard served <root>/results, so a "successful" pipeline produced a dashboard
that stayed empty. One results directory, or none of the numbers mean anything.
"""

from __future__ import annotations

from pathlib import Path


def test_pipeline_and_dashboard_resolve_same_results_dir(repo_root: Path):
    import run_pipeline
    import app  # dashboard/app.py (dashboard/ is on sys.path)
    from config import presets

    expected = (repo_root / "results").resolve()
    actual = {
        "config.presets.RESULTS_DIR": Path(presets.RESULTS_DIR).resolve(),
        "run_pipeline.RESULTS_DIR": Path(run_pipeline.RESULTS_DIR).resolve(),
        "dashboard.app.RESULTS": Path(app.RESULTS).resolve(),
    }

    bad = {name: str(p) for name, p in actual.items() if p != expected}
    assert not bad, (
        "every entry point must resolve the SAME results directory "
        f"{expected}; these disagree: {bad}. Fix config/presets.py and import "
        "it, do not re-derive the path locally."
    )

    assert Path(run_pipeline.RESULTS_DIR).parent.resolve() == repo_root.resolve(), (
        "run_pipeline.py must write into the project-root results/ directory, "
        f"not {run_pipeline.RESULTS_DIR} — a scripts/results directory is invisible "
        "to the dashboard"
    )
    for name, value in actual.items():
        assert Path(value).parent.resolve() == repo_root.resolve(), (
            f"{name} points at {value}, which is not <repo>/results; the "
            "dashboard would serve an empty directory"
        )


def test_every_stage_output_lands_in_the_shared_dirs(
    tmp_path: Path, monkeypatch, repo_root: Path
):
    """No --output in run_pipeline.py may point anywhere but the shared dirs.

    The original bug was a single re-derived path (scripts/results) that the
    dashboard never read. This walks the argv the orchestrator actually builds,
    so a future stage that invents its own directory fails here instead of
    silently producing artifacts nobody serves.
    """
    import run_pipeline
    from config import presets

    data = tmp_path / "data"
    results = tmp_path / "results"
    report = tmp_path / "report"
    data.mkdir(parents=True)
    results.mkdir(parents=True)
    report.mkdir(parents=True)
    monkeypatch.setattr(run_pipeline, "DATA_DIR", data)
    monkeypatch.setattr(run_pipeline, "RESULTS_DIR", results)
    monkeypatch.setattr(run_pipeline, "REPORT_DIR", report)
    monkeypatch.setattr(run_pipeline.sys, "argv", [
        "run_pipeline.py",
        "--area", presets.AREA,
        "--center-lat", str(presets.CENTER_LAT),
        "--center-lon", str(presets.CENTER_LON),
        "--before-date", presets.BEFORE_DATE,
        "--after-date", presets.AFTER_DATE,
        # The event date is required: run_pipeline no longer assumes the event
        # happened on the post-event acquisition date.
        "--event-date", presets.EVENT_DATE,
        "--osm-snapshot", presets.OSM_SNAPSHOT_DATE,
    ])

    commands: list[list[str]] = []

    def fake_run(cmd, cwd=None):  # noqa: ANN001 - mimics run_pipeline.run
        commands.append([str(part) for part in cmd])
        return True

    monkeypatch.setattr(run_pipeline, "run", fake_run)
    run_pipeline.main()

    assert len(commands) >= 7, (
        f"expected the download/inference/damage/cutoff/report stages, captured "
        f"{len(commands)}: {commands}"
    )

    outputs = [
        cmd[i + 1]
        for cmd in commands
        for i, part in enumerate(cmd[:-1])
        if part == "--output"
    ]
    assert outputs, "no --output arguments captured"
    for path in outputs:
        resolved = Path(path).resolve()
        assert tmp_path.resolve() in resolved.parents, (
            f"run_pipeline.py passes --output {path}, which is outside the shared "
            "results/ and data/ directories it was pointed at. Every stage output "
            "must come from config.presets, or the dashboard will not serve it."
        )

    # The dead path from the original bug, and the real trees, must be untouched.
    dead = [
        (repo_root / "scripts" / "results").resolve(),
        (repo_root / "results").resolve(),
        (repo_root / "data").resolve(),
        (repo_root / "report").resolve(),
    ]
    for cmd in commands:
        for arg in cmd:
            if arg.startswith("-"):
                continue
            try:
                resolved = Path(arg).resolve()
            except OSError:  # pragma: no cover - defensive
                continue
            for target in dead:
                assert resolved != target and target not in resolved.parents, (
                    f"run_pipeline.py passes {arg!r}, which resolves inside "
                    f"{target} instead of the results/data/report directories it "
                    "was routed to"
                )


def test_data_dir_is_shared_too(repo_root: Path):
    import run_pipeline
    import app
    from config import presets

    expected = (repo_root / "data").resolve()
    for name, value in (
        ("config.presets", presets.DATA_DIR),
        ("run_pipeline", run_pipeline.DATA_DIR),
        ("dashboard.app", app.DATA),
    ):
        assert Path(value).resolve() == expected, (
            f"{name} resolves data/ to {value}, but the pipeline downloads into "
            f"{expected}; inputs read from one place and written to another is a "
            "silent empty-dashboard bug"
        )
