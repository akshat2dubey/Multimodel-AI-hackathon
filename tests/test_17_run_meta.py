"""17. One config: CLI/preset -> results/run_meta.json -> every stage.

The area name, the bbox and the dates used to be literals in a dozen files, and
they disagreed: the preset said before/after 2026-08-14/2026-08-28, the README
said 2026-08-24/2026-08-31, and the Sentinel-1 pair actually downloaded is
2026-08-24/2026-09-05. A report whose header names a corridor and a date the data
does not support is worse than no report.

These tests pin the single resolved artifact that replaced them (run_meta.json),
the bbox-first / center-extent-fallback rule, and the fact that each stage reads
that artifact instead of restating the values.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from config import run_meta as rm


# ─── resolution rules ────────────────────────────────────────────────────────

def test_preset_resolves_and_round_trips_through_run_meta_json(tmp_path):
    meta = rm.resolve_run_meta(preset="trishuli")
    assert meta["area"] and meta["before_date"] and meta["after_date"]
    # The case-study AOI was checked against the EMSR927 AOI01 source.
    assert meta["bbox"] == [85.117358, 27.972395, 85.581515, 28.077922]
    assert meta["bbox_source"] == "preset"
    assert meta["aoi_confirmed"] is True

    path = rm.write_run_meta(tmp_path, meta)
    assert path.name == rm.RUN_META_FILENAME and path.is_file()

    # Found from the directory, from the file, and from a sibling artifact — the
    # last is what lets a stage discover it from the --flood-mask it already has.
    for where in (tmp_path, path, tmp_path / "flood_mask.tif", tmp_path / "nope"):
        assert rm.load_run_meta(where)["area"] == meta["area"]


def test_cli_values_override_the_preset():
    meta = rm.resolve_run_meta(
        preset="trishuli", area="Other Valley",
        before_date="2026-01-01", after_date="2026-01-02",
        event_date="2026-01-02", osm_snapshot_date="2025-12-01",
    )
    assert meta["area"] == "Other Valley"
    assert meta["before_date"] == "2026-01-01"
    assert meta["after_date"] == "2026-01-02"
    assert meta["osm_snapshot_date"] == "2025-12-01"


def test_bbox_is_primary_and_the_center_follows_it():
    meta = rm.resolve_run_meta(preset="trishuli", bbox="85.00,28.00,85.40,28.40")
    assert meta["bbox"] == [85.0, 28.0, 85.4, 28.4]
    assert meta["bbox_source"] == "cli"
    assert meta["aoi_confirmed"] is False
    # The center is the BOX's midpoint, not the preset's unverified fallback: a
    # center that disagrees with the AOI is exactly the drift being removed.
    assert (meta["center_lat"], meta["center_lon"]) == (28.2, 85.2)

    confirmed = rm.resolve_run_meta(
        preset="trishuli", bbox="85.00,28.00,85.40,28.40", confirm_aoi=True,
    )
    assert confirmed["aoi_confirmed"] is True

    custom_confirmed = rm.resolve_run_meta(
        area="Verified Valley", bbox="85.00,28.00,85.40,28.40", confirm_aoi=True,
        before_date="2026-08-14", after_date="2026-08-28",
        event_date="2026-08-26", osm_snapshot_date="2026-07-27",
    )
    assert custom_confirmed["aoi_confirmed"] is True


def test_center_extent_is_only_a_fallback_and_is_marked_unconfirmed():
    meta = rm.resolve_run_meta(
        area="Unconfirmed Valley", center_lat=27.5, center_lon=84.9, extent=0.2,
        before_date="2026-08-14", after_date="2026-08-28",
        event_date="2026-08-26", osm_snapshot_date="2026-07-27",
    )
    assert meta["bbox"] == [84.7, 27.3, 85.1, 27.7]
    assert meta["bbox_source"] == "center_extent_fallback"
    assert meta["aoi_confirmed"] is False


def test_unknown_preset_and_missing_values_are_refused():
    with pytest.raises(ValueError, match="atlantis"):
        rm.resolve_run_meta(preset="atlantis")

    # No preset and no explicit values: refuse rather than invent a case study.
    with pytest.raises(ValueError):
        rm.resolve_run_meta()


def test_post_event_osm_snapshot_is_refused():
    with pytest.raises(ValueError, match="not before the event"):
        rm.resolve_run_meta(preset="trishuli", osm_snapshot_date="2026-09-01")


def test_malformed_bboxes_are_refused():
    for bad in (
        "85,28,86", "86,28,85,28.6", "a,b,c,d", "nan,28,86,29",
        "-181,28,86,29", "85,91,86,92",
    ):
        with pytest.raises(ValueError):
            rm.parse_bbox(bad)


# ─── the orchestrator publishes it and hands it to every stage ───────────────

def test_run_pipeline_publishes_run_meta_and_hands_it_to_every_stage(tmp_path, monkeypatch):
    import run_pipeline

    data, results, report = (tmp_path / n for n in ("data", "results", "report"))
    for d in (data, results, report):
        d.mkdir(parents=True)

    monkeypatch.setattr(run_pipeline, "DATA_DIR", data)
    monkeypatch.setattr(run_pipeline, "RESULTS_DIR", results)
    monkeypatch.setattr(run_pipeline, "REPORT_DIR", report)
    monkeypatch.setattr(run_pipeline.sys, "argv", ["run_pipeline.py", "--preset", "trishuli"])

    # The preprocessing stage is mandatory and fail-closed: without the archive a
    # label actually downloaded, run_pipeline stops (exit 4) instead of analysing
    # stale rasters. run() is stubbed below, so these stand-ins are never opened.
    import toydata

    for label in ("before", "after"):
        toydata.placeholder_safe_archive(data, label)

    commands: list[list[str]] = []

    def fake_run(cmd, cwd=None):  # noqa: ANN001 - mirrors run_pipeline.run
        commands.append([str(part) for part in cmd])
        if any("fetch_osm.py" in part for part in cmd):
            osm_dir = data / "osm"
            osm_dir.mkdir(parents=True, exist_ok=True)
            (osm_dir / "osm_metadata.json").write_text(json.dumps({
                "bbox": [85.117358, 27.972395, 85.581515, 28.077922],
                "snapshot_date": "2026-07-27",
                "failed_required_categories": [],
            }), encoding="utf-8")
        return True

    monkeypatch.setattr(run_pipeline, "run", fake_run)
    run_pipeline.main()

    meta_path = results / "run_meta.json"
    assert meta_path.is_file(), "run_pipeline must publish the resolved run metadata"
    published = json.loads(meta_path.read_text(encoding="utf-8"))
    assert published["preset"] == "trishuli"
    assert published["bbox"] and published["before_date"] and published["after_date"]

    def flag_values(cmd, flag):
        return [cmd[i + 1] for i, part in enumerate(cmd[:-1]) if part == flag]

    # Every download stage is handed the SAME bbox string.
    reached = [flag_values(c, "--bbox") for c in commands]
    drawn = {v[0] for v in reached if v}
    assert len(drawn) == 1, f"stages disagree about the AOI: {drawn}"
    assert len([v for v in reached if v]) >= 4, (
        "all four download stages (sentinel-1, sentinel-2, OSM, DEM) must get --bbox"
    )


def test_pipeline_stops_when_required_sentinel_download_fails(tmp_path, monkeypatch):
    import run_pipeline

    data, results, report = (tmp_path / n for n in ("data", "results", "report"))
    for directory in (data, results, report):
        directory.mkdir(parents=True)
    monkeypatch.setattr(run_pipeline, "DATA_DIR", data)
    monkeypatch.setattr(run_pipeline, "RESULTS_DIR", results)
    monkeypatch.setattr(run_pipeline, "REPORT_DIR", report)
    monkeypatch.setattr(run_pipeline.sys, "argv", ["run_pipeline.py", "--preset", "trishuli"])

    calls = []

    def fail_sentinel(cmd, cwd=None):
        calls.append(cmd)
        return not any("download_sentinel.py" in str(part) for part in cmd)

    monkeypatch.setattr(run_pipeline, "run", fail_sentinel)
    with pytest.raises(SystemExit) as exc:
        run_pipeline.main()

    assert exc.value.code == 1
    assert len(calls) == 1
    assert not (results / "manifest.json").exists()


def test_pipeline_stops_when_osm_bounds_do_not_match_run_aoi(tmp_path, monkeypatch):
    import run_pipeline

    data, results, report = (tmp_path / n for n in ("data", "results", "report"))
    for directory in (data, results, report):
        directory.mkdir(parents=True)
    monkeypatch.setattr(run_pipeline, "DATA_DIR", data)
    monkeypatch.setattr(run_pipeline, "RESULTS_DIR", results)
    monkeypatch.setattr(run_pipeline, "REPORT_DIR", report)
    monkeypatch.setattr(run_pipeline.sys, "argv", ["run_pipeline.py", "--preset", "trishuli"])

    def write_osm_metadata(cmd, cwd=None):
        if any("fetch_osm.py" in str(part) for part in cmd):
            osm_dir = data / "osm"
            osm_dir.mkdir(parents=True, exist_ok=True)
            (osm_dir / "osm_metadata.json").write_text(json.dumps({
                "bbox": [85.4, 28.0, 86.0, 28.6],
                "snapshot_date": "2026-07-27",
                "failed_required_categories": [],
            }), encoding="utf-8")
        return True

    monkeypatch.setattr(run_pipeline, "run", write_osm_metadata)
    with pytest.raises(SystemExit) as exc:
        run_pipeline.main()

    assert exc.value.code == 1
    assert not (results / "manifest.json").exists()

# ─── the artifacts describe their own run ────────────────────────────────────

def _stats_inputs(tmp_path: Path):
    """Minimal but real report inputs, written into tmp_path."""
    import toydata

    mask_path = toydata.write_mask_tif(
        tmp_path / "flood_mask.tif", toydata.cutoff_flood_mask()
    )
    (tmp_path / "damage.geojson").write_text(
        json.dumps({
            "type": "FeatureCollection", "features": [],
            "metadata": {"assessment_method": "flood_mask_overlay"},
        }),
        encoding="utf-8",
    )
    (tmp_path / "cutoff.geojson").write_text(
        json.dumps({
            "type": "FeatureCollection", "features": [], "metadata": {"status": "ok",
            "summary": {
                "total_settlements": 0, "total_hospitals": 1, "reachable_settlements": 0,
                "cutoff_settlements": 0, "cutoff_percentage": 0.0,
                "already_isolated_settlements": 0, "flooded_road_segments": 0,
            },
            "cutoff_settlements": []},
        }),
        encoding="utf-8",
    )
    return mask_path, tmp_path / "damage.geojson", tmp_path / "cutoff.geojson"


def test_report_header_and_data_label_come_from_run_meta(tmp_path):
    import generate_report

    mask_path, damage_path, cutoff_path = _stats_inputs(tmp_path)
    meta = rm.resolve_run_meta(
        area="Toy Valley", bbox="85.00,28.00,85.40,28.40",
        before_date="2026-01-01", after_date="2026-01-02",
        event_date="2026-01-02", osm_snapshot_date="2025-12-01",
        confirm_aoi=True,
    )
    out = tmp_path / "situation_report.html"
    generate_report.generate_html_report(
        meta["area"], meta["before_date"], meta["after_date"],
        generate_report.load_flood_mask_stats(mask_path),
        generate_report.load_damage_stats(damage_path),
        generate_report.load_cutoff_stats(cutoff_path),
        out, synthetic=False, run_meta=meta,
    )
    html = out.read_text(encoding="utf-8")

    assert "Toy Valley" in html, "the header must name the run's area"
    assert "Flood Event Analysis • 2026-01-02" in html, (
        "the header must state the EVENT date from run_meta, not the after-acquisition date"
    )
    assert "2025-12-01" in html, (
        "the data label must state the run's own OSM snapshot date, not a literal"
    )
    assert "AREA OF INTEREST NOT CONFIRMED" not in html, (
        "a CLI-supplied (confirmed) bbox must not be labelled unconfirmed"
    )


def test_report_states_an_unconfirmed_aoi_plainly(tmp_path):
    import generate_report

    mask_path, damage_path, cutoff_path = _stats_inputs(tmp_path)
    # A custom bbox is not confirmation by itself, even when it overrides a preset.
    meta = rm.resolve_run_meta(
        preset="trishuli", bbox="85.2,28.0,85.4,28.2",
        area="Unconfirmed AOI Test",
    )
    assert meta["aoi_confirmed"] is False

    out = tmp_path / "report.html"
    generate_report.generate_html_report(
        meta["area"], meta["before_date"], meta["after_date"],
        generate_report.load_flood_mask_stats(mask_path),
        generate_report.load_damage_stats(damage_path),
        generate_report.load_cutoff_stats(cutoff_path),
        out, synthetic=True, run_meta=meta,
    )
    assert "AREA OF INTEREST NOT CONFIRMED" in out.read_text(encoding="utf-8"), (
        "a custom bbox without explicit AOI confirmation must be labelled unconfirmed"
    )


def test_damage_and_cutoff_metadata_carry_the_run_metadata(toy_scene, tmp_path):
    import geopandas as gpd

    import cutoff_analysis as ca
    import damage_assessment as da

    meta = rm.resolve_run_meta(
        area="Toy Valley", bbox="85.00,28.00,85.40,28.40",
        before_date="2026-01-01", after_date="2026-01-02",
        event_date="2026-01-02", osm_snapshot_date="2025-12-01",
    )

    empty = gpd.GeoDataFrame(geometry=[], crs=toy_scene["crs"])
    damage_out = tmp_path / "damage.geojson"
    da.save_damage_assessment(
        gpd.read_file(toy_scene["buildings"]), empty, empty, damage_out,
        transform=toy_scene["transform"], crs=toy_scene["crs"], run_meta=meta,
    )
    md = json.loads(damage_out.read_text(encoding="utf-8"))["metadata"]
    assert md["area"] == "Toy Valley"
    assert md["bbox"] == [85.0, 28.0, 85.4, 28.4]
    assert md["event_date"] == "2026-01-02"
    assert md["osm_snapshot_date"].startswith("2025-12-01")

    mask, transform, crs = ca.load_flood_mask(toy_scene["mask_path"])
    buildings = gpd.read_file(toy_scene["buildings"])
    graph, positions, types = ca.build_road_graph(
        gpd.read_file(toy_scene["roads"]), buildings, mask, transform, crs
    )
    cutoff_out = tmp_path / "cutoff_settlements.geojson"
    ca.analyze_cutoffs(
        graph, positions, types, buildings, None, mask, transform, crs, cutoff_out,
        run_meta=meta,
    )
    cdata = json.loads(cutoff_out.read_text(encoding="utf-8"))
    assert cdata["metadata"]["area"] == "Toy Valley"
    assert cdata["metadata"]["bbox"] == [85.0, 28.0, 85.4, 28.4]
    assert cdata["metadata"]["event_date"] == "2026-01-02"
