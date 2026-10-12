#!/usr/bin/env python3
"""
Jalnetra — Pipeline Orchestrator
Run the complete flood damage mapping pipeline end-to-end.

Usage (case-study values come from a preset — see config/presets.py):
    python run_pipeline.py --preset trishuli

Or give the area explicitly. --bbox is the PRIMARY area-of-interest input;
--center-lat/--center-lon (with an optional --extent) is only a fallback:
    python run_pipeline.py \
        --area "<area name>" \
        --bbox <min_lon,min_lat,max_lon,max_lat> \
        --before-date <YYYY-MM-DD> \
        --after-date <YYYY-MM-DD> \
        --event-date <YYYY-MM-DD> \
        --osm-snapshot <YYYY-MM-DD>

Whatever the inputs, they are resolved ONCE into results/run_meta.json
(config/run_meta.py), and every stage reads them from there — so the report, the
damage metadata and the cut-off list cannot disagree about the area or dates.

Credentials: CDSE_CLIENT_ID / CDSE_CLIENT_SECRET must be set before
any Sentinel download. The helper `scripts/cdse_env.py` reads the
git-ignored file `.secrets/cdse.env` so the secret never appears on a
command line or in a log:

    python -c "import cdse_env; cdse_env.load_cdse_env()"
    python run_pipeline.py --area ... --center-lon <lon> ...

(Without credentials the Sentinel-1 download exits non-zero and the
pipeline continues on Sentinel-1 imagery only if available.)
"""

import argparse

import sys
import subprocess
from pathlib import Path

# Make `scripts/` AND the project root importable regardless of the working
# directory, so `config.presets` (the single source of truth for paths and
# case-study constants) resolves either way.
_scripts_root = Path(__file__).resolve().parent
_project_root = _scripts_root.parent
for _p in (str(_scripts_root), str(_project_root)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config.presets import (  # noqa: E402
    DATA_DIR,
    MODELS_DIR,
    PRESETS,
    PROJECT_ROOT as ROOT,
    REPORT_DIR,
    RESULTS_DIR,
)
from config.run_meta import (  # noqa: E402
    RUN_META_FILENAME,
    describe,
    resolve_run_meta,
    write_run_meta,
)

# Provenance + the disqualification boundary (T34). The rules are evaluated at
# the start (dates, and where the inputs come from) and again at the end, when
# the acquisitions and their hashes are known.
from manifest import (  # noqa: E402
    RuleViolation,
    build_manifest,
    describe_rule_failure,
    enforce_rules,
    input_paths_from,
    output_paths_from,
    rules_check,
    write_manifest,
)

# Pre-load Copernicus credentials from the git-ignored secret file so
# the cdse / CDSEClient libraries (which read os.environ) can
# authenticate without the secret ever touching a command line, a shell
# history entry, or a log.
try:
    import cdse_env
    cdse_env.load_cdse_env()
except (Exception, SystemExit) as exc:  # pylint: disable=broad-except
    # Don't fail the whole pipeline for a credential problem — Sentinel-1
    # is optional in this orchestrator, and a missing secret is an
    # operational issue the operator can fix without losing the rest of
    # the run.
    print(
        f"⚠ Credentials not loaded: {exc}",
        file=sys.stderr,
    )
    print(
        "   Sentinel-1 downloads require a Copernicus account. See scripts/cdse_env.py.",
        file=sys.stderr,
    )


if hasattr(sys.stdout, "reconfigure"):  # never crash on a cp1252 Windows console
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def run(cmd: list[str], cwd: Path | None = None) -> bool:
    """Run a command and return True on success."""
    print(f"\n{'='*70}")
    print(f"▶ Running: {' '.join(cmd)}")
    print(f"{'='*70}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        print(f"\n❌ Command failed with exit code {result.returncode}")
        return False
    print(f"\n✅ Success")
    return True


def main():
    parser = argparse.ArgumentParser(description="Jalnetra — Full Pipeline")
    parser.add_argument(
        "--preset",
        default=None,
        help=(
            "Case-study preset supplying default area/bbox/dates. Available: "
            + ", ".join(sorted(PRESETS))
            + ". An explicit flag below always overrides the preset."
        ),
    )
    parser.add_argument("--area", default=None, help="Area description")
    parser.add_argument(
        "--bbox",
        help=(
            "PRIMARY area of interest: min_lon,min_lat,max_lon,max_lat. "
            "Falls back to --center-lat/--center-lon/--extent when omitted."
        ),
    )
    parser.add_argument("--center-lat", type=float, default=None,
                        help="Fallback AOI centre latitude (prefer --bbox)")
    parser.add_argument("--center-lon", type=float, default=None,
                        help="Fallback AOI centre longitude (prefer --bbox)")
    parser.add_argument("--extent", type=float, default=None,
                        help="Fallback AOI half-width in degrees")
    parser.add_argument("--before-date", default=None, help="Before-event date (YYYY-MM-DD)")
    parser.add_argument("--after-date", default=None, help="After-event date (YYYY-MM-DD)")
    parser.add_argument("--event-date", default=None, help="Event date (YYYY-MM-DD)")
    parser.add_argument("--osm-snapshot", default=None, help="OSM snapshot date (YYYY-MM-DD)")
    args = parser.parse_args()

    # Resolve the run's metadata ONCE (CLI over preset) and publish it next to
    # the artifacts. Every stage below — and every stage run later by hand —
    # reads this one file, which is what stops the area, the dates and the bbox
    # from drifting apart between the mask, the damage assessment and the report.
    try:
        run_meta = resolve_run_meta(
            preset=args.preset,
            area=args.area,
            bbox=args.bbox,
            center_lat=args.center_lat,
            center_lon=args.center_lon,
            extent=args.extent,
            event_date=args.event_date,
            before_date=args.before_date,
            after_date=args.after_date,
            osm_snapshot_date=args.osm_snapshot,
        )
    except ValueError as exc:
        print(f"❌ {exc}", file=sys.stderr)
        sys.exit(2)

    # Anchor everything to the scripts/ directory so the pipeline can be
    # invoked from any working directory without turning
    #   scripts/scripts/download_sentinel.py
    # into a real filename.
    root = Path(__file__).resolve().parent
    # Paths come from config.presets (imported at module level) so the
    # orchestrator, the offline runner and the dashboard all agree on exactly one
    # results/ directory.
    data_dir = DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir = RESULTS_DIR

    # Hard data rules first, before anything is written or fetched (T34): a run
    # that would use a post-event OSM snapshot or an EMS/UNOSAT input must leave
    # no artifact behind that could be mistaken for a result. The checks that
    # need the acquisitions are re-evaluated on the manifest at the end.
    try:
        enforce_rules(rules_check(run_meta))
    except RuleViolation as exc:
        print(f"❌ {exc}", file=sys.stderr)
        sys.exit(3)

    meta_path = write_run_meta(results_dir, run_meta)
    # One string, used verbatim by all four download stages, so they cannot
    # disagree about the AOI.
    bbox_str = ",".join(f"{v:.6f}" for v in run_meta["bbox"])

    _scripts = lambda name: str(root / name)

    print("=" * 70)
    print("  🌊  Jalnetra — Mapping Flood Damage from Space")
    print("=" * 70)
    print(f"  {describe(run_meta)}")
    print(
        "  Credentials: "
        + ("set" if __import__('os').environ.get('CDSE_CLIENT_ID')
           else "NOT SET (Sentinel downloads will fail)")
    )
    print(f"  run_meta:    {meta_path}")
    print("=" * 70)
    if not run_meta.get("aoi_confirmed"):
        print(
            "\n⚠️ AOI NOT CONFIRMED: the bbox came from the preset fallback "
            f"({run_meta['bbox_source']}), not a confirmed case-study AOI.\n"
            "   Pass --bbox once the authoritative AOI is confirmed (see the TODO "
            "in config/presets.py):\n"
            "     python scripts/run_pipeline.py --preset trishuli --bbox min_lon,min_lat,max_lon,max_lat\n"
            "   Every artifact from this run is stamped aoi_confirmed: false.",
            file=sys.stderr,
        )

    # Step 1: Download satellite data
    if not run([
        sys.executable, _scripts("download_sentinel.py"),
        "--bbox", bbox_str,
        "--before-date", run_meta["before_date"],
        "--after-date", run_meta["after_date"],
        "--output", str(data_dir / "sentinel1"),
    ]):
        print("\n⚠️ Sentinel-1 download failed — continuing with preprocessing step (may fail if no data)")

    # Step 2: Download Sentinel-2 (optical, optional — cloud dependent)
    if not run([
        sys.executable, _scripts("download_sentinel2.py"),
        "--bbox", bbox_str,
        "--before-date", run_meta["before_date"],
        "--after-date", run_meta["after_date"],
        "--output", str(data_dir / "sentinel2"),
    ]):
        print("\n⚠️ Sentinel-2 download failed — cloud cover likely. Relying on Sentinel-1.")

    # Step 3: Fetch OSM pre-event data
    if not run([
        sys.executable, _scripts("fetch_osm.py"),
        "--bbox", bbox_str,
        "--snapshot-date", run_meta["osm_snapshot_date"],
        "--event-date", run_meta["event_date"],
        "--output", str(data_dir / "osm"),
    ]):
        print("\n❌ OSM data fetch failed — cannot proceed without OSM data")
        sys.exit(1)

    # Step 4: Download Copernicus DEM
    if not run([
        sys.executable, _scripts("download_dem.py"),
        "--bbox", bbox_str,
        "--output", str(data_dir / "dem"),
    ]):
        print("\n⚠️ DEM download failed — flood path tracing will be skipped")

    # Step 5: Pick the flood-mask engine. THIS ORCHESTRATOR NEVER TRAINS.
    # It used to run train_segmentation.py --epochs 50 inline whenever the
    # checkpoint was missing; with an empty dataset that script then fabricated
    # 240 synthetic tiles, so a "trained" model and a confident-looking flood map
    # could both come out of data the pipeline invented. Training needs a real
    # dataset and is now an explicit, separate, human decision:
    #
    #   python scripts/train_segmentation.py --dataset <kuro_siwo|sen1floods11>
    #
    # Without a checkpoint, inference falls back to the log-ratio change-detection
    # baseline, which still produces a real mask from the real imagery.
    model_path = MODELS_DIR / "flood_unet.pt"
    use_baseline = not model_path.exists()
    if use_baseline:
        print(
            f"\n⚠️ No trained model at {model_path}\n"
            "   Falling back to the log-ratio change-detection baseline "
            "(after-minus-before VV drop).\n"
            "   A flood mask is still produced from the real imagery; it is a band "
            "difference,\n   not a segmentation model. This pipeline never trains — "
            "train separately:\n"
            "     python scripts/train_segmentation.py --dataset <real dataset>\n"
        )
    else:
        print(f"\nℹ️ Using trained model: {model_path}")

    # Step 6: Run inference (flood segmentation)
    # NOTE: inference.py appends "before"/"after" itself, so both flags take the
    # sentinel1 root (which download_sentinel.py filled with before/ + after/).
    results_dir.mkdir(parents=True, exist_ok=True)
    inference_cmd = [
        sys.executable, _scripts("inference.py"),
        "--before-dir", str(data_dir / "sentinel1"),
        "--after-dir", str(data_dir / "sentinel1"),
        "--output", str(results_dir / "flood_mask.tif"),
    ]
    if use_baseline:
        inference_cmd.append("--baseline")
    else:
        inference_cmd += ["--model", str(model_path)]
    if not run(inference_cmd):
        print("\n❌ Inference failed — cannot proceed")
        sys.exit(1)

    # Step 7: Damage assessment (overlay flood mask on OSM)
    if not run([
        sys.executable, _scripts("damage_assessment.py"),
        "--flood-mask", str(results_dir / "flood_mask.tif"),
        "--osm-buildings", str(data_dir / "osm" / "buildings.geojson"),
        "--osm-roads", str(data_dir / "osm" / "roads.geojson"),
        "--osm-bridges", str(data_dir / "osm" / "bridges.geojson"),
        "--run-meta", str(meta_path),
        "--output", str(results_dir / "damage_assessment.geojson"),
    ]):
        print("\n⚠️ Damage assessment failed")

    # Step 8: Cut-off settlement analysis
    if not run([
        sys.executable, _scripts("cutoff_analysis.py"),
        "--roads", str(data_dir / "osm" / "roads.geojson"),
        "--buildings", str(data_dir / "osm" / "buildings.geojson"),
        "--flood-mask", str(results_dir / "flood_mask.tif"),
        "--hospitals", str(data_dir / "osm" / "hospitals.geojson"),
        "--run-meta", str(meta_path),
        "--output", str(results_dir / "cutoff_settlements.geojson"),
    ]):
        print("\n⚠️ Cut-off analysis failed")

    # Step 9: Bonus — flood path tracing
    dem_path = data_dir / "dem" / "dem.tif"
    if dem_path.exists():
        if not run([
            sys.executable, _scripts("flood_path.py"),
            "--dem", str(dem_path),
            # The centre comes from the resolved run metadata, not from a
            # separately-passed flag that could disagree with the bbox.
            "--start-lat", str(run_meta["center_lat"] + 0.1),
            "--start-lon", str(run_meta["center_lon"]),
            "--output", str(results_dir / "flood_path.geojson"),
        ]):
            print("\n⚠️ Flood path tracing failed")

    # Step 10: Generate situation report. The report's area, dates and header
    # come from run_meta.json, so its header can no longer name a corridor the
    # data does not support.
    if not run([
        sys.executable, _scripts("generate_report.py"),
        "--run-meta", str(meta_path),
        "--flood-mask", str(results_dir / "flood_mask.tif"),
        "--damage", str(results_dir / "damage_assessment.geojson"),
        "--cutoffs", str(results_dir / "cutoff_settlements.geojson"),
        "--output", str(REPORT_DIR / "situation_report.html"),
    ]):
        print("\n⚠️ Report generation failed")

    # Step 11: Provenance manifest (T34). Every run leaves one, including a run
    # whose later steps failed — the inputs it used are still evidence.
    manifest = build_manifest(
        run_meta=run_meta,
        inputs=input_paths_from(data_dir),
        outputs=output_paths_from(results_dir),
        repo_root=ROOT,
        method="unet" if not use_baseline else "log_ratio_baseline",
        model=None if use_baseline else model_path,
        sentinel_dir=data_dir / "sentinel1",
    )
    manifest_path = write_manifest(results_dir, manifest)
    print(f"\n📜 Provenance manifest: {manifest_path}")
    for name, ok in manifest["rules_check"].items():
        print(f"   {name}: {ok}")

    failure = describe_rule_failure(manifest["rules_check"])
    if failure:
        print(f"\n❌ {failure}", file=sys.stderr)
        sys.exit(3)

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║  ✅ Pipeline complete!                                       ║
║                                                              ║
║  Results in: {results_dir}                         ║
║  Dashboard:  cd dashboard && python app.py                   ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
