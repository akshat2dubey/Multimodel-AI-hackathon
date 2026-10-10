#!/usr/bin/env python3
"""
Jalnetra — Pipeline Orchestrator
Run the complete flood damage mapping pipeline end-to-end.

Usage:
    python run_pipeline.py \
        --area "Bhote Koshi corridor" \
        --center-lat 28.3 \
        --center-lon 85.7 \
        --before-date 2026-08-24 \
        --after-date 2026-08-31 \
        --osm-snapshot 2026-07-27

Credentials: CDSE_CLIENT_ID / CDSE_CLIENT_SECRET must be set before
any Sentinel download. The helper `scripts/cdse_env.py` reads the
git-ignored file `.secrets/cdse.env` so the secret never appears on a
command line or in a log:

    python -c "import cdse_env; cdse_env.load_cdse_env()"
    python run_pipeline.py --area ... --center-lon 85.7 ...

(Without credentials the Sentinel-1 download exits non-zero and the
pipeline continues on Sentinel-1 imagery only if available.)
"""

import argparse

import sys
import subprocess
from pathlib import Path

# Pre-load Copernicus credentials from the git-ignored secret file so
# the cdse / CDSEClient libraries (which read os.environ) can
# authenticate without the secret ever touching a command line, a shell
# history entry, or a log.
try:
    # Make `scripts/` importable regardless of the working directory.
    _scripts_root = Path(__file__).resolve().parent
    if str(_scripts_root) not in sys.path:
        sys.path.insert(0, str(_scripts_root))
    import cdse_env
    cdse_env.load_cdse_env()
except Exception as exc:  # pylint: disable=broad-except
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
    parser.add_argument("--area", required=True, help="Area description")
    parser.add_argument("--center-lat", type=float, required=True, help="Center latitude")
    parser.add_argument("--center-lon", type=float, required=True, help="Center longitude")
    parser.add_argument("--before-date", required=True, help="Before-event date (YYYY-MM-DD)")
    parser.add_argument("--after-date", required=True, help="After-event date (YYYY-MM-DD)")
    parser.add_argument("--osm-snapshot", required=True, help="OSM snapshot date (YYYY-MM-DD)")
    parser.add_argument("--bbox", help="Bounding box: min_lon,min_lat,max_lon,max_lat")
    parser.add_argument("--skip-training", action="store_true", help="Skip model training (use existing model)")
    args = parser.parse_args()

    # Anchor everything to the scripts/ directory so the pipeline can be
    # invoked from any working directory without turning
    #   scripts/scripts/download_sentinel.py
    # into a real filename.
    root = Path(__file__).resolve().parent
    data_dir = root.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    _scripts = lambda name: str(root / name)

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     🌊  Jalnetra — Mapping Flood Damage from Space       ║
║                                                              ║
║     Area:  {args.area:<47}║
║     Center: {args.center_lat}, {args.center_lon}               ║
║     Before: {args.before_date}   After: {args.after_date}      ║
║     OSM snapshot: {args.osm_snapshot}                      ║
║     Credentials: {'set' if __import__('os').environ.get('CDSE_CLIENT_ID') else 'NOT SET (Sentinel downloads will fail)'}                 ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")

    # Step 1: Download satellite data
    if not run([
        sys.executable, _scripts("download_sentinel.py"),
        "--center-lat", str(args.center_lat),
        "--center-lon", str(args.center_lon),
        "--before-date", args.before_date,
        "--after-date", args.after_date,
        "--output", str(data_dir / "sentinel1"),
    ]):
        print("\n⚠️ Sentinel-1 download failed — continuing with preprocessing step (may fail if no data)")
    
    # Step 2: Download Sentinel-2 (optical, optional — cloud dependent)
    if not run([
        sys.executable, _scripts("download_sentinel2.py"),
        "--center-lat", str(args.center_lat),
        "--center-lon", str(args.center_lon),
        "--before-date", args.before_date,
        "--after-date", args.after_date,
        "--output", str(data_dir / "sentinel2"),
    ]):
        print("\n⚠️ Sentinel-2 download failed — cloud cover likely. Relying on Sentinel-1.")

    # Step 3: Fetch OSM pre-event data
    bbox_arg = []
    if args.bbox:
        bbox_arg = ["--bbox", args.bbox]
    if not run([
        sys.executable, _scripts("fetch_osm.py"),
        "--center-lat", str(args.center_lat),
        "--center-lon", str(args.center_lon),
        "--snapshot-date", args.osm_snapshot,
        "--output", str(data_dir / "osm"),
        *bbox_arg,
    ]):
        print("\n❌ OSM data fetch failed — cannot proceed without OSM data")
        sys.exit(1)

    # Step 4: Download Copernicus DEM
    if not run([
        sys.executable, _scripts("download_dem.py"),
        "--center-lat", str(args.center_lat),
        "--center-lon", str(args.center_lon),
        "--output", str(data_dir / "dem"),
    ]):
        print("\n⚠️ DEM download failed — flood path tracing will be skipped")

    # Step 5: Train segmentation model (if not skipped)
    model_path = root.parent / "models" / "flood_unet.pt"
    if not args.skip_training and not model_path.exists():
        if not run([
            sys.executable, _scripts("train_segmentation.py"),
            "--output", str(model_path),
            "--epochs", "50",
        ]):
            print("\n⚠️ Model training failed — using placeholder model for demo")
            model_path = None
    else:
        print(f"\nℹ️ Using existing model: {model_path}")

    # Step 6: Run inference (flood segmentation)
    # NOTE: inference.py appends "before"/"after" itself, so both flags take the
    # sentinel1 root (which download_sentinel.py filled with before/ + after/).
    results_dir = root / "results"
    results_dir.mkdir(exist_ok=True)
    if not run([
        sys.executable, _scripts("inference.py"),
        "--before-dir", str(data_dir / "sentinel1"),
        "--after-dir", str(data_dir / "sentinel1"),
        "--model", str(model_path) if model_path else "",
        "--output", str(results_dir / "flood_mask.tif"),
    ]):
        print("\n❌ Inference failed — cannot proceed")
        sys.exit(1)

    # Step 7: Damage assessment (overlay flood mask on OSM)
    if not run([
        sys.executable, _scripts("damage_assessment.py"),
        "--flood-mask", str(results_dir / "flood_mask.tif"),
        "--osm-buildings", str(data_dir / "osm" / "buildings.geojson"),
        "--osm-roads", str(data_dir / "osm" / "roads.geojson"),
        "--osm-bridges", str(data_dir / "osm" / "bridges.geojson"),
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
        "--output", str(results_dir / "cutoff_settlements.geojson"),
    ]):
        print("\n⚠️ Cut-off analysis failed")

    # Step 9: Bonus — flood path tracing
    dem_path = data_dir / "dem" / "dem.tif"
    if dem_path.exists():
        if not run([
            sys.executable, _scripts("flood_path.py"),
            "--dem", str(dem_path),
            "--start-lat", str(args.center_lat + 0.1),
            "--start-lon", str(args.center_lon),
            "--output", str(results_dir / "flood_path.geojson"),
        ]):
            print("\n⚠️ Flood path tracing failed")

    # Step 10: Generate situation report
    if not run([
        sys.executable, _scripts("generate_report.py"),
        "--area", args.area,
        "--before-date", args.before_date,
        "--after-date", args.after_date,
        "--flood-mask", str(results_dir / "flood_mask.tif"),
        "--damage", str(results_dir / "damage_assessment.geojson"),
        "--cutoffs", str(results_dir / "cutoff_settlements.geojson"),
        "--output", str(root.parent / "report" / "situation_report.html"),
    ]):
        print("\n⚠️ Report generation failed")

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
