#!/usr/bin/env python3
"""
Offline pipeline runner — exercise the whole backend chain on synthetic data.

Generated inputs land in data_demo/ and outputs in results_demo/ (see
config.presets), so a demo run can neither read nor overwrite the real data/ and
results/ trees.

Runs every downstream stage in order using the reference flood mask produced by
make_demo_data.py, so damage assessment, cut-off analysis, flood-path tracing,
the situation report and the dashboard all get real artifacts without needing a
Copernicus Data Space account.

Stages skipped by design (they need real satellite data):
    download_sentinel.py, download_sentinel2.py, download_dem.py, fetch_osm.py,
    train_segmentation.py, inference.py

Usage:
    python scripts/run_offline.py
    python scripts/run_offline.py --skip-data-generation
    python scripts/run_offline.py --area "Bhote Koshi – Trishuli Corridor"
"""

from __future__ import annotations

import argparse

import sys
import subprocess
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):  # never crash on a cp1252 Windows console
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
for _p in (str(SCRIPTS), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config.presets import (  # noqa: E402  (path set up above)
    AFTER_DATE,
    AREA,
    BEFORE_DATE,
    CENTER_LAT,
    CENTER_LON,
    DATA_DEMO_DIR,
    DEFAULT_PRESET,
    FIXTURE_EXTENT,
    OSM_SNAPSHOT_DATE,
    RESULTS_DEMO_DIR,
    RESULTS_NAME,
)
from config.run_meta import resolve_run_meta, write_run_meta  # noqa: E402

BANNER = "*** SYNTHETIC DEMO DATA — NOT REAL EARTH OBSERVATION DATA ***"


def run(label: str, cmd: list[str]) -> bool:
    print(f"\n{'=' * 72}\n {label}\n{'=' * 72}")
    print("  $ " + " ".join(cmd))
    # Decode the child's output as UTF-8 with errors="replace" for the same
    # reason run_pipeline.py does: these scripts print ✓/⚠/❌, and reading them
    # through the console's cp1252 codec killed the run with UnicodeDecodeError.
    result = subprocess.run(
        cmd, cwd=ROOT, text=True, encoding="utf-8", errors="replace"
    )
    if result.returncode != 0:
        print(f"\n  FAILED (exit {result.returncode}): {label}")
        return False
    print(f"  OK: {label}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the offline (synthetic) pipeline")
    parser.add_argument("--area", default=AREA)
    parser.add_argument("--before-date", default=BEFORE_DATE)
    parser.add_argument("--after-date", default=AFTER_DATE)
    parser.add_argument("--center-lat", type=float, default=CENTER_LAT)
    parser.add_argument("--center-lon", type=float, default=CENTER_LON)
    parser.add_argument("--extent", type=float, default=FIXTURE_EXTENT)
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--skip-data-generation", action="store_true",
                        help=f"Reuse the fixture already in {DATA_DEMO_DIR.name}/ and "
                             f"{RESULTS_DEMO_DIR.name}/")
    args = parser.parse_args()

    py = sys.executable
    # Every artifact of an offline run lands in results_demo/ and every generated
    # input in data_demo/ (both imported from config.presets), never in results/
    # or data/ — the real runs write there, and the dashboard reads results/ for
    # them and shows a banner when it is pointed at the demo directory instead.
    results = RESULTS_DEMO_DIR
    data = DATA_DEMO_DIR
    ok = True

    print(f"\n{'#' * 72}\n#  {BANNER}\n#  Offline run: validates plumbing, not scientific results.\n"
          f"#  Outputs -> {results}\n{'#' * 72}")

    # 1. synthetic scene + reference flood mask
    if not args.skip_data_generation:
        ok &= run("1/5  Generate synthetic scene + reference flood mask",
                  [py, str(SCRIPTS / "make_demo_data.py"),
                   "--center-lat", str(args.center_lat),
                   "--center-lon", str(args.center_lon),
                   "--extent", str(args.extent),
                   "--size", str(args.size),
                   "--root", str(ROOT),
                   "--data-dir", str(data),
                   "--results-dir", str(results)])
        if not ok:
            return 1

    mask = results / "flood_mask.tif"
    if not mask.exists():
        print(f"\nMissing {mask} — run without --skip-data-generation first.")
        return 1
    if not (data / "osm" / "buildings.geojson").exists():
        print(f"\nMissing generated inputs in {data} — run without "
              f"--skip-data-generation first.")
        return 1
    if Path(mask).parent.name == RESULTS_NAME:
        print("\n  ✋ refusing to run: the fixture mask is inside results/. Real runs "
              "write there; a demo must not overwrite them.")
        return 1

    # 0. publish this demo run's metadata next to its artifacts, exactly as a
    # real run does, so damage/cutoff/report describe the same area and dates
    # instead of falling back to the preset. This writes to results_demo/.
    run_meta = resolve_run_meta(
        preset=DEFAULT_PRESET,
        area=args.area,
        center_lat=args.center_lat,
        center_lon=args.center_lon,
        extent=args.extent,
        before_date=args.before_date,
        after_date=args.after_date,
    )
    meta_path = write_run_meta(results, run_meta)
    print(f"\n  Run metadata -> {meta_path}")

    # 2. damage assessment (buildings / roads / bridges)
    ok &= run("2/5  Damage assessment — flood mask x OSM features",
              [py, str(SCRIPTS / "damage_assessment.py"),
               "--flood-mask", str(mask),
               "--osm-buildings", str(data / "osm" / "buildings.geojson"),
               "--osm-roads", str(data / "osm" / "roads.geojson"),
               "--osm-bridges", str(data / "osm" / "bridges.geojson"),
               "--run-meta", str(meta_path),
               "--output", str(results / "damage_assessment.geojson")])

    # 3. cut-off settlement analysis
    ok &= run("3/5  Cut-off analysis — road-network reachability",
              [py, str(SCRIPTS / "cutoff_analysis.py"),
               "--roads", str(data / "osm" / "roads.geojson"),
               "--buildings", str(data / "osm" / "buildings.geojson"),
               "--flood-mask", str(mask),
               "--hospitals", str(data / "osm" / "hospitals.geojson"),
               "--run-meta", str(meta_path),
               "--output", str(results / "cutoff_settlements.geojson")])

    # 4. bonus: flood path tracing down the valley
    start_lat, start_lon = _upstream_point(args)
    dem = data / "dem" / "dem.tif"
    if dem.exists():
        ok &= run("4/5  Bonus — DEM flood-path tracing (+ settlements along path)",
                  [py, str(SCRIPTS / "flood_path.py"),
                   "--dem", str(dem),
                   "--start-lat", f"{start_lat:.6f}",
                   "--start-lon", f"{start_lon:.6f}",
                   "--settlements", str(data / "osm" / "settlements.geojson"),
                   "--output", str(results / "flood_path.geojson")])
    else:
        print(f"\n  SKIP 4/5 — no DEM at {dem}")

    # 5. one-page situation report
    ok &= run("5/5  Situation report",
              [py, str(SCRIPTS / "generate_report.py"),
               "--run-meta", str(meta_path),
               "--area", args.area,
               "--before-date", args.before_date,
               "--after-date", args.after_date,
               "--flood-mask", str(mask),
               "--damage", str(results / "damage_assessment.geojson"),
               "--cutoffs", str(results / "cutoff_settlements.geojson"),
               # The report states THIS run's numbers, so it belongs to this run's
               # results directory. report/ is the published handout copy and must
               # not be overwritten with synthetic figures.
               "--output", str(results / "situation_report.html")])

    print(f"\n{'=' * 72}")
    print(f" {'Offline pipeline complete.' if ok else ' Offline pipeline finished with errors.'}")
    print(f"{'=' * 72}")
    print(f"  Reminder: {BANNER}")
    print("\n  Artifacts:")
    for p in (results / "flood_mask.tif", results / "damage_assessment.geojson",
              results / "cutoff_settlements.geojson", results / "flood_path.geojson",
              results / "situation_report.html"):
        try:
            rel = p.relative_to(ROOT)
        except ValueError:
            rel = p
        print(f"    {'OK ' if p.exists() else 'MISS'}  {rel}")
    print("\n  Dashboard:   cd dashboard && python app.py")
    print("  Dashboard (these results):  "
          "JALNETRA_RESULTS_DIR=results_demo python app.py   -> a DEMO banner is shown")
    return 0 if ok else 1


def _upstream_point(args) -> tuple[float, float]:
    """Top of the river, derived from the same scene geometry as the generator."""
    import numpy as np

    from make_demo_data import make_grid, river_centerline, river_lonlat

    transform, _, _ = make_grid(args.center_lat, args.center_lon, args.extent, args.size)
    rows, cols, progress = river_centerline(args.size)
    i = int(round(0.03 * (len(rows) - 1)))  # just below the glacier (Kodari)
    xs, ys = river_lonlat(np.array([rows[i]]), np.array([cols[i]]), transform)
    return float(ys[0]), float(xs[0])


if __name__ == "__main__":
    raise SystemExit(main())
