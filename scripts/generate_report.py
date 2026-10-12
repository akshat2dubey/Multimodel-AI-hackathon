#!/usr/bin/env python3
"""
Situation Report Generator

Generates a 1-page situation report (HTML) for rescue teams based on
the flood mapping results. Every number comes from the system's maps,
never invented by language models.

The report includes:
  - Event overview (area, date)
  - Flood extent statistics (from flood mask)
  - Damaged buildings summary (from damage assessment)
  - Cut-off settlements (from cut-off analysis)
  - List of worst-affected areas
  - Map reference for dashboard

Usage:
    python generate_report.py \
        --run-meta ./results/run_meta.json \
        --flood-mask ./results/flood_mask.tif \
        --damage ./results/damage_assessment.geojson \
        --cutoffs ./results/cutoff_settlements.geojson \
        --output ./report/situation_report.html

The report's area, event date and data label come from run_meta.json (written by
run_pipeline.py), NOT from strings baked into this file — so the header can no
longer name a corridor or a snapshot date the data does not support. --area,
--before-date and --after-date override it for a one-off run.
"""

import argparse

import sys
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio

# Project root on sys.path so `config.*` resolves whether this is run as
# `python scripts/generate_report.py` or imported by a test.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.run_meta import run_meta_for_script  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):  # never crash on a cp1252 Windows console
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def load_flood_mask_stats(mask_path: Path) -> dict:
    """Compute statistics from flood mask GeoTIFF."""
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        transform = src.transform
        crs = src.crs
        width = src.width
        height = src.height
        bounds = src.bounds

    if crs is not None and getattr(crs, "is_geographic", False):
        # Pixel size is in DEGREES here. Reporting it as metres (and squaring it
        # for area) is what made every hectare figure in the report meaningless.
        # Convert at the scene-centre latitude instead.
        lat_mid = (bounds.bottom + bounds.top) / 2.0
        m_per_deg = 111320.0
        px_w_m = abs(transform.a) * m_per_deg * math.cos(math.radians(lat_mid))
        px_h_m = abs(transform.e) * m_per_deg
        resolution = round((px_w_m + px_h_m) / 2.0, 2)
    else:
        px_w_m = abs(transform.a)
        px_h_m = abs(transform.e)
        resolution = round((px_w_m + px_h_m) / 2.0, 2)

    area_per_pixel_ha = (px_w_m * px_h_m) / 10000  # in hectares
    total_area_ha = width * height * area_per_pixel_ha

    flooded = mask > 0.5
    flood_pixels = int(flooded.sum())
    flood_ha = flood_pixels * area_per_pixel_ha
    flood_pct = flood_pixels / flooded.size * 100

    # Compute flood intensity (average probability in flooded areas)
    mean_prob = float(mask[flooded].mean()) if flood_pixels > 0 else 0

    return {
        "width_px": width,
        "height_px": height,
        "resolution_m": resolution,
        "total_area_ha": round(total_area_ha, 1),
        "flood_pixels": flood_pixels,
        "flood_area_ha": round(flood_ha, 1),
        "flood_percentage": round(flood_pct, 2),
        "mean_flood_probability": round(mean_prob, 3),
        "crs": str(crs),
    }


def load_damage_stats(damage_path: Path) -> dict:
    """Load damage assessment stats from GeoJSON."""
    with open(damage_path) as f:
        data = json.load(f)

    features = data.get("features", [])
    buildings = [f for f in features if f["properties"]["feature_type"] == "building"]
    roads = [f for f in features if f["properties"]["feature_type"] == "road"]
    bridges = [f for f in features if f["properties"]["feature_type"] == "bridge"]

    building_damage = {}
    for b in buildings:
        level = b["properties"]["damage_level"]
        building_damage[level] = building_damage.get(level, 0) + 1

    road_damage = {}
    for r in roads:
        level = r["properties"]["damage_level"]
        road_damage[level] = road_damage.get(level, 0) + 1

    bridge_damage = {}
    for br in bridges:
        level = br["properties"]["damage_level"]
        bridge_damage[level] = bridge_damage.get(level, 0) + 1

    total_buildings = len(buildings)
    affected_buildings = sum(
        v for k, v in building_damage.items()
        if k in ("PARTIAL", "SEVERELY_DAMAGED", "DESTROYED")
    )
    destroyed_buildings = building_damage.get("DESTROYED", 0)

    return {
        "total_buildings": total_buildings,
        "affected_buildings": affected_buildings,
        "affected_buildings_pct": round(affected_buildings / max(total_buildings, 1) * 100, 1),
        "destroyed_buildings": destroyed_buildings,
        "severely_damaged_buildings": building_damage.get("SEVERELY_DAMAGED", 0),
        "partially_damaged_buildings": building_damage.get("PARTIAL", 0),
        "building_damage_breakdown": building_damage,
        "total_road_segments": len(roads),
        "cut_road_segments": road_damage.get("SEGMENT_CUT", 0),
        "affected_road_segments": road_damage.get("SEGMENT_AFFECTED", 0) + road_damage.get("SEGMENT_CUT", 0),
        "total_bridges": len(bridges),
        "damaged_bridges": bridge_damage.get("BRIDGE_DAMAGED", 0),
    }


def load_cutoff_stats(cutoff_path: Path) -> dict:
    """Load cut-off analysis stats from GeoJSON."""
    with open(cutoff_path) as f:
        data = json.load(f)

    cutoff_list = data.get("cutoff_settlements", [])
    summary = data.get("summary", {})

    return {
        "total_settlements": summary.get("total_settlements", 0),
        "cutoff_settlements": summary.get("cutoff_settlements", len(cutoff_list)),
        "cutoff_percentage": summary.get("cutoff_percentage", 0),
        "reachable_settlements": summary.get("reachable_settlements", 0),
        "flooded_road_segments": summary.get("flooded_road_segments", 0),
        "already_isolated_settlements": summary.get("already_isolated_settlements", 0),
        # Without a reference facility the cut-off analysis deliberately produced
        # an empty list. Carrying the status through stops the report from
        # presenting that as a reassuring "0% cut off".
        "status": data.get("status", "ok"),
        "reference_facility_note": data.get("reference_facility_note"),
        "cutoff_list": cutoff_list,
    }


def generate_html_report(
    area: str,
    before_date: str,
    after_date: str,
    flood_stats: dict,
    damage_stats: dict,
    cutoff_stats: dict,
    output_path: Path,
    synthetic: bool = False,
    run_meta: dict | None = None,
):
    """Generate the HTML situation report.

    ``run_meta`` is the resolved run metadata (config/run_meta.py): when given,
    the event date, the bbox in the header and the data label come from it, so
    the report describes the run that actually produced its inputs.
    """

    today = datetime.now().strftime("%d %B %Y")
    meta = run_meta or {}
    # The event date is the event, not the post-event acquisition date: those
    # differ by design (the after scene may be days later), and using the
    # acquisition date silently moved the reported event.
    event_date = meta.get("event_date") or after_date

    bbox = meta.get("bbox")
    bbox_text = ", ".join(f"{v:.3f}" for v in bbox) if bbox else ""
    meta_line = f"Flood Event Analysis • {event_date}" + (
        f" • bbox {bbox_text}" if bbox_text else ""
    )

    # An unconfirmed AOI must be visible in the artifact itself, not only in a
    # console warning nobody keeps.
    if meta and not meta.get("aoi_confirmed", False):
        aoi_banner = """
  <div style="background:#fff8e1; border:1px solid #ffe082; border-left:4px solid #f9a825;
              padding:12px 18px; margin:0 0 18px; border-radius:4px; font-size:12px; color:#5d4037;">
    <strong>AREA OF INTEREST NOT CONFIRMED.</strong> This run used a fallback
    bounding box (""" + str(meta.get("bbox_source", "unknown")) + """), not the
    confirmed case-study AOI. Treat the area covered here as provisional.
  </div>"""
    else:
        aoi_banner = ""

    if synthetic:
        data_label = "SYNTHETIC DEMO DATA — not real Earth observation data"
        synthetic_banner = """
  <div style="background:#fff3e0; border:1px solid #ffb74d; border-left:4px solid #e65100;
              padding:14px 18px; margin:0 0 20px; border-radius:4px;">
    <strong style="color:#e65100;">DEMONSTRATION OUTPUT — SYNTHETIC DATA</strong>
    <div style="font-size:12px; color:#6d4c41; margin-top:4px;">
      Every figure in this report was computed from a fabricated test fixture
      (scripts/make_demo_data.py), not from Copernicus Sentinel-1/-2 or a real DEM.
      It validates the pipeline, and must not be quoted as a measurement of the
      August 2026 Nepal event.
    </div>
  </div>"""
    else:
        snapshot = meta.get("osm_snapshot_date")
        data_label = "Sentinel-1 SAR + OSM pre-event snapshot" + (
            f" ({snapshot})" if snapshot else ""
        )
        synthetic_banner = ""

    # A cut-off analysis that had no reference facility must not read as "0%".
    if cutoff_stats.get("status", "ok") != "ok" or cutoff_stats.get("reference_facility_note"):
        note = (cutoff_stats.get("reference_facility_note")
                or "Cut-off analysis did not run for this area.")
        reference_note = (
            '<div style="margin-top:10px; padding:8px 10px; background:#fff8e1; '
            'border-left:3px solid #f9a825; font-size:12px; color:#5d4037;">'
            f"{note}</div>"
        )
    else:
        reference_note = ""

    # Format large numbers
    def fmt(n):
        if n >= 1000000:
            return f"{n/1000000:.1f}M"
        elif n >= 1000:
            return f"{n/1000:.1f}K"
        return str(n)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Situation Report — {area}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
    background: #f5f5f5;
    color: #222;
    line-height: 1.5;
  }}
  .page {{
    max-width: 800px;
    margin: 0 auto;
    background: white;
    padding: 40px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
  }}
  .header {{
    border-bottom: 3px solid #c62828;
    padding-bottom: 20px;
    margin-bottom: 24px;
  }}
  .header .badge {{
    display: inline-block;
    background: #c62828;
    color: white;
    padding: 4px 12px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }}
  .header h1 {{
    font-size: 24px;
    margin: 8px 0 4px;
    color: #1a1a1a;
  }}
  .header .meta {{
    color: #666;
    font-size: 14px;
    margin-top: 4px;
  }}
  .header .report-date {{
    color: #999;
    font-size: 12px;
    margin-top: 8px;
  }}
  .alert-box {{
    background: #fff3f3;
    border: 1px solid #ef9a9a;
    border-left: 4px solid #c62828;
    padding: 16px 20px;
    margin: 20px 0;
    border-radius: 4px;
  }}
  .alert-box .alert-num {{
    font-size: 32px;
    font-weight: 700;
    color: #c62828;
  }}
  .grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin: 20px 0;
  }}
  .card {{
    background: #fafafa;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 16px;
  }}
  .card h3 {{
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #888;
    margin-bottom: 8px;
  }}
  .card .value {{
    font-size: 28px;
    font-weight: 700;
    color: #222;
  }}
  .card .value.danger {{ color: #c62828; }}
  .card .value.warning {{ color: #e65100; }}
  .card .value.ok {{ color: #2e7d32; }}
  .card .sub {{
    font-size: 12px;
    color: #999;
    margin-top: 2px;
  }}
  .full-width {{ grid-column: 1 / -1; }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
    margin: 12px 0;
  }}
  th {{
    background: #f5f5f5;
    text-align: left;
    padding: 8px 12px;
    font-weight: 600;
    font-size: 12px;
    text-transform: uppercase;
    color: #666;
    border-bottom: 2px solid #e0e0e0;
  }}
  td {{
    padding: 8px 12px;
    border-bottom: 1px solid #eee;
  }}
  tr:nth-child(even) td {{ background: #fafafa; }}
  .cutoff-list {{
    max-height: 200px;
    overflow-y: auto;
  }}
  .footer {{
    margin-top: 32px;
    padding-top: 16px;
    border-top: 1px solid #e0e0e0;
    font-size: 11px;
    color: #999;
    line-height: 1.6;
  }}
  .rating {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
  }}
  .rating.critical {{ background: #ffcdd2; color: #b71c1c; }}
  .rating.high {{ background: #ffe0b2; color: #e65100; }}
  .rating.moderate {{ background: #fff9c4; color: #f57f17; }}
  .rating.low {{ background: #c8e6c9; color: #2e7d32; }}
</style>
</head>
<body>
<div class="page">
  <div class="header">
    <span class="badge">Situation Report • Automated</span>
    <h1>{area}</h1>
    <div class="meta">
      {meta_line}
    </div>
    <div class="report-date">
      Generated: {today} • By Jalnetra System • Data: {data_label}
    </div>
  </div>
{synthetic_banner}
{aoi_banner}

  <div class="alert-box">
    <span class="alert-num">{cutoff_stats['cutoff_settlements']}</span>
    settlements cut off from hospitals and towns due to flood-damaged roads.
    <span style="margin-left: 16px; color: #666; font-size: 13px;">
      Rescue teams should prioritize road clearance on the segments listed below.
    </span>
  </div>

  <div class="grid">
    <div class="card">
      <h3>Flood Extent</h3>
      <div class="value danger">{flood_stats['flood_area_ha']:,} <span style="font-size:14px;color:#888;">hectares</span></div>
      <div class="sub">{flood_stats['flood_pixels']:,} pixels • {flood_stats['flood_percentage']}% of area • {flood_stats['resolution_m']}m resolution</div>
    </div>

    <div class="card">
      <h3>Buildings Affected</h3>
      <div class="value warning">{damage_stats['affected_buildings']} <span style="font-size:14px;color:#888;">of {damage_stats['total_buildings']}</span></div>
      <div class="sub">{damage_stats['affected_buildings_pct']}% affected • {damage_stats['destroyed_buildings']} destroyed</div>
    </div>

    <div class="card">
      <h3>Road Segments Blocked</h3>
      <div class="value danger">{cutoff_stats['flooded_road_segments']}</div>
      <div class="sub">network edges cut by flooding &middot;
        {damage_stats['affected_road_segments']} of {damage_stats['total_road_segments']}
        OSM ways touched, {damage_stats['cut_road_segments']} fully cut</div>
    </div>

    <div class="card">
      <h3>Bridges Damaged</h3>
      <div class="value {'danger' if damage_stats['damaged_bridges'] > 0 else 'ok'}">{damage_stats['damaged_bridges']}</div>
      <div class="sub">of {damage_stats['total_bridges']} total bridges</div>
    </div>
  </div>

  <div class="full-width">
    <h2 style="font-size:16px; margin-bottom:12px; color:#444;">Cut-Off Settlements — Priority Rescue Targets</h2>
    <table>
      <thead>
        <tr>
          <th>#</th>
          <th>Settlement</th>
          <th>Location (lat, lon)</th>
          <th>Nearest Reachable Town</th>
          <th>Distance (km)</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>
"""

    cutoff_list = cutoff_stats["cutoff_list"]
    if cutoff_list:
        for i, s in enumerate(cutoff_list[:20], 1):
            lat = s["coordinates"]["latitude"]
            lon = s["coordinates"]["longitude"]
            nearest = (s.get("nearest_reachable_town_name")
                       or s.get("nearest_reachable_town") or "Unknown")
            dist = s.get("nearest_reachable_distance_km", "—")
            html += f"""
        <tr>
          <td>{i}</td>
          <td><strong>{s.get('settlement_name', f'Settlement #{i}')}</strong></td>
          <td>{lat:.4f}, {lon:.4f}</td>
          <td>{str(nearest)[:30]}</td>
          <td>{dist}</td>
          <td><span class="rating critical">CUT OFF</span></td>
        </tr>"""
    else:
        html += """
        <tr>
          <td colspan="6" style="text-align:center; color:#999; padding:24px;">
            No cut-off settlements identified in this analysis area.
          </td>
        </tr>"""

    html += f"""
      </tbody>
    </table>
  </div>

  <div class="full-width" style="margin-top:20px;">
    <h2 style="font-size:16px; margin-bottom:12px; color:#444;">Damage Breakdown</h2>
    <div class="grid" style="grid-template-columns: 1fr 1fr 1fr;">
      <div class="card">
        <h3>Buildings by Damage Level</h3>
        <table style="font-size:13px;">
          <tbody>
            <tr><td>Destroyed</td><td><span class="rating critical">{damage_stats.get('destroyed_buildings', 0)}</span></td></tr>
            <tr><td>Severely damaged</td><td><span class="rating high">{damage_stats.get('severely_damaged_buildings', 0)}</span></td></tr>
            <tr><td>Partially flooded</td><td><span class="rating moderate">{damage_stats.get('partially_damaged_buildings', 0)}</span></td></tr>
            <tr><td>Unaffected</td><td><span class="rating low">{damage_stats.get('total_buildings', 0) - damage_stats.get('affected_buildings', 0)}</span></td></tr>
          </tbody>
        </table>
      </div>
      <div class="card">
        <h3>Road Network Status</h3>
        <table style="font-size:13px;">
          <tbody>
            <tr><td>Total segments</td><td>{damage_stats['total_road_segments']}</td></tr>
            <tr><td>OSM ways fully cut (feature-level)</td><td><span class="rating critical">{damage_stats['cut_road_segments']}</span></td></tr>
            <tr><td>Road-network edges cut (graph-level)</td><td><span class="rating critical">{cutoff_stats['flooded_road_segments']}</span></td></tr>
            <tr><td>Segments affected</td><td><span class="rating high">{damage_stats['affected_road_segments']}</span></td></tr>
            <tr><td>Bridges damaged</td><td><span class="rating {'critical' if damage_stats['damaged_bridges'] > 0 else 'low'}">{damage_stats['damaged_bridges']}</span></td></tr>
          </tbody>
        </table>
      </div>
      <div class="card">
        <h3>Connectivity</h3>
        <table style="font-size:13px;">
          <tbody>
            <tr><td>Total settlements</td><td>{cutoff_stats['total_settlements']}</td></tr>
            <tr><td>Reachable</td><td><span class="rating low">{cutoff_stats['reachable_settlements']}</span></td></tr>
            <tr><td>Cut off</td><td><span class="rating critical">{cutoff_stats['cutoff_settlements']}</span></td></tr>
            <tr><td>Cut-off rate</td><td><strong>{cutoff_stats['cutoff_percentage']}%</strong></td></tr>
            <tr><td>Already isolated before the event</td><td>{cutoff_stats['already_isolated_settlements']}</td></tr>
          </tbody>
        </table>
{reference_note}
      </div>
    </div>
  </div>

  <div class="footer">
    <strong>Data sources:</strong> Contains modified Copernicus Sentinel data 2026 •
    Produced using Copernicus WorldDEM-30 — DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH
    2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved •
    © OpenStreetMap contributors (pre-event snapshot: 27 July 2026) •
    Training data: Bountos et al., 2024 (Kuro Siwo, NeurIPS 2024) •
    <br>
    <strong>Disclaimer:</strong> This is an educational prototype, not an operational disaster response tool.
    Sentinel satellite revisit times (several days) mean this system cannot provide real-time warnings.
    Results should be verified by ground teams before rescue operations.
    Generated by Jalnetra — Multimodal AI Hackathon 2026, Track B.
  </div>
</div>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"  ✅ Situation report saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate situation report")
    # run_meta.json is the source of truth for area/dates; these flags are
    # overrides for a one-off run that has no metadata file.
    parser.add_argument("--run-meta", type=Path, default=None,
                        help="run_meta.json written by run_pipeline.py (defaults to the one beside --flood-mask)")
    parser.add_argument("--area", default=None, help="Area name (defaults to run_meta)")
    parser.add_argument("--before-date", default=None, help="Before-event date (defaults to run_meta)")
    parser.add_argument("--after-date", default=None, help="After-event date (defaults to run_meta)")
    parser.add_argument("--flood-mask", type=Path, required=True, help="Flood mask GeoTIFF")
    parser.add_argument("--damage", type=Path, required=True, help="Damage assessment GeoJSON")
    parser.add_argument("--cutoffs", type=Path, required=True, help="Cut-off analysis GeoJSON")
    parser.add_argument("--output", type=Path, default=Path("report/situation_report.html"))
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("Generating Situation Report")
    print(f"{'='*70}")

    # Area and dates: run_meta.json first, CLI override second.
    run_meta, meta_source = run_meta_for_script(args.run_meta, args.flood_mask)
    area = args.area or run_meta.get("area")
    before_date = args.before_date or run_meta.get("before_date")
    after_date = args.after_date or run_meta.get("after_date")
    missing = [n for n, v in (("area", area), ("before-date", before_date),
                              ("after-date", after_date)) if not v]
    if missing:
        print(
            f"❌ no {', '.join(missing)}: pass --run-meta (results/run_meta.json) "
            "or the matching flag. The report must not invent an area or a date.",
            file=sys.stderr,
        )
        sys.exit(2)
    where = f"from {meta_source}" if meta_source else "(preset fallback)"
    print(f"  Run metadata {where}: {area} | {before_date} → {after_date}")

    print("\n📊 Loading flood mask statistics...")
    flood_stats = load_flood_mask_stats(args.flood_mask)
    print(f"  Flood area: {flood_stats['flood_area_ha']} ha ({flood_stats['flood_percentage']}%)")

    print("\n📊 Loading damage assessment...")
    damage_stats = load_damage_stats(args.damage)
    print(f"  Buildings affected: {damage_stats['affected_buildings']} of {damage_stats['total_buildings']}")
    print(f"  Roads cut: {damage_stats['cut_road_segments']} of {damage_stats['total_road_segments']}")

    print("\n📊 Loading cut-off analysis...")
    cutoff_stats = load_cutoff_stats(args.cutoffs)
    print(f"  Cut-off settlements: {cutoff_stats['cutoff_settlements']}")
    print(f"  Cut-off rate: {cutoff_stats['cutoff_percentage']}%")

    # Flag synthetic inputs so a demo fixture can never be mistaken for a
    # real event measurement in the published report.
    synthetic = False
    change_json = args.flood_mask.parent / "change_detection.json"
    if change_json.exists():
        try:
            synthetic = bool(json.loads(
                change_json.read_text(encoding="utf-8")).get("synthetic"))
        except Exception:
            pass
    if (args.flood_mask.parent / "DEMO_DATA_NOTICE.txt").exists():
        synthetic = True
    if synthetic:
        print("  NOTE: inputs are SYNTHETIC demo data — report will be marked")

    print("\n📝 Generating HTML report...")
    generate_html_report(
        area, before_date, after_date,
        flood_stats, damage_stats, cutoff_stats,
        args.output,
        synthetic=synthetic,
        run_meta=run_meta,
    )

    print(f"\n✅ Situation report complete!")
    print(f"   Open: {args.output}")


if __name__ == "__main__":
    main()
