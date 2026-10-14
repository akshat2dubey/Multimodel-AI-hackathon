#!/usr/bin/env python3
"""
Damage Assessment — Overlay Flood Mask on OSM Features

Takes the flood mask produced by inference.py and overlays it on:
  - Buildings from OpenStreetMap (pre-event)
  - Roads from OpenStreetMap (pre-event)
  - Bridges from OpenStreetMap (pre-event)

For each feature, reports flood overlap rather than structural damage:
  - UNAFFECTED, FLOODED_PARTIAL, FLOODED_MAJOR, or UNKNOWN

Usage:
    python damage_assessment.py \
        --flood-mask ./results/flood_mask.tif \
        --osm-buildings ./data/osm/buildings.geojson \
        --osm-roads ./data/osm/roads.geojson \
        --osm-bridges ./data/osm/bridges.geojson \
        --output ./results/damage_assessment.geojson
"""

import argparse

import sys
import json
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.features
from scipy.ndimage import binary_dilation
from rasterio.mask import mask as rasterio_mask
from rasterio.windows import Window
from shapely.geometry import mapping, shape
from shapely.prepared import prep

# The project root must be importable so config.presets resolves whether this is
# run as `python scripts/damage_assessment.py` or imported by a test.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.presets import FLOOD_FRACTION_THRESHOLD, case_study  # noqa: E402
from config.run_meta import run_meta_for_script  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):  # never crash on a cp1252 Windows console
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def load_flood_mask(mask_path: Path) -> tuple[np.ndarray, rasterio.profiles.Profile]:
    """Load flood mask GeoTIFF and return array + profile."""
    with rasterio.open(mask_path) as src:
        mask = src.read(1).astype(np.float32)
        profile = src.profile
        transform = src.transform
        crs = src.crs
    return mask, profile, transform, crs


def load_valid_mask(mask_path: Path) -> np.ndarray | None:
    """Return a boolean layer marking pixels the flood mask may be trusted on.

    The mask itself is produced by the segmentation model, which has no idea what
    it does not know: cloud, shadow, layover/shadow in the SAR geometry and the
    nodata collar all pass through as numbers. Two conventions are understood:

      * a second raster band containing the validity flag (1 = observed);
      * no second band -> anything equal to the mask's nodata value is invalid.

    Returns None when the raster carries no validity information at all, in which
    case callers treat every pixel as observed (the historical behaviour).
    """
    with rasterio.open(mask_path) as src:
        if src.count >= 2:
            return src.read(2) > 0.5
        nodata = src.nodata
        if nodata is None:
            return None
        return src.read(1) != nodata


def pixel_area_ha(transform: rasterio.transform.Affine, crs=None) -> float:
    """Area of one pixel in hectares, derived from the raster transform.

    A geographic CRS stores pixel size in degrees, so it must be converted to
    metres at the scene latitude before it can be squared into an area. Getting
    this wrong silently multiplies every hectare figure in the report by a
    constant, which is worse than crashing.
    """
    px_w = abs(getattr(transform, "a", 0.0))
    px_h = abs(getattr(transform, "e", 0.0))
    if px_w == 0 or px_h == 0:
        raise ValueError(f"degenerate raster transform (no pixel size): {transform}")

    if crs is not None and getattr(crs, "is_geographic", False):
        # Centre latitude of the raster, from the transform alone.
        m_per_deg = 111320.0
        lat_mid = abs(getattr(transform, "f", 0.0))
        px_w_m = px_w * m_per_deg * math.cos(math.radians(lat_mid))
        px_h_m = px_h * m_per_deg
    else:
        px_w_m = px_w
        px_h_m = px_h

    return (px_w_m * px_h_m) / 10000.0  # 1 ha = 10 000 m^2


def _geometry_window(geom, transform, shape, pad: int = 1):
    """Pixel window covering a geometry's bounds (clipped to the raster).

    Rasterising against the full grid for every feature is O(features x raster):
    on a 3000x3000 mask with 5 000 footprints that alone takes minutes, and it
    allocates a 9 MB array per call. Working on the feature's own window keeps
    the cost proportional to the footprint instead.
    """
    if geom is None or geom.is_empty:
        return None
    minx, miny, maxx, maxy = geom.bounds
    if not all(math.isfinite(v) for v in (minx, miny, maxx, maxy)):
        return None

    row_a, col_a = rasterio.transform.rowcol(transform, minx, maxy)
    row_b, col_b = rasterio.transform.rowcol(transform, maxx, miny)
    r0 = max(0, min(row_a, row_b) - pad)
    r1 = min(shape[0], max(row_a, row_b) + pad + 1)
    c0 = max(0, min(col_a, col_b) - pad)
    c1 = min(shape[1], max(col_a, col_b) + pad + 1)
    if r0 >= r1 or c0 >= c1:
        return None
    return r0, r1, c0, c1


def estimate_feature_damage(
    feature_geom,
    flood_mask: np.ndarray,
    transform: rasterio.transform.Affine,
    crs,
    feature_type: str = "building",
    valid_mask: np.ndarray | None = None,
    permanent_water_mask: np.ndarray | None = None,
) -> dict:
    """
    Estimate damage level for a single OSM feature.

    Args:
        feature_geom: Shapely geometry of the feature
        flood_mask: 2D numpy array of flood probability (0-1)
        transform: Rasterio affine transform (pixel → world coords)
        crs: Coordinate reference system
        feature_type: 'building', 'road', or 'bridge'
        valid_mask: optional boolean layer; where it is False the flood mask was
            not observed (cloud/shadow/nodata) and the feature is UNKNOWN rather
            than silently reported as unaffected.

    Returns:
        Dict with damage assessment
    """
    if feature_geom is None or feature_geom.is_empty:
        return {
            "damage_level": "UNKNOWN",
            "reason": "Empty geometry",
            "overlap_ratio": 0.0,
            "feature_pixels": 0,
            "flood_pixels": 0,
            "method": "empty_geometry",
        }

    window = _geometry_window(feature_geom, transform, flood_mask.shape,
                               pad=3 if feature_type == "bridge" else 1)
    if window is None:
        return {
            "damage_level": "UNKNOWN",
            "reason": "Feature outside image area",
            "overlap_ratio": 0.0,
            "feature_pixels": 0,
            "flood_pixels": 0,
            "method": "out_of_bounds",
        }
    r0, r1, c0, c1 = window

    sub_transform = rasterio.windows.transform(
        Window(c0, r0, c1 - c0, r1 - r0), transform
    )
    try:
        geom_rasterized = rasterio.features.rasterize(
            [(feature_geom, 1)],
            out_shape=(r1 - r0, c1 - c0),
            transform=sub_transform,
            fill=0,
            dtype=np.uint8,
        )
    except Exception as e:
        return {
            "damage_level": "UNKNOWN",
            "reason": f"Could not rasterize: {e}",
            "overlap_ratio": 0.0,
            "feature_pixels": 0,
            "flood_pixels": 0,
            "method": "rasterize_failed",
        }

    # Compute intersection inside the window (cheap) instead of the whole grid.
    feature_pixels = (geom_rasterized > 0)
    flood_pixels = (flood_mask[r0:r1, c0:c1] > 0.5)

    method = "raster_overlap"
    if feature_type == "bridge" and feature_pixels.any():
        deck_pixels = feature_pixels
        feature_pixels = binary_dilation(deck_pixels, iterations=2) & ~deck_pixels
        if permanent_water_mask is not None:
            feature_pixels &= ~(permanent_water_mask[r0:r1, c0:c1] > 0.5)
        method = "bridge_approach_ring"
        if not feature_pixels.any():
            return {
                "damage_level": "UNKNOWN",
                "reason": "No observed bridge approach pixels outside permanent water",
                "overlap_ratio": 0.0,
                "feature_pixels": 0,
                "flood_pixels": 0,
                "method": "no_bridge_approaches",
            }
    if feature_pixels.sum() == 0:
        # The footprint is thinner than a pixel (routine for OSM buildings on a
        # 10 m Sentinel-1 grid). Rasterising leaves nothing to measure, so sample
        # the mask at the centroid instead of declaring the feature unassessable.
        row, col = rasterio.transform.rowcol(
            transform, feature_geom.centroid.x, feature_geom.centroid.y)
        if not (0 <= row < flood_mask.shape[0] and 0 <= col < flood_mask.shape[1]):
            return {
                "damage_level": "UNKNOWN",
                "reason": "Feature outside image area",
                "overlap_ratio": 0.0,
                "feature_pixels": 0,
                "flood_pixels": 0,
                "method": "out_of_bounds",
            }
        if valid_mask is not None and not bool(valid_mask[row, col]):
            return {
                "damage_level": "UNKNOWN",
                "reason": "No valid observation at this pixel (cloud/shadow/nodata)",
                "overlap_ratio": 0.0,
                "feature_pixels": 0,
                "flood_pixels": 0,
                "method": "no_valid_observations",
            }
        overlap_ratio = float(flood_mask[row, col])
        intersection_sum = 0
        method = "centroid_sample"
    else:
        sub_valid = None if valid_mask is None else valid_mask[r0:r1, c0:c1]
        if sub_valid is not None:
            # The validity layer may arrive as a bool array (read from a raster
            # band) or as a float band of 1.0/0.0 — compare rather than & so both
            # work.
            observed = feature_pixels & (np.asarray(sub_valid) > 0.5)
            if not observed.any():
                # The footprint exists on the grid but every pixel covering it was
                # cloud/shadow/nodata — that is "we cannot say", not "undamaged".
                return {
                    "damage_level": "UNKNOWN",
                    "reason": "No valid observation over this feature (cloud/shadow/nodata)",
                    "overlap_ratio": 0.0,
                    "feature_pixels": int(feature_pixels.sum()),
                    "flood_pixels": 0,
                    "method": "no_valid_observations",
                }
            feature_pixels = observed
        intersection_sum = int((feature_pixels & flood_pixels).sum())
        overlap_ratio = intersection_sum / feature_pixels.sum()

    # Damage classification

    # Damage classification
    damage = (
        "FLOODED_MAJOR" if overlap_ratio >= FLOOD_FRACTION_THRESHOLD
        else "FLOODED_PARTIAL" if overlap_ratio > 0
        else "UNAFFECTED"
    )

    return {
        "damage_level": damage,
        "overlap_ratio": round(float(overlap_ratio), 3),
        "overlap_ratio_raw": float(overlap_ratio),
        "feature_pixels": int(feature_pixels.sum()),
        "flood_pixels": intersection_sum,
        "method": method,
    }


def _result_frame(results: list[dict], crs) -> gpd.GeoDataFrame:
    """Build the assessment frame.

    An empty result list cannot be handed a CRS without an explicit geometry
    column — geopandas raises, so a zero-feature area would crash the stage
    instead of reporting zeros.
    """
    if results:
        return gpd.GeoDataFrame(results, crs=crs)
    return gpd.GeoDataFrame(geometry=gpd.GeoSeries([], dtype="geometry"), crs=crs)


def assess_buildings(
    buildings_gdf: gpd.GeoDataFrame,
    flood_mask: np.ndarray,
    transform: rasterio.transform.Affine,
    crs,
    valid_mask: np.ndarray | None = None,
) -> gpd.GeoDataFrame:
    """Assess flood damage for all buildings."""
    results = []
    for idx, row in buildings_gdf.iterrows():
        geom = row.geometry
        damage_info = estimate_feature_damage(
            geom, flood_mask, transform, crs, "building", valid_mask
        )

        result = row.to_dict()
        result["damage_level"] = damage_info["damage_level"]
        result["damage_overlap_ratio"] = damage_info["overlap_ratio"]
        # "raster_overlap" or "centroid_sample" — the latter means the footprint was
        # sub-pixel, which is worth disclosing in the report's limitations.
        result["damage_method"] = damage_info.get("method", "raster_overlap")
        result["damage_reason"] = damage_info.get("reason", "")
        result["damage_assessment_method"] = "flood_mask_overlay"

        # Add flood flag for quick filtering (UNKNOWN is deliberately not "affected")
        result["flood_affected"] = damage_info["damage_level"] in (
            "FLOODED_PARTIAL", "FLOODED_MAJOR"
        )

        # Count buildings by damage
        results.append(result)

    result_gdf = _result_frame(results, crs)

    # Summary stats
    total = len(result_gdf)
    # An empty GeoDataFrame has no columns at all, so every column read below
    # must be guarded by `total` — otherwise the zero-feature run raises
    # KeyError instead of reporting zero counts.
    if total:
        affected = int(result_gdf["damage_level"].isin(
            ["FLOODED_PARTIAL", "FLOODED_MAJOR"]
        ).sum())
        major = int((result_gdf["damage_level"] == "FLOODED_MAJOR").sum())
        partial = int((result_gdf["damage_level"] == "FLOODED_PARTIAL").sum())
        unknown = int((result_gdf["damage_level"] == "UNKNOWN").sum())
    else:
        affected = major = partial = unknown = 0

    denom = max(total, 1)
    print(f"\n  🏗️  Buildings Assessment:")
    print(f"     Total buildings:     {total}")
    print(f"     Flood-affected:      {affected} ({affected / denom * 100:.1f}%)" if total
          else "     Flood-affected:      0 (no buildings)")
    print(f"     Major flooding:      {major}")
    print(f"     Partial flooding:    {partial}")
    print(f"     Unobserved (UNKNOWN):{unknown}")
    print(f"     Unaffected:          {total - affected - unknown}")

    return result_gdf


def assess_roads(
    roads_gdf: gpd.GeoDataFrame,
    flood_mask: np.ndarray,
    transform: rasterio.transform.Affine,
    crs,
    valid_mask: np.ndarray | None = None,
) -> gpd.GeoDataFrame:
    """Assess flood impact on road segments."""
    results = []
    for idx, row in roads_gdf.iterrows():
        geom = row.geometry
        damage_info = estimate_feature_damage(
            geom, flood_mask, transform, crs, "road", valid_mask
        )

        result = _safe_row_dict(row)
        result["damage_level"] = damage_info["damage_level"]
        result["damage_overlap_ratio"] = damage_info["overlap_ratio"]
        result["damage_method"] = damage_info.get("method", "")
        result["damage_reason"] = damage_info.get("reason", "")

        # Road is cut if > 50% of its length is in flood
        ratio = damage_info.get("overlap_ratio_raw", damage_info["overlap_ratio"])
        result["road_cut"] = ratio > FLOOD_FRACTION_THRESHOLD
        result["road_affected"] = ratio > 0

        results.append(result)

    result_gdf = _result_frame(results, crs)

    total = len(result_gdf)
    if total:
        cut = int(result_gdf["road_cut"].sum())
        affected = int(result_gdf["road_affected"].sum())
    else:
        cut = affected = 0

    denom = max(total, 1)
    print(f"\n  🛣️  Roads Assessment:")
    print(f"     Total road segments: {total}")
    print(f"     Road segments cut:   {cut} ({cut / denom * 100:.1f}%)" if total
          else "     Road segments cut:   0 (no road segments)")
    print(f"     Road segments affected (any): {affected}")

    return result_gdf


def assess_bridges(
    bridges_gdf: gpd.GeoDataFrame,
    flood_mask: np.ndarray,
    transform: rasterio.transform.Affine,
    crs,
    valid_mask: np.ndarray | None = None,
    permanent_water_mask: np.ndarray | None = None,
) -> gpd.GeoDataFrame:
    """Assess flood damage to bridges."""
    results = []
    for idx, row in bridges_gdf.iterrows():
        geom = row.geometry
        damage_info = estimate_feature_damage(
            geom, flood_mask, transform, crs, "bridge", valid_mask,
            permanent_water_mask,
        )

        result = _safe_row_dict(row)
        result["damage_level"] = damage_info["damage_level"]
        result["damage_overlap_ratio"] = damage_info["overlap_ratio"]
        result["damage_method"] = damage_info.get("method", "")
        result["damage_reason"] = damage_info.get("reason", "")
        result["bridge_damaged"] = damage_info.get("overlap_ratio_raw", damage_info["overlap_ratio"]) >= FLOOD_FRACTION_THRESHOLD

        results.append(result)

    result_gdf = _result_frame(results, crs)

    total = len(result_gdf)
    damaged = int(result_gdf["bridge_damaged"].sum()) if total else 0

    print(f"\n  🌉  Bridges Assessment:")
    print(f"     Total bridges:    {total}")
    print(f"     Damaged bridges:  {damaged}")

    return result_gdf


def _safe_row_dict(row) -> dict:
    """Build an assessment result dict from a GeoDataFrame row without risking a
    **row.to_dict() collision with our own keys.

    ``graph.add_node(..., **tags)`` and ``dict(**row.to_dict())`` both explode when the
    source row contains a key named ``type``, ``pos``, ``geometry`` or anything we later
    assign by keyword. The safe path is: copy every non-geometry field explicitly, then
    patch the geometry in under our own key if we need it at all. This is the same family
    of bug as the cutoff_analysis settlement-cluster path below.
    """
    out: dict = {}
    for k, v in row.items():
        if k == "geometry":
            continue
        out[str(k)] = v
    if hasattr(row, "geometry") and row.geometry is not None:
        out["geometry"] = row.geometry
    return out


def _json_safe(value):
    """Recursively make a value JSON-legal.

    OSM feature rows are assembled with pandas, and because buildings, roads and
    bridges carry different tag columns, every row ends up with NaN for the tags
    it does not have. json.dumps() happily writes a bare NaN token, which is NOT
    valid JSON — the browser then fails to parse /api/damage and the whole
    dashboard silently falls back to its empty state. Map NaN/inf to null and
    numpy scalars to plain Python types instead.
    """
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)


def save_damage_assessment(
    buildings: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
    bridges: gpd.GeoDataFrame,
    output_path: Path,
    transform=None,
    crs=None,
    run_meta: dict | None = None,
):
    """Save complete damage assessment as GeoJSON.

    ``run_meta`` (config/run_meta.py) supplies the area, bbox and dates that the
    metadata records, so this file describes the run that actually produced it.
    A direct caller that passes none gets the preset's values rather than
    nothing, so the published product still names its area and dates.
    """
    meta = run_meta if run_meta else case_study()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Combine into one FeatureCollection with feature type attribute
    all_features = []

    for gdf, feature_type in [(buildings, "building"), (roads, "road"), (bridges, "bridge")]:
        for _, row in gdf.iterrows():
            feat = {
                "type": "Feature",
                "properties": {
                    "feature_type": feature_type,
                    "damage_level": row.get("damage_level", "UNKNOWN"),
                    "damage_overlap_ratio": row.get("damage_overlap_ratio", 0),
                    "damage_method": row.get("damage_method", ""),
                    "damage_reason": row.get("damage_reason", ""),
                    "flood_affected": row.get("flood_affected", False),
                    "road_cut": row.get("road_cut", False),
                    "bridge_damaged": row.get("bridge_damaged", False),
                    "osm_id": str(row.get("id", "")),
                    "osm_tags": {k: v for k, v in row.items()
                                 if k not in ("geometry", "damage_level", "damage_overlap_ratio",
                                             "flood_affected", "road_cut", "bridge_damaged", "id",
                                             "index", "osm_id", "_geom") and not k.startswith("_")},
                },
                "geometry": mapping(row.geometry) if hasattr(row.geometry, "geom_type") else row.geometry,
            }
            all_features.append(feat)

    # Record only the artifact's own name, never an absolute local path: the
    # GeoJSON is a published product and a machine-specific path leaks the
    # operator's directory layout into it.
    metadata = {
        "description": "Flood damage assessment — overlay of flood mask on OSM features",
        "assessment_method": "flood_mask_overlay",
        "flood_mask_source": "flood_mask.tif",
        "area": meta.get("area"),
        "bbox": meta.get("bbox"),
        "aoi_confirmed": bool(meta.get("aoi_confirmed", False)),
        "osm_snapshot_date": (
            f"{meta['osm_snapshot_date']} (pre-event)"
            if meta.get("osm_snapshot_date") else None
        ),
        "event_date": meta.get("event_date"),
        "before_date": meta.get("before_date"),
        "after_date": meta.get("after_date"),
        "generated_by": "Jalnetra damage_assessment.py",
    }
    if transform is not None:
        try:
            metadata["pixel_area_ha"] = round(pixel_area_ha(transform, crs), 6)
        except ValueError:
            pass

    collection = {
        "type": "FeatureCollection",
        "features": all_features,
        "metadata": metadata,
    }

    # allow_nan=False so an unserialisable value fails loudly here rather than
    # producing a file the dashboard cannot parse.
    output_path.write_text(
        json.dumps(_json_safe(collection), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"\n  ✅ Damage assessment saved: {output_path}")
    print(f"     Total features assessed: {len(all_features)}")
    print(f"       Buildings: {len(buildings)}")
    print(f"       Roads:     {len(roads)}")
    print(f"       Bridges:   {len(bridges)}")


def main():
    parser = argparse.ArgumentParser(description="Assess flood damage to OSM features")
    parser.add_argument("--flood-mask", type=Path, required=True, help="Flood mask GeoTIFF")
    parser.add_argument("--osm-buildings", type=Path, required=True, help="OSM buildings GeoJSON")
    parser.add_argument("--osm-roads", type=Path, required=True, help="OSM roads GeoJSON")
    parser.add_argument("--osm-bridges", type=Path, required=True, help="OSM bridges GeoJSON")
    parser.add_argument("--run-meta", type=Path, default=None,
                        help="run_meta.json written by run_pipeline.py (defaults to the one beside --flood-mask)")
    parser.add_argument("--output", type=Path, default=Path("results/damage_assessment.geojson"))
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("Damage Assessment — Flood Mask Overlay on OSM Features")
    print(f"{'='*70}")

    # Which run this assessment belongs to (area, bbox, dates).
    run_meta, meta_source = run_meta_for_script(args.run_meta, args.flood_mask)
    print(f"  Run metadata: {run_meta.get('area')} "
          f"({'from ' + str(meta_source) if meta_source else 'preset fallback'})")

    # Load flood mask
    print("\n📥 Loading flood mask...")
    flood_mask, profile, transform, crs = load_flood_mask(args.flood_mask)
    print(f"  Flood mask shape: {flood_mask.shape}")
    print(f"  Flooded pixels:   {int((flood_mask > 0.5).sum())}")
    print(f"  Flood %:          {(flood_mask > 0.5).sum() / flood_mask.size * 100:.2f}%")

    # Optional validity layer (cloud/shadow/nodata) — features that fall entirely
    # inside it become UNKNOWN instead of being reported as undamaged.
    valid_mask = load_valid_mask(args.flood_mask)
    if valid_mask is None:
        print("  Validity layer:   none (every pixel treated as observed)")
    else:
        print(f"  Valid pixels:     {int(valid_mask.sum()):,} / {valid_mask.size:,}")

    # Load OSM features
    print("\n📥 Loading OSM features...")

    buildings = gpd.read_file(args.osm_buildings)
    print(f"  Buildings: {len(buildings)} loaded")

    roads = gpd.read_file(args.osm_roads)
    print(f"  Roads:     {len(roads)} loaded")

    bridges = gpd.read_file(args.osm_bridges)
    print(f"  Bridges:   {len(bridges)} loaded")

    # Ensure same CRS. An empty FeatureCollection has crs=None and cannot be
    # transformed, so empty inputs must be skipped rather than crash the run —
    # "nothing to assess" is a valid, exit-0 outcome.
    # Keep the empty frame's reported crs in sync with the raster so downstream code
    # that inspects crs does not see None for an "empty but in-scene" layer.
    if buildings.empty and buildings.crs is None:
        buildings = buildings.set_crs(crs, allow_override=True)
    elif not buildings.empty and buildings.crs != crs:
        buildings = buildings.to_crs(crs)
    if roads.empty and roads.crs is None:
        roads = roads.set_crs(crs, allow_override=True)
    elif not roads.empty and roads.crs != crs:
        roads = roads.to_crs(crs)
    if bridges.empty and bridges.crs is None:
        bridges = bridges.set_crs(crs, allow_override=True)
    elif not bridges.empty and bridges.crs != crs:
        bridges = bridges.to_crs(crs)

    # Assess damage
    print(f"\n{'─'*70}")
    print(" Assessing damage...")

    buildings_damaged = assess_buildings(buildings, flood_mask, transform, crs, valid_mask)
    roads_damaged = assess_roads(roads, flood_mask, transform, crs, valid_mask)
    permanent_water_path = args.flood_mask.parent / "flood_mask_before.tif"
    permanent_water_mask = None
    if permanent_water_path.is_file():
        with rasterio.open(permanent_water_path) as src:
            permanent_water_mask = src.read(1) > 0.5
            if (permanent_water_mask.shape != flood_mask.shape
                    or src.transform != transform or src.crs != crs):
                raise ValueError("pre-event water mask does not match the flood-mask grid")
    bridges_damaged = assess_bridges(
        bridges, flood_mask, transform, crs, valid_mask, permanent_water_mask
    )

    # Save
    save_damage_assessment(
        buildings_damaged, roads_damaged, bridges_damaged, args.output,
        transform=transform, crs=crs,
        run_meta=run_meta,
    )

    print(f"\n✅ Damage assessment complete!")


if __name__ == "__main__":
    main()
