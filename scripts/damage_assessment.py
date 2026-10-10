#!/usr/bin/env python3
"""
Damage Assessment — Overlay Flood Mask on OSM Features

Takes the flood mask produced by inference.py and overlays it on:
  - Buildings from OpenStreetMap (pre-event)
  - Roads from OpenStreetMap (pre-event)
  - Bridges from OpenStreetMap (pre-event)

For each feature, estimates damage level:
  - UNAFFECTED:    No overlap with flood mask
  - PARTIAL:      Partial overlap (< 50% of feature area in flood)
  - SEVERELY hit:  > 50% of feature area in flood
  - DESTROYED:     Feature fully within flood extent (for buildings)
  - UNKNOWN:       Feature covered by cloud/shadow in satellite imagery

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
from rasterio.mask import mask as rasterio_mask
from shapely.geometry import mapping, shape
from shapely.prepared import prep

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


def estimate_feature_damage(
    feature_geom,
    flood_mask: np.ndarray,
    transform: rasterio.transform.Affine,
    crs,
    feature_type: str = "building",
) -> dict:
    """
    Estimate damage level for a single OSM feature.

    Args:
        feature_geom: Shapely geometry of the feature
        flood_mask: 2D numpy array of flood probability (0-1)
        transform: Rasterio affine transform (pixel → world coords)
        crs: Coordinate reference system
        feature_type: 'building', 'road', or 'bridge'

    Returns:
        Dict with damage assessment
    """
    # Rasterize the feature geometry to the flood mask grid
    try:
        # Convert geometry to pixel coordinates
        geom_rasterized = rasterio.features.rasterize(
            [(feature_geom, 1)],
            out_shape=flood_mask.shape,
            transform=transform,
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

    # Compute intersection
    feature_pixels = (geom_rasterized > 0)
    flood_pixels = (flood_mask > 0.5)

    method = "raster_overlap"
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
        overlap_ratio = float(flood_mask[row, col])
        intersection_sum = 0
        method = "centroid_sample"
    else:
        intersection_sum = int((feature_pixels & flood_pixels).sum())
        overlap_ratio = intersection_sum / feature_pixels.sum()

    # Damage classification
    if feature_type == "building":
        if overlap_ratio >= 0.8:
            damage = "DESTROYED"
        elif overlap_ratio >= 0.3:
            damage = "SEVERELY_DAMAGED"
        elif overlap_ratio > 0:
            damage = "PARTIAL"
        else:
            damage = "UNAFFECTED"
    elif feature_type == "road":
        if overlap_ratio >= 0.5:
            damage = "SEGMENT_CUT"
        elif overlap_ratio > 0:
            damage = "SEGMENT_AFFECTED"
        else:
            damage = "UNAFFECTED"
    elif feature_type == "bridge":
        if overlap_ratio >= 0.3:
            damage = "BRIDGE_DAMAGED"
        else:
            damage = "BRIDGE_INTACT"
    else:
        if overlap_ratio >= 0.5:
            damage = "SEVERELY_DAMAGED"
        elif overlap_ratio > 0:
            damage = "PARTIAL"
        else:
            damage = "UNAFFECTED"

    return {
        "damage_level": damage,
        "overlap_ratio": round(float(overlap_ratio), 3),
        "feature_pixels": int(feature_pixels.sum()),
        "flood_pixels": intersection_sum,
        "method": method,
    }


def assess_buildings(
    buildings_gdf: gpd.GeoDataFrame,
    flood_mask: np.ndarray,
    transform: rasterio.transform.Affine,
    crs,
) -> gpd.GeoDataFrame:
    """Assess flood damage for all buildings."""
    results = []
    for idx, row in buildings_gdf.iterrows():
        geom = row.geometry
        damage_info = estimate_feature_damage(geom, flood_mask, transform, crs, "building")

        result = row.to_dict()
        result["damage_level"] = damage_info["damage_level"]
        result["damage_overlap_ratio"] = damage_info["overlap_ratio"]
        # "raster_overlap" or "centroid_sample" — the latter means the footprint was
        # sub-pixel, which is worth disclosing in the report's limitations.
        result["damage_method"] = damage_info.get("method", "raster_overlap")
        result["damage_assessment_method"] = "flood_mask_overlay"

        # Add flood flag for quick filtering
        result["flood_affected"] = damage_info["damage_level"] in (
            "PARTIAL", "SEVERELY_DAMAGED", "DESTROYED"
        )

        # Count buildings by damage
        results.append(result)

    result_gdf = gpd.GeoDataFrame(results, crs=crs)

    # Summary stats
    total = len(result_gdf)
    affected = result_gdf["damage_level"].isin(
        ["PARTIAL", "SEVERELY_DAMAGED", "DESTROYED"]
    ).sum()
    destroyed = (result_gdf["damage_level"] == "DESTROYED").sum()
    severe = (result_gdf["damage_level"] == "SEVERELY_DAMAGED").sum()
    partial = (result_gdf["damage_level"] == "PARTIAL").sum()

    print(f"\n  🏗️  Buildings Assessment:")
    print(f"     Total buildings:     {total}")
    print(f"     Flood-affected:      {affected} ({affected/total*100:.1f}%)")
    print(f"     Destroyed:           {destroyed}")
    print(f"     Severely damaged:    {severe}")
    print(f"     Partially flooded:   {partial}")
    print(f"     Unaffected:          {total - affected}")

    return result_gdf


def assess_roads(
    roads_gdf: gpd.GeoDataFrame,
    flood_mask: np.ndarray,
    transform: rasterio.transform.Affine,
    crs,
) -> gpd.GeoDataFrame:
    """Assess flood impact on road segments."""
    results = []
    for idx, row in roads_gdf.iterrows():
        geom = row.geometry
        damage_info = estimate_feature_damage(geom, flood_mask, transform, crs, "road")

        result = row.to_dict()
        result["damage_level"] = damage_info["damage_level"]
        result["damage_overlap_ratio"] = damage_info["overlap_ratio"]
        result["damage_method"] = damage_info.get("method", "")

        # Road is cut if > 50% of its length is in flood
        result["road_cut"] = damage_info["damage_level"] in ("SEGMENT_CUT",)
        result["road_affected"] = damage_info["damage_level"] in (
            "SEGMENT_CUT", "SEGMENT_AFFECTED"
        )

        results.append(result)

    result_gdf = gpd.GeoDataFrame(results, crs=crs)

    total = len(result_gdf)
    cut = result_gdf["road_cut"].sum()
    affected = result_gdf["road_affected"].sum()

    print(f"\n  🛣️  Roads Assessment:")
    print(f"     Total road segments: {total}")
    print(f"     Road segments cut:   {cut} ({cut/total*100:.1f}%)")
    print(f"     Road segments affected (any): {affected}")

    return result_gdf


def assess_bridges(
    bridges_gdf: gpd.GeoDataFrame,
    flood_mask: np.ndarray,
    transform: rasterio.transform.Affine,
    crs,
) -> gpd.GeoDataFrame:
    """Assess flood damage to bridges."""
    results = []
    for idx, row in bridges_gdf.iterrows():
        geom = row.geometry
        damage_info = estimate_feature_damage(geom, flood_mask, transform, crs, "bridge")

        result = row.to_dict()
        result["damage_level"] = damage_info["damage_level"]
        result["damage_overlap_ratio"] = damage_info["overlap_ratio"]
        result["damage_method"] = damage_info.get("method", "")
        result["bridge_damaged"] = damage_info["damage_level"] in (
            "BRIDGE_DAMAGED",
        )

        results.append(result)

    result_gdf = gpd.GeoDataFrame(results, crs=crs)

    total = len(result_gdf)
    damaged = result_gdf["bridge_damaged"].sum()

    print(f"\n  🌉  Bridges Assessment:")
    print(f"     Total bridges:    {total}")
    print(f"     Damaged bridges:  {damaged}")

    return result_gdf


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
):
    """Save complete damage assessment as GeoJSON."""
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
                    "flood_affected": row.get("flood_affected", False),
                    "road_cut": row.get("road_cut", False),
                    "bridge_damaged": row.get("bridge_damaged", False),
                    "osm_id": str(row.get("id", "")),
                    "osm_tags": {k: v for k, v in row.items()
                                 if k not in ("geometry", "damage_level", "damage_overlap_ratio",
                                             "flood_affected", "road_cut", "bridge_damaged", "id",
                                             "index", "osm_id") and not k.startswith("_")},
                },
                "geometry": mapping(row.geometry) if hasattr(row.geometry, "geom_type") else row.geometry,
            }
            all_features.append(feat)

    collection = {
        "type": "FeatureCollection",
        "features": all_features,
        "metadata": {
            "description": "Flood damage assessment — overlay of flood mask on OSM features",
            "assessment_method": "flood_mask_overlay",
            "flood_mask_source": str(output_path.parent / "flood_mask.tif"),
            "osm_snapshot_date": "2026-07-27 (pre-event)",
            "event_date": "2026-08-26",
            "generated_by": "Jalnetra damage_assessment.py",
        },
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
    parser.add_argument("--output", type=Path, default=Path("results/damage_assessment.geojson"))
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("Damage Assessment — Flood Mask Overlay on OSM Features")
    print(f"{'='*70}")

    # Load flood mask
    print("\n📥 Loading flood mask...")
    flood_mask, profile, transform, crs = load_flood_mask(args.flood_mask)
    print(f"  Flood mask shape: {flood_mask.shape}")
    print(f"  Flooded pixels:   {int((flood_mask > 0.5).sum())}")
    print(f"  Flood %:          {(flood_mask > 0.5).sum() / flood_mask.size * 100:.2f}%")

    # Load OSM features
    print("\n📥 Loading OSM features...")

    buildings = gpd.read_file(args.osm_buildings)
    print(f"  Buildings: {len(buildings)} loaded")

    roads = gpd.read_file(args.osm_roads)
    print(f"  Roads:     {len(roads)} loaded")

    bridges = gpd.read_file(args.osm_bridges)
    print(f"  Bridges:   {len(bridges)} loaded")

    # Ensure same CRS
    if buildings.crs != crs:
        buildings = buildings.to_crs(crs)
    if roads.crs != crs:
        roads = roads.to_crs(crs)
    if bridges.crs != crs:
        bridges = bridges.to_crs(crs)

    # Assess damage
    print(f"\n{'─'*70}")
    print(" Assessing damage...")

    buildings_damaged = assess_buildings(buildings, flood_mask, transform, crs)
    roads_damaged = assess_roads(roads, flood_mask, transform, crs)
    bridges_damaged = assess_bridges(bridges, flood_mask, transform, crs)

    # Save
    save_damage_assessment(buildings_damaged, roads_damaged, bridges_damaged, args.output)

    print(f"\n✅ Damage assessment complete!")


if __name__ == "__main__":
    main()
