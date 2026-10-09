#!/usr/bin/env python3
"""
Cut-off Settlement Analysis

Uses the road network from OSM (pre-event) and the flood mask to determine
which settlements have been cut off — i.e., no longer have a road connection
to the nearest town or hospital.

Algorithm:
  1. Build a road network graph from OSM road segments
  2. Remove road segments that intersect flooded areas (or are "destroyed"/"cut")
  3. From each town/hospital node, run BFS to find all reachable nodes
  4. Settlements not reachable = cut off
  5. For each cut-off settlement, report nearest reachable town and road segment blockage

This directly addresses Track B requirement #3:
  "Who is cut off? Using the road network, find settlements that no longer
   have a road connection to the nearest town or hospital."

Usage:
    python cutoff_analysis.py \
        --roads ./data/osm/roads.geojson \
        --buildings ./data/osm/buildings.geojson \
        --flood-mask ./results/flood_mask.tif \
        --hospitals ./data/osm/hospitals.geojson \
        --output ./results/cutoff_settlements.geojson
"""

import argparse

import sys
import json
import math
from collections import deque
from pathlib import Path
from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
import rasterio
from shapely.geometry import LineString, Point, Polygon, mapping
from shapely.ops import linemerge

if hasattr(sys.stdout, "reconfigure"):  # never crash on a cp1252 Windows console
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def load_flood_mask(mask_path: Path) -> tuple[np.ndarray, rasterio.Affine, Any]:
    """Load flood mask and return array, transform, crs."""
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        transform = src.transform
        crs = src.crs
    return mask, transform, crs


def is_road_segment_affected(
    road_geom: LineString,
    flood_mask: np.ndarray,
    transform: rasterio.Affine,
    threshold: float = 0.3,
) -> bool:
    """
    Check if a road segment intersects flooded areas significantly.

    A road is considered "cut" if > threshold fraction of its length
    passes through flooded pixels.
    """
    try:
        # Rasterize the road line
        from rasterio.features import rasterize
        road_raster = rasterize(
            [(road_geom, 1)],
            out_shape=flood_mask.shape,
            transform=transform,
            fill=0,
            dtype=np.uint8,
        )
        road_pixels = road_raster > 0
        flooded_pixels = flood_mask > 0.5

        if road_pixels.sum() == 0:
            return False

        overlap = (road_pixels & flooded_pixels).sum()
        ratio = overlap / road_pixels.sum()
        return ratio > threshold
    except Exception:
        # Fallback: check if any vertex of the line is in flood
        flood_extent = None
        try:
            from rasterio.features import geometry_mask
            # Approximate: sample points along the line
            coords = list(road_geom.coords)
            for x, y in coords[::max(1, len(coords) // 10)]:
                row, col = rasterio.transform.rowcol(transform, x, y)
                if 0 <= row < flood_mask.shape[0] and 0 <= col < flood_mask.shape[1]:
                    if flood_mask[row, col] > 0.5:
                        return True
        except Exception:
            pass
        return False


def build_road_graph(
    roads_gdf: gpd.GeoDataFrame,
    buildings_gdf: gpd.GeoDataFrame,
    flood_mask: np.ndarray,
    transform: rasterio.Affine,
    crs,
) -> tuple[nx.Graph, dict, dict]:
    """
    Build a NetworkX graph from the road network.

    Nodes: road endpoints + building locations (settlements)
    Edges: road segments (weighted by length)

    Returns:
      - graph: NetworkX graph
      - node_positions: dict of node_id → (lon, lat)
      - node_types: dict of node_id → type ('road_junction', 'settlement', 'hospital')
    """
    graph = nx.Graph()
    node_positions = {}
    node_types = {}

    # Add building locations as settlement nodes
    for _, row in buildings_gdf.iterrows():
        geom = row.geometry
        if geom.geom_type == "Point":
            node_id = f"building_{row.get('id', id(geom))}"
            pos = (geom.x, geom.y)
            graph.add_node(node_id, pos=pos, type="settlement", **row.to_dict())
            node_positions[node_id] = pos
            node_types[node_id] = "settlement"
        elif geom.geom_type in ("Polygon", "MultiPolygon"):
            centroid = geom.centroid
            node_id = f"building_{row.get('id', id(geom))}"
            pos = (centroid.x, centroid.y)
            graph.add_node(node_id, pos=pos, type="settlement", **row.to_dict())
            node_positions[node_id] = pos
            node_types[node_id] = "settlement"

    # Add road segments as edges between junction nodes
    for _, row in roads_gdf.iterrows():
        geom = row.geometry

        if geom.geom_type == "LineString":
            coords = list(geom.coords)
        elif geom.geom_type == "MultiLineString":
            coords = []
            for line in geom.geoms:
                coords.extend(list(line.coords))
        else:
            continue

        if len(coords) < 2:
            continue

        road_cut = is_road_segment_affected(geom, flood_mask, transform)

        for i in range(len(coords) - 1):
            p1 = coords[i]
            p2 = coords[i + 1]
            node1 = f"junction_{int(p1[0]*1e6)}_{int(p1[1]*1e6)}"
            node2 = f"junction_{int(p2[0]*1e6)}_{int(p2[1]*1e6)}"

            # Ensure nodes exist
            if node1 not in graph:
                graph.add_node(node1, pos=(p1[0], p1[1]), type="road_junction")
                node_positions[node1] = (p1[0], p1[1])
                node_types[node1] = "road_junction"
            if node2 not in graph:
                graph.add_node(node2, pos=(p2[0], p2[1]), type="road_junction")
                node_positions[node2] = (p2[0], p2[1])
                node_types[node2] = "road_junction"

            # Add edge with length and flood status
            segment_geom = LineString([p1, p2])
            seg_cut = is_road_segment_affected(
                segment_geom, flood_mask, transform
            )
            graph.add_edge(
                node1, node2,
                length=_segment_length_m(p1, p2, crs),
                road_cut=seg_cut or road_cut,
                flooded=seg_cut or road_cut,
            )

    # Snap each settlement (building) node onto the road network.
    # Without this, settlement nodes are isolated islands in the graph and every
    # settlement looks "cut off" regardless of the flood mask.
    junction_positions = {
        nid: pos for nid, pos in node_positions.items()
        if node_types.get(nid) == "road_junction"
    }
    snapped = 0
    for node_id, ntype in list(node_types.items()):
        if ntype != "settlement":
            continue
        pos = node_positions.get(node_id)
        if pos is None:
            continue
        nearest = _find_nearest_node(graph, junction_positions, pos, max_dist_km=2.0)
        if nearest:
            graph.add_edge(
                node_id, nearest,
                length=0.0,
                connector=True,
                road_cut=False,
                flooded=False,
            )
            snapped += 1
    n_settlements = sum(1 for t in node_types.values() if t == "settlement")
    print(f"  Settlements snapped to road network: {snapped}/{n_settlements}")

    return graph, node_positions, node_types


def _segment_length_m(p1, p2, crs) -> float:
    """Segment length in metres.

    On a geographic CRS, Shapely's .length is in DEGREES; dividing that by 1000
    to get "km" reported every road segment as 0.0 km in the report.
    """
    if crs is not None and getattr(crs, "is_geographic", False):
        return _haversine(p1[0], p1[1], p2[0], p2[1])
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def find_reachable_nodes(
    graph: nx.Graph,
    source_nodes: list[str],
    exclude_flooded: bool = True,
) -> set[str]:
    """BFS from source nodes, optionally avoiding flooded road segments."""
    reachable = set(source_nodes)

    if not source_nodes:
        return reachable

    queue = deque(source_nodes)

    while queue:
        node = queue.popleft()
        for neighbor in graph.neighbors(node):
            if neighbor in reachable:
                continue
            if exclude_flooded:
                edge_data = graph[node][neighbor]
                if edge_data.get("flooded", False):
                    continue  # Can't pass through flooded road
            reachable.add(neighbor)
            queue.append(neighbor)

    return reachable


def analyze_cutoffs(
    graph: nx.Graph,
    node_positions: dict,
    node_types: dict,
    settlements_gdf: gpd.GeoDataFrame | None,
    hospitals_gdf: gpd.GeoDataFrame | None,
    flood_mask: np.ndarray,
    transform: rasterio.Affine,
    crs,
    output_path: Path,
) -> dict[str, Any]:
    """
    Find cut-off settlements and produce results.

    A settlement is cut off if it's not reachable from any hospital
    via the road network (with flooded segments removed).
    """
    # Identify hospital nodes
    hospital_nodes = []
    if hospitals_gdf is not None and len(hospitals_gdf) > 0:
        for _, row in hospitals_gdf.iterrows():
            geom = row.geometry
            if geom.geom_type == "Point":
                pos = (geom.x, geom.y)
                node_id = f"hospital_{row.get('id', id(geom))}"
                # Find nearest road junction (must be a junction, not a building node)
                nearest = _find_nearest_node(
                    graph, node_positions, pos, node_type="road_junction"
                )
                if nearest:
                    hospital_nodes.append(nearest)
                    graph.add_node(node_id, pos=pos, type="hospital",
                                   name=row.get("name", node_id))
                    node_positions[node_id] = pos
                    node_types[node_id] = "hospital"
                    if nearest in graph:
                        graph.add_edge(node_id, nearest, length=0.001, flooded=False)
                    print(f"  🏥 Hospital node: {node_id} → nearest junction: {nearest}")

    print(f"\n  🏥  Hospitals in network: {len(hospital_nodes)}")

    # Identify settlement nodes
    settlement_nodes = [
        nid for nid, ntype in node_types.items()
        if ntype == "settlement"
    ]
    print(f"  🏘️  Settlements in network: {len(settlement_nodes)}")

    if not hospital_nodes and not settlement_nodes:
        print("  ⚠️ No hospitals or settlements in network — creating fallback analysis")
        return {"cutoff_settlements": [], "warning": "No hospitals or settlements found"}

    # Find reachable nodes from hospitals (with flood cuts)
    reachable = find_reachable_nodes(graph, hospital_nodes, exclude_flooded=True)
    unreachable = set(settlement_nodes) - reachable

    print(f"  ✅ Reachable from hospitals: {len(reachable)} nodes")
    print(f"  🚫 Cut-off settlements:       {len(unreachable)}")

    # For each cut-off settlement, find details
    cutoff_details = []
    for node_id in sorted(unreachable):
        pos = node_positions.get(node_id, (0, 0))
        # Find nearest reachable node
        nearest_reachable = _find_nearest_reachable(
            graph, node_positions, reachable, pos
        )
        # Find blocking road segments
        blocking_segments = _find_blocking_segments(
            graph, node_id, reachable
        )

        cutoff_details.append({
            "settlement_id": node_id,
            "settlement_name": (graph.nodes.get(node_id) or {}).get("name"),
            "coordinates": {
                "longitude": round(pos[0], 6),
                "latitude": round(pos[1], 6),
            },
            "nearest_reachable_town": nearest_reachable["node_id"] if nearest_reachable else None,
            "nearest_reachable_town_name": (nearest_reachable or {}).get("name"),
            "nearest_reachable_distance_km": round(nearest_reachable["distance_km"], 2) if nearest_reachable else None,
            "blocking_segments": [
                {
                    "from": seg.get("from"),
                    "to": seg.get("to"),
                    "length_km": round(seg.get("length_km", 0), 3),
                    "flooded": seg.get("flooded", True),
                }
                for seg in (blocking_segments or [])[:5]  # Top 5 blockers
            ],
            "cutoff_reason": "No road connection to hospital/town due to flood damage",
        })

    # Also compute: which road segments are flooded/cut
    cut_road_segments = []
    for u, v, data in graph.edges(data=True):
        if data.get("flooded", False):
            cut_road_segments.append({
                "from_node": u,
                "to_node": v,
                "length_km": round(data.get("length", 0) / 1000, 3),
                "flooded": True,
            })

    result = {
        "analysis_timestamp": str(Path.cwd()),
        "method": "road_network_graph_bfs",
        "flood_mask": str(output_path.parent / "flood_mask.tif"),
        "summary": {
            "total_settlements": len(settlement_nodes),
            "total_hospitals": len(hospital_nodes),
            "reachable_settlements": len(reachable & set(settlement_nodes)),
            "cutoff_settlements": len(unreachable),
            "cutoff_percentage": round(len(unreachable) / max(len(settlement_nodes), 1) * 100, 1),
            "flooded_road_segments": len(cut_road_segments),
        },
        "cutoff_settlements": cutoff_details,
        "flooded_roads": cut_road_segments[:50],  # Top 50
        "note": "Cut-off settlements are those with no road connection to any hospital or major town after removing flooded road segments.",
    }

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"\n  ✅ Cut-off analysis saved: {output_path}")
    print(f"     Cut-off settlements: {len(cutoff_details)}")
    print(f"     Flooded road segments: {len(cut_road_segments)}")

    return result


def _find_nearest_node(
    graph: nx.Graph,
    node_positions: dict,
    target_pos: tuple[float, float],
    max_dist_km: float = 5.0,
    node_type: str | None = None,
) -> str | None:
    """Find the nearest graph node to a target position.

    If node_type is given, only nodes already registered as that type in the
    graph are considered (e.g. only snap onto road junctions).
    """
    best_node = None
    best_dist = float("inf")

    for node_id, pos in node_positions.items():
        if node_type is not None and graph.nodes.get(node_id, {}).get("type") != node_type:
            continue
        dist = _haversine(pos[0], pos[1], target_pos[0], target_pos[1])
        if dist < best_dist and dist < max_dist_km * 1000:
            best_dist = dist
            best_node = node_id

    return best_node


def _find_nearest_reachable(
    graph: nx.Graph,
    node_positions: dict,
    reachable: set[str],
    target_pos: tuple[float, float],
) -> dict | None:
    """Find the nearest reachable node to a target position.

    Prefers nodes that carry a name (a village or a hospital) so the report can
    say "nearest reachable town: Barhabise" instead of an opaque junction id.
    """
    best = None
    best_dist = float("inf")
    best_named = None
    best_named_dist = float("inf")

    for node_id in reachable:
        pos = node_positions.get(node_id)
        if pos is None:
            continue
        dist = _haversine(pos[0], pos[1], target_pos[0], target_pos[1])
        name = (graph.nodes.get(node_id) or {}).get("name")
        candidate = {"node_id": node_id, "distance_km": dist / 1000, "name": name}
        if dist < best_dist:
            best_dist = dist
            best = candidate
        if name and dist < best_named_dist:
            best_named_dist = dist
            best_named = candidate

    return best_named or best


def _find_blocking_segments(
    graph: nx.Graph,
    target_node: str,
    reachable: set[str],
    max_segments: int = 5,
) -> list[dict]:
    """Find road segments that block access from target to reachable set."""
    blockers = []

    # BFS from target, stop at first unreachable edge
    visited = set()
    queue = deque([(target_node, None)])

    while queue:
        node, parent = queue.popleft()
        if node in visited:
            continue
        visited.add(node)

        if node in reachable:
            # Found a path — the edge from parent to node might be the blocker
            if parent and parent not in reachable:
                edge_data = graph[parent][node]
                blockers.append({
                    "from": parent,
                    "to": node,
                    "length_km": round(edge_data.get("length", 0) / 1000, 3),
                    "flooded": edge_data.get("flooded", True),
                })
            continue

        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                queue.append((neighbor, node))

        if len(blockers) >= max_segments:
            break

    return blockers


def _haversine(lon1, lat1, lon2, lat2) -> float:
    """Haversine distance in meters."""
    R = 6371000  # Earth radius in meters
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def main():
    parser = argparse.ArgumentParser(description="Cut-off settlement analysis")
    parser.add_argument("--roads", type=Path, required=True, help="OSM roads GeoJSON")
    parser.add_argument("--buildings", type=Path, required=True, help="OSM buildings GeoJSON")
    parser.add_argument("--flood-mask", type=Path, required=True, help="Flood mask GeoTIFF")
    parser.add_argument("--hospitals", type=Path, default=None, help="OSM hospitals GeoJSON")
    parser.add_argument("--output", type=Path, default=Path("results/cutoff_settlements.geojson"))
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("Cut-off Settlement Analysis")
    print(f"{'='*70}")
    print(f"  Roads:       {args.roads}")
    print(f"  Buildings:   {args.buildings}")
    print(f"  Flood mask:  {args.flood_mask}")
    print(f"  Hospitals:   {args.hospitals or 'Not provided — using settlements only'}")
    print(f"  Output:      {args.output}")

    # Load flood mask
    print("\n📥 Loading flood mask...")
    flood_mask, transform, crs = load_flood_mask(args.flood_mask)
    flooded_pixels = (flood_mask > 0.5).sum()
    print(f"  Flooded pixels: {flooded_pixels:,}")
    print(f"  Flood coverage: {flooded_pixels / flood_mask.size * 100:.2f}%")

    # Load OSM data
    print("\n📥 Loading OSM road network...")
    roads = gpd.read_file(args.roads)
    print(f"  Road segments: {len(roads)}")

    buildings = gpd.read_file(args.buildings)
    print(f"  Buildings:     {len(buildings)}")

    hospitals = None
    if args.hospitals and args.hospitals.exists():
        hospitals = gpd.read_file(args.hospitals)
        print(f"  Hospitals:     {len(hospitals)}")

    # Align CRS
    if roads.crs != crs:
        roads = roads.to_crs(crs)
    if buildings.crs != crs:
        buildings = buildings.to_crs(crs)
    if hospitals is not None and hospitals.crs != crs:
        hospitals = hospitals.to_crs(crs)

    # Build road graph
    print(f"\n{'─'*70}")
    print(" Building road network graph...")
    graph, node_positions, node_types = build_road_graph(
        roads, buildings, flood_mask, transform, crs
    )
    print(f"  Graph nodes: {graph.number_of_nodes()}")
    print(f"  Graph edges: {graph.number_of_edges()}")

    # Analyze cut-offs
    print(f"\n{'─'*70}")
    print(" Analyzing cut-offs...")
    result = analyze_cutoffs(
        graph, node_positions, node_types,
        buildings, hospitals,
        flood_mask, transform, crs,
        args.output,
    )

    print(f"\n{'='*70}")
    print(" Cut-off Analysis Summary")
    print(f"{'='*70}")
    print(f"  Total settlements analyzed:  {result['summary']['total_settlements']}")
    print(f"  Hospitals in network:        {result['summary']['total_hospitals']}")
    print(f"  Reachable settlements:       {result['summary']['reachable_settlements']}")
    print(f"  🚫 Cut-off settlements:      {result['summary']['cutoff_settlements']} "
          f"({result['summary']['cutoff_percentage']}%)")
    print(f"  Flooded road segments:       {result['summary']['flooded_road_segments']}")
    print(f"\n💡 Top cut-off settlements:")
    for i, s in enumerate(result["cutoff_settlements"][:5]):
        print(f"   {i+1}. ({s['coordinates']['latitude']:.4f}, {s['coordinates']['longitude']:.4f}) "
              f"— nearest town: {s.get('nearest_reachable_town', 'N/A')} "
              f"({s.get('nearest_reachable_distance_km', '?')} km)")

    print(f"\n✅ Cut-off analysis complete!")


if __name__ == "__main__":
    main()
