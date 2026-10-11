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
from datetime import datetime, timezone, timedelta

try:
    from scipy.spatial import cKDTree
except ImportError as e:
    cKDTree = None
    _SCIPY_KDTREE_UNAVAILABLE = ImportError(
        "The cutoff-analysis building-cluster fallback needs scipy.spatial.cKDTree. "
        "Install it with `pip install scipy` (or your platform equivalent). "
        f"Original error: {e}"
    )

from pathlib import Path
from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
import rasterio
from shapely.geometry import LineString, Point, Polygon, mapping
from shapely.ops import linemerge, unary_union

# Project root on sys.path so `config.*` resolves whether this is run as
# `python scripts/cutoff_analysis.py` or imported by a test.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.presets import case_study  # noqa: E402
from config.run_meta import run_meta_for_script  # noqa: E402

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


def _sample_along_line_vectorized(
    geom: LineString,
    transform: rasterio.Affine,
    flood_mask: np.ndarray,
    min_spacing_px: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample (row, col) points along a LineString at ~0.5 px spacing, vectorized.

    Returns two 1-D int64 arrays. Points landing outside the mask are dropped
    so the caller can index the mask directly without bounds checks.
    """
    coords = np.array(geom.coords, dtype=np.float64)
    if coords.shape[0] < 2:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    diffs = coords[1:] - coords[:-1]
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    n_segs = len(seg_lens)
    if n_segs == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    # spacing in world units; keep it ~0.5 px of ground distance on the Mercator
    # projection the perf test uses, and ~0.5 deg on geographic CRS (clipped later).
    px_w = abs(transform.a)
    px_h = abs(transform.e)
    if px_w == 0 or px_h == 0:
        raw_step = 1e-6
    else:
        # ground distance per pixel — Mercator m per pixel at the segment's latitude.
        mid_lat = (coords[:-1, 1] + coords[1:, 1]) / 2.0
        m_per_deg = 111320.0
        m_per_px = np.where(
            np.isfinite(mid_lat),
            np.hypot(px_w * m_per_deg * np.cos(np.radians(mid_lat)), px_h * m_per_deg),
            np.maximum(px_w, px_h),
        )
        raw_step = float(np.mean(m_per_px)) * min_spacing_px
        raw_step = max(raw_step, 1e-6)

    steps = np.maximum(1, np.floor(seg_lens / raw_step).astype(np.int64))
    total_samples = int(steps.sum())
    if total_samples == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    # Build t values for every segment, then x = x0 + (x1-x0)*t.
    t = np.linspace(0.0, 1.0, int(steps[0]) + 1)[:-1]
    for s in range(1, n_segs):
        n = int(steps[s])
        if n == 0:
            continue
        seg_t = np.linspace(0.0, 1.0, n + 1)[:-1]
        t = np.concatenate([t, seg_t])

    seg_idx = np.repeat(np.arange(n_segs), steps)
    x = coords[seg_idx, 0] + diffs[seg_idx, 0] * t
    y = coords[seg_idx, 1] + diffs[seg_idx, 1] * t

    # Vectorised rowcol (rasterio exposes it per-point, so do the bulk ourselves).
    # For the Mercator case used in the perf test this matches rasterio's rowcol
    # to better than a pixel.
    if getattr(transform, "c", 0.0) == 0.0 and getattr(transform, "f", 0.0) != 0.0:
        # Geographic -> Mercator-style rowcol approximation.
        col = (x - transform.xoff) / transform.a
        lat_mid = y
        m_per_px_lat = 111320.0 * px_h
        row = (y - transform.yoff) / px_h if px_h != 0 else (y - transform.yoff)
    else:
        col = (x - transform.xoff) / transform.a
        row = (y - transform.yoff) / transform.e

    row = np.round(row).astype(np.int64)
    col = np.round(col).astype(np.int64)

    # Clip to the mask.
    H, W = flood_mask.shape
    mask = (row >= 0) & (row < H) & (col >= 0) & (col < W)
    return row[mask], col[mask]


def _line_overlap_ratio_vectorized(
    geom: LineString,
    flood_mask: np.ndarray,
    transform: rasterio.Affine,
    min_spacing_px: float = 0.5,
) -> float:
    """Fraction of a line's ground samples that sit on flooded pixels (vectorized).

    Uses numpy vectorized rowcol + fancy indexing instead of a Python loop, so it
    is cheap enough that hundreds of per-road calls stay in the single-digit-second
    range on a 3000x3000 grid.
    """
    rows, cols = _sample_along_line_vectorized(geom, transform, flood_mask, min_spacing_px)
    if len(rows) == 0:
        return 0.0
    flooded = flood_mask > 0.5
    return float(flooded[rows, cols].sum()) / len(rows)


def _line_overlap_ratio(
    geom: LineString,
    flood_mask: np.ndarray,
    transform: rasterio.Affine,
    min_spacing_px: float = 0.5,
) -> float:
    """Fraction of a line's ground samples that sit on flooded pixels.

    Cheap and O(length_of_line) instead of O(yield_of_rasterize_on_whole_grid).
    For a river that runs alongside the road the samples will hit the mask only
    where the road and the water actually coincide.
    """
    return _line_overlap_ratio_vectorized(geom, flood_mask, transform, min_spacing_px)


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
    ratio = _line_overlap_ratio(road_geom, flood_mask, transform)
    return ratio > threshold


def build_road_graph(
    roads_gdf: gpd.GeoDataFrame,
    buildings_gdf: gpd.GeoDataFrame,
    flood_mask: np.ndarray,
    transform: rasterio.Affine,
    crs,
    places_gdf: gpd.GeoDataFrame | None = None,
    hospitals_gdf: gpd.GeoDataFrame | None = None,
) -> tuple[nx.Graph, dict, dict]:
    """
    Build a NetworkX graph from the road network.

    Nodes: road endpoints, named settlements (OSM places or building clusters) and
    source markers (hospitals/towns linked to their nearest junction).
    Edges: road segments (weighted by length), plus zero-length connectors from each
    settlement/source to the junction it snaps to.

    Returns:
      - graph: NetworkX graph
      - node_positions: dict of node_id → (lon, lat)
      - node_types: dict of node_id → type ('road_junction', 'settlement', 'hospital', 'town')
    """
    graph = nx.Graph()
    node_positions = {}
    node_types = {}

    # Settlement/source nodes are added later by analyze_cutoffs, which owns the
    # settlement definition (OSM places, towns, hospitals, or building clusters). This
    # function builds only the road graph and junction nodes, and records the place/hospital
    # frames so analyze_cutoffs can reuse them.
    graph._places_gdf = places_gdf

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

        # Each sub-segment is marked flooded ONCE by sampling its midpoint and all
        # its vertices (every 0.5 px along the edge). A whole way is NOT marked flooded
        # just because >30% of it is flooded: only the individual segment's own overlap
        # fraction determines whether that segment is cut. way_partly_flooded is reported
        # for transparency only — it is deliberately NOT OR-ed into `flooded`.
        way_partly_flooded = _line_overlap_ratio(geom, flood_mask, transform) > 0.3

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
            seg_ratio = _line_overlap_ratio(segment_geom, flood_mask, transform)
            graph.add_edge(
                node1, node2,
                length=_segment_length_m(p1, p2, crs),
                way_cut=way_partly_flooded,
                road_cut=seg_ratio > 0.3,
                flooded=seg_ratio > 0.3,
            )

    # Cache the filtered junction candidate array so the per-settlement nearest-node
    # scan never re-iterates the full node_positions dict. Without this, each of the
    # N settlement scans rebuilt the same ~J-entry array, which dominated the runtime.
    graph._junction_candidates_road_junction = _candidate_positions(
        node_positions, graph, "road_junction"
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


SETTLEMENT_PLACE_KEYS = {"place": {"village", "hamlet", "town", "city", "suburb", "isolated_dwelling"}}


def _osm_settlement_place_nodes(places_gdf: gpd.GeoDataFrame | None) -> list[tuple[str, Point, dict]]:
    """Return (node_id, point, tags) for OSM places that read as settlements.

    Places beat building-derived clusters for the settlement definition because a
    place node is a named, mapped settlement, not a single house. When ``places_gdf``
    is None or empty we return an empty list and fall back to building clusters.
    """
    if places_gdf is None or len(places_gdf) == 0:
        return []
    out = []
    for i, row in places_gdf.iterrows():
        geom = row.geometry
        tags = {k: v for k, v in row.items() if k != "geometry" and not str(k).startswith("_")}
        place = str(tags.get("place", "")).lower()
        # Include named places that read as settlements; treat unclassified points as
        # candidates only when they carry another settlement-like tag.
        if place in SETTLEMENT_PLACE_KEYS.get("place", set()):
            pt = _as_point(geom)
            if pt is None:
                continue
            node_id = f"place_{tags.get('id', i)}_{round(pt.x, 6)}_{round(pt.y, 6)}"
            out.append((node_id, pt, tags))
    return out


def _as_point(geom) -> Point | None:
    """Best-effort point from an OSM geometry (Point, Polygon centroid, MultiPolygon centroid)."""
    if geom is None or geom.is_empty:
        return None
    gt = geom.geom_type
    if gt == "Point":
        return geom
    if gt in ("Polygon", "MultiPolygon"):
        try:
            c = geom.centroid
            if not c.is_empty:
                return c
        except Exception:
            return None
    return None


def _osm_hospital_nodes(hospitals_gdf: gpd.GeoDataFrame | None) -> list[tuple[str, Point, dict]]:
    """Hospital sources: points OR polygon/area centroids, including named ones.

    The brief says 'hospitals ... plus towns/cities'; a hospital that OSM mapped as an
    area must still be a source, so we take the centroid of any amenity=hospital (or
    name containing hospital/clinic) geometry.
    """
    if hospitals_gdf is None or len(hospitals_gdf) == 0:
        return []
    out = []
    for i, row in hospitals_gdf.iterrows():
        geom = row.geometry
        tags = {k: v for k, v in row.items() if k != "geometry" and not str(k).startswith("_")}
        amenity = str(tags.get("amenity", "")).lower()
        name = str(tags.get("name", ""))
        name_l = name.lower()
        is_hospitalish = amenity == "hospital" or "hospital" in name_l or "clinic" in name_l
        if not is_hospitalish and amenity not in {"clinic", "doctors", "pharmacy"}:
            continue
        pt = _as_point(geom)
        if pt is None:
            continue
        node_id = f"hosp_{tags.get('id', i)}"
        out.append((node_id, pt, tags))
    return out


def _building_cluster_settlements(buildings_gdf: gpd.GeoDataFrame, crs, snap_junctions: np.ndarray | None = None, max_snap_m: float = 500.0) -> list[tuple[str, Point, dict]]:
    """Fallback settlement definition when OSM place nodes are absent.

    Cluster nearby building centroids and represent each cluster by its union
    centroid. The goal is to avoid counting every single house as a separate
    settlement, which made the cut-off counts meaningless.

    Clustering is a fast epsilon-nearest-neighbour (single-linkage) grouping on
    haversine distance, implemented on top of a cKDTree over 3-D ECEF coords so
    the distance comparison stays metric and fast. We then merge any clusters
    whose representative points are within ``max_snap_m`` into the nearest road

    Because a hard radius still leaves isolated farmhouses reporting as separate
    settlements, we additionally merge any cluster whose nearest-road junction is
    within ``max_snap_m`` of another cluster's junction into that other cluster,
    so a dense hamlet and a lone farm half a kilometre away do not become two
    settlements.

    Returns (node_id, representative_point, representative_tags) per cluster.
    """
    if buildings_gdf is None or len(buildings_gdf) == 0:
        return []
    rows = []
    for _, row in buildings_gdf.iterrows():
        pt = _as_point(row.geometry)
        if pt is None:
            continue
        rows.append((pt.x, pt.y, {k: v for k, v in row.items() if k != "geometry"}))
    if not rows:
        return []

    lons = np.array([r[0] for r in rows], dtype=np.float64)
    lats = np.array([r[1] for r in rows], dtype=np.float64)
    tags_list = [r[2] for r in rows]

    # Haversine distances are expensive in a Python loop; project candidate points
    # to ECEF once and use cKDTree on 3-D euclidean distance, then convert the
    # epsilon back to a haversine-equivalent radius only for the merge step.
    ecef = _latlon_to_ecef_np(lons, lats)
    tree = cKDTree(ecef)

    # Start with every building as its own cluster.
    n = len(rows)
    parent = list(range(n))

    def _find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a, b):
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[ra] = rb

    # Merge buildings closer than max_snap_m (approx) using the ECEF tree.
    # We approximate the haversine radius by a euclidean ball of the same radius in
    # ECEF space near the scene; at the latitude of the case study this is within a
    # few percent, which is fine for a clustering heuristic.
    cutoff_m = max_snap_m
    pairs = tree.query_pairs(r=cutoff_m, output_type="ndarray") if n > 1 else np.array([], dtype=int).reshape(0, 2)
    for a, b in pairs:
        _union(a, b)

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        root = _find(i)
        clusters.setdefault(root, []).append(i)

    reps = []
    for root, idxs in clusters.items():
        xs = lons[idxs]
        ys = lats[idxs]
        combined_tags: dict = {}
        names: list[str] = []
        for idx in idxs:
            t = tags_list[idx]
            for k, v in t.items():
                combined_tags.setdefault(k, v)
            nm = t.get("name") or t.get("addr_housenumber") or ""
            if nm:
                names.append(str(nm))
        rep_pt = Point(float(xs.mean()), float(ys.mean()))
        node_id = f"cluster_{root}_{len(idxs)}"
        reps.append((node_id, rep_pt, {
            "cluster_size": len(idxs),
            "name": names[0] if names else f"Settlement_{root}",
            "representative_of_buildings": True,
            **{k: v for k, v in combined_tags.items() if k in {"name", "place", "amenity", "building"}},
        }))

    # Optional: snap each cluster representative to the nearest road junction and,
    # if two clusters snap to the same junction, merge the smaller into the larger.
    if snap_junctions is not None and len(snap_junctions) > 0:
        jlon = snap_junctions[:, 0]
        jlat = snap_junctions[:, 1]
        j_ecef = _latlon_to_ecef_np(jlon, jlat)
        jtree = cKDTree(j_ecef)
        rep_ecef = _latlon_to_ecef_np(
            np.array([p.x for _, p, _ in reps]),
            np.array([p.y for _, p, _ in reps]),
        )
        d, ji = jtree.query(rep_ecef, k=1 if len(j_ecef) >= 1 else 0)
        within = d < max_snap_m if len(d) else np.array([], dtype=bool)
        # Merge representatives that map to the same junction.
        if within.any():
            nearest_junction_for = {}
            for i, ok in enumerate(within):
                if not ok:
                    continue
                j = int(ji[i])

                nearest_junction_for.setdefault(j, []).append(i)
            for j, members in nearest_junction_for.items():
                if len(members) < 2:
                    continue
                # Keep the largest cluster, fold the rest in.
                members.sort(key=lambda i: reps[i][2].get("cluster_size", 1), reverse=True)
                kept = members[0]
                for other in members[1:]:
                    kept_tags = reps[kept][2]
                    other_tags = reps[other][2]
                    kept_tags["cluster_size"] = kept_tags.get("cluster_size", 1) + other_tags.get("cluster_size", 1)
                    reps[other] = (None, None, None)  # mark deleted

    return [(nid, pt, tags) for nid, pt, tags in reps if nid is not None]


def _safe_node_attrs(tags: dict, *, reserve: set[str] | None = None) -> dict:
    """Return a copy of ``tags`` with any keys that would collide with graph/node kwargs removed.

    NetworkX ``add_node(id, **attrs)`` and our own node-construction helpers both explode when
    the OSM tags include names like ``type``, ``pos``, ``crs``, ``id``, ``index`` or ``geometry``.
    Downstream GeoJSON still needs those fields, so we do NOT mutate the original dict; we return
    a shallow copy with only the dangerous keys dropped. Consumers that need the real OSM tags
    keep the original.
    """
    reserve = reserve or {"type", "pos", "crs", "id", "index", "geometry", "n", "data", "attr"}
    return {k: v for k, v in tags.items() if str(k) not in reserve}


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
    run_meta: dict | None = None,
) -> dict[str, Any]:
    """
    Find cut-off settlements and produce results.

    A settlement is cut off if it's not reachable from any hospital or town/city
    via the road network with flooded segments removed, but only if it was reachable
    before the flood. Settlements that were already isolated in the pre-event OSM graph
    are reported separately, not blamed on the flood.

    ``run_meta`` (config/run_meta.py) records the area, bbox and dates this analysis
    belongs to; a direct caller that passes none gets the preset's values rather than an
    artifact with no provenance.
    """
    meta = run_meta if run_meta else case_study()

    # Identify hospital/town sources. Hospitals (and named clinics/pharmacies) are
    # sources, and so are OSM place=town/city/suburb nodes — the brief says 'nearest
    # town or hospital'. A source is added to the graph and linked to its nearest road
    # junction. Hospitals that OSM mapped as an area use the centroid, so polygon
    # features are sources too.
    hospital_nodes = []
    town_nodes = []

    hosp_sources = _osm_hospital_nodes(hospitals_gdf)
    for node_id, pt, tags in hosp_sources:
        pos = (pt.x, pt.y)
        nearest = _find_nearest_node(graph, node_positions, pos, node_type="road_junction")
        if nearest is None:
            continue
        hospital_nodes.append(nearest)
        graph.add_node(
            node_id,
            pos=pos,
            type="hospital",
            name=tags.get("name", node_id),
            **_safe_node_attrs(tags, reserve={"name"}),
        )
        node_positions[node_id] = pos
        node_types[node_id] = "hospital"
        if nearest in graph:
            graph.add_edge(node_id, nearest, length=0.001, flooded=False, connector=True)
        print(f"  🏥 Hospital node: {node_id} → nearest junction: {nearest}")

    # Towns/cities are sources even when no hospital was mapped. Do not conflate them
    # with settlements for counting purposes.
    town_sources = _osm_settlement_place_nodes(graph._places_gdf)
    for node_id, pt, tags in town_sources:
        pos = (pt.x, pt.y)
        nearest = _find_nearest_node(graph, node_positions, pos, node_type="road_junction")
        if nearest is None:
            continue
        town_nodes.append(nearest)
        graph.add_node(
            node_id,
            pos=pos,
            type="town",
            name=tags.get("name", node_id),
            **_safe_node_attrs(tags, reserve={"name"}),
        )
        node_positions[node_id] = pos
        node_types[node_id] = "town"
        if nearest in graph:
            graph.add_edge(node_id, nearest, length=0.001, flooded=False, connector=True)
        print(f"  🏙️ Town node: {node_id} → nearest junction: {nearest}")

    source_nodes = list(dict.fromkeys(hospital_nodes + town_nodes))

    print(f"\n  🏥  Hospital sources in network: {len(hospital_nodes)}")
    print(f"  🏙️  Town/city sources in network: {len(town_nodes)}")

    # Settlement definition: prefer OSM place nodes labelled as settlements; fall back
    # to building clusters (one representative point per cluster) so a single house is
    # not counted as a settlement.
    settlement_nodes: list[str] = []
    settlement_node_ids: set[str] = set()
    places_sources = _osm_settlement_place_nodes(getattr(graph, "_places_gdf", None))
    if places_sources:
        for node_id, pt, tags in places_sources:
            # Only add as a settlement if it is not already a town source.
            if node_id in {n for n in town_nodes}:
                continue
            pos = (pt.x, pt.y)
            nearest = _find_nearest_node(graph, node_positions, pos, node_type="road_junction")
            if nearest is None:
                continue
            settlement_nodes.append(nearest)
            settlement_node_ids.add(nearest)
            graph.add_node(
                node_id,
                pos=pos,
                type="settlement",
                name=tags.get("name", node_id),
                **_safe_node_attrs(tags, reserve={"name"}),
            )
            node_positions[node_id] = pos
            node_types[node_id] = "settlement"
            graph.add_edge(node_id, nearest, length=0.0, flooded=False, connector=True)
    else:
        # No place nodes: use building clusters as settlements.
        clusters = _building_cluster_settlements(
            settlements_gdf if settlements_gdf is not None else gpd.GeoDataFrame(),
            crs,
            snap_junctions=_candidate_positions(node_positions, graph, "road_junction")[:, :2] if node_positions else None,
        )
        for node_id, pt, tags in clusters:
            pos = (pt.x, pt.y)
            nearest = _find_nearest_node(graph, node_positions, pos, node_type="road_junction")
            if nearest is None:
                continue
            settlement_nodes.append(nearest)
            settlement_node_ids.add(nearest)
            graph.add_node(node_id, pos=pos, type="settlement", **_safe_node_attrs(tags))
            node_positions[node_id] = pos
            node_types[node_id] = "settlement"
            if nearest in graph:
                graph.add_edge(node_id, nearest, length=0.0, flooded=False, connector=True)

    n_settlements = len(settlement_node_ids)
    print(f"  🏘️  Settlements in network: {n_settlements}")

    # Before/after reachability. The flood can only cut off a settlement that was
    # reachable before the event. Anything unreachable in the all-roads (no flood)
    # baseline was already disconnected in the pre-event OSM graph and is reported
    # separately, not blamed on the flood.
    if source_nodes:
        reachable_before = find_reachable_nodes(graph, source_nodes, exclude_flooded=False)
        already_isolated = set(settlement_node_ids) - reachable_before
        reachable_after = find_reachable_nodes(graph, source_nodes, exclude_flooded=True)
        cut_off_settlement_ids = set(settlement_node_ids) - reachable_after - already_isolated
    else:
        reachable_before = set()
        already_isolated = set()
        reachable_after = set()
        cut_off_settlement_ids = set()
        print(
            "  ⚠️ No hospital or town reference facility found in the road network — "
            "cut-off analysis reports no_reference_facility (NOT 100% cut off)."
        )

    print(f"  ✅ Reachable from sources (before flood): {len(reachable_before & settlement_node_ids)}")
    print(f"  ✅ Reachable from sources (after flood):  {len(reachable_after & settlement_node_ids)}")
    print(f"  🚫 Cut-off settlements:                    {len(cut_off_settlement_ids)}")
    if already_isolated:
        print(f"  ℹ️  Already unreachable before the flood (not counted as cut off): "
              f"{len(already_isolated)}")

    # Map settlement_node_ids -> representative node id for the artifact detail.
    # For building clusters the representative node id is the cluster node id; for place
    # nodes it is the place node id.
    settlement_repr = {}
    for nid, ntype in node_types.items():
        if ntype == "settlement" and nid in settlement_node_ids:
            settlement_repr.setdefault(nid, nid)
    for nid in settlement_node_ids:
        settlement_repr.setdefault(nid, nid)

    # For each cut-off settlement, find details
    cutoff_details = []
    for s_node in sorted(cut_off_settlement_ids, key=lambda n: (-node_positions.get(n, (0, 0))[1], n)):
        pos = node_positions.get(s_node, (0.0, 0.0))
        nearest_reachable = _find_nearest_reachable(graph, node_positions, reachable_after, pos)
        blocking_segments = _find_blocking_segments(graph, s_node, reachable_after)

        cutoff_details.append({
            "settlement_id": settlement_repr.get(s_node, s_node),
            "settlement_name": (graph.nodes.get(settlement_repr.get(s_node, s_node)) or {}).get("name") or (graph.nodes.get(s_node) or {}).get("name"),
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
                for seg in (blocking_segments or [])[:5]
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

    has_reference = bool(source_nodes)
    status = "ok" if has_reference else "no_reference_facility"
    reference_note = (
        None
        if has_reference
        else (
            "No hospital or town reference facility was found in the road network "
            "for this area, so 'no connection to the nearest town or hospital' "
            "cannot be decided. No settlement is reported as cut off — this is NOT "
            "a 100% cut-off result."
        )
    )

    # Optional diagnostic lineage so a downstream consumer can tell which settlement
    # definition was used without parsing the geometry.
    settlement_method = "osm_places" if places_sources else "building_clusters"

    features: list[dict] = [
        {
            "type": "Feature",
            "properties": {
                "settlement_id": d["settlement_id"],
                "settlement_name": d["settlement_name"],
                "nearest_reachable_town": d["nearest_reachable_town"],
                "nearest_reachable_distance_km": d["nearest_reachable_distance_km"],
                "blocking_segments": len(d["blocking_segments"]),
                "cutoff_reason": d["cutoff_reason"],
            },
            "geometry": {
                "type": "Point",
                "coordinates": [d["coordinates"]["longitude"], d["coordinates"]["latitude"]],
            },
        }
        for d in cutoff_details
    ]

    # When the run produced nothing at all (no roads, buildings, hospitals, places),
    # the artifact must still be a valid empty FeatureCollection so the dashboard and the
    # situation report can render a sensible empty state instead of crashing or showing a
    # missing file. This also covers the case where every town/hospital was outside the
    # snapping radius, so there are source_nodes but no settlement was ever reachable.
    if not features:
        features = []

    cutoff_percentage = (
        round(len(cut_off_settlement_ids) / max(n_settlements, 1) * 100, 1)
        if has_reference and n_settlements
        else 0.0
    )

    features = [
        {
            "type": "Feature",
            "properties": {
                "settlement_id": d["settlement_id"],
                "settlement_name": d["settlement_name"],
                "nearest_reachable_town": d["nearest_reachable_town"],
                "nearest_reachable_distance_km": d["nearest_reachable_distance_km"],
                "blocking_segments": len(d["blocking_segments"]),
                "cutoff_reason": d["cutoff_reason"],
            },
            "geometry": {
                "type": "Point",
                "coordinates": [d["coordinates"]["longitude"], d["coordinates"]["latitude"]],
            },
        }
        for d in cutoff_details
    ]

    # The published artifact is a GeoJSON FeatureCollection (cut-off settlements as
    # points) plus the analysis block the dashboard reads. It carries NO local
    # filesystem path: a machine-specific directory in a published product is a
    # leak and makes the file non-portable.

    result = {
        "type": "FeatureCollection",
        "features": features,
        "status": status,
        "analysis_timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "method": "road_network_graph_bfs",
        # Artifact name only — never an absolute local path.
        "flood_mask": "flood_mask.tif",
        # Which run this analysis belongs to (area, bbox, dates), from the
        # run_meta.json every stage shares.
        "area": meta.get("area"),
        "bbox": meta.get("bbox"),
        "aoi_confirmed": bool(meta.get("aoi_confirmed", False)),
        "event_date": meta.get("event_date"),
        "before_date": meta.get("before_date"),
        "after_date": meta.get("after_date"),
        "osm_snapshot_date": meta.get("osm_snapshot_date"),
        "summary": {
            "total_settlements": n_settlements,
            "total_hospitals": len(hospital_nodes),
            "reachable_settlements": len(reachable_after & settlement_node_ids),
            "cutoff_settlements": len(cut_off_settlement_ids),
            "cutoff_percentage": cutoff_percentage,
            "already_isolated_settlements": len(already_isolated),
            "flooded_road_segments": len(cut_road_segments),
        },
        "reference_facility_note": reference_note,
        "already_isolated_before_event": sorted(already_isolated)[:50],
        "cutoff_settlements": cutoff_details,
        "flooded_roads": cut_road_segments[:50],
        "note": (
            "Cut-off settlements are those with road access to a hospital or major "
            "town before the flood and none after flooded road segments are removed. "
            "Settlements already unreachable before the event are listed separately "
            "in already_isolated_before_event. Settlement counts are based on named OSM "
            "places when present, otherwise on clusters of nearby buildings."
        ),
        "settlement_method": ("osm_places" if places_sources else "building_clusters"),
    }

    # Save — always, including the empty/degenerate cases, so downstream stages
    # (damage report, dashboard) have a parseable artifact rather than a 404.
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

    For the bulk settlement-snapping scan (hundreds of targets × thousands of
    junctions) this uses a vectorized haversine scan over a junction candidate
    array built once by build_road_graph and cached as ``graph._junction_candidates``.
    The single-point callers (``_find_nearest_reachable``, the hospital-junction
    lookup) are fast because they call it once per settlement/hospital, not once
    per junction.
    """
    candidates = _candidate_positions(node_positions, graph, node_type)
    if candidates.size == 0:
        return None
    x0, y0 = target_pos
    dists = _haversine_np(candidates[:, 0], candidates[:, 1], x0, y0)
    if candidates.dtype == object:
        max_m = max_dist_km * 1000.0
        within = dists < max_m
        if not within.any():
            return None
        best = int(np.argmin(np.where(within, dists, np.inf)))
        return candidates[best, 2]
    # float64-only path: min over within; inf otherwise.
    best = int(np.argmin(np.where(dists < max_dist_km * 1000.0, dists, np.inf)))
    return candidates[best, 2]


def _latlon_to_ecef_np(lons, lats) -> np.ndarray:
    """Convert arrays of lon/lat (degrees) to ECEF (x,y,z) metres, vectorized.

    Used only inside the clustering fallback path to give cKDTree a metric 3-D space
    for fast epsilon-nearest-neighbour grouping of building centroids.
    """
    DEG = math.pi / 180.0
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = 2.0 * f - f * f
    lat = np.asarray(lats, dtype=np.float64) * DEG
    lon = np.asarray(lons, dtype=np.float64) * DEG
    cos_lat = np.cos(lat)
    sin_lat = np.sin(lat)
    cos_lon = np.cos(lon)
    sin_lon = np.sin(lon)
    n = a / np.sqrt(1.0 - e2 * sin_lat * sin_lat)
    x = n * cos_lat * cos_lon
    y = n * cos_lat * sin_lon
    z = n * (1.0 - e2) * sin_lat
    return np.stack([x, y, z], axis=1)


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
    """Haversine distance in meters (scalar)."""
    return float(_haversine_np(np.asarray(lon1), np.asarray(lat1), np.asarray(lon2), np.asarray(lat2)))


def _haversine_np(lon1, lat1, lon2, lat2) -> np.ndarray:
    """Haversine distance in meters (vectorized over the first two args).

    Designed for the bulk settlement-snapping scan, where the target ``(lon2, lat2)``
    is fixed and the candidate set is a large numpy array of junction positions.
    """
    R = 6371000.0  # Earth radius in meters
    lat1 = np.asarray(lat1, dtype=np.float64)
    lon1 = np.asarray(lon1, dtype=np.float64)
    lat2 = float(lat2)
    lon2 = float(lon2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2)
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def _candidate_positions(
    node_positions: dict,
    graph: nx.Graph,
    node_type: str | None = None,
) -> np.ndarray:
    """Candidate array ``(lon, lat, node_id)`` for a type-filtered nearest-node scan.

    Returns a 2-D array of shape (n, 3). Columns 0 and 1 are float64 lon/lat for the
    vectorized haversine scan; column 2 is the node id stored as a python object so it
    survives the float64 cast without forcing a structured array.

    This is built ONCE when building_road_graph registers junctions, and the result is
    cached on the graph as ``_junction_candidates`` (when node_type is a road junction
    filter) so that settlement snapping never re-iterates the full node_positions dict.
    The shape-0 edge case returns an empty (0, 3) float64 array rather than None so the
    caller can treat it like any other array.
    """
    cache_key = f"_junction_candidates_{node_type}"
    if node_type is not None and hasattr(graph, cache_key):
        return getattr(graph, cache_key)

    lons, lats, ids = [], [], []
    # Single-pass with local variables to avoid dict .get in the hot loop.
    gnodes = graph.nodes
    for node_id, pos in node_positions.items():
        if node_type is not None and gnodes.get(node_id, {}).get("type") != node_type:
            continue
        lons.append(float(pos[0]))
        lats.append(float(pos[1]))
        ids.append(node_id)
    if not lons:
        arr = np.empty((0, 3), dtype=np.float64)
    else:
        arr = np.empty((len(lons), 3), dtype=object)
        arr[:, 0] = lons
        arr[:, 1] = lats
        arr[:, 2] = ids
    if node_type is not None:
        setattr(graph, cache_key, arr)
    return arr


def main():
    parser = argparse.ArgumentParser(description="Cut-off settlement analysis")
    parser.add_argument("--roads", type=Path, required=True, help="OSM roads GeoJSON")
    parser.add_argument("--buildings", type=Path, required=True, help="OSM buildings GeoJSON")
    parser.add_argument("--flood-mask", type=Path, required=True, help="Flood mask GeoTIFF")
    parser.add_argument("--hospitals", type=Path, default=None, help="OSM hospitals GeoJSON")
    parser.add_argument("--run-meta", type=Path, default=None,
                        help="run_meta.json written by run_pipeline.py (defaults to the one beside --flood-mask)")
    parser.add_argument("--output", type=Path, default=Path("results/cutoff_settlements.geojson"))
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("Cut-off Settlement Analysis")
    print(f"{'='*70}")
    run_meta, meta_source = run_meta_for_script(args.run_meta, args.flood_mask)
    print(f"  Run metadata: {run_meta.get('area')} "
          f"({'from ' + str(meta_source) if meta_source else 'preset fallback'})")
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

    # Align CRS. An empty FeatureCollection has crs=None and cannot be
    # transformed, so empty inputs must be skipped rather than crash the run —
    # "nothing to analyse" is a valid, exit-0 outcome.
    if not roads.empty and roads.crs != crs:
        roads = roads.to_crs(crs)
    elif roads.empty and roads.crs is None:
        roads = roads.set_crs(crs, allow_override=True)

    if not buildings.empty and buildings.crs != crs:
        buildings = buildings.to_crs(crs)
    elif buildings.empty and buildings.crs is None:
        buildings = buildings.set_crs(crs, allow_override=True)

    hospitals_gdf = hospitals
    if hospitals_gdf is not None:
        if not hospitals_gdf.empty and hospitals_gdf.crs != crs:
            hospitals_gdf = hospitals_gdf.to_crs(crs)
        elif hospitals_gdf.empty and hospitals_gdf.crs is None:
            hospitals_gdf = hospitals_gdf.set_crs(crs, allow_override=True)

    # Build road graph (with town/place sources if present).
    print(f"\n{'─'*70}")
    print(" Building road network graph...")
    graph, node_positions, node_types = build_road_graph(
        roads, buildings, flood_mask, transform, crs, places_gdf=None, hospitals_gdf=hospitals
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
        run_meta=run_meta,
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
