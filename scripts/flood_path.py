#!/usr/bin/env python3
"""
Flood Path Tracing — Bonus Feature

Given an upstream point, trace the flood path down the valley using
Copernicus DEM elevation data and list settlements along it.

Uses steepest-descent flow routing on the DEM:
  1. For each pixel, compute flow direction (steepest downhill neighbor)
  2. Starting from the upstream point, follow the flow path downstream
  3. Intersect the path with settlement locations
  4. Output: GeoJSON of the path + list of settlements along it

Usage:
    python flood_path.py \
        --dem ./data/dem/dem.tif \
        --start-lat <lat> --start-lon <lon>   # upstream point to trace from \
        --output ./results/flood_path.geojson

    run_pipeline.py derives the start point from the resolved run metadata
    (results/run_meta.json), so it always matches the run's own bbox.
"""

import argparse

import sys
import json
import math
import heapq
from pathlib import Path

import numpy as np
import rasterio
import rasterio.features
import rasterio.warp
from scipy import ndimage
from shapely.geometry import LineString, Point, mapping
from tqdm import tqdm

if hasattr(sys.stdout, "reconfigure"):  # never crash on a cp1252 Windows console
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def load_dem(dem_path: Path) -> tuple[np.ndarray, rasterio.Affine, Any, str]:
    """Load Copernicus DEM and return elevation array, transform, crs, nodata."""
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float32)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata
    print(f"  DEM shape: {dem.shape}")
    print(f"  DEM resolution: {abs(transform.a):.1f}m")
    print(f"  DEM CRS: {crs}")
    print(f"  Elevation range: {np.nanmin(dem):.0f}m – {np.nanmax(dem):.0f}m")
    return dem, transform, crs, nodata


def compute_flow_direction(dem: np.ndarray) -> np.ndarray:
    """
    Compute D8 flow direction grid.

    For each pixel, find the steepest downhill neighbor (among 8 neighbors).
    Returns array of same shape as dem with values 0-7 indicating direction:
      0=E, 1=SE, 2=S, 3=SW, 4=W, 5=NW, 6=N, 7=NE
      (or 0=flat/no-flow for NODATA/flat areas)
    """
    rows, cols = dem.shape
    flow_dir = np.full((rows, cols), -1, dtype=np.int8)  # -1 = no flow (flat/NODATA)
    flow_drop = np.full((rows, cols), 0.0, dtype=np.float32)  # vertical drop to neighbor

    # Neighbor offsets (row, col) for 8 directions: E, SE, S, SW, W, NW, N, NE
    neighbors = [
        (0, 1),   # E  → direction 0
        (1, 1),   # SE → direction 1
        (1, 0),   # S  → direction 2
        (1, -1),  # SW → direction 3
        (0, -1),  # W  → direction 4
        (-1, -1), # NW → direction 5
        (-1, 0),  # N  → direction 6
        (-1, 1),  # NE → direction 7
    ]

    for d, (dr, dc) in enumerate(neighbors):
        # Shifted elevation (pad with NaN)
        shifted = np.full((rows + 2, cols + 2), np.nan, dtype=np.float32)
        shifted[1:-1, 1:-1] = dem
        neighbor_elev = shifted[
            1 + dr : 1 + dr + rows,
            1 + dc : 1 + dc + cols,
        ]

        # Vertical drop from current pixel to this neighbor
        drop = dem - neighbor_elev

        # Valid flow: must go downhill (positive drop) and neighbor is valid
        valid_flow = (drop > 0) & (~np.isnan(neighbor_elev)) & (~np.isnan(dem))

        # Update if this direction is steeper than previous best
        steeper = valid_flow & (drop > flow_drop)
        flow_dir[steeper] = d
        flow_drop[steeper] = drop[steeper]

    # Handle flat areas: pixels where no downhill neighbor found
    no_flow = flow_dir == -1
    if no_flow.any():
        print(f"  ⚠ {no_flow.sum()} flat/no-data pixels (no downhill neighbor)")

    return flow_dir


def _escape_pit(
    dem: np.ndarray,
    row: int,
    col: int,
    radius: int,
    eps: float = 0.0,
) -> list[tuple[int, int]] | None:
    """
    Find the route out of a local minimum (sink) via the lowest surrounding saddle.

    Real DEMs contain sinks: pixels with no strictly lower neighbour, where a
    strict D8 walk correctly stops even though the river plainly continues.
    This runs a minimax (lowest-saddle) Dijkstra inside a window around the
    sink: the route that minimises the *highest* elevation it must cross. That
    is the spill path water would take when the basin overflows — the standard
    "breaching" fix for D8 on sink-laden terrain (cf. Barnes et al., 2014).

    Returns the list of (row, col) from the sink to the first pixel that is
    strictly lower than the sink, or None if no escape exists within radius.
    """
    rows, cols = dem.shape
    r0, r1 = max(0, row - radius), min(rows, row + radius + 1)
    c0, c1 = max(0, col - radius), min(cols, col + radius + 1)

    start = (row, col)
    start_elev = float(dem[row, col])

    best: dict[tuple[int, int], float] = {start: start_elev}
    prev: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    heap: list[tuple[float, int, int]] = [(start_elev, row, col)]

    neigh = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]

    target: tuple[int, int] | None = None
    while heap:
        cost, r, c = heapq.heappop(heap)
        if cost > best.get((r, c), float("inf")):
            continue
        if (r, c) != start and dem[r, c] < start_elev - eps:
            target = (r, c)
            break
        for dr, dc in neigh:
            nr, nc = r + dr, c + dc
            if not (r0 <= nr < r1 and c0 <= nc < c1):
                continue
            e = dem[nr, nc]
            if np.isnan(e):
                continue
            ncost = cost if cost > e else float(e)
            if ncost < best.get((nr, nc), float("inf")):
                best[(nr, nc)] = ncost
                prev[(nr, nc)] = (r, c)
                heapq.heappush(heap, (ncost, nr, nc))

    if target is None:
        return None

    route: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = target
    while cur is not None:
        route.append(cur)
        cur = prev[cur]
    route.reverse()
    return route


def trace_flow_path(
    dem: np.ndarray,
    flow_dir: np.ndarray,
    start_row: int,
    start_col: int,
    max_steps: int = 50000,
    min_elevation: float = 100.0,
    pit_radius: int = 200,
    max_breaches: int = 5000,
) -> tuple[list[tuple[int, int]], list[dict]]:
    """
    Trace the flow path from a starting pixel following D8 flow directions.

    When the walk reaches a pixel with no downhill neighbour (a sink), and
    pit_radius > 0, it breaches the sink via the lowest surrounding saddle and
    carries on — so the trace follows the valley instead of stopping dead at
    the first local minimum.

    Returns (path, breaches): path is a list of (row, col) tuples, breaches
    records every sink that had to be breached (for honest reporting).
    """
    path = [(start_row, start_col)]
    seen = {(start_row, start_col)}
    row, col = start_row, start_col
    rows, cols = dem.shape

    # Direction deltas (row change, col change) for D8
    dir_deltas = [
        (0, 1),    # 0: E
        (1, 1),    # 1: SE
        (1, 0),    # 2: S
        (1, -1),   # 3: SW
        (0, -1),   # 4: W
        (-1, -1),  # 5: NW
        (-1, 0),   # 6: N
        (-1, 1),   # 7: NE
    ]

    breaches: list[dict] = []

    for step in range(max_steps):
        d = flow_dir[row, col]

        if d < 0 or d > 7:
            # Sink (no downhill neighbour) or edge. Try to breach and continue.
            if pit_radius > 0 and len(breaches) < max_breaches:
                route = _escape_pit(dem, row, col, pit_radius)
                if route and len(route) > 1:
                    added = 0
                    for rr, cc in route[1:]:
                        if (rr, cc) in seen:
                            continue
                        path.append((rr, cc))
                        seen.add((rr, cc))
                        added += 1
                        if dem[rr, cc] < min_elevation:
                            break
                    if added:
                        pit_elev = float(dem[row, col])
                        esc_r, esc_c = path[-1]
                        breaches.append({
                            "pit_rowcol": [int(row), int(col)],
                            "pit_elevation_m": round(pit_elev, 1),
                            "escape_rowcol": [int(esc_r), int(esc_c)],
                            "escape_elevation_m": round(float(dem[esc_r, esc_c]), 1),
                            "route_pixels": added,
                        })
                        row, col = esc_r, esc_c
                        if dem[row, col] < min_elevation:
                            break
                        continue
            break  # genuine end of flow (edge / NODATA / unbreachable sink)

        dr, dc = dir_deltas[d]
        row += dr
        col += dc

        # Check bounds
        if not (0 <= row < rows and 0 <= col < cols):
            break

        # Check for NODATA
        if np.isnan(dem[row, col]):
            break

        # Check for loop (rare but possible in flat areas)
        if (row, col) in seen:
            break

        path.append((row, col))
        seen.add((row, col))

        # Stop if elevation is very low (reached valley floor / river)
        if dem[row, col] < min_elevation:
            break

    return path, breaches


def rowcol_to_lonlat(
    row: int,
    col: int,
    transform: rasterio.Affine,
    crs,
) -> tuple[float, float]:
    """Convert pixel (row, col) to geographic (lon, lat)."""
    x, y = transform * (col, row)
    if str(crs) == "EPSG:4326":
        return (x, y)
    # Reproject to lat/lon if needed
    from rasterio.warp import transform as reproject
    lon_lats = reproject(
        [x], [y],
        src_crs=crs, dst_crs="EPSG:4326",
    )
    return (lon_lats[0][0], lon_lats[1][0])


def intersect_settlements(
    path_coords: list[tuple[float, float]],
    settlements_gdf,
) -> list[dict]:
    """
    Find settlements that lie along the flood path.

    Uses a buffer around the path line and checks intersection with settlement points/polygons.
    """
    if len(path_coords) < 2:
        return []

    path_line = LineString(path_coords)

    # Create a buffer (e.g., 500m) around the path to catch nearby settlements
    # Buffer distance depends on DEM resolution — 30m DEM → buffer ~ some meters
    buffer_dist = 0.005  # ~500m in degrees at this latitude
    path_buffer = path_line.buffer(buffer_dist)

    settlements_along_path = []
    for _, row in settlements_gdf.iterrows():
        geom = row.geometry
        # Check if settlement intersects the buffered path
        if geom.intersects(path_buffer):
            dist = path_line.project(geom.centroid) if geom.geom_type == "Point" else \
                   path_line.project(geom.representative_point())
            settlements_along_path.append({
                "settlement_name": row.get("name", row.get("amenity", "Unnamed settlement")),
                "coordinates": {
                    "longitude": round(geom.x if hasattr(geom, "x") else geom.centroid.x, 6),
                    "latitude": round(geom.y if hasattr(geom, "y") else geom.centroid.y, 6),
                },
                "distance_along_path_km": round(dist * 111000 / 1000, 1),  # Approx km
                "osm_type": row.get("place", row.get("amenity", "unknown")),
            })

    return settlements_along_path


def path_to_geojson(
    path_rowcols: list[tuple[int, int]],
    transform: rasterio.Affine,
    crs,
    settlements: list[dict] | None = None,
) -> dict:
    """Convert pixel path to GeoJSON FeatureCollection."""
    path_lonlats = [
        rowcol_to_lonlat(r, c, transform, crs) for r, c in path_rowcols
    ]

    features = []

    # Path line
    if len(path_lonlats) >= 2:
        features.append({
            "type": "Feature",
            "properties": {
                "type": "flood_path",
                "length_pixels": len(path_rowcols),
                "description": "Steepest-descent flow path from upstream point",
            },
            "geometry": {
                "type": "LineString",
                "coordinates": [[lon, lat] for lon, lat in path_lonlats],
            },
        })

    # Start point
    if path_lonlats:
        features.append({
            "type": "Feature",
            "properties": {
                "type": "start_point",
                "description": "Upstream starting point (glacier collapse area)",
            },
            "geometry": {
                "type": "Point",
                "coordinates": list(path_lonlats[0][::-1]),  # [lon, lat]
            },
        })

    # End point
    if len(path_lonlats) > 1:
        features.append({
            "type": "Feature",
            "properties": {
                "type": "end_point",
                "description": "Downstream terminus of flow path",
            },
            "geometry": {
                "type": "Point",
                "coordinates": list(path_lonlats[-1][::-1]),
            },
        })

    # Settlements along path
    if settlements:
        for s in settlements:
            features.append({
                "type": "Feature",
                "properties": {
                    "type": "settlement",
                    "name": s["settlement_name"],
                    "osm_type": s["osm_type"],
                    "distance_along_path_km": s["distance_along_path_km"],
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [s["coordinates"]["longitude"], s["coordinates"]["latitude"]],
                },
            })

    return {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "method": "D8 steepest-descent flow routing",
            "dem_source": DEM_SOURCE_REAL,
            "description": "Flood path tracing from upstream point downstream through valley",
            "note": "This is a simplified D8 flow model. Real flood routing requires hydrological modeling (e.g., FLDWAV, HEC-RAS) with hydraulic parameters.",
        },
    }


DEM_SOURCE_REAL = (
    "Copernicus WorldDEM-30 — DLR e.V. 2010-2014, Airbus Defence and Space GmbH "
    "2014-2018, provided under COPERNICUS by the European Union and ESA")
DEM_SOURCE_SYNTHETIC = (
    "*** SYNTHETIC DEM (demo fixture) — not WorldDEM-30. Do not cite as Copernicus DEM.")


def dem_is_synthetic(dem_path: Path) -> bool:
    """True when the DEM came from demo/make_demo_data.py.

    Labelling a fabricated elevation model as "Copernicus WorldDEM-30" would put
    a false attribution into the submission, so the sidecar decides the label.
    """
    for candidate in (dem_path.parent / "PRODUCT_INFO.json",
                      dem_path.parent / "DEM_SOURCE.json"):
        if candidate.exists():
            try:
                if json.loads(candidate.read_text(encoding="utf-8")).get("synthetic"):
                    return True
            except Exception:
                pass
    return "SYNTHETIC" in dem_path.name.upper()


def main():
    parser = argparse.ArgumentParser(description="Flood path tracing via DEM flow routing")
    parser.add_argument("--dem", type=Path, required=True, help="Copernicus DEM GeoTIFF")
    parser.add_argument("--start-lat", type=float, required=True, help="Starting latitude")
    parser.add_argument("--start-lon", type=float, required=True, help="Starting longitude")
    parser.add_argument("--settlements", type=Path, default=None, help="OSM settlements GeoJSON")
    parser.add_argument("--output", type=Path, default=Path("results/flood_path.geojson"))
    parser.add_argument(
        "--min-elevation", type=float, default=100.0,
        help="Stop the trace once elevation drops below this many metres",
    )
    parser.add_argument(
        "--pit-radius", type=int, default=200,
        help="Breach DEM sinks within this many pixels via the lowest saddle "
             "(0 disables breaching - strict D8, stops at first sink)",
    )
    parser.add_argument(
        "--max-breaches", type=int, default=5000,
        help="Safety cap on how many DEM sinks may be breached in one trace",
    )
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("Flood Path Tracing — D8 Steepest-Descent Flow Routing")
    print(f"{'='*70}")
    print(f"  Start point:  ({args.start_lat:.4f}, {args.start_lon:.4f})")
    print(f"  DEM:           {args.dem}")
    print(f"  Settlements:   {args.settlements or 'Not provided'}")
    print(f"  Output:        {args.output}")

    # Load DEM
    print("\n📥 Loading DEM...")
    dem, transform, crs, nodata = load_dem(args.dem)
    dem_valid = dem[~np.isnan(dem)]
    print(f"  Valid pixels: {len(dem_valid):,} / {dem.size:,} ({len(dem_valid)/dem.size*100:.1f}%)")

    # Compute flow direction
    print("\n🧮 Computing D8 flow direction...")
    flow_dir = compute_flow_direction(dem)
    flow_pixels = (flow_dir >= 0).sum()
    print(f"  Pixels with flow direction: {flow_pixels:,} ({flow_pixels/dem.size*100:.1f}%)")

    # Convert start point to pixel coordinates
    # NB: rasterio's rowcol() returns (row, col) — not (col, row). Swapping these
    # silently transposes the start point, which on a non-square DEM lands the
    # trace in a completely different place.
    start_row, start_col = rasterio.transform.rowcol(transform, args.start_lon, args.start_lat)
    print(f"\n📍 Starting pixel: row={start_row}, col={start_col}")
    print(f"   Elevation at start: {dem[start_row, start_col]:.1f}m")

    # Trace path
    print("\n🔄 Tracing flow path...")
    path, breaches = trace_flow_path(
        dem,
        flow_dir,
        start_row,
        start_col,
        min_elevation=args.min_elevation,
        pit_radius=args.pit_radius,
        max_breaches=args.max_breaches,
    )
    print(f"  Path length: {len(path)} pixels ({len(path) * abs(transform.a):.1f}m)")

    # Convert to lon/lat
    path_lonlats = [
        rowcol_to_lonlat(r, c, transform, crs) for r, c in path
    ]
    if path_lonlats:
        start_latlon = path_lonlats[0]
        end_latlon = path_lonlats[-1]
        print(f"  Start (lat/lon):  ({start_latlon[1]:.4f}, {start_latlon[0]:.4f})")
        print(f"  End   (lat/lon):  ({end_latlon[1]:.4f}, {end_latlon[0]:.4f})")
        elev_start = dem[path[0][0], path[0][1]]
        elev_end = dem[path[-1][0], path[-1][1]]
        print(f"  Elevation drop:   {elev_start:.0f}m → {elev_end:.0f}m ({elev_start - elev_end:.0f}m drop)")
        if breaches:
            _b0 = breaches[0]
            if len(breaches) >= args.max_breaches:
                print(f"  ⚠ breach cap reached ({args.max_breaches}) — the trace "
                      f"stopped at a sink, not at the valley floor",
                      file=sys.stderr)
            print(f"  Sinks breached:   {len(breaches)} "
                  f"(first at {_b0['pit_elevation_m']:.0f} m -> "
                  f"{_b0['escape_elevation_m']:.0f} m via lowest saddle)")

    # Find settlements along path
    settlements_along = []
    if args.settlements and args.settlements.exists():
        print("\n🏘️  Finding settlements along path...")
        try:
            import geopandas as gpd
            settlements_gdf = gpd.read_file(args.settlements)
            if settlements_gdf.crs != crs:
                settlements_gdf = settlements_gdf.to_crs(crs)
            settlements_along = intersect_settlements(path_lonlats, settlements_gdf)
            print(f"  Settlements along path: {len(settlements_along)}")
            for s in settlements_along:
                print(f"    • {s['settlement_name']} "
                      f"({s['coordinates']['latitude']:.4f}, {s['coordinates']['longitude']:.4f}) "
                      f"— {s['distance_along_path_km']:.1f}km along path")
        except Exception as e:
            print(f"  ⚠️ Could not load settlements: {e}")

    # Save GeoJSON
    print(f"\n💾 Saving results...")
    geojson = path_to_geojson(path, transform, crs, settlements_along)
    geojson["metadata"]["sinks_breached"] = len(breaches)
    geojson["metadata"]["breach_cap"] = args.max_breaches
    geojson["metadata"]["breach_cap_reached"] = (
        len(breaches) >= args.max_breaches
    )
    geojson["metadata"]["pit_breach_radius_px"] = args.pit_radius
    geojson["metadata"]["min_elevation_stop_m"] = args.min_elevation
    if breaches:
        geojson["metadata"]["breach_note"] = (
            "Strict D8 stops at the first DEM sink. Each sink here was breached "
            "along its lowest surrounding saddle so the trace continues "
            "down-valley (standard pit-breaching for sink-laden DEMs)."
        )
    if dem_is_synthetic(args.dem):
        geojson["metadata"]["dem_source"] = DEM_SOURCE_SYNTHETIC
        geojson["metadata"]["synthetic"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(geojson, indent=2), encoding="utf-8")
    print(f"  ✅ Flood path saved: {args.output}")

    print(f"\n{'='*70}")
    print(" Bonus Feature: Flood Path Tracing — Complete")
    print(f"{'='*70}")
    px_m = abs(transform.a)
    if crs is not None and getattr(crs, "is_geographic", False):
        # degrees, not metres — convert at the path's mean latitude
        # 0.0 as the fallback, not a case-study latitude: an empty path has no
        # location, and borrowing one from config/ would be a hidden assumption.
        mean_lat = float(np.mean([lat for _, lat in path_lonlats])) if path_lonlats else 0.0
        px_m = abs(transform.a) * 111320.0 * math.cos(math.radians(mean_lat))
    print(f"  Path length:     {len(path)} pixels (~{len(path) * px_m / 1000:.1f} km)")
    print(f"  Settlements hit: {len(settlements_along)}")
    print(f"  Output:          {args.output}")


if __name__ == "__main__":
    main()
