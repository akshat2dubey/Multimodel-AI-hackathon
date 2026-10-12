"""9. The cut-off output must be a valid GeoJSON carrying no local paths.

results/cutoff_settlements.geojson is a published artifact: it is served by the
dashboard, embedded in the report and handed to judges. It previously carried
`"analysis_timestamp": str(Path.cwd())` and an absolute `flood_mask` path, which
leaks the operator's directory layout and makes the file non-portable.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import geopandas as gpd

import cutoff_analysis as ca

ABSOLUTE_PATH_RE = re.compile(r"^([A-Za-z]:[\\/]|/|\\\\)")


def _walk_strings(value):
    if isinstance(value, dict):
        for v in value.values():
            yield from _walk_strings(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _walk_strings(v)
    elif isinstance(value, str):
        yield value


def _produce(toy_scene, tmp_path, hospitals="default"):
    mask, transform, crs = ca.load_flood_mask(toy_scene["mask_path"])
    roads = gpd.read_file(toy_scene["roads"])
    buildings = gpd.read_file(toy_scene["buildings"])
    hospitals_gdf = gpd.read_file(toy_scene["hospitals"]) if hospitals == "default" else None
    graph, positions, types = ca.build_road_graph(roads, buildings, mask, transform, crs)
    out = tmp_path / "cutoff_settlements.geojson"
    ca.analyze_cutoffs(graph, positions, types, buildings, hospitals_gdf,
                       mask, transform, crs, out)
    return out


def test_output_is_valid_geojson(toy_scene, tmp_path):
    out = _produce(toy_scene, tmp_path)
    data = json.loads(out.read_text(encoding="utf-8"))

    assert data["type"] == "FeatureCollection", (
        f"a published .geojson must be a FeatureCollection at the top level, got "
        f"{data.get('type')!r}"
    )
    assert isinstance(data["features"], list), "features must be a list"

    # The dashboard reads these two custom members; keep them alongside GeoJSON.
    assert "cutoff_settlements" in data, "the dashboard reads cutoff_settlements"
    assert "summary" in data, "the report reads summary"

    for feature in data["features"]:
        assert feature["type"] == "Feature", f"bad feature type: {feature.get('type')!r}"
        assert feature["geometry"]["type"] == "Point", (
            f"cut-off settlements must be point features, got "
            f"{feature['geometry']['type']!r}"
        )
        lon, lat = feature["geometry"]["coordinates"]
        assert -180 <= lon <= 180 and -90 <= lat <= 90, (
            f"coordinates {lon},{lat} are not lon/lat — check the CRS handling"
        )

    # A strict GeoJSON reader (e.g. a browser, or ogr2ogr) must accept it.
    reloaded = gpd.read_file(out)
    assert len(reloaded) == len(data["features"]), (
        "geopandas could not read back as many features as were written"
    )


def test_output_contains_no_local_filesystem_path(toy_scene, tmp_path):
    out = _produce(toy_scene, tmp_path)
    data = json.loads(out.read_text(encoding="utf-8"))

    leaked = []
    for value in _walk_strings(data):
        if str(tmp_path) in value or ABSOLUTE_PATH_RE.match(value):
            leaked.append(value)

    assert not leaked, (
        "the published cut-off GeoJSON contains local filesystem path(s): "
        f"{leaked[:5]}. Record artifact names (e.g. 'flood_mask.tif') and an ISO "
        "timestamp instead — an operator's directory layout is not part of the "
        "product."
    )

    raw = out.read_text(encoding="utf-8")
    for needle in ("Path.cwd", str(Path.cwd()), str(tmp_path)):
        assert needle not in raw, (
            f"the raw output text contains {needle!r}; the file must be portable"
        )


def test_no_reference_facility_output_is_still_valid_geojson(toy_scene, tmp_path):
    out = _produce(toy_scene, tmp_path, hospitals=None)
    data = json.loads(out.read_text(encoding="utf-8"))

    assert data["type"] == "FeatureCollection"
    assert data["features"] == []
    assert data["status"] == "no_reference_facility"
    assert data["reference_facility_note"], (
        "an empty result must explain itself in the artifact, not only on stdout"
    )
