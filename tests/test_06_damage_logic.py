"""6. Damage classification on a toy scene.

The rules under test:
  * a building fully inside NEW flood water is flagged;
  * a building on the permanent river (water in both scenes) is not — the river
    is not flood damage;
  * a bridge over permanent water stays intact, a bridge across the new flood
    does not;
  * a feature sitting entirely under invalid observations is UNKNOWN, never
    "undamaged".
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np

import damage_assessment as da
import inference
import toydata

# A cloud/observation gap that covers one of the toy buildings.
CLOUD_ROWS = (30, 40)
CLOUD_COLS = (150, 160)


def _delivered_mask():
    """The flood mask the pipeline actually publishes (new water only)."""
    before, after = toydata.river_and_patch()
    return inference.derive_change_mask(before, after)


def _valid_mask(shape=(toydata.SIZE, toydata.SIZE)):
    valid = np.ones(shape, dtype="float32")
    r0, r1 = CLOUD_ROWS
    c0, c1 = CLOUD_COLS
    valid[r0:r1, c0:c1] = 0.0
    return valid


def _gdf(features):
    return gpd.GeoDataFrame(
        [{k: v for k, v in f.items() if k != "geometry"} for f in features],
        geometry=[f["geometry"] for f in features],
        crs=f"EPSG:{toydata.CRS_EPSG}",
    )


def _levels(result) -> dict:
    return dict(zip(result["id"].astype(str), result["damage_level"].astype(str)))


def test_building_in_new_flood_is_flagged_building_on_river_is_not():
    mask = _delivered_mask()
    buildings = _gdf(toydata.toy_damage_buildings_features(valid_block=CLOUD_ROWS + CLOUD_COLS))
    out = da.assess_buildings(buildings, mask, toydata.toy_transform(), toydata.toy_crs(),
                              _valid_mask())
    levels = _levels(out)

    assert levels["b/flood"] in ("FLOODED_PARTIAL", "FLOODED_MAJOR"), (
        f"a building inside the new flood patch must be flagged, got "
        f"{levels['b/flood']!r} (overlap={dict(zip(out['id'], out['damage_overlap_ratio']))})"
    )
    assert out.loc[out["id"] == "b/flood", "flood_affected"].iloc[0], (
        "the flood_affected flag must be true for a building in the new flood"
    )
    assert levels["b/river"] == "UNAFFECTED", (
        f"a building on the permanent river must NOT be reported as flood-damaged, "
        f"got {levels['b/river']!r} — the delivered mask excludes pre-existing water"
    )
    assert not out.loc[out["id"] == "b/river", "flood_affected"].iloc[0], (
        "the river building must not be counted in flood_affected"
    )


def test_feature_with_no_valid_observation_is_unknown_not_undamaged():
    mask = _delivered_mask()
    buildings = _gdf(toydata.toy_damage_buildings_features(valid_block=CLOUD_ROWS + CLOUD_COLS))
    out = da.assess_buildings(buildings, mask, toydata.toy_transform(), toydata.toy_crs(),
                              _valid_mask())
    levels = _levels(out)
    reasons = dict(zip(out["id"].astype(str), out["damage_reason"].astype(str)))

    assert levels["b/cloud"] == "UNKNOWN", (
        f"a building entirely under valid_mask == 0 must be UNKNOWN, got "
        f"{levels['b/cloud']!r}. Cloud/nodata is 'we cannot say', not 'undamaged'."
    )
    assert "valid" in reasons["b/cloud"].lower(), (
        f"the UNKNOWN reason must name the missing observation, got {reasons['b/cloud']!r}"
    )
    assert not out.loc[out["id"] == "b/cloud", "flood_affected"].iloc[0], (
        "UNKNOWN must not be counted as flood-affected"
    )


def test_bridge_over_permanent_water_is_intact_bridge_over_new_flood_is_not():
    mask = _delivered_mask()
    bridges = _gdf(toydata.toy_bridges_features())
    out = da.assess_bridges(bridges, mask, toydata.toy_transform(), toydata.toy_crs(),
                            _valid_mask())
    levels = _levels(out)

    assert levels["way/bridge-river"] == "UNAFFECTED", (
        f"a bridge crossing the permanent river must stay intact, got "
        f"{levels['way/bridge-river']!r} — the river was already there before the event"
    )
    assert levels["way/bridge-patch"] in ("FLOODED_PARTIAL", "FLOODED_MAJOR"), (
        f"a bridge crossing the new flood patch must be damaged, got "
        f"{levels['way/bridge-patch']!r}"
    )
    damaged = dict(zip(out["id"].astype(str), out["bridge_damaged"].astype(bool)))
    assert not damaged["way/bridge-river"] and damaged["way/bridge-patch"]


def test_road_crossing_new_flood_is_cut():
    mask = _delivered_mask()
    roads = _gdf(toydata.toy_damage_roads_features())
    out = da.assess_roads(roads, mask, toydata.toy_transform(), toydata.toy_crs(),
                          _valid_mask())
    levels = _levels(out)

    assert levels["way/road"] == "FLOODED_MAJOR", (
        f"a road whose whole length lies in the new flood must be FLOODED_MAJOR, got "
        f"{levels['way/road']!r}"
    )
    assert bool(out.loc[out["id"] == "way/road", "road_cut"].iloc[0])


def test_empty_feature_types_do_not_crash_and_report_zero():
    mask = _delivered_mask()
    empty = gpd.GeoDataFrame(
        {"id": []}, geometry=gpd.GeoSeries([], dtype="geometry"),
        crs=f"EPSG:{toydata.CRS_EPSG}",
    )

    buildings = da.assess_buildings(empty, mask, toydata.toy_transform(), toydata.toy_crs())
    roads = da.assess_roads(empty, mask, toydata.toy_transform(), toydata.toy_crs())
    bridges = da.assess_bridges(empty, mask, toydata.toy_transform(), toydata.toy_crs())

    for name, gdf in (("buildings", buildings), ("roads", roads), ("bridges", bridges)):
        assert len(gdf) == 0, f"{name}: expected zero rows, got {len(gdf)}"
