"""8. Cut-off settlement logic on a toy road network.

The toy scene is one road (J0-J1-J2-J3) with a settlement at each junction, a
hospital at J0, and one settlement that no road ever reached. The flood covers
the J2->J3 sub-segment only.

Four failure modes this guards against:
  (a) no hospital   -> must report "no reference facility", not 100% cut off;
  (b) a settlement already unreachable before the flood must not be blamed on it;
  (c) a settlement whose only road is flooded must be reported;
  (d) a road flooded only partly must not sever the segments that stay dry.
"""

from __future__ import annotations

import geopandas as gpd
import pytest

import cutoff_analysis as ca


@pytest.fixture
def network(toy_scene):
    mask, transform, crs = ca.load_flood_mask(toy_scene["mask_path"])
    roads = gpd.read_file(toy_scene["roads"])
    buildings = gpd.read_file(toy_scene["buildings"])
    hospitals = gpd.read_file(toy_scene["hospitals"])
    graph, positions, types = ca.build_road_graph(roads, buildings, mask, transform, crs)
    return {
        "mask": mask, "transform": transform, "crs": crs,
        "roads": roads, "buildings": buildings, "hospitals": hospitals,
        "graph": graph, "positions": positions, "types": types,
    }


def _analyze(network, tmp_path, hospitals_gdf="default"):
    out = tmp_path / "cutoff.json"
    hospitals = network["hospitals"] if hospitals_gdf == "default" else hospitals_gdf
    return ca.analyze_cutoffs(
        network["graph"], network["positions"], network["types"],
        network["buildings"], hospitals,
        network["mask"], network["transform"], network["crs"],
        out,
    )


def _names(result) -> set[str]:
    return {d.get("settlement_name") for d in result["cutoff_settlements"]}


def test_no_hospitals_reports_no_reference_facility_not_100_percent_cutoff(network, tmp_path):
    result = _analyze(network, tmp_path, hospitals_gdf=None)

    assert result["status"] == "no_reference_facility", (
        "without a hospital or town there is no reference facility to be cut off "
        f"from; status must say so, got {result['status']!r}"
    )
    assert result["cutoff_settlements"] == [], (
        "no settlement may be reported as cut off when the reference facility is "
        f"missing; got {len(result['cutoff_settlements'])} cut-off settlements"
    )
    assert result["summary"]["cutoff_percentage"] != 100, (
        "a missing hospital must never surface as a 100% cut-off rate — that is "
        "the single most misleading number this analysis can print"
    )
    assert result["summary"]["cutoff_percentage"] == 0.0, (
        f"expected 0.0%, got {result['summary']['cutoff_percentage']}"
    )
    assert result["reference_facility_note"], (
        "the output must explain why no cut-off list was produced"
    )


def test_settlement_isolated_before_the_flood_is_not_reported(network, tmp_path):
    result = _analyze(network, tmp_path)

    assert "Nowhere" not in _names(result), (
        "\"Nowhere\" has no road within the 2 km snapping radius in the PRE-event "
        "snapshot, so it is not a flood casualty and must not appear in "
        f"cutoff_settlements; got {sorted(n for n in _names(result) if n)}"
    )
    isolated_ids = result["already_isolated_before_event"]
    assert any("iso" in str(i) for i in isolated_ids), (
        f"the already-isolated settlement must be reported separately under "
        f"already_isolated_before_event, got {isolated_ids}"
    )
    assert result["summary"]["already_isolated_settlements"] >= 1, (
        "the summary must state how many settlements were already unreachable "
        "before the flood, so the cut-off count cannot be inflated silently"
    )


def test_settlement_whose_only_road_is_flooded_is_reported(network, tmp_path):
    result = _analyze(network, tmp_path)

    assert "Village3" in _names(result), (
        "the settlement at J3 loses its only road (J2->J3 is flooded) and must be "
        f"reported as cut off; got {sorted(n for n in _names(result) if n)}"
    )
    assert result["summary"]["flooded_road_segments"] >= 1, (
        "the flooded sub-segment must be counted, got "
        f"{result['summary']['flooded_road_segments']}"
    )
    # 4 villages along the road + "Nowhere", which never had road access and is
    # tracked as a settlement node too (but excluded from the cut-off count).
    assert result["summary"]["total_settlements"] == 5, (
        f"expected 5 settlement nodes, got {result['summary']['total_settlements']}"
    )


def test_partly_flooded_road_does_not_cut_the_whole_way(network, tmp_path):
    result = _analyze(network, tmp_path)
    names = _names(result)

    for reachable in ("Village0", "Village1", "Village2"):
        assert reachable not in names, (
            f"{reachable} is still connected to the hospital by dry road "
            "(J0->J1->J2); only the flooded J2->J3 segment may be cut. Reporting "
            "the whole OSM way as closed is what produced false cut-off lists."
        )
    assert result["summary"]["cutoff_settlements"] == 1, (
        f"exactly one settlement is cut off in this scene, got "
        f"{result['summary']['cutoff_settlements']}: {sorted(n for n in names if n)}"
    )
