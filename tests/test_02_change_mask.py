"""2. The delivered flood mask must be NEW water, not "water in the after scene".

A before scene with a river and an after scene with the river plus a flood patch
must produce a mask that contains the patch and nothing of the river. Otherwise
the permanent river is reported as flood damage.

These tests drive the same function inference.py uses to write results/flood_mask.tif
(`derive_change_mask`) and then write/read the GeoTIFF through the real
`save_flood_mask`, so the artifact is checked, not just the arithmetic.
"""

from __future__ import annotations

import numpy as np

import toydata


def test_change_mask_contains_only_the_new_patch():
    import inference

    before, after = toydata.river_and_patch()
    delivered = inference.derive_change_mask(before, after)

    river_rows = slice(*toydata.RIVER_ROWS)
    river = delivered[river_rows, :]
    assert river.sum() == 0, (
        "the permanent river is still present in the delivered flood mask "
        f"({int(river.sum())} river pixels). The mask must be after AND NOT before "
        "— otherwise every riverside building reads as destroyed."
    )

    pr0, pr1 = toydata.PATCH_ROWS
    pc0, pc1 = toydata.PATCH_COLS
    expected_patch = (pr1 - pr0) * (pc1 - pc0)
    assert int(delivered.sum()) == expected_patch, (
        f"the change mask has {int(delivered.sum())} pixels but the only new water "
        f"is the {expected_patch}-pixel patch at rows {toydata.PATCH_ROWS}, "
        f"cols {toydata.PATCH_COLS}"
    )
    assert delivered[pr0:pr1, pc0:pc1].all(), (
        "every pixel of the new flood patch must be 1 in the change mask"
    )
    assert set(np.unique(delivered)) <= {0.0, 1.0}, (
        f"the change mask must be binary, got values {np.unique(delivered)[:5]}"
    )


def test_change_statistics_separate_new_water_from_persistent_water():
    import inference

    before, after = toydata.river_and_patch()
    stats = inference.compute_change_detection(before, after)

    r0, r1 = toydata.RIVER_ROWS
    pr0, pr1 = toydata.PATCH_ROWS
    pc0, pc1 = toydata.PATCH_COLS
    river_px = (r1 - r0) * toydata.SIZE
    patch_px = (pr1 - pr0) * (pc1 - pc0)

    assert stats["flood_extent_pixels"] == patch_px, (
        f"flood extent should be the {patch_px}-pixel new patch, got "
        f"{stats['flood_extent_pixels']}"
    )
    assert stats["persistent_water_pixels"] == river_px, (
        f"the river ({river_px} px) should be counted as persistent water, got "
        f"{stats['persistent_water_pixels']}"
    )
    assert stats["total_water_after_pixels"] == river_px + patch_px, (
        "total water in the after scene must include the river and the patch "
        f"({river_px + patch_px} px), got {stats['total_water_after_pixels']}"
    )


def test_written_flood_mask_tif_is_the_change_mask(tmp_path):
    import rasterio

    import inference

    before, after = toydata.river_and_patch()
    delivered = inference.derive_change_mask(before, after)

    out = tmp_path / "flood_mask.tif"
    profile = {
        "driver": "GTiff",
        "crs": toydata.toy_crs(),
        "transform": toydata.toy_transform(),
        "height": delivered.shape[0],
        "width": delivered.shape[1],
        "count": 1,
        "dtype": "float32",
    }
    inference.save_flood_mask(delivered, None, profile, out)

    with rasterio.open(out) as src:
        written = src.read(1)

    assert int(written.sum()) == int(delivered.sum()), (
        "the GeoTIFF on disk must contain exactly the change mask "
        f"({int(delivered.sum())} px), got {int(written.sum())} px"
    )
    r0, r1 = toydata.RIVER_ROWS
    assert written[r0:r1, :].sum() == 0, (
        "the river must not appear in results/flood_mask.tif — the delivered "
        "product is the flood, not the river"
    )
