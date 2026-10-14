#!/usr/bin/env python3
"""Post-processing shared by the real inference and the synthetic demo fixture.

Kept in its own module so both producers of ``flood_mask.tif`` (inference.py and
demo/make_demo_data.py) apply exactly the same speckle rule. Without that, the demo
fixture would quietly stop being a faithful stand-in for a real run.

No torch, no rasterio — just the array operation.
"""

from __future__ import annotations

import numpy as np

# Connected components smaller than this many pixels are dropped from a flood
# mask. At a 10 m Sentinel-1 pixel that is ~800 m^2: a real flood is a connected
# patch, so anything smaller is speckle, and every spurious blob becomes a
# "flooded" building or a cut road downstream.
MIN_COMPONENT_PIXELS = 8


def remove_small_components(
    mask: np.ndarray,
    min_size: int = MIN_COMPONENT_PIXELS,
) -> tuple[np.ndarray, int]:
    """Drop connected blobs smaller than ``min_size`` pixels (speckle control).

    Returns the cleaned binary float32 mask and how many blobs were removed, so
    the count can be reported (and asserted in tests) instead of being silently
    applied.
    """
    from scipy import ndimage

    binary = np.asarray(mask) > 0.5
    if not binary.any():
        return binary.astype(np.float32), 0

    # 8-connectivity: diagonal touching is still one patch.
    structure = np.ones((3, 3), dtype=bool)
    labels, count = ndimage.label(binary, structure=structure)
    if count == 0:
        return binary.astype(np.float32), 0

    sizes = np.bincount(labels.ravel())
    keep = np.zeros(sizes.shape, dtype=bool)
    keep[1:] = sizes[1:] >= int(min_size)
    cleaned = keep[labels]
    removed = int(count - int(keep[1:].sum()))
    return cleaned.astype(np.float32), removed
