"""3. Sliding-window inference must cover every pixel.

If the tile grid is not edge-aligned, the strip between the last full tile and
the image border is never written. The overlap-averaging step then divides by a
zero count there, and those pixels silently become "not flooded" — a band of
false negatives along the east and south edges of every scene, which is exactly
where the stitched-mosaic seams are hardest to notice.

The model is a stub that returns a constant probability: this test is about the
tiling geometry, not about segmentation quality.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import inference

SIZES = [200, 256, 300, 700]
TILE = 256
OVERLAP = 64


class _ConstantModel(torch.nn.Module):
    """Returns a fixed probability for every pixel (0.9 -> above threshold)."""

    def __init__(self, value: float = 0.9):
        super().__init__()
        self.value = value

    def forward(self, x):  # (B, C, H, W) -> (B, 1, H, W)
        return torch.full(
            (x.shape[0], 1, x.shape[2], x.shape[3]), self.value, dtype=torch.float32
        )


@pytest.mark.parametrize("size", SIZES)
def test_every_pixel_is_covered_by_at_least_one_tile(size: int):
    model = _ConstantModel(0.9)
    image = np.zeros((2, size, size), dtype=np.float32)

    binary, prob = inference.tile_inference(
        model, image, TILE, OVERLAP, device=torch.device("cpu"), threshold=0.5
    )

    assert prob.shape == (size, size), (
        f"tile_inference returned a {prob.shape} probability map for a "
        f"{size}x{size} image"
    )
    uncovered = int((prob <= 0).sum())
    assert uncovered == 0, (
        f"{uncovered} of {size * size} pixels have zero tile coverage "
        f"(image {size}x{size}, tile {TILE}, overlap {OVERLAP}). The tile grid "
        "must be edge-aligned: append a final tile flush with the border "
        "instead of relying on range(0, extent - tile + 1, stride)."
    )

    assert int(binary.sum()) == size * size, (
        f"with a constant 0.9 probability every pixel should be flooded, but only "
        f"{int(binary.sum())} of {size * size} are — the uncovered strip is being "
        "thresholded as dry"
    )


def test_tile_starts_cover_the_extent_exactly():
    starts = inference._tile_starts(768, 256, 192)
    covered = set()
    for s in starts:
        covered.update(range(s, s + 256))
    assert covered == set(range(768)), (
        f"tile starts {starts} (tile 256, stride 192) do not cover 0..767; "
        f"missing: {sorted(set(range(768)) - covered)[:10]}"
    )
