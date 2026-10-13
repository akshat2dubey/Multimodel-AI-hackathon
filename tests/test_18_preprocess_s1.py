"""18. Sentinel-1 preprocessing bridge (T07).

Tests that preprocess_s1 builds the four GeoTIFFs (s1_before_VV, s1_before_VH,
s1_after_VV, s1_after_VH) and sidecar S1_PROCESSED.json from a mocked Process API
response without needing credentials or external network access.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds

import preprocess_s1


def _create_mock_crop_bytes(width=64, height=64) -> bytes:
    """Generate in-memory 2-band GeoTIFF bytes simulating Process API output."""
    mem = MemoryFile()
    transform = from_bounds(85.4, 28.0, 86.0, 28.6, width, height)
    with mem.open(
        driver="GTiff",
        height=height,
        width=width,
        count=2,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        # Linear sigma0 values
        vv_linear = np.full((height, width), 0.05, dtype="float32")
        vh_linear = np.full((height, width), 0.01, dtype="float32")
        dst.write(vv_linear, 1)
        dst.write(vh_linear, 2)
    return bytes(mem.getbuffer())


def test_mocked_process_api_builds_geotiffs_and_sidecar(tmp_path: Path, monkeypatch):
    """A mocked-API test builds the GeoTIFFs and sidecar from a fake response."""
    output_dir = tmp_path / "sentinel1"
    crop_bytes = _create_mock_crop_bytes()

    # Mock credentials so no environment variables are needed
    monkeypatch.setattr(preprocess_s1, "_credentials", lambda: ("fake_client_id", "fake_client_secret"))

    # Mock requests to token and process endpoints
    mock_post_resp = MagicMock()
    mock_post_resp.status_code = 200
    mock_post_resp.json.return_value = {"access_token": "mocked_jwt_token"}

    mock_get_resp = MagicMock()
    mock_get_resp.status_code = 200
    mock_get_resp.content = crop_bytes

    import requests

    def fake_post(url, data=None, **kwargs):
        return mock_post_resp

    def fake_get(url, params=None, headers=None, **kwargs):
        return mock_get_resp

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(requests, "get", fake_get)

    bbox = (85.4, 28.0, 86.0, 28.6)
    acq_time = datetime(2026, 8, 24, 0, 48, tzinfo=timezone.utc)

    # 1. Process "before" product
    sidecar_before = preprocess_s1.preprocess_product(
        safe_zip=Path("fake_before.zip"),
        output_dir=output_dir,
        label="before",
        bbox=bbox,
        acquisition=acq_time,
        orbit=19,
        target_pixels=64,
    )

    # 2. Process "after" product
    sidecar_after = preprocess_s1.preprocess_product(
        safe_zip=Path("fake_after.zip"),
        output_dir=output_dir,
        label="after",
        bbox=bbox,
        acquisition=acq_time,
        orbit=19,
        target_pixels=64,
    )

    # Assert outputs for before
    before_dir = output_dir / "before"
    assert (before_dir / "s1_before_VV.tif").is_file()
    assert (before_dir / "s1_before_VH.tif").is_file()
    assert (before_dir / "S1_PROCESSED.json").is_file()

    # Assert outputs for after
    after_dir = output_dir / "after"
    assert (after_dir / "s1_after_VV.tif").is_file()
    assert (after_dir / "s1_after_VH.tif").is_file()
    assert (after_dir / "S1_PROCESSED.json").is_file()

    # Validate sidecar contents
    sidecar_data = json.loads(sidecar_after.read_text(encoding="utf-8"))
    assert sidecar_data["units"] == "dB"
    assert sidecar_data["bands"] == ["VV", "VH"]
    assert sidecar_data["relative_orbit"] == 19

    # Validate output GeoTIFF values are in dB (linear 0.05 -> ~ -13 dB)
    with rasterio.open(before_dir / "s1_before_VV.tif") as src:
        data = src.read(1)
        assert data.shape == (64, 64)
        assert -20.0 < data.mean() < 0.0
