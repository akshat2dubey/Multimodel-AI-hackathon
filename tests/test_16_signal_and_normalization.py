"""16. One input scaling, one honest checkpoint, and a real fallback mask.

Four claims are checked here, and each has bitten this project in a different way:

* training on an empty dataset fails, and the opt-in smoke run writes a file whose
  NAME says it is a fixture (`*_smoke.pt`) so it can never be mistaken for a
  trained model;
* the dB -> 0..1 mapping is literally one function shared by training and
  inference, and its window travels inside the checkpoint;
* inference honours that window instead of inventing its own scaling;
* with no checkpoint at all, inference falls back to the log-ratio baseline and
  writes the same four-raster contract (flood_mask.tif = NEW water only,
  water_after.tif = all water, flood_mask_before.tif = permanent water,
  valid_mask.tif = observed pixels) with the statistics describing the delivered
  file.

Everything runs on toy data in tmp_path: no network, no credentials, no real
model. A green run says the plumbing is honest, not that a flood map is correct.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import toydata

TIMEOUT_S = 60

# Toy scene geometry: land 28.38/85.60-ish, ~20 m pixels.
LAND_DB = -8.0
WATER_DB = -20.0


# ─── helpers ─────────────────────────────────────────────────


def _write_s1_pair(root: Path) -> tuple[Path, Path]:
    """Write a before/after Sentinel-1 pair (dB GeoTIFFs) that mimics the case study.

    * a permanent river, dark before AND after -> not a flood;
    * a new flood patch, dark only after -> the flood;
    * one isolated speckle pixel of new water -> noise that must be dropped;
    * a nodata block -> must be excluded by valid_mask.tif.
    """
    import rasterio

    transform = toydata.toy_transform()
    crs = toydata.toy_crs()
    size = toydata.SIZE

    before = np.full((size, size), LAND_DB, dtype="float32")
    after = np.full((size, size), LAND_DB, dtype="float32")

    r0, r1 = toydata.RIVER_ROWS
    before[r0:r1, :] = WATER_DB
    after[r0:r1, :] = WATER_DB

    p_r0, p_r1 = toydata.PATCH_ROWS
    p_c0, p_c1 = toydata.PATCH_COLS
    after[p_r0:p_r1, p_c0:p_c1] = WATER_DB

    NODATA = -9999.0

    # Isolated single-pixel "flood" -> speckle (kept clear of the nodata block).
    after[20, 20] = WATER_DB

    # Nodata block in the after scene (top-left 10x10) -> invalid, not land.
    after[0:10, 0:10] = NODATA

    def write(folder: Path, label: str, vv: np.ndarray, vh: np.ndarray) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        for band, arr in (("VV", vv), ("VH", vh)):
            with rasterio.open(
                folder / f"s1_{label}_{band}.tif",
                "w",
                driver="GTiff",
                height=size,
                width=size,
                count=1,
                dtype="float32",
                crs=crs,
                transform=transform,
                nodata=NODATA,
                compress="lzw",
            ) as dst:
                dst.write(arr, 1)

    before_dir = root / "sentinel1" / "before"
    after_dir = root / "sentinel1" / "after"
    # VH is a few dB darker than VV in this fixture; only VV is used by the baseline.
    write(before_dir, "before", before, before - 3.0)
    write(after_dir, "after", after, after - 3.0)
    return before_dir, after_dir


def _run_inference(scripts_dir: Path, cwd: Path, args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(scripts_dir / "inference.py"), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(cwd),
        timeout=TIMEOUT_S,
    )


def _read(path: Path) -> np.ndarray:
    import rasterio

    with rasterio.open(path) as src:
        return src.read(1)


# ─── (a) the checkpoint can never masquerade as a model ──────


def test_smoke_test_writes_a_checkpoint_named_smoke(tmp_path, scripts_dir):
    """--smoke-test is opt-in AND self-labelling: the file it writes is *_smoke.pt."""
    dataset = tmp_path / "empty_dataset"
    dataset.mkdir()
    out = tmp_path / "models" / "flood_unet.pt"

    proc = subprocess.run(
        [
            sys.executable, str(scripts_dir / "train_segmentation.py"),
            "--dataset", str(dataset),
            "--output", str(out),
            "--epochs", "1",
            "--batch-size", "8",
            # 128 px: the smallest size the 4-level U-Net can actually consume.
            "--img-size", "128",
            "--model", "unet",
            "--smoke-test",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(tmp_path),
        timeout=TIMEOUT_S,
    )

    combined = (proc.stdout or "") + (proc.stderr or "")
    assert proc.returncode == 0, (
        f"--smoke-test must produce a checkpoint; exit={proc.returncode}\n{combined[-1500:]}"
    )

    smoke_path = out.with_name(out.stem + "_smoke" + out.suffix)
    assert smoke_path.is_file(), (
        f"a smoke-test run must write {smoke_path.name}, not the trained-model "
        f"filename. Files present: {sorted(p.name for p in out.parent.glob('*'))}"
    )
    assert not out.exists(), (
        f"{out.name} exists after a --smoke-test run. The checkpoint trained on "
        "fabricated tiles must never occupy the trained-model path."
    )

    info_path = out.parent / "model_info_smoke.json"
    assert info_path.is_file(), (
        "the smoke run must describe itself in model_info_smoke.json "
        f"(found {sorted(p.name for p in out.parent.glob('*.json'))})"
    )
    info = json.loads(info_path.read_text(encoding="utf-8"))
    assert info.get("synthetic_training_data") is True, (
        "model_info_smoke.json must record that the training data was synthetic, "
        f"got synthetic_training_data={info.get('synthetic_training_data')!r}"
    )


# ─── (c) one normalisation, shared and persisted ─────────────


def test_normalisation_is_one_function_shared_by_training_and_inference(scripts_dir):
    import inference
    import s1_normalization
    import train_segmentation

    assert train_segmentation.normalize_bands is s1_normalization.normalize_bands, (
        "training must use s1_normalization.normalize_bands, not its own scaling"
    )
    assert inference.normalize_bands is s1_normalization.normalize_bands, (
        "inference must use s1_normalization.normalize_bands, not its own scaling"
    )

    cfg = s1_normalization.normalization_config()
    stacked = np.array([[[-30.0, 0.0]], [[-30.0, 0.0]], [[-10.0, 10.0]]], dtype="float32")
    scaled = s1_normalization.normalize_bands(stacked, ["VV", "VH", "ratio"], cfg)
    assert scaled.shape == stacked.shape
    assert scaled.min() >= 0.0 and scaled.max() <= 1.0, (
        f"normalisation must land in 0..1, got [{scaled.min()}, {scaled.max()}]"
    )
    assert scaled[0, 0, 0] == pytest.approx(0.0) and scaled[0, 0, 1] == pytest.approx(1.0), (
        "the dB window endpoints must map to 0 and 1 (VV -30 dB -> 0, 0 dB -> 1)"
    )
    assert scaled[2, 0, 0] == pytest.approx(0.0) and scaled[2, 0, 1] == pytest.approx(1.0), (
        "the ratio channel must use the ratio window, not the backscatter one"
    )

    # Clipping outside the window (a -40 dB shadow or a +10 dB target).
    clipped = s1_normalization.normalize_band(np.array([-40.0, 10.0], dtype="float32"), "VV", cfg)
    assert list(clipped) == [0.0, 1.0], f"out-of-window dB must clip, got {clipped}"

    # An optical band is already 0..1 reflectance and must not be rescaled.
    optical = s1_normalization.normalize_band(np.array([0.25], dtype="float32"), "B04", cfg)
    assert optical[0] == pytest.approx(0.25), (
        "Sentinel-2 reflectance must pass through, not be treated as dB"
    )


def test_normalisation_window_is_stored_in_and_read_from_the_checkpoint(scripts_dir):
    import s1_normalization

    stored = {"normalization": {"db_min": -22.0, "db_max": -2.0, "kind": "db_linear"}}
    cfg = s1_normalization.normalization_from_checkpoint(stored)
    assert cfg["source"] == "checkpoint", "a stored window must be reported as coming from the checkpoint"
    assert (cfg["db_min"], cfg["db_max"]) == (-22.0, -2.0), (
        f"the checkpoint's own window must win, got {cfg['db_min']}..{cfg['db_max']}"
    )

    # A checkpoint from before this existed must say so rather than pretend.
    legacy = s1_normalization.normalization_from_checkpoint({})
    assert legacy["source"] != "checkpoint", (
        "a checkpoint with no normalisation block must be flagged as using the "
        "default window, not silently treated as if it carried one"
    )
    assert (legacy["db_min"], legacy["db_max"]) == (
        s1_normalization.DB_MIN, s1_normalization.DB_MAX
    )


def test_inference_uses_the_window_from_the_model_not_its_own(monkeypatch):
    """tile_inference must scale inputs with the config the model carries."""
    import torch

    import inference

    class _MeanModel(torch.nn.Module):
        """Returns the mean of its input as the flood probability."""

        def forward(self, x):  # (B, C, H, W) -> (B, 1, H, W)
            m = x.mean(dim=(1, 2, 3)).clamp(0.0, 1.0)
            return m.view(-1, 1, 1, 1).expand(-1, 1, x.shape[2], x.shape[3]).contiguous()

    image = np.full((1, 32, 32), -20.0, dtype="float32")  # -20 dB everywhere

    wide = _MeanModel()
    wide.normalization = {"db_min": -40.0, "db_max": 0.0, "ratio_min": -10.0, "ratio_max": 10.0}
    wide.bands = ["VV"]
    _, prob_wide = inference.tile_inference(wide, image, 32, 0, device=torch.device("cpu"))

    narrow = _MeanModel()
    narrow.normalization = {"db_min": -20.0, "db_max": 0.0, "ratio_min": -10.0, "ratio_max": 10.0}
    narrow.bands = ["VV"]
    _, prob_narrow = inference.tile_inference(narrow, image, 32, 0, device=torch.device("cpu"))

    # tile_inference feeds the model through a sigmoid, so the expectation is
    # sigmoid(scaled value).
    assert prob_wide[0, 0] == pytest.approx(float(torch.sigmoid(torch.tensor(0.5))), abs=1e-6), (
        "-20 dB in the window [-40, 0] must scale to 0.5; if this is wrong the "
        "model's own window was ignored"
    )
    assert prob_narrow[0, 0] == pytest.approx(float(torch.sigmoid(torch.tensor(0.0))), abs=1e-6), (
        "-20 dB in the window [-20, 0] must scale to 0.0 — proof that the "
        "per-model window is actually applied"
    )


# ─── (b) no checkpoint -> a real baseline mask, same contract ─


def test_baseline_fallback_writes_the_four_raster_contract(tmp_path, scripts_dir):
    before_dir, after_dir = _write_s1_pair(tmp_path)
    results = tmp_path / "results"

    proc = _run_inference(
        scripts_dir,
        tmp_path,
        [
            "--before-dir", str(before_dir),
            "--after-dir", str(after_dir),
            "--baseline",
            "--output", str(results / "flood_mask.tif"),
            "--img-size", "64",
        ],
    )
    combined = (proc.stdout or "") + (proc.stderr or "")
    assert proc.returncode == 0, f"baseline inference failed:\n{combined[-2000:]}"

    flood = _read(results / "flood_mask.tif")
    water_after = _read(results / "water_after.tif")
    permanent = _read(results / "flood_mask_before.tif")
    valid = _read(results / "valid_mask.tif")

    r0, r1 = toydata.RIVER_ROWS
    p_r0, p_r1 = toydata.PATCH_ROWS
    p_c0, p_c1 = toydata.PATCH_COLS
    patch_px = (p_r1 - p_r0) * (p_c1 - p_c0)

    assert flood[r0:r1, :].sum() == 0, (
        "the permanent river must not be in flood_mask.tif: a river is not a flood, "
        "and treating it as one flags every riverside building and every bridge"
    )
    assert flood[p_r0:p_r1, p_c0:p_c1].all(), (
        "the new flood patch must be fully present in flood_mask.tif"
    )
    assert flood[20, 20] == 0, (
        "a single speckle pixel of 'new water' must be removed by the speckle filter"
    )
    assert int(flood.sum()) == patch_px, (
        f"flood_mask.tif should hold exactly the {patch_px}-pixel patch, got "
        f"{int(flood.sum())}"
    )

    assert water_after[r0:r1, :].all(), (
        "water_after.tif must contain the permanent river — it is the full "
        "post-event water extent, kept separate from the flood"
    )
    assert permanent[r0:r1, :].all() and permanent[p_r0:p_r1, p_c0:p_c1].sum() == 0, (
        "flood_mask_before.tif must be the PRE-event water: the river yes, the new "
        "patch no"
    )

    assert valid[0:10, 0:10].sum() == 0, (
        "the nodata block must be marked invalid: the mask may not claim to have "
        "observed pixels the sensor never returned"
    )
    assert valid[100, 100] == 1, "ordinary observed pixels must be valid"

    change = json.loads((results / "change_detection.json").read_text(encoding="utf-8"))
    assert change["method"] == "log_ratio_baseline", (
        f"the JSON must record which engine produced the mask, got {change.get('method')!r}"
    )
    assert change["flood_extent_pixels"] == int(flood.sum()), (
        "the statistics must describe the DELIVERED raster, not an in-memory mask "
        f"({change['flood_extent_pixels']} != {int(flood.sum())})"
    )
    assert change["flood_extent_raw_pixels"] > change["flood_extent_pixels"], (
        "the speckle pixel must show up as a difference between the raw change mask "
        "and the delivered one"
    )
    assert change["small_components_removed_pixels"] >= 1
    assert change["min_component_pixels"] >= 1
    assert change["delivered_mask"] == "flood_mask.tif"
    assert change["water_after_mask"] == "water_after.tif"
    assert change["permanent_water_mask"] == "flood_mask_before.tif"
    assert change["valid_mask"] == "valid_mask.tif"
    # Aliases the dashboard reads, kept consistent with the delivered raster.
    assert change["flood_pixels"] == int(flood.sum())
    assert change["newly_flooded_pixels"] == int(flood.sum())
    assert change["flood_area_ha"] > 0


def test_missing_checkpoint_falls_back_instead_of_stopping(tmp_path, scripts_dir):
    """No --baseline flag either: a missing checkpoint must not abort the pipeline."""
    before_dir, after_dir = _write_s1_pair(tmp_path)
    results = tmp_path / "results"

    proc = _run_inference(
        scripts_dir,
        tmp_path,
        [
            "--before-dir", str(before_dir),
            "--after-dir", str(after_dir),
            "--model", str(tmp_path / "models" / "flood_unet.pt"),  # does not exist
            "--output", str(results / "flood_mask.tif"),
            "--img-size", "64",
        ],
    )
    combined = (proc.stdout or "") + (proc.stderr or "")
    assert proc.returncode == 0, f"a missing checkpoint must fall back, not fail:\n{combined[-2000:]}"
    assert "baseline" in combined.lower(), (
        "the fallback must be announced, not silent:\n" + combined[-1200:]
    )
    assert (results / "flood_mask.tif").exists()
    change = json.loads((results / "change_detection.json").read_text(encoding="utf-8"))
    assert change["method"] == "log_ratio_baseline"


def test_run_pipeline_never_trains_the_model(scripts_dir):
    """The orchestrator must not train: that fabricated a dataset and a checkpoint."""
    source = (scripts_dir / "run_pipeline.py").read_text(encoding="utf-8")
    # A real invocation would build the command: `_scripts("train_segmentation.py")`
    # or `str(scripts_dir / "train_segmentation.py")` passed to run([...]). Matching
    # bare mentions would flag the advice printed to the operator, and even a
    # comment; matching the command construction is the thing that matters.
    offender = [
        line.strip()
        for line in source.splitlines()
        if "train_segmentation.py" in line
        and ("_scripts(" in line or "scripts_dir /" in line or "sys.executable" in line)
    ]
    assert not offender, (
        "run_pipeline.py still invokes training (it did so inline with --epochs 50 "
        "whenever models/flood_unet.pt was missing, which trained on synthetic tiles):\n"
        + "\n".join(offender)
    )
    assert "--baseline" in source, (
        "run_pipeline.py must fall back to inference.py's log-ratio baseline when no "
        "checkpoint exists, so a real mask is still produced"
    )
