#!/usr/bin/env python3
"""
Flood Segmentation Inference Pipeline

Runs the trained U-Net model on Sentinel-1/2 imagery to produce a flood probability mask.
Compares before/after imagery to detect flood extent change.

Uses the model trained by train_segmentation.py.
Works for any area/date: the area and acquisition dates come from
results/run_meta.json, not from a literal in this file.

Usage:
    python inference.py \
        --before-dir ./data/sentinel1/before \
        --after-dir ./data/sentinel1/after \
        --model ./models/flood_unet.pt \
        --output ./results/flood_mask.tif

Input:  Sentinel-1 GRD products (VV + VH polarization bands) or Sentinel-2 L2A
Output: GeoTIFF flood probability mask (0=land, 1=water/debris) + change detection map
"""

import argparse

import sys
import json
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from rasterio.windows import Window
from tqdm import tqdm

if hasattr(sys.stdout, "reconfigure"):  # never crash on a cp1252 Windows console
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    from train_segmentation import UNet, ResNetUNet, BCEDiceLoss
    MODEL_CLASSES = {"unet": UNet, "resnet_unet": ResNetUNet}
except ImportError:
    MODEL_CLASSES = {}
    print("⚠️ Cannot import model classes — ensure scripts/ is in PYTHONPATH")

# The single dB -> 0..1 input mapping, shared with train_segmentation.py. The
# window comes from the checkpoint config when it has one; see load_model().
from s1_normalization import (
    DB_MIN,
    describe_normalization,
    normalize_bands,
    normalization_config,
    normalization_from_checkpoint,
)

# Speckle control for the delivered flood mask, shared with make_demo_data.py so
# the fixture applies the same rule the real run does.
from mask_postprocess import MIN_COMPONENT_PIXELS, remove_small_components  # noqa: E402

# Checkpoints are pickles: safe_model is the only place this project loads one,
# and it enforces weights_only=True plus a matching .sha256 sidecar (T35).
from safe_model import ModelTrustError, load_checkpoint  # noqa: E402

# Log-ratio change-detection baseline (used when no trained checkpoint exists).
# A drop of this many dB between the before and after VV acquisitions is new
# water: calm open water is specular, so it returns far less backscatter.
BASELINE_THRESHOLD_DB = 1.5

# Backscatter below this (dB, VV) is water rather than land. Typical calm water
# sits near -20 dB; vegetated or built terrain sits far above it.
WATER_BACKSCATTER_DB = -15.0

# Result-filename contract. damage_assessment.py and cutoff_analysis.py read
# flood_mask.tif, and nothing else, so these names are not cosmetic.
FLOOD_MASK_NAME = "flood_mask.tif"
WATER_AFTER_NAME = "water_after.tif"
PERMANENT_WATER_NAME = "flood_mask_before.tif"
VALID_MASK_NAME = "valid_mask.tif"


def load_model(
    model_path: Path,
    device: torch.device,
    allow_unverified: bool = False,
) -> torch.nn.Module:
    """Load trained model from checkpoint, plus the scaling it was trained with.

    The file is hash-checked and deserialised with weights_only=True before
    anything from it is used; a tampered or unverifiable checkpoint stops the run
    with a message rather than loading different weights than the ones evaluated.
    """
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        sys.exit(1)

    try:
        checkpoint = load_checkpoint(
            model_path, device=device, allow_unverified=allow_unverified
        )
    except ModelTrustError as exc:
        print(f"\n❌ {exc}")
        sys.exit(1)

    config = checkpoint.get("config", {})
    model_type = config.get("model_type", "unet")
    in_channels = config.get("in_channels", 2)
    if not isinstance(model_type, str):
        model_type = model_type.__name__ if hasattr(model_type, "__name__") else "unet"
    model_type = model_type.lower()  # "ResNetUNet" or "UNet" -> "resnetunet" / "unet"
    print(f"  Checkpoint config: model_type={model_type} in_channels={in_channels}")
    if model_type == "resnetunet":
        model_type = "resnet_unet"

    if model_type in MODEL_CLASSES:
        model = MODEL_CLASSES[model_type](
            in_channels=in_channels,
            num_classes=1,
            pretrained=False,
        )
    else:
        from train_segmentation import UNet as _UNet
        model = _UNet(in_channels=in_channels, num_classes=1)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # The scaling the weights were trained against. A checkpoint without the
    # block predates the shared normalisation, so the canonical window is used
    # and the mismatch is printed instead of being hidden.
    normalization = normalization_from_checkpoint(config)
    model.normalization = normalization
    model.bands = config.get("bands") or ["VV", "VH"]

    print(f"  Loaded model: {model_type}")
    print(f"  Input channels: {in_channels}")
    print(f"  Input norm: {describe_normalization(normalization)}")
    print(f"  Best val Dice: {checkpoint.get('best_val_dice', 'N/A'):.4f}")

    return model


def _load_s1_sidecar(image_path: Path) -> dict | None:
    """Read the S1_PROCESSED.json sidecar that preprocess_s1.py writes.

    Consumers (this file, damage_assessment, flood_path, the dashboard) should
    read the sidecar instead of guessing whether the tifs are linear sigma0 or
    dB, or what the CRS / pixel size is.
    """
    sidecar = image_path / "S1_PROCESSED.json"
    if not sidecar.exists():
        return None
    try:
        return json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"  ⚠️ Sidecar present but unreadable: {sidecar}: {exc}")
        return None


def _valid_mask(vv: np.ndarray, vh: np.ndarray, units: str | None) -> np.ndarray:
    """Pixels the imagery actually observed: no nodata, no radar shadow.

    Two things make a Sentinel-1 pixel untrustworthy for flood mapping: the
    nodata collar (already NaN here) and radar shadow / layover, where the sensor
    receives almost no return and the value is meaningless. Either way the model
    is guessing, so the pixel is marked invalid rather than reported as land.
    """
    finite = np.isfinite(vv) & np.isfinite(vh)
    is_linear = units not in (None, "dB", "db", "decibel", "decibels")
    if is_linear:
        # sigma0 <= 0 is not a measurement; -50 dB is the shadow floor used by _to_db.
        above_floor = (np.nan_to_num(vv, nan=0.0) > 1e-5) & (np.nan_to_num(vh, nan=0.0) > 1e-5)
    else:
        above_floor = (np.nan_to_num(vv, nan=-999.0) > -50.0) & (np.nan_to_num(vh, nan=-999.0) > -50.0)
    return (finite & above_floor)


def preprocess_sentinel1(
    image_path: Path | None,
    return_valid: bool = False,
):
    """
    Load and preprocess Sentinel-1 GRD data.

    Expected: VV and VH polarization bands as separate GeoTIFFs or a combined file.
    Preprocessing:
      1. Radiometric calibration (convert to backscatter coefficient σ° in dB)
      2. Speckle filtering (simple 3x3 median filter for demo; Lee filter for production)
      3. Create input tensor: [VV_dB, VH_dB, VV/VH_ratio]

    Returns: numpy array of shape (C, H, W), or ``(array, valid)`` with
    ``return_valid=True`` where ``valid`` is the (H, W) boolean observed-pixel
    mask (nodata / radar-shadow excluded).
    """
    if image_path is None or not image_path.exists():
        return None

    def _ret(img: np.ndarray) -> np.ndarray:
        """Return the image, or (image, valid) when the caller asked for it."""
        if return_valid:
            return img, _valid_mask(img[0], img[1], units)
        return img

    print(f"  Loading Sentinel-1 data from: {image_path}")

    # If a sidecar exists, let it decide units and spot-fill missing metadata;
    # otherwise fall back to the old behaviour of assuming the tifs are already
    # in dB (the synthetic fixture and the Preprocess API crop both arrive that
    # way, but a bare linear sigma0 download would not).
    sidecar = _load_s1_sidecar(image_path)
    units = None
    if sidecar is not None:
        units = sidecar.get("units")
        if units and str(units).lower() != "db":
            print(f"  ⚠️ Sidecar says units={units!r} — treating as linear sigma0, converting to dB")
    else:
        print("  ⚠️ No S1_PROCESSED.json sidecar — assuming tifs are already dB")

    # Try to find band files
    vv_path = None
    vh_path = None

    # Case 1: Separate band files
    for f in image_path.iterdir():
        name = f.name.lower()
        if "vv" in name and f.suffix in (".tif", ".tiff"):
            vv_path = f
        if "vh" in name and f.suffix in (".tif", ".tiff"):
            vh_path = f

    # Case 2: Combined file with multiple bands
    if vv_path is None and vh_path is None:
        tif_files = list(image_path.glob("*.tif")) + list(image_path.glob("*.tiff"))
        if tif_files:
            # Assume first tif contains all bands
            with rasterio.open(tif_files[0]) as src:
                data = src.read()  # (bands, H, W)
                if data.shape[0] >= 2:
                    vv = data[0]
                    vh = data[1]
                    print(f"  Loaded combined file: {src.count} bands, {src.width}x{src.height}")
                    # Respect the sidecar's units when present; otherwise assume dB.
                    vv_db, vh_db = _to_db(vv, vh, units)
                    ratio = vv_db - vh_db  # VV/VH ratio
                    return _ret(np.stack([vv_db, vh_db, ratio], axis=0))

    # Case 3: Individual band files
    if vv_path and vh_path:
        with rasterio.open(vv_path) as src_vv:
            vv = src_vv.read(1).astype(np.float32)
            nodata = src_vv.nodata
            if nodata is not None:
                vv[vv == nodata] = np.nan
        with rasterio.open(vh_path) as src_vh:
            vh = src_vh.read(1).astype(np.float32)
            nodata = src_vh.nodata
            if nodata is not None:
                vh[vh == nodata] = np.nan
        vv_db, vh_db = _to_db(vv, vh, units)
        ratio = vv_db - vh_db
        return _ret(np.stack([vv_db, vh_db, ratio], axis=0))

    print(f"  ⚠️ Could not find VV/VH bands")
    return None


def _to_db(vv: np.ndarray, vh: np.ndarray, units: str | None) -> tuple[np.ndarray, np.ndarray]:
    """Convert one or both bands to dB if the sidecar says they are linear, else pass through.

    The two-band path keeps the array layout callers expect: an (H,W) per band. When one
    input is already dB and the other is linear, this still converts only the linear one
    so both end up with the same units.
    """
    import numpy as np

    def _maybe_to_db(arr: np.ndarray) -> np.ndarray:
        masked = np.ma.masked_invalid(arr)
        if masked.count() == 0:
            return arr.copy()
        is_linear = units not in (None, "dB", "db", "decibel", "decibels")
        if is_linear:
            out = np.where(masked.filled(0) > 0, 10.0 * np.log10(np.clip(masked.filled(1e-6), 1e-6, None)), -999.0)
            out = out.astype(np.float32)
            out[masked.mask] = np.nan
            return out
        return arr.astype(np.float32)

    vv_db = _maybe_to_db(vv)
    vh_db = _maybe_to_db(vh)
    # Clip to a sane no-data range so the mask engine is not fed -9999 noise.
    # The preprocess_s1.py output already lands in a reasonable dB window, so this
    # is mostly a safety net for legacy / ambiguous inputs.
    for arr, db in ((vv, vv_db), (vh, vh_db)):
        db[db < -50] = np.ma.median(db[db >= -50]) if (db >= -50).any() else -20.0
    return vv_db, vh_db


def preprocess_sentinel2(image_path: Path | None) -> np.ndarray | None:
    """
    Load and preprocess Sentinel-2 L2A data.

    Uses bands: B02 (Blue), B03 (Green), B04 (Red), B08 (NIR), B11 (SWIR), B12 (SWIR)
    Returns: numpy array of shape (6, H, W)
    """
    if image_path is None or not image_path.exists():
        return None

    print(f"  Loading Sentinel-2 data from: {image_path}")
    # Implementation would read band files and stack them
    # For hackathon: placeholder
    bands = []
    for band_id in ["B02", "B03", "B04", "B08", "B11", "B12"]:
        band_file = list(image_path.glob(f"*{band_id}*.tif")) + list(image_path.glob(f"*{band_id}*.tiff"))
        if band_file:
            with rasterio.open(band_file[0]) as src:
                band_data = src.read(1).astype(np.float32)
                # Sentinel-2 L2A is surface reflectance (0-10000 scale, divide by 10000)
                band_data = band_data / 10000.0
                bands.append(band_data)
        else:
            # Placeholder zero band
            bands.append(np.zeros((512, 512), dtype=np.float32))

    # Cloud mask (if available from SCL band)
    scl_file = list(image_path.glob("*SCL*.tif")) + list(image_path.glob("*SCL*.tiff"))
    if scl_file:
        with rasterio.open(scl_file[0]) as src:
            scl = src.read(1)
            # Mask: 0=cloud, 1=shadow, 2=vegetation, 3=bare, 4=water, 5=unclassified
            cloud_mask = (scl <= 1)  # cloud + shadow
            # Set cloud pixels to NaN in all bands
            for i in range(len(bands)):
                bands[i][cloud_mask] = np.nan
            print(f"  Applied cloud mask (SCL band) — masked {cloud_mask.sum()} pixels")

    return np.stack(bands, axis=0)


def _tile_starts(extent: int, tile_size: int, stride: int) -> list[int]:
    """Tile start offsets that cover [0, extent) with no gaps.

    Stops after the last fitting position, then appends an edge-aligned tile when
    the regular stride would leave a strip short of the boundary.
    """
    if extent <= tile_size:
        return [0]
    starts = list(range(0, extent - tile_size + 1, stride))
    last = extent - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def derive_change_mask(
    before_mask: np.ndarray | None,
    after_mask: np.ndarray,
) -> np.ndarray:
    """The delivered flood mask: water that appeared since the before scene.

    The after scene contains the permanent river, so writing it as flood_mask.tif
    reports the whole river as "flooded" — every riverside building reads as
    DESTROYED and every bridge over the river as damaged. The product this
    pipeline promises is the *change*: after AND NOT before.
    """
    after_bin = (np.asarray(after_mask) > 0.5)
    if before_mask is None:
        return after_bin.astype(np.float32)
    before_bin = (np.asarray(before_mask) > 0.5)
    return (after_bin & ~before_bin).astype(np.float32)


def log_ratio_baseline(
    before_img: np.ndarray | None,
    after_img: np.ndarray,
    threshold_db: float = BASELINE_THRESHOLD_DB,
    min_size: int = MIN_COMPONENT_PIXELS,
    water_db: float = WATER_BACKSCATTER_DB,
) -> tuple[np.ndarray, int]:
    """Change-detection baseline for when no trained checkpoint exists.

    Not a segmentation model — a physical measurement. Calm open water is
    specular, so its backscatter collapses: a pixel whose VV drops by more than
    ``threshold_db`` between the two acquisitions has almost certainly become
    water. Newly flooded = drop larger than the threshold AND not water before.

    This keeps a real (if coarse) mask flowing through the pipeline when
    ``models/flood_unet.pt`` is absent, instead of the pipeline stalling or, worse,
    emitting a placeholder. Returns (mask, removed_small_blobs).
    """
    if after_img is None:
        raise ValueError("log-ratio baseline needs an after image")
    after_vv = np.asarray(after_img[0], dtype=np.float32)
    if before_img is None:
        before_vv = np.full_like(after_vv, np.nan)
    else:
        before_vv = np.asarray(before_img[0], dtype=np.float32)
        if before_vv.shape != after_vv.shape:
            h = min(before_vv.shape[0], after_vv.shape[0])
            w = min(before_vv.shape[1], after_vv.shape[1])
            before_vv, after_vv = before_vv[:h, :w], after_vv[:h, :w]

    # In dB the log ratio is a plain difference: after - before.
    drop = before_vv - after_vv  # positive = darker after = new water
    after_is_water = after_vv < float(water_db)
    # Pre-event water is the permanent-water mask: a river that was already
    # there is not flood damage, however dark it looks afterwards.
    was_water = np.nan_to_num(before_vv, nan=0.0) < float(water_db)
    newly_flooded = (drop > float(threshold_db)) & after_is_water & ~was_water

    newly_flooded = np.nan_to_num(newly_flooded, nan=0.0).astype(bool)
    mask, removed = remove_small_components(newly_flooded, min_size)
    return mask, removed


def tile_inference(
    model: torch.nn.Module,
    image: np.ndarray,
    tile_size: int = 256,
    overlap: int = 64,
    device: torch.device = torch.device("cpu"),
    threshold: float = 0.5,
    normalization: dict | None = None,
    band_names=None,
) -> np.ndarray:
    """
    Run inference on a large image using sliding window tiling.

    Args:
        model: Trained U-Net model
        image: (C, H, W) numpy array of input bands (Sentinel-1 dB)
        tile_size: Size of each tile (pixels)
        overlap: Overlap between tiles (for seamless stitching)
        device: torch device
        threshold: Probability threshold for binary mask
        normalization: dB -> 0..1 config; defaults to the one baked into the
            checkpoint (``model.normalization``), so inference always scales
            inputs exactly as training did
        band_names: channel names for the normalisation windows

    Returns:
        Binary flood mask of shape (H, W)
    """
    C, H, W = image.shape
    output_mask = np.zeros((H, W), dtype=np.float32)
    output_count = np.zeros((H, W), dtype=np.float32)  # For averaging overlaps

    # Normalise with the SAME function training used, with the SAME window: the
    # config stored in the checkpoint. Feeding raw dB to a model trained on
    # 0..1-scaled inputs (or the reverse) silently produces garbage.
    norm_cfg = normalization or getattr(model, "normalization", None) or normalization_config()
    names = band_names or getattr(model, "bands", None)
    image_norm = normalize_bands(image, names, norm_cfg)

    # Pad to make divisible by tile_size
    pad_h = (tile_size - H % tile_size) % tile_size
    pad_w = (tile_size - W % tile_size) % tile_size
    if pad_h > 0 or pad_w > 0:
        image_norm = np.pad(image_norm, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")
        H_pad, W_pad = image_norm.shape[1], image_norm.shape[2]
    else:
        H_pad, W_pad = H, W

    # Slide window. Positions are edge-aligned: a plain range(0, H_pad - tile + 1,
    # stride) leaves an uncovered strip at the far edge whenever the padding is
    # not an exact multiple of the stride (e.g. 700 px padded to 768, stride 192:
    # tiles at 0/192/384 stop at 640 and columns 640-767 are never written).
    stride = tile_size - overlap

    for y in tqdm(_tile_starts(H_pad, tile_size, stride), desc="  Tiling", leave=False):
        for x in _tile_starts(W_pad, tile_size, stride):
            tile = image_norm[:, y : y + tile_size, x : x + tile_size]
            tile_tensor = torch.from_numpy(tile).unsqueeze(0).to(device)

            with torch.no_grad():
                pred = model(tile_tensor)  # (1, 1, tile_size, tile_size)
                pred_prob = F.sigmoid(pred).squeeze().cpu().numpy()

            # Add to output with overlap averaging. The slice may be smaller
            # than tile_size when x+tile_size exceeds the padded width — crop
            # the prediction to the same window.
            region = output_mask[y : y + tile_size, x : x + tile_size]
            output_mask[y : y + tile_size, x : x + tile_size] = region + pred_prob[: region.shape[0], : region.shape[1]]
            count_region = output_count[y : y + tile_size, x : x + tile_size]
            output_count[y : y + tile_size, x : x + tile_size] = count_region + 1.0

    # Average overlaps — masks exist in padded space (H_pad, W_pad)
    output_mask = np.divide(
        output_mask, output_count,
        out=np.zeros_like(output_mask), where=output_count > 0,
    )

    # Crop back to original size FIRST, then threshold on the valid region.
    output_mask = output_mask[:H, :W]
    binary_mask = (output_mask > threshold).astype(np.float32)

    return binary_mask, output_mask


def compute_change_detection(
    before_mask: np.ndarray,
    after_mask: np.ndarray,
    before_prob: np.ndarray | None = None,
    after_prob: np.ndarray | None = None,
    transform=None,
    crs=None,
    delivered_mask: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
    min_component_pixels: int | None = None,
    method: str = "unet",
) -> dict:
    """
    Compute flood change detection from before/after masks.

    Returns dict with:
      - flood_extent: pixels newly flooded (after=1, before=0)
      - disappeared: pixels that were water before but not after (rare)
      - persistent_water: water in both images (rivers, lakes)
      - total_flooded: all water pixels in after image
      - flood_area_ha: flood extent in hectares

    ``delivered_mask``, when given, is the artifact actually written to
    ``flood_mask.tif`` (change mask + speckle removal). The extent figures then
    describe THAT file, so the numbers in this JSON and the raster the dashboard
    and damage assessment read can never disagree; ``flood_extent_raw_pixels``
    keeps the pre-speckle count and ``small_components_removed_pixels`` the
    difference.

    Areas are derived from the raster transform when one is supplied; the old
    hard-coded 0.01 ha (a 10 m pixel) silently mis-scaled every hectare figure on
    gridded data that is not exactly 10 m.
    """
    before_bin = (before_mask > 0.5).astype(np.int32) if before_mask is not None else np.zeros_like(after_mask)
    after_bin = (after_mask > 0.5).astype(np.int32)

    flood_extent = after_bin & (1 - before_bin)       # Newly flooded
    persistent = after_bin & before_bin                # Water in both
    disappeared = (1 - after_bin) & before_bin         # Water disappeared (drought/drainage)

    # Default remains the historical 10 m assumption so callers that pass no
    # transform keep working; the pipeline always passes one.
    area_per_pixel_ha = 0.01
    if transform is not None:
        from damage_assessment import pixel_area_ha as _pixel_area_ha
        area_per_pixel_ha = _pixel_area_ha(transform, crs)

    result = {
        "flood_extent_pixels": int(flood_extent.sum()),
        "flood_extent_ha": round(flood_extent.sum() * area_per_pixel_ha, 2),
        "persistent_water_pixels": int(persistent.sum()),
        "persistent_water_ha": round(persistent.sum() * area_per_pixel_ha, 2),
        "total_water_after_pixels": int(after_bin.sum()),
        "total_water_after_ha": round(after_bin.sum() * area_per_pixel_ha, 2),
        "disappeared_pixels": int(disappeared.sum()),
        "flood_percentage": round(
            flood_extent.sum() / max(after_bin.sum(), 1) * 100, 1
        ),
        "pixel_area_ha": round(area_per_pixel_ha, 6),
        # The files this run wrote, so a consumer knows where each number lives.
        "delivered_mask": "flood_mask.tif",
        "water_after_mask": "water_after.tif",
        "permanent_water_mask": PERMANENT_WATER_NAME,
        "valid_mask": VALID_MASK_NAME,
        # How the mask was produced: "unet" or "log_ratio_baseline".
        "method": method,
        # Explicit, unambiguous aliases for the three quantities above.
        "newly_flooded_pixels": int(flood_extent.sum()),
        "permanent_water_pixels": int(persistent.sum()),
        "water_after_pixels": int(after_bin.sum()),
        # Keys the dashboard already reads (templates/index.html).
        "flood_pixels": int(flood_extent.sum()),
        "total_pixels": int(after_mask.size),
    }
    result["newly_flooded_ha"] = result["flood_extent_ha"]
    result["permanent_water_ha"] = result["persistent_water_ha"]
    result["water_after_ha"] = result["total_water_after_ha"]
    result["flood_area_ha"] = result["flood_extent_ha"]

    # Describe the file that was actually written, not the raw inference mask.
    if delivered_mask is not None:
        delivered_px = int((np.asarray(delivered_mask) > 0.5).sum())
        raw_px = result["flood_extent_pixels"]
        result["flood_extent_pixels"] = delivered_px
        result["flood_extent_ha"] = round(delivered_px * area_per_pixel_ha, 2)
        result["flood_extent_raw_pixels"] = raw_px
        result["small_components_removed_pixels"] = max(raw_px - delivered_px, 0)
        result["min_component_pixels"] = (
            int(min_component_pixels) if min_component_pixels is not None
            else int(MIN_COMPONENT_PIXELS)
        )
        result["flood_percentage"] = round(
            delivered_px / max(result["total_water_after_pixels"], 1) * 100, 1
        )
        result["newly_flooded_pixels"] = delivered_px
        result["newly_flooded_ha"] = result["flood_extent_ha"]
        result["flood_pixels"] = delivered_px
        result["flood_area_ha"] = result["flood_extent_ha"]

    # Observation coverage: how much of the scene the imagery actually observed.
    if valid_mask is not None:
        valid = np.asarray(valid_mask) > 0.5
        result["valid_pixels"] = int(valid.sum())
        result["invalid_pixels"] = int(valid.size - valid.sum())
        result["valid_percentage"] = round(valid.sum() / max(valid.size, 1) * 100, 1)

    return result


def save_valid_mask(
    valid: np.ndarray,
    profile: dict,
    output_path: Path,
    before_profile_or_dir: dict | Path | None = None,
) -> None:
    """Write the observed-pixel mask (1 = usable observation, 0 = nodata/shadow).

    A separate raster on purpose: it describes the *input imagery*, not the flood,
    and damage_assessment.load_valid_mask() can consume it to report UNKNOWN
    instead of guessing on pixels the sensor never really saw.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged = dict(profile) if profile else {}
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=valid.shape[0],
        width=valid.shape[1],
        count=1,
        dtype="float32",
        crs=merged.get("crs", "EPSG:4326"),
        transform=merged.get("transform"),
        compress="lzw",
    ) as dst:
        dst.write((np.asarray(valid) > 0.5).astype(np.float32), 1)
    print(f"  Valid-data mask: {output_path} "
          f"({int((np.asarray(valid) > 0.5).sum()):,} observed px)")


def save_flood_mask(
    mask: np.ndarray,
    prob_map: np.ndarray | None,
    profile: dict,
    output_path: Path,
    before_profile_or_dir: dict | Path | None = None,
    prob_name: str | None = "flood_probability.tif",
):
    """Save flood mask as GeoTIFF with proper georeferencing.

    ``before_profile_or_dir`` is either a S1_PROCESSED.json dict (from a sidecar)
    or the before-directory Path (so this signature still accepts the old
    ``save_flood_mask(..., before_dir)`` calls without type errors). When it is a
    Path we read the sidecar from that directory and merge its CRS/nodata into
    the profile before writing, so the mask inherits the grid the preprocess step
    actually produced.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    merged = dict(profile) if profile is not None else {
        "driver": "GTiff",
        "height": mask.shape[0],
        "width": mask.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": rasterio.transform.from_bounds(
            0, 0, mask.shape[1], mask.shape[0], mask.shape[1], mask.shape[0]
        ),
    }

    # Merge CRS / nodata from a sidecar when the caller passed one.
    if before_profile_or_dir is not None:
        if isinstance(before_profile_or_dir, dict):
            sidecar = before_profile_or_dir
        elif isinstance(before_profile_or_dir, Path) and (before_profile_or_dir / "S1_PROCESSED.json").exists():
            try:
                sidecar = json.loads((before_profile_or_dir / "S1_PROCESSED.json").read_text(encoding="utf-8"))
            except Exception:
                sidecar = None
        else:
            sidecar = None
        if sidecar is not None:
            crs_raw = sidecar.get("crs")
            if isinstance(crs_raw, str) and crs_raw.startswith("EPSG:"):
                try:
                    from rasterio.crs import CRS
                    merged["crs"] = CRS.from_epsg(int(crs_raw.split(":", 1)[-1]))
                except Exception:
                    pass
            nodata = sidecar.get("nodata_value")
            if nodata is not None:
                merged["nodata"] = float(nodata)

    # Binary mask
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=mask.shape[0],
        width=mask.shape[1],
        count=1,
        dtype="float32",
        crs=merged.get("crs", "EPSG:4326"),
        transform=merged.get("transform"),
        compress="lzw",
        nodata=merged.get("nodata"),
    ) as dst:
        dst.write(mask.astype(np.float32), 1)

    # Probability map (if available). Only the primary mask passes a prob_name:
    # the reference (before-water) raster used to overwrite this file.
    if prob_map is not None and prob_name:
        prob_path = output_path.parent / prob_name
        with rasterio.open(
            prob_path,
            "w",
            driver="GTiff",
            height=prob_map.shape[0],
            width=prob_map.shape[1],
            count=1,
            dtype="float32",
            crs=merged.get("crs", "EPSG:4326"),
            transform=merged.get("transform"),
            compress="lzw",
            nodata=merged.get("nodata"),
        ) as dst:
            dst.write(prob_map.astype(np.float32), 1)
        print(f"  Probability map: {prob_path}")

    print(f"  Flood mask saved: {output_path}")
    print(f"  Mask size: {mask.shape[1]}x{mask.shape[0]} pixels")
    print(f"  Flooded pixels: {int(mask.sum())}")
    print(f"  Flood percentage: {mask.sum() / mask.size * 100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Run flood segmentation inference")
    parser.add_argument("--before-dir", type=Path, required=True, help="Before-event image directory")
    parser.add_argument("--after-dir", type=Path, required=True, help="After-event image directory")
    parser.add_argument("--model", type=Path, default=Path("models/flood_unet.pt"), help="Model path")
    parser.add_argument("--output", type=Path, default=Path("results/flood_mask.tif"), help="Output flood mask")
    parser.add_argument("--img-size", type=int, default=256, help="Tile size for inference")
    parser.add_argument("--threshold", type=float, default=0.5, help="Flood probability threshold")
    parser.add_argument("--device", default="auto", help="Device (cuda/cpu/auto)")
    parser.add_argument("--sentinel1-bands", nargs="+", default=["VV", "VH"], help="Sentinel-1 bands")
    parser.add_argument(
        "--baseline",
        action="store_true",
        help=(
            "Skip the U-Net entirely and use the log-ratio change-detection "
            "baseline (after-minus-before VV drop above --baseline-threshold). "
            "Used automatically when no checkpoint exists."
        ),
    )
    parser.add_argument(
        "--baseline-threshold", type=float, default=BASELINE_THRESHOLD_DB,
        help=f"dB drop that counts as new water in baseline mode (default {BASELINE_THRESHOLD_DB})",
    )
    parser.add_argument(
        "--trust-unverified-model",
        action="store_true",
        help=(
            "Load a checkpoint that has no .sha256 sidecar (only for weights you "
            "produced yourself; train_segmentation.py writes the sidecar)"
        ),
    )
    parser.add_argument(
        "--min-component-pixels", type=int, default=MIN_COMPONENT_PIXELS,
        help=(
            f"Drop flood blobs smaller than this many pixels as speckle "
            f"(default {MIN_COMPONENT_PIXELS})"
        ),
    )
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # ── Which engine produces the mask ───────────────────────────────────────
    # The trained U-Net when its checkpoint exists; otherwise the log-ratio
    # threshold baseline, which still yields a mask from the real imagery. No
    # branch here fabricates a mask or silently substitutes a placeholder model.
    model = None
    method = "unet"
    if args.baseline:
        method = "log_ratio_baseline"
        print(
            f"\n▶ Baseline mode: log-ratio change detection "
            f"(>{args.baseline_threshold} dB VV drop), no U-Net."
        )
    elif args.model is not None and Path(args.model).is_file():
        model = load_model(
            args.model, device, allow_unverified=args.trust_unverified_model
        )
    else:
        method = "log_ratio_baseline"
        print(
            f"\n⚠ No trained checkpoint at {args.model} — falling back to the "
            "log-ratio change-detection baseline.\n"
            "  A real mask is still produced from the imagery, but it is a band-drop "
            "threshold,\n  not a segmentation model. Train one with "
            "scripts/train_segmentation.py --dataset <real dataset>\n  (see the "
            "script's --help; synthetic tiles are opt-in via --smoke-test)."
        )

    # Load before/after imagery
    print(f"\n{'='*70}")
    print("Loading and preprocessing imagery")
    print(f"{'='*70}")

    # The download layout IS before-dir/before (set by download_sentinel.py), but
    # callers may also pass either a plain directory containing the VV/VH tifs
    # directly, or the download root. Handle both without guessing.
    before_dir = args.before_dir
    if not (before_dir / "s1_before_VV.tif").exists() and (before_dir / "before").is_dir():
        before_dir = before_dir / "before"
    after_dir = args.after_dir
    if not (after_dir / "s1_after_VV.tif").exists() and (after_dir / "after").is_dir():
        after_dir = after_dir / "after"
    # return_valid=True also yields the (H, W) observed-pixel mask: nodata collar
    # and radar shadow excluded, so the delivered valid_mask.tif is honest about
    # where the imagery can be trusted.
    loaded_before = preprocess_sentinel1(before_dir, return_valid=True)
    loaded_after = preprocess_sentinel1(after_dir, return_valid=True)
    before_img, before_valid = loaded_before if isinstance(loaded_before, tuple) else (loaded_before, None)
    after_img, after_valid = loaded_after if isinstance(loaded_after, tuple) else (loaded_after, None)

    # Fallback to Sentinel-2 if Sentinel-1 not available. Optical imagery has no
    # SAR shadow/nodata geometry, so there is no observed-pixel mask to carry.
    if before_img is None:
        print("  ⚠ No Sentinel-1 data — trying Sentinel-2...")
        before_img = preprocess_sentinel2(args.before_dir / "before")
        after_img = preprocess_sentinel2(args.after_dir / "after")
        before_valid = None
        after_valid = None

    if before_img is None and after_img is None:
        print("❌ No imagery found in either before or after directories")
        sys.exit(1)

    print(f"  Before image shape: {before_img.shape if before_img is not None else 'None'}")
    print(f"  After image shape:  {after_img.shape if after_img is not None else 'None'}")

    # Ensure same number of channels
    if before_img is not None and after_img is not None:
        n_bands = min(before_img.shape[0], after_img.shape[0])
        before_img = before_img[:n_bands]
        after_img = after_img[:n_bands]

    # Match the model's expected channel count. preprocess_sentinel1 returns
    # [VV, VH, ratio] (3 ch) but the checkpoint may be trained on fewer. Crop
    # from the front: VV+VH kept, ratio dropped as it is derivable.
    chk_in = getattr(model, "in_channels", None) if model is not None else None
    if model is not None and chk_in is None:
        for m_ in model.modules():
            if isinstance(m_, torch.nn.Conv2d):
                chk_in = m_.in_channels
                break
    if before_img is not None and chk_in and before_img.shape[0] != chk_in:
        print(f"  Adjusting channels {before_img.shape[0]} → {chk_in} (model was trained with {chk_in})")
        before_img = before_img[:chk_in]
        after_img = after_img[:chk_in]

    # Load geospatial profile from after image (for georeferencing output).
    # Prefer the CRS / pixel size / nodata from the sentinel-1 sidecar when it exists
    # so the mask inherits the same projected grid the preprocess step wrote, instead of
    # inheriting a possibly-stale or synthetic TIFF profile.
    profile = None
    after_tif_dir = after_dir if (after_dir / "s1_after_VV.tif").exists() else args.after_dir / "after"
    sidecar = _load_s1_sidecar(after_tif_dir)
    if sidecar is not None:
        crs_raw = sidecar.get("crs")
        if isinstance(crs_raw, str) and crs_raw.startswith("EPSG:"):
            try:
                from rasterio.crs import CRS
                profile = {
                    "crs": CRS.from_epsg(int(crs_raw.split(":", 1)[-1])),
                    "nodata": sidecar.get("nodata_value", -999.0),
                }
            except Exception as exc:  # pylint: disable=broad-except
                print(f"  ⚠️ Could not parse sidecar CRS {crs_raw!r}: {exc}")
    for tif in after_tif_dir.glob("*.tif"):
        try:
            with rasterio.open(tif) as src:
                profile = src.profile if profile is None else dict(profile, **src.profile)
            break
        except Exception:
            pass

    # Run inference
    print(f"\n{'='*70}")
    print("Running flood segmentation inference")
    print(f"{'='*70}")

    if after_img is None:
        print("❌ No after-event imagery — a flood mask needs the post-event scene.")
        sys.exit(1)

    if method == "unet":
        def _model_mask(img):
            if img is None:
                return None, None
            return tile_inference(
                model, img, args.img_size, device=device, threshold=args.threshold
            )

        before_mask, before_prob = _model_mask(before_img)
        after_mask, after_prob = _model_mask(after_img)
        if before_mask is not None:
            print(f"  Before water mask (U-Net): {before_mask.sum():.0f} px")
        print(f"  After water mask (U-Net):  {after_mask.sum():.0f} px")

        # Delivered mask = new water only. The permanent river in the after scene
        # is not a flood.
        raw_change = derive_change_mask(before_mask, after_mask)
    else:
        # Baseline: an absolute backscatter threshold defines water, and the
        # log-ratio drop between the two acquisitions defines what BECAME water.
        def _water_threshold(img):
            if img is None:
                return None, None
            water = np.asarray(img[0], dtype=np.float32) < WATER_BACKSCATTER_DB
            return np.nan_to_num(water, nan=0.0).astype(np.float32), None

        before_mask, before_prob = _water_threshold(before_img)
        after_mask, after_prob = _water_threshold(after_img)
        if before_mask is not None:
            print(f"  Before water mask (threshold): {before_mask.sum():.0f} px")
        print(f"  After water mask (threshold):  {after_mask.sum():.0f} px")

        raw_change, _ = log_ratio_baseline(
            before_img, after_img, threshold_db=args.baseline_threshold
        )

    # Speckle control on the delivered mask. A flood is a connected patch; a
    # couple of dark pixels is sensor noise, and each one becomes a "flooded"
    # building or a cut road downstream.
    change_mask, removed_blobs = remove_small_components(
        raw_change, args.min_component_pixels
    )
    if removed_blobs:
        print(f"  Speckle filter: removed {removed_blobs} blobs smaller than "
              f"{args.min_component_pixels} px")
    change_prob = None
    if after_prob is not None:
        change_prob = np.where(change_mask > 0.5, after_prob, 0.0).astype(np.float32)

    results_dir = args.output.parent
    results_dir.mkdir(parents=True, exist_ok=True)

    # Compute change detection. The delivered_mask argument makes the JSON
    # describe the raster that was actually written (post-speckle), so the
    # numbers here and flood_mask.tif cannot disagree.
    change = compute_change_detection(
        before_mask if before_mask is not None else np.zeros_like(after_mask),
        after_mask, before_prob, after_prob,
        transform=(profile or {}).get("transform"),
        crs=(profile or {}).get("crs"),
        delivered_mask=change_mask,
        valid_mask=after_valid,
        min_component_pixels=args.min_component_pixels,
        method=method,
    )
    print(f"\n{'─'*70}")
    print("Flood Change Detection Results")
    print(f"{'─'*70}")
    for k, v in change.items():
        print(f"  {k}: {v}")
    (results_dir / "change_detection.json").write_text(
        json.dumps(change, indent=2),
        encoding="utf-8",
    )
    print(f"  Change detection: {results_dir / 'change_detection.json'}")

    # ── The four rasters ─────────────────────────────────────────────────────
    # 1. flood_mask.tif — THE delivered product: newly flooded only (after AND
    #    NOT before), speckle removed. damage_assessment.py and
    #    cutoff_analysis.py read this file and need no change.
    save_flood_mask(change_mask, change_prob, profile, args.output, after_dir)
    print(f"  flood_mask.tif = newly flooded only "
          f"({int(change_mask.sum()):,} px; permanent water excluded)")

    # 2. water_after.tif — all water in the after scene, permanent water included.
    #    Kept so the two can be compared (and so "flooded" can never again be
    #    confused with "has a river").
    if after_mask is not None:
        save_flood_mask(
            after_mask, None, profile, results_dir / WATER_AFTER_NAME, after_dir,
            prob_name=None,
        )

    # 3. flood_mask_before.tif — the pre-event water extent: the permanent-water
    #    mask the change detection subtracts.
    if before_mask is not None:
        before_profile = _load_s1_sidecar(before_dir)
        if before_profile and before_profile.get("crs"):
            tmp = dict(profile)
            crs_raw = before_profile.get("crs")
            if isinstance(crs_raw, str) and crs_raw.startswith("EPSG:"):
                try:
                    from rasterio.crs import CRS
                    tmp["crs"] = CRS.from_epsg(int(crs_raw.split(":", 1)[-1]))
                except Exception:
                    pass
            profile = tmp
        save_flood_mask(
            before_mask, None, profile, results_dir / PERMANENT_WATER_NAME, before_dir,
            prob_name=None,
        )
        print(f"  {PERMANENT_WATER_NAME} = pre-event (permanent) water: "
              f"{int(before_mask.sum()):,} px")

    # 4. valid_mask.tif — where the imagery was actually observed (nodata and
    #    radar shadow removed), for consumers that must not guess on blind pixels.
    if after_valid is not None:
        save_valid_mask(after_valid, profile, results_dir / VALID_MASK_NAME, after_dir)

    print(f"\n✅ Inference complete!  (method: {method})")
    print(f"   Flood mask:          {args.output}")
    print(f"   Water after:         {results_dir / WATER_AFTER_NAME}")
    print(f"   Permanent water:     {results_dir / PERMANENT_WATER_NAME}")
    print(f"   Valid data mask:     {results_dir / VALID_MASK_NAME}")


if __name__ == "__main__":
    main()
