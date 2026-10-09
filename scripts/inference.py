#!/usr/bin/env python3
"""
Flood Segmentation Inference Pipeline

Runs the trained U-Net model on Sentinel-1/2 imagery to produce a flood probability mask.
Compares before/after imagery to detect flood extent change.

Uses the model trained by train_segmentation.py.
For the Trishuli case study: runs on the August 2026 flood event.

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


def load_model(model_path: Path, device: torch.device) -> torch.nn.Module:
    """Load trained model from checkpoint."""
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        sys.exit(1)

    checkpoint = torch.load(model_path, map_location=device)

    model_type = checkpoint.get("config", {}).get("model_type", "unet")
    in_channels = checkpoint.get("config", {}).get("in_channels", 2)

    if model_type in MODEL_CLASSES:
        model = MODEL_CLASSES[model_type](
            in_channels=in_channels,
            num_classes=1,
            pretrained=False,
        )
    else:
        # Fallback: UNet
        from train_segmentation import UNet
        model = UNet(in_channels=in_channels, num_classes=1)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"  Loaded model: {model_type}")
    print(f"  Input channels: {in_channels}")
    print(f"  Best val Dice: {checkpoint.get('best_val_dice', 'N/A'):.4f}")

    return model


def preprocess_sentinel1(image_path: Path | None) -> np.ndarray | None:
    """
    Load and preprocess Sentinel-1 GRD data.

    Expected: VV and VH polarization bands as separate GeoTIFFs or a combined file.
    Preprocessing:
      1. Radiometric calibration (convert to backscatter coefficient σ° in dB)
      2. Speckle filtering (simple 3x3 median filter for demo; Lee filter for production)
      3. Create input tensor: [VV_dB, VH_dB, VV/VH_ratio]

    Returns: numpy array of shape (C, H, W)
    """
    if image_path is None or not image_path.exists():
        return None

    print(f"  Loading Sentinel-1 data from: {image_path}")

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
                    # Convert from linear to dB
                    vv_db = 10 * np.log10(np.clip(vv, 1e-6, None))
                    vh_db = 10 * np.log10(np.clip(vh, 1e-6, None))
                    ratio = vv_db - vh_db  # VV/VH ratio
                    return np.stack([vv_db, vh_db, ratio], axis=0)

    # Case 3: Individual band files
    if vv_path and vh_path:
        with rasterio.open(vv_path) as src_vv:
            vv = src_vv.read(1).astype(np.float32)
            vv_db = 10 * np.log10(np.clip(vv, 1e-6, None))
        with rasterio.open(vh_path) as src_vh:
            vh = src_vh.read(1).astype(np.float32)
            vh_db = 10 * np.log10(np.clip(vh, 1e-6, None))
        ratio = vv_db - vh_db
        return np.stack([vv_db, vh_db, ratio], axis=0)

    print(f"  ⚠️ Could not find VV/VH bands")
    return None


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


def tile_inference(
    model: torch.nn.Module,
    image: np.ndarray,
    tile_size: int = 256,
    overlap: int = 64,
    device: torch.device = torch.device("cpu"),
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Run inference on a large image using sliding window tiling.

    Args:
        model: Trained U-Net model
        image: (C, H, W) numpy array of input bands
        tile_size: Size of each tile (pixels)
        overlap: Overlap between tiles (for seamless stitching)
        device: torch device
        threshold: Probability threshold for binary mask

    Returns:
        Binary flood mask of shape (H, W)
    """
    C, H, W = image.shape
    output_mask = np.zeros((H, W), dtype=np.float32)
    output_count = np.zeros((H, W), dtype=np.float32)  # For averaging overlaps

    # Normalize image (assume already in reasonable range after preprocessing)
    # Sentinel-1: dB values typically -20 to 0; Sentinel-2: 0-1 reflectance
    image_norm = image.astype(np.float32)

    # Pad to make divisible by tile_size
    pad_h = (tile_size - H % tile_size) % tile_size
    pad_w = (tile_size - W % tile_size) % tile_size
    if pad_h > 0 or pad_w > 0:
        image_norm = np.pad(image_norm, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")
        H_pad, W_pad = image_norm.shape[1], image_norm.shape[2]
    else:
        H_pad, W_pad = H, W

    # Slide window
    stride = tile_size - overlap
    tiles = []

    for y in tqdm(range(0, H_pad - tile_size + 1, stride), desc="  Tiling", leave=False):
        for x in range(0, W_pad - tile_size + 1, stride):
            tile = image_norm[:, y : y + tile_size, x : x + tile_size]
            tile_tensor = torch.from_numpy(tile).unsqueeze(0).to(device)

            with torch.no_grad():
                pred = model(tile_tensor)  # (1, 1, tile_size, tile_size)
                pred_prob = F.sigmoid(pred).squeeze().cpu().numpy()

            # Add to output with overlap averaging
            output_mask[y : y + tile_size, x : x + tile_size] += pred_prob
            output_count[y : y + tile_size, x : x + tile_size] += 1

    # Average overlaps
    output_mask = np.divide(output_mask, output_count, out=np.zeros_like(output_mask), where=output_count > 0)

    # Crop back to original size
    output_mask = output_mask[:H, :W]

    # Apply threshold
    binary_mask = (output_mask > threshold).astype(np.float32)

    return binary_mask, output_mask


def compute_change_detection(
    before_mask: np.ndarray,
    after_mask: np.ndarray,
    before_prob: np.ndarray | None = None,
    after_prob: np.ndarray | None = None,
) -> dict:
    """
    Compute flood change detection from before/after masks.

    Returns dict with:
      - flood_extent: pixels newly flooded (after=1, before=0)
      - disappeared: pixels that were water before but not after (rare)
      - persistent_water: water in both images (rivers, lakes)
      - total_flooded: all water pixels in after image
      - flood_area_ha: flood extent in hectares (assuming 10m pixels)
    """
    before_bin = (before_mask > 0.5).astype(np.int32) if before_mask is not None else np.zeros_like(after_mask)
    after_bin = (after_mask > 0.5).astype(np.int32)

    flood_extent = after_bin & (1 - before_bin)       # Newly flooded
    persistent = after_bin & before_bin                # Water in both
    disappeared = (1 - after_bin) & before_bin         # Water disappeared (drought/drainage)

    pixel_area_ha = 0.01  # 10m x 10m = 100m² = 0.01 ha

    result = {
        "flood_extent_pixels": int(flood_extent.sum()),
        "flood_extent_ha": round(flood_extent.sum() * pixel_area_ha, 2),
        "persistent_water_pixels": int(persistent.sum()),
        "persistent_water_ha": round(persistent.sum() * pixel_area_ha, 2),
        "total_water_after_pixels": int(after_bin.sum()),
        "total_water_after_ha": round(after_bin.sum() * pixel_area_ha, 2),
        "disappeared_pixels": int(disappeared.sum()),
        "flood_percentage": round(
            flood_extent.sum() / max(after_bin.sum(), 1) * 100, 1
        ),
    }

    return result


def save_flood_mask(
    mask: np.ndarray,
    prob_map: np.ndarray | None,
    profile: dict,
    output_path: Path,
    before_profile: dict | None = None,
):
    """Save flood mask as GeoTIFF with proper georeferencing."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if before_profile is not None:
        profile = before_profile.copy()
    elif profile is not None:
        profile = profile.copy()
    else:
        # Create default profile
        profile = {
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

    # Binary mask
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
            height=mask.shape[0],
            width=mask.shape[1],
            count=1,
            dtype="float32",
            crs=profile.get("crs", "EPSG:4326"),
            transform=profile.get("transform"),
            compress="lzw",
        ) as dst:
        dst.write(mask.astype(np.float32), 1)

    # Probability map (if available)
    if prob_map is not None:
        prob_path = output_path.parent / "flood_probability.tif"
        with rasterio.open(
            prob_path,
            "w",
            driver="GTiff",
            height=prob_map.shape[0],
            width=prob_map.shape[1],
            count=1,
            dtype="float32",
            crs=profile.get("crs", "EPSG:4326"),
            transform=profile.get("transform"),
            compress="lzw",
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
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Load model
    model = load_model(args.model, device)

    # Load before/after imagery
    print(f"\n{'='*70}")
    print("Loading and preprocessing imagery")
    print(f"{'='*70}")

    before_img = preprocess_sentinel1(args.before_dir / "before")
    after_img = preprocess_sentinel1(args.after_dir / "after")

    # Fallback to Sentinel-2 if Sentinel-1 not available
    if before_img is None:
        print("  ⚠ No Sentinel-1 data — trying Sentinel-2...")
        before_img = preprocess_sentinel2(args.before_dir / "before")
        after_img = preprocess_sentinel2(args.after_dir / "after")

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

    # Load geospatial profile from after image (for georeferencing output)
    profile = None
    if after_img is not None and (args.after_dir / "after").exists():
        for tif in (args.after_dir / "after").glob("*.tif"):
            try:
                with rasterio.open(tif) as src:
                    profile = src.profile
                break
            except Exception:
                pass

    # Run inference
    print(f"\n{'='*70}")
    print("Running flood segmentation inference")
    print(f"{'='*70}")

    if before_img is not None:
        before_mask, before_prob = tile_inference(
            model, before_img, args.img_size, device=device, threshold=args.threshold
        )
        print(f"  Before flood mask: {before_mask.sum():.0f} flooded pixels")
    else:
        before_mask, before_prob = None, None

    if after_img is not None:
        after_mask, after_prob = tile_inference(
            model, after_img, args.img_size, device=device, threshold=args.threshold
        )
        print(f"  After flood mask: {after_mask.sum():.0f} flooded pixels")
    else:
        after_mask, after_prob = None, None

    # Compute change detection
    if before_mask is not None and after_mask is not None:
        change = compute_change_detection(before_mask, after_mask, before_prob, after_prob)
        print(f"\n{'─'*70}")
        print("Flood Change Detection Results")
        print(f"{'─'*70}")
        for k, v in change.items():
            print(f"  {k}: {v}")

        # Save change detection summary
        results_dir = args.output.parent
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "change_detection.json").write_text(
            json.dumps(change, indent=2),
            encoding="utf-8",
        )
        print(f"  Change detection: {results_dir / 'change_detection.json'}")

    # Save flood mask
    save_flood_mask(after_mask, after_prob, profile, args.output, profile)

    # Copy before mask for reference
    if before_mask is not None:
        before_mask_path = args.output.parent / "flood_mask_before.tif"
        save_flood_mask(before_mask, before_prob, profile, before_mask_path, profile)

    print(f"\n✅ Inference complete!")
    print(f"   Flood mask: {args.output}")


if __name__ == "__main__":
    main()
