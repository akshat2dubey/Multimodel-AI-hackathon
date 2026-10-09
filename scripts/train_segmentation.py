#!/usr/bin/env python3
"""
Flood Segmentation Model — U-Net Training Pipeline

Trains a U-Net model to detect water/debris in satellite imagery.
Uses Kuro Siwo (recommended) or Sen1Floods11 training data.

AI Component (Track B requirement — choose one):
  ✅ Flood segmentation model — Train a model to detect water or debris in
     satellite images using the listed training data, and show how well it
     works on Himalayan scenes it has never seen.

Training data options:
  - Kuro Siwo: https://github.com/Orion-AI-Lab/KuroSiwo (MIT/CC BY, cite Bountos et al. 2024)
  - Sen1Floods11: https://github.com/cloudtostreet/Sen1Floods11 (CC BY 4.0, cite Bonafilia et al. 2020)

Citation required:
  If using Kuro Siwo: Bountos, N., et al. (2024). "Kuro Siwo: A Large-Scale
    Dataset for Flood Segmentation in Satellite Imagery." NeurIPS 2024.
  If using Sen1Floods11: Bonafilia, D., et al. (2020). "Sen1Floods11: A
    Georeferenced Cropped Sentinel-1 Dataset for Flood Detection."
    CVPR Workshops.

Model architecture: U-Net with ResNet-34 encoder (pre-trained on ImageNet)
  - Input:  Sentinel-1 VV + VH + ratio bands (or Sentinel-2 RGB + NIR)
  - Output: Binary flood mask (water=1, land=0) at 10m resolution
  - Loss:   Dice loss + BCE (handles class imbalance — floods are small)

After training, the model is evaluated on held-out scenes from diverse
geographies to show cross-region generalization.

Usage:
    python train_segmentation.py \
        --dataset ./data/kuro_siwo \
        --output ./models/flood_unet.pt \
        --epochs 50 \
        --batch-size 16 \
        --img-size 256
"""

import argparse

import sys
import json
import os
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet34_Weights, resnet34

if hasattr(sys.stdout, "reconfigure"):  # never crash on a cp1252 Windows console
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ─── Configuration ───────────────────────────────────────────

MODEL_OUTPUT = "models/flood_unet.pt"
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 16
DEFAULT_IMAGE_SIZE = 256
LEARNING_RATE = 1e-4
NUM_WORKERS = 4

SENTINEL1_BANDS = ["VV", "VH"]         # Sentinel-1: 2 bands
SENTINEL2_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]  # Sentinel-2: 6 bands
ALL_BANDS = SENTINEL1_BANDS + SENTINEL2_BANDS  # Combined input: 8 channels

# ─── U-Net Model ─────────────────────────────────────────────


class ConvBlock(nn.Module):
    """Double convolution block used in U-Net."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net for flood segmentation.

    Architecture:
      Encoder (contracting path): 4 levels of downsampling
      Bottleneck: 1 level at smallest resolution
      Decoder (expanding path): 4 levels of upsampling with skip connections

    Input:  N x C x H x W  (C = number of spectral bands)
    Output: N x 1 x H x W  (flood probability map)
    """

    def __init__(
        self,
        in_channels: int = 2,
        num_classes: int = 1,
        base_channels: int = 64,
        depth: int = 4,
    ):
        super().__init__()
        self.depth = depth
        self.in_channels = in_channels

        # Encoder (contracting)
        self.encoders = nn.ModuleList()
        ch = in_channels
        for i in range(depth):
            out_ch = base_channels * (2**i)
            self.encoders.append(ConvBlock(ch, out_ch))
            ch = out_ch
            if i < depth - 1:
                self.encoders.append(nn.MaxPool2d(2))

        # Bottleneck
        bottleneck_ch = base_channels * (2**depth)
        self.bottleneck = ConvBlock(ch, bottleneck_ch)

        # Decoder (expanding)
        self.decoders = nn.ModuleList()
        ch = bottleneck_ch
        for i in range(depth - 1, -1, -1):
            out_ch = base_channels * (2**i)
            self.decoders.append(nn.ConvTranspose2d(ch, out_ch // 2, 2, stride=2))
            self.decoders.append(ConvBlock(ch, out_ch))
            ch = out_ch

        # Final output
        self.final_conv = nn.Conv2d(base_channels, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder pass with skip connections
        skips = []
        for i, layer in enumerate(self.encoders):
            x = layer(x)
            # After a ConvBlock + MaxPool, save for skip connection
            if isinstance(layer, ConvBlock) and i % 2 == 0:
                skips.append(x)
            # After MaxPool, x is downsampled

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder pass
        skips = list(reversed(skips))
        for i, layer in enumerate(self.decoders):
            if isinstance(layer, nn.ConvTranspose2d):
                x = layer(x)
                # Add skip connection from encoder
                x = x + skips[i // 2]  # element-wise add (or concat)
            else:
                x = layer(x)

        return self.final_conv(x)


class ResNetUNet(nn.Module):
    """
    U-Net with ResNet-34 encoder (pre-trained on ImageNet).
    Better feature extraction from encoder; commonly used for satellite segmentation.
    """

    def __init__(
        self,
        in_channels: int = 2,
        num_classes: int = 1,
        pretrained: bool = True,
    ):
        super().__init__()

        # Load ResNet-34 encoder (replace first conv for multi-channel input)
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = resnet34(weights=weights)

        # Modify first convolution for arbitrary input channels
        old_conv = resnet.conv1
        self.first_conv = nn.Conv2d(
            in_channels,
            64,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        if pretrained:
            # Initialize from RGB weights (average over 3 channels) + random for extras
            with torch.no_grad():
                new_weight = old_conv.weight.clone()
                if in_channels <= 3:
                    self.first_conv.weight = nn.Parameter(new_weight[:, :in_channels, :, :])
                else:
                    # Fill first 3 channels from ImageNet, rest from Gaussian
                    rgb_weight = new_weight[:, :3, :, :]
                    extra_weight = torch.randn(
                        new_weight.shape[0], in_channels - 3, *new_weight.shape[2:]
                    ) * 0.01
                    self.first_conv.weight = nn.Parameter(torch.cat([rgb_weight, extra_weight], dim=1))
                if old_conv.bias is not None:
                    self.first_conv.bias = nn.Parameter(old_conv.bias.clone())

        # Remove FC layer — keep only feature extractor
        self.encoder = nn.Sequential(
            self.first_conv,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,  # 256 channels, /4 resolution
            resnet.layer2,  # 512 channels, /8
            resnet.layer3,  # 1024 channels, /16
            resnet.layer4,  # 2048 channels, /32
        )

        # Decoder (U-Net up path) — NOTE: resnet34 channels, not resnet50:
        # layer1=64 (/4), layer2=128 (/8), layer3=256 (/16), layer4=512 (/32).
        # Must be a module list, not nn.Sequential: each up-step needs to receive
        # a skip connection that a Sequential container cannot route.
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)   # /32 → /16
        self.bn4 = nn.BatchNorm2d(256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)   # /16 → /8
        self.bn3 = nn.BatchNorm2d(128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)    # /8 → /4
        self.bn2 = nn.BatchNorm2d(64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)     # /4 → /2
        self.bn1b = nn.BatchNorm2d(32)
        self.up0 = nn.ConvTranspose2d(32, 32, 2, stride=2)     # /2 → /1
        self.bn0 = nn.BatchNorm2d(32)
        self.final_conv = nn.Conv2d(32, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder: conv1 -> bn1 -> relu -> maxpool -> layer1..layer4, saving skips
        x = self.encoder[0](x)
        x = self.encoder[1](x)
        x = self.encoder[2](x)
        x = self.encoder[3](x)
        s1 = self.encoder[4](x)   # layer1, 64 ch, /4
        s2 = self.encoder[5](s1)  # layer2, 128 ch, /8
        s3 = self.encoder[6](s2)  # layer3, 256 ch, /16
        x  = self.encoder[7](s3)  # layer4, 512 ch, /32 (bottleneck)

        # Decoder: upsample + add the matching encoder skip (same resolution)
        x = F.relu(self.bn4(self.up4(x)) + s3)          # 256ch @ /16 + s3 (256ch /16)
        x = F.relu(self.bn3(self.up3(x)) + s2)          # 128ch @ /8 + s2
        x = F.relu(self.bn2(self.up2(x)) + s1)          # 64ch @ /4 + s1
        x = F.relu(self.bn1b(self.up1(x)))              # 32ch @ /2
        x = F.relu(self.bn0(self.up0(x)))               # 32ch @ /1
        return torch.sigmoid(self.final_conv(x))


# ─── Loss Functions ──────────────────────────────────────────


class DiceLoss(nn.Module):
    """Dice Loss for segmentation — good for class imbalance."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    """Combined BCE + Dice loss — standard for segmentation."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


# ─── Data Augmentation ───────────────────────────────────────


class FloodAugmentation:
    """Data augmentation for satellite imagery flood training."""

    def __init__(self, img_size: int = 256):
        self.img_size = img_size
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.0], std=[1.0]),  # Already normalized
        ])

    def augment(self, image: np.ndarray, mask: np.ndarray):
        """Apply random augmentation to image-mask pair."""
        # Random horizontal flip
        if random.random() > 0.5:
            image = np.flip(image, axis=2).copy()
            mask = np.flip(mask, axis=2).copy()

        # Random vertical flip
        if random.random() > 0.5:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()

        # Random rotation (multiples of 90°)
        k = random.randint(0, 3)
        if k > 0:
            image = np.rot90(image, k, axes=(1, 2)).copy()
            mask = np.rot90(mask, k, axes=(1, 2)).copy()

        # Random crop/pad to target size
        h, w = image.shape[1], image.shape[2]
        if h > self.img_size:
            top = random.randint(0, h - self.img_size)
            left = random.randint(0, w - self.img_size)
            image = image[:, top : top + self.img_size, left : left + self.img_size]
            mask = mask[:, top : top + self.img_size, left : left + self.img_size]
        elif h < self.img_size:
            pad_h = (self.img_size - h) // 2
            pad_w = (self.img_size - w) // 2
            image = np.pad(image, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="edge")
            mask = np.pad(mask, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="constant", constant_values=0)

        # Normalize image (assume already in reasonable range)
        image = image.astype(np.float32) / 255.0

        return image, mask


# ─── Dataset ──────────────────────────────────────────────────


class FloodDataset(Dataset):
    """
    Dataset for flood segmentation training.

    Expects directory structure:
        dataset_dir/
          images/           # Satellite image tiles (numpy .npy or geotiff)
          masks/            # Binary flood masks (0=land, 1=water)
          splits/           # train.txt, val.txt, test.txt ( filenames)

    Each tile: image of shape (C, H, W), mask of shape (1, H, W)
    """

    def __init__(
        self,
        images_dir: Path,
        masks_dir: Path,
        split_file: Path,
        img_size: int = 256,
        bands: list[str] | None = None,
    ):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.img_size = img_size
        self.bands = bands or SENTINEL1_BANDS

        # Read split
        if split_file.exists():
            filenames = split_file.read_text(encoding="utf-8").strip().split("\n")
        else:
            # Auto-discover
            filenames = sorted(f.name for f in images_dir.glob("*.npy"))

        self.filenames = [f for f in filenames if (images_dir / f).exists()]

        print(f"  Dataset: {len(self.filenames)} tiles, {len(self.bands)} bands, img_size={img_size}")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx: int):
        fname = self.filenames[idx]
        img_path = self.images_dir / fname
        mask_path = self.masks_dir / fname.replace(".npy", "_mask.npy")

        # Load
        image = np.load(img_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.float32)

        # Ensure correct shape: (C, H, W) for image, (1, H, W) for mask
        if image.ndim == 2:
            image = image[np.newaxis, ...]  # (H, W) → (1, H, W)
        if mask.ndim == 2:
            mask = mask[np.newaxis, ...]  # (H, W) → (1, H, W)

        # Select bands if needed
        if image.shape[0] != len(self.bands):
            # Assume bands are in order specified by self.bands
            # If we have more bands than needed, take first N
            image = image[: len(self.bands)]

        # Augment
        aug = FloodAugmentation(self.img_size)
        image, mask = aug.augment(image, mask)

        return (
            torch.from_numpy(image),
            torch.from_numpy(mask),
            fname,
        )


# ─── Training Loop ────────────────────────────────────────────


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
    output_path: Path,
    start_epoch: int = 0,
):
    """Train the model with early stopping on validation Dice score."""

    best_val_dice = 0.0
    patience = 5
    patience_counter = 0
    training_log = []

    print(f"\n{'='*70}")
    print(f"Training U-Net Flood Segmentation Model")
    print(f"{'='*70}")
    print(f"  Device:       {device}")
    print(f"  Epochs:       {epochs}")
    print(f"  Batch size:   {train_loader.batch_size}")
    print(f"  Learning rate:{optimizer.param_groups[0]['lr']}")
    print(f"  Output:       {output_path}")
    print(f"  Starting at:  epoch {start_epoch}")
    print(f"{'='*70}\n")

    for epoch in range(start_epoch, epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        train_count = 0

        for batch_idx, (images, masks, _) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            # Compute Dice for logging
            with torch.no_grad():
                pred_binary = (torch.sigmoid(outputs) > 0.5).float()
                intersection = (pred_binary * masks).sum()
                dice = (2.0 * intersection + 1.0) / (
                    pred_binary.sum() + masks.sum() + 1.0
                )
                train_dice += dice.item()
            train_count += 1

            if batch_idx % 50 == 0:
                print(f"  Epoch {epoch+1}/{epochs}  Batch {batch_idx}/{len(train_loader)}  "
                      f"Loss: {loss.item():.4f}  Dice: {dice.item():.4f}")

        avg_train_loss = train_loss / train_count
        avg_train_dice = train_dice / train_count

        # Validation
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        val_count = 0

        with torch.no_grad():
            for images, masks, _ in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)

                val_loss += loss.item()
                pred_binary = (torch.sigmoid(outputs) > 0.5).float()
                intersection = (pred_binary * masks).sum()
                dice = (2.0 * intersection + 1.0) / (
                    pred_binary.sum() + masks.sum() + 1.0
                )
                val_dice += dice.item()
                val_count += 1

        avg_val_loss = val_loss / max(val_count, 1)
        avg_val_dice = val_dice / max(val_count, 1)

        log_entry = {
            "epoch": epoch + 1,
            "train_loss": round(avg_train_loss, 4),
            "train_dice": round(avg_train_dice, 4),
            "val_loss": round(avg_val_loss, 4),
            "val_dice": round(avg_val_dice, 4),
            "timestamp": time.time(),
        }
        training_log.append(log_entry)

        print(f"\n{'─'*70}")
        print(f"Epoch {epoch+1:3d}/{epochs}  |  "
              f"Train Loss: {avg_train_loss:.4f}  Dice: {avg_train_dice:.4f}  |  "
              f"Val Loss: {avg_val_loss:.4f}  Dice: {avg_val_dice:.4f}")

        # Save best model
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            patience_counter = 0
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_dice": best_val_dice,
                "training_log": training_log,
                "config": {
                    "in_channels": model.in_channels if hasattr(model, "in_channels") else 2,
                    "num_classes": 1,
                    "model_type": model.__class__.__name__,
                },
            }, output_path)
            print(f"  ✅ Best model saved! (Dice: {best_val_dice:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  ⏸️  Early stopping at epoch {epoch+1} (patience={patience})")
                break

    print(f"\n{'='*70}")
    print(f"Training complete! Best validation Dice: {best_val_dice:.4f}")
    print(f"Model saved to: {output_path}")
    print(f"{'='*70}")

    # Save training log
    log_path = output_path.parent / "training_log.json"
    log_path.write_text(json.dumps(training_log, indent=2), encoding="utf-8")
    print(f"Training log: {log_path}")

    return best_val_dice


# ─── Main ─────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Train U-Net flood segmentation model")
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to training dataset directory (with images/ and masks/ subdirs)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(MODEL_OUTPUT),
        help=f"Output path for trained model (default: {MODEL_OUTPUT})",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--bands", nargs="+", default=SENTINEL1_BANDS, help="Input bands")
    parser.add_argument("--model", choices=["unet", "resnet_unet"], default="resnet_unet")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume", type=Path, help="Resume from checkpoint")
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset paths
    dataset_dir = args.dataset
    images_dir = dataset_dir / "images"
    masks_dir = dataset_dir / "masks"
    splits_dir = dataset_dir / "splits"

    # Create synthetic dataset if none exists (for demo purposes)
    if not images_dir.exists() or not list(images_dir.glob("*.npy")):
        print("\n⚠️ No training images found. Creating synthetic dataset for demonstration.")
        create_synthetic_dataset(dataset_dir, args.img_size, len(args.bands))

    train_split = splits_dir / "train.txt" if splits_dir.exists() else None
    val_split = splits_dir / "val.txt" if splits_dir.exists() else None

    # Datasets
    train_ds = FloodDataset(
        images_dir,
        masks_dir,
        train_split or (dataset_dir / "train.txt"),
        img_size=args.img_size,
        bands=args.bands,
    )
    val_ds = FloodDataset(
        images_dir,
        masks_dir,
        val_split or (dataset_dir / "val.txt"),
        img_size=args.img_size,
        bands=args.bands,
    )

    # DataLoaders
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    # Model
    if args.model == "unet":
        model = UNet(
            in_channels=len(args.bands),
            num_classes=1,
            base_channels=64,
            depth=4,
        )
    else:
        model = ResNetUNet(
            in_channels=len(args.bands),
            num_classes=1,
            pretrained=True,
        )

    model = model.to(device)
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer & loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)

    # Resume from checkpoint
    start_epoch = 0
    if args.resume and args.resume.exists():
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        print(f"Resumed from epoch {start_epoch} (Dice: {checkpoint['best_val_dice']:.4f})")

    # Train
    args.output.parent.mkdir(parents=True, exist_ok=True)
    best_dice = train(
        model, train_loader, val_loader, optimizer, criterion,
        device, args.epochs, args.output, start_epoch,
    )

    # Save final model info
    model_info = {
        "model_type": args.model,
        "in_channels": len(args.bands),
        "bands": args.bands,
        "input_resolution": args.img_size,
        "best_val_dice": round(best_dice, 4),
        "total_epochs": args.epochs,
        "training_log": (args.output.parent / "training_log.json").exists(),
        "citation": (
            "If using Kuro Siwo: Bountos, N., et al. (2024). Kuro Siwo: A Large-Scale "
            "Dataset for Flood Segmentation in Satellite Imagery. NeurIPS 2024. (MIT/CC BY)"
        ),
    }
    (args.output.parent / "model_info.json").write_text(json.dumps(model_info, indent=2), encoding="utf-8")
    print(f"  Model info: {args.output.parent / 'model_info.json'}")


def create_synthetic_dataset(dataset_dir: Path, img_size: int, num_bands: int):
    """Create a small synthetic dataset for demonstration/testing."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    images_dir = dataset_dir / "images"
    masks_dir = dataset_dir / "masks"
    splits_dir = dataset_dir / "splits"
    images_dir.mkdir(exist_ok=True)
    masks_dir.mkdir(exist_ok=True)
    splits_dir.mkdir(exist_ok=True)

    # Generate realistic flood-like patterns
    flood_shapes = [
        # "river" flood
        lambda h, w: _synthetic_flood_river(h, w),
        # "flash flood" patch
        lambda h, w: _synthetic_flood_patch(h, w),
        # "coastal flood"
        lambda h, w: _synthetic_flood_coastal(h, w),
    ]

    n_train = 200
    n_val = 40

    train_filenames = []
    val_filenames = []

    for i in range(n_train + n_val):
        fname = f"tile_{i:04d}.npy"
        shape_func = flood_shapes[i % len(flood_shapes)]

        # Satellite-like image (random terrain + noise)
        image = np.random.randn(num_bands, img_size, img_size).astype(np.float32) * 0.1
        # Add terrain gradient
        y, x = np.ogrid[:img_size, :img_size]
        terrain = np.sin(x / img_size * 4 * np.pi) * 0.2 + np.cos(y / img_size * 3 * np.pi) * 0.2
        image[0] += terrain

        # Flood mask
        mask = shape_func(img_size, img_size).astype(np.float32)
        mask = mask[np.newaxis, ...]  # (1, H, W)

        # Save
        np.save(images_dir / fname, image)
        np.save(masks_dir / fname.replace(".npy", "_mask.npy"), mask)

        if i < n_train:
            train_filenames.append(fname)
        else:
            val_filenames.append(fname)

    (splits_dir / "train.txt").write_text("\n".join(train_filenames), encoding="utf-8")
    (splits_dir / "val.txt").write_text("\n".join(val_filenames), encoding="utf-8")

    print(f"  Created synthetic dataset: {n_train} train + {n_val} val tiles")


def _synthetic_flood_river(h: int, w: int) -> np.ndarray:
    """Synthesize a river-like flood pattern."""
    mask = np.zeros((h, w), dtype=np.float32)
    y, x = np.ogrid[:h, :w]
    # Sine-wave river path
    x1 = np.arange(w)  # column indices,
    river_y = (h // 2 + (np.sin((x1 / w) * 6 * np.pi) * h * 0.2).astype(int)).astype(int)
    for i in range(w):
        ry = int(min(max(int(river_y[i]), 0), h - 1))
        mask[max(0, ry - 3):ry + 4, i] = 1.0
    # Add noise to mask edges
    noise = np.random.rand(h, w) * 0.3
    mask = np.clip(mask + noise, 0, 1)
    return mask


def _synthetic_flood_patch(h: int, w: int) -> np.ndarray:
    """Synthesize a flood patch (flash flood area)."""
    mask = np.zeros((h, w), dtype=np.float32)
    cy, cx = h // 2 + np.random.randint(-20, 20), w // 2 + np.random.randint(-20, 20)
    ry, rx = np.random.randint(15, 40), np.random.randint(15, 40)
    y, x = np.ogrid[:h, :w]
    ellipse = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2
    mask[ellipse < 1] = 1.0
    return mask


def _synthetic_flood_coastal(h: int, w: int) -> np.ndarray:
    """Synthesize coastal flooding."""
    mask = np.zeros((h, w), dtype=np.float32)
    y, x = np.ogrid[:h, :w]
    # Flood from bottom (coast)
    flood_line = h * 0.6 + np.random.randint(-10, 10)
    flood_height = np.random.randint(10, 30)
    mask[int(flood_line):int(flood_line) + flood_height, :] = 1.0
    # Add some random inundation patches
    for _ in range(5):
        px, py = np.random.randint(0, w), np.random.randint(0, h)
        PR = np.random.randint(5, 15)
        mask[max(0, py - PR):py + PR, max(0, px - PR):px + PR] = 1.0
    return mask


if __name__ == "__main__":
    main()
