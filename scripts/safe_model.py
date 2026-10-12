#!/usr/bin/env python3
"""
Safe checkpoint loading (T35 — model supply chain).

A ``.pt`` file is a pickle: ``torch.load`` with ``weights_only=False`` runs
whatever the file tells it to, so a checkpoint from anywhere but our own trainer
is remote code execution. This module is the only place the project
deserialises a checkpoint, and it enforces three rules:

1. ``weights_only=True`` — torch refuses anything that is not a plain container
   of tensors, numbers, strings and booleans, so a pickled class cannot run.
2. A sibling ``<model>.pt.sha256`` written by ``train_segmentation.py`` must
   match. A checkpoint with no sidecar is refused unless the caller explicitly
   says ``allow_unverified=True`` (used for a resume file the user just wrote).
3. The loaded object must have the shape the pipeline expects
   (``model_state_dict`` plus an optional ``config`` block) — a checkpoint that
   is not a dict is refused with a clear message, not an AttributeError later.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

CHUNK = 1024 * 1024


class ModelTrustError(RuntimeError):
    """The checkpoint could not be trusted, so it was not loaded."""


def sha256_file(path: Path) -> str:
    """SHA-256 of a file, read in chunks so a 250 MB checkpoint is fine."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def sidecar_path(model_path: Path) -> Path:
    """Where the hash of ``model_path`` is recorded, e.g. ``flood_unet.pt.sha256``."""
    return model_path.with_name(model_path.name + ".sha256")


def write_sidecar(model_path: Path) -> Path:
    """Record the hash next to a checkpoint we just produced."""
    path = sidecar_path(model_path)
    path.write_text(sha256_file(model_path) + "\n", encoding="utf-8")
    return path


def verify_checkpoint(model_path: Path, allow_unverified: bool = False) -> str:
    """Check the checkpoint's hash before anything deserialises it.

    Returns the verified digest, or a warning string when no sidecar exists and
    the caller accepted an unverified file. Raises :class:`ModelTrustError`
    otherwise, so a mismatch stops the run instead of loading different weights
    than the ones that were evaluated.
    """
    model_path = Path(model_path)
    if not model_path.is_file():
        raise ModelTrustError(f"model checkpoint not found: {model_path.name}")

    sidecar = sidecar_path(model_path)
    if not sidecar.is_file():
        if allow_unverified:
            return "unverified: no .sha256 sidecar"
        raise ModelTrustError(
            f"no hash for {model_path.name}: refusing to load an unverified "
            f"checkpoint. Expected {sidecar.name} beside it. Train with "
            f"scripts/train_segmentation.py (which writes it), or pass "
            f"--trust-unverified-model if you produced this file yourself."
        )

    expected = sidecar.read_text(encoding="utf-8").strip().split()[0].lower()
    actual = sha256_file(model_path)
    if expected != actual:
        raise ModelTrustError(
            f"checkpoint hash mismatch for {model_path.name}: the file does not "
            f"match {sidecar.name} (expected {expected[:12]}…, found "
            f"{actual[:12]}…). Refusing to load it."
        )
    return actual


def load_checkpoint(
    model_path: Path,
    device: str | torch.device = "cpu",
    allow_unverified: bool = False,
) -> dict:
    """Verify, then deserialise a checkpoint with ``weights_only=True``.

    Every failure path raises :class:`ModelTrustError` with a message a human can
    act on — never a raw UnpicklingError or a half-loaded model.
    """
    model_path = Path(model_path)
    verify_checkpoint(model_path, allow_unverified=allow_unverified)

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError as exc:  # torch < 2.0 has no weights_only argument
        raise ModelTrustError(
            "this torch build cannot load checkpoints safely (needs torch>=2.0 "
            "for weights_only=True). Upgrade torch, or use the log-ratio "
            "baseline with --baseline."
        ) from exc
    except Exception as exc:  # noqa: BLE001 — pickle errors carry many types
        raise ModelTrustError(
            f"{model_path.name} is not a plain tensor checkpoint "
            f"({type(exc).__name__}). Refusing to load it."
        ) from exc

    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ModelTrustError(
            f"{model_path.name} is not a Jalnetra checkpoint: expected a dict "
            f"with 'model_state_dict'."
        )
    return checkpoint
