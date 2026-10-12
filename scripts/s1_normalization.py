#!/usr/bin/env python3
"""The one input normalisation for the flood-segmentation model.

Why this file exists
--------------------
Training and inference must feed the network *identically scaled* inputs. They
did not: the training dataset divided every band by 255 (a leftover from 8-bit
imagery) while inference handed the model raw dB values around -20..0. A model
trained that way sees a completely different input distribution at inference
time, which is one of the quietest ways for a flood map to become nonsense.

So the mapping lives here, in exactly one function (`normalize_bands`), used by
BOTH `train_segmentation.py` (in `FloodAugmentation`) and `inference.py` (in
`tile_inference`). The exact ranges are written into the checkpoint's `config`
block by training and read back by inference (`normalization_from_checkpoint`),
so a checkpoint can never be evaluated against a different scaling than the one
it was trained with — a mismatch is detectable instead of invisible.

Conventions
-----------
* Sentinel-1 backscatter (VV, VH) arrives in dB; the fixed window is
  ``DB_MIN..DB_MAX`` (default -30..0 dB).
* ``ratio`` is ``VV_dB - VH_dB``, itself a dB difference, with its own window.
* Optical bands (Sentinel-2 B02..B12) are surface reflectance already in 0..1,
  so they are passed through untouched.
"""

from __future__ import annotations

import numpy as np

# ─── The fixed windows ───────────────────────────────────────

DB_MIN = -30.0          # dB: below this is radar shadow / no-return
DB_MAX = 0.0            # dB: 0 dB is an extremely bright target
RATIO_MIN = -10.0       # dB difference VV - VH
RATIO_MAX = 10.0

# Band names that are backscatter in dB vs already-normalised reflectance.
S1_DB_BANDS = ("VV", "VH")
RATIO_BANDS = ("ratio",)
OPTICAL_BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")

# Guard against silently re-scaling with a different convention.
NORMALIZATION_VERSION = 1


def normalization_config() -> dict:
    """The normalisation block stored in a checkpoint's ``config``.

    Written by training, read by inference. Keep every key JSON-serialisable.
    """
    return {
        "kind": "db_linear",
        "version": NORMALIZATION_VERSION,
        "db_min": DB_MIN,
        "db_max": DB_MAX,
        "ratio_min": RATIO_MIN,
        "ratio_max": RATIO_MAX,
        "clip": [0.0, 1.0],
        "s1_db_bands": list(S1_DB_BANDS),
        "ratio_bands": list(RATIO_BANDS),
        "optical_passthrough_bands": list(OPTICAL_BANDS),
    }


def normalization_from_checkpoint(config: dict | None) -> dict:
    """Read the normalisation block out of a checkpoint's ``config``.

    Older checkpoints (trained before this module existed) have no block. Rather
    than inventing a range for them, fall back to the canonical window and mark
    the result with ``source`` so callers can print where the numbers came from.
    """
    cfg = dict(normalization_config())
    stored = (config or {}).get("normalization") if isinstance(config, dict) else None
    if isinstance(stored, dict) and "db_min" in stored and "db_max" in stored:
        for key in ("kind", "version", "db_min", "db_max", "ratio_min", "ratio_max"):
            if key in stored:
                cfg[key] = stored[key]
        cfg["source"] = "checkpoint"
    else:
        cfg["source"] = "default (checkpoint carries no normalisation block)"
    return cfg


def describe_normalization(config: dict | None = None) -> str:
    """One line for the console: which windows are in force, and where from."""
    cfg = config or normalization_config()
    return (
        f"dB [{cfg['db_min']:g}, {cfg['db_max']:g}] -> 0..1, "
        f"ratio [{cfg.get('ratio_min', RATIO_MIN):g}, {cfg.get('ratio_max', RATIO_MAX):g}] "
        f"[{cfg.get('source', 'default')}]"
    )


# ─── The mapping ─────────────────────────────────────────────

def _band_names_for(channels: int, band_names=None) -> list[str]:
    """Band names for a (C, H, W) stack.

    ``preprocess_sentinel1`` returns ``[VV, VH, ratio]``; a 2-channel stack is
    ``[VV, VH]``. Anything else falls back to positional names.
    """
    if band_names:
        names = list(band_names)
    elif channels == 1:
        names = ["VV"]
    elif channels == 2:
        names = ["VV", "VH"]
    elif channels == 3:
        names = ["VV", "VH", "ratio"]
    else:
        names = [f"band_{i}" for i in range(channels)]
    if len(names) < channels:
        names += [f"band_{i}" for i in range(len(names), channels)]
    return names[:channels]


def _range_for(name: str, config: dict) -> tuple[float, float] | None:
    """The (min, max) window for one band, or None when it must be left alone."""
    key = str(name).strip()
    if key in config.get("optical_passthrough_bands", OPTICAL_BANDS):
        return None
    if key in config.get("ratio_bands", RATIO_BANDS):
        return (
            float(config.get("ratio_min", RATIO_MIN)),
            float(config.get("ratio_max", RATIO_MAX)),
        )
    return (float(config.get("db_min", DB_MIN)), float(config.get("db_max", DB_MAX)))


def normalize_band(band: np.ndarray, name: str = "VV", config: dict | None = None) -> np.ndarray:
    """Clip one band to its fixed window and scale it into 0..1."""
    cfg = config or normalization_config()
    window = _range_for(name, cfg)
    arr = np.asarray(band, dtype=np.float32)
    if window is None:
        # Already 0..1 reflectance; keep it, but don't pretend it was mapped.
        return np.clip(arr, 0.0, 1.0)
    lo, hi = window
    span = hi - lo
    if span <= 0:
        raise ValueError(f"empty normalisation window for band {name!r}: [{lo}, {hi}]")
    return np.clip((arr - lo) / span, 0.0, 1.0).astype(np.float32)


def normalize_bands(
    image: np.ndarray,
    band_names=None,
    config: dict | None = None,
) -> np.ndarray:
    """Map a (C, H, W) — or single (H, W) — stack into 0..1, one window per band.

    THE shared entry point: training and inference both call this, so the two
    can only disagree if they pass different ``config``.
    """
    cfg = config or normalization_config()
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:
        return normalize_band(arr, (band_names or ["VV"])[0], cfg)

    channels = arr.shape[0]
    names = _band_names_for(channels, band_names)
    out = np.empty_like(arr, dtype=np.float32)
    for i, name in enumerate(names):
        out[i] = normalize_band(arr[i], name, cfg)
    return out
