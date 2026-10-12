"""5. Training on an empty dataset must fail, not invent data.

train_segmentation.py used to fabricate 240 synthetic tiles whenever --dataset
was empty and train on them, producing a checkpoint that looks like a model. A
judge running the documented command would then be shown "Dice 0.44" from tiles
the script made up. Synthetic data is opt-in via --smoke-test, and the default
must be a loud, non-zero failure.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Importing torch/torchvision dominates the runtime; the check itself is quick.
TIMEOUT_S = 180


def _run(scripts_dir: Path, args: list[str], cwd: Path):
    return subprocess.run(
        [sys.executable, str(scripts_dir / "train_segmentation.py"), *args],
        capture_output=True,
        text=True,
        # The scripts print emoji and re-configure stdout to UTF-8; decode the
        # same way instead of letting the platform codec fail on their output.
        encoding="utf-8",
        errors="replace",
        cwd=str(cwd),
        timeout=TIMEOUT_S,
    )


def test_empty_dataset_dir_fails_without_creating_synthetic_tiles(tmp_path, scripts_dir):
    dataset = tmp_path / "empty_dataset"
    dataset.mkdir()
    model_out = tmp_path / "models" / "flood_unet.pt"

    proc = _run(
        scripts_dir,
        ["--dataset", str(dataset), "--output", str(model_out), "--epochs", "1"],
        cwd=tmp_path,
    )

    combined = (proc.stdout or "") + (proc.stderr or "")
    assert proc.returncode != 0, (
        "train_segmentation.py exited 0 on an empty --dataset. It must fail "
        "instead of fabricating a training set. Missing --smoke-test should be "
        f"an error.\nstdout: {proc.stdout[-500:]}\nstderr: {proc.stderr[-500:]}"
    )
    assert "No training images" in combined, (
        "the failure must say what is wrong ('No training images found in ...') "
        f"so the operator knows to supply a dataset; got:\n{combined[-800:]}"
    )
    assert not model_out.exists(), (
        f"{model_out} was written even though no real dataset exists — no "
        "checkpoint may be produced from fabricated tiles by default"
    )
    assert not (dataset / "images").exists(), (
        f"{dataset / 'images'} was created: the script generated synthetic tiles "
        "without being asked to. Synthetic data requires --smoke-test."
    )


def test_smoke_test_flag_is_available_and_documented(tmp_path, scripts_dir):
    proc = subprocess.run(
        [sys.executable, str(scripts_dir / "train_segmentation.py"), "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(tmp_path),
        timeout=TIMEOUT_S,
    )
    assert proc.returncode == 0, f"--help must work; got {proc.returncode}"
    assert "--smoke-test" in proc.stdout, (
        "there must be an explicit opt-in flag (--smoke-test) for training on "
        "generated tiles; without it the only path is a real dataset"
    )
