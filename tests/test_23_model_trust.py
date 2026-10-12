"""T35 — safe model loading (model supply chain).

A checkpoint is a pickle: loading one with ``weights_only=False`` executes
whatever the file contains. These tests pin the three defences — a matching
``.sha256`` sidecar, ``weights_only=True`` deserialisation, and a shape check —
against tampering, against a missing sidecar, and against a genuine pickle
payload.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

torch = pytest.importorskip("torch")

import safe_model  # noqa: E402


def _write_checkpoint(path: Path, value: float = 1.0) -> Path:
    """A minimal checkpoint in the shape train_segmentation.py writes."""
    torch.save(
        {
            "epoch": 1,
            "model_state_dict": {"weight": torch.full((2, 2), value)},
            "best_val_dice": 0.5,
            "config": {"in_channels": 2, "bands": ["VV", "VH"]},
        },
        path,
    )
    return path


def test_sidecar_is_written_and_verified(tmp_path):
    ckpt = _write_checkpoint(tmp_path / "flood_unet.pt")
    sidecar = safe_model.write_sidecar(ckpt)
    assert sidecar.name == "flood_unet.pt.sha256"
    assert safe_model.verify_checkpoint(ckpt) == safe_model.sha256_file(ckpt)

    loaded = safe_model.load_checkpoint(ckpt, device="cpu")
    assert "model_state_dict" in loaded
    assert loaded["config"]["bands"] == ["VV", "VH"]


def test_tampered_checkpoint_is_refused(tmp_path):
    """Weights swapped after the fact must not load, even though the file is valid torch."""
    ckpt = _write_checkpoint(tmp_path / "flood_unet.pt", value=1.0)
    safe_model.write_sidecar(ckpt)
    _write_checkpoint(ckpt, value=2.0)          # same path, different bytes

    with pytest.raises(safe_model.ModelTrustError, match="hash mismatch"):
        safe_model.load_checkpoint(ckpt, device="cpu")


def test_missing_sidecar_is_refused_unless_explicitly_trusted(tmp_path):
    ckpt = _write_checkpoint(tmp_path / "downloaded.pt")

    with pytest.raises(safe_model.ModelTrustError, match="no hash"):
        safe_model.load_checkpoint(ckpt, device="cpu")

    # The escape hatch is explicit, named, and reported as unverified.
    assert safe_model.load_checkpoint(ckpt, device="cpu", allow_unverified=True)
    assert "unverified" in safe_model.verify_checkpoint(ckpt, allow_unverified=True)


class _Pickled:
    """A custom class: exactly what weights_only=True must refuse to unpickle."""

    def __reduce__(self):  # pragma: no cover - proves the payload is executable
        return (pickle.loads, (b"b'ran'",))


def test_pickled_object_is_refused_even_with_a_valid_sidecar(tmp_path):
    ckpt = tmp_path / "evil.pt"
    torch.save({"model_state_dict": {}, "payload": _Pickled()}, ckpt, pickle_protocol=4)
    safe_model.write_sidecar(ckpt)              # hash is fine; content is not

    with pytest.raises(safe_model.ModelTrustError, match="not a plain tensor checkpoint"):
        safe_model.load_checkpoint(ckpt, device="cpu")


def test_non_checkpoint_dict_is_refused(tmp_path):
    ckpt = tmp_path / "wrong.pt"
    torch.save({"something": torch.zeros(1)}, ckpt)
    safe_model.write_sidecar(ckpt)

    with pytest.raises(safe_model.ModelTrustError, match="model_state_dict"):
        safe_model.load_checkpoint(ckpt, device="cpu")


def test_missing_file_is_refused(tmp_path):
    with pytest.raises(safe_model.ModelTrustError, match="not found"):
        safe_model.load_checkpoint(tmp_path / "nope.pt", device="cpu")


def test_inference_refuses_an_untrusted_checkpoint_before_loading(tmp_path, monkeypatch):
    """load_model must exit non-zero (no traceback) rather than import a bad file."""
    import inference

    ckpt = _write_checkpoint(tmp_path / "flood_unet.pt")
    safe_model.write_sidecar(ckpt)
    _write_checkpoint(ckpt, value=9.0)

    with pytest.raises(SystemExit) as excinfo:
        inference.load_model(ckpt, torch.device("cpu"))
    assert excinfo.value.code == 1
