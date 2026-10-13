"""T35 — a run must only claim the outputs it actually wrote.

The results directory is shared by every run, and the synthetic fixture used to
drop its artifacts there too. A manifest built by globbing the whole directory
therefore listed `flood_path_SYNTHETIC_BACKUP.geojson` (a fabricated path) and a
leftover `flood_mask_real.tif` as outputs of a *real* run, each with a real
SHA-256, which is exactly the kind of evidence a judge would trust.

These tests pin the boundary: an artifact older than the run is reported as
"not from this run" and never enters the outputs list; and a real run refuses to
write into a directory still stamped as the synthetic fixture.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import manifest  # noqa: E402
from config.presets import (  # noqa: E402
    DEMO_NOTICE_FILENAME,
    demo_notice_blocker,
)


def _meta() -> dict:
    return {
        "run_id": "run-fresh",
        "area": "toy",
        "bbox": [85.4, 28.0, 85.5, 28.1],
        "event_date": "2026-08-26",
        "before_date": "2026-08-24",
        "after_date": "2026-09-05",
        "osm_snapshot_date": "2026-07-27",
    }


def _write(path: Path, text: str, *, mtime: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    os.utime(path, (mtime, mtime))
    return path


def test_a_stale_file_is_not_claimed_as_this_runs_output(tmp_path):
    """The synthetic backup predates the run, so it cannot be an output of it."""
    stale = _write(tmp_path / "flood_path_SYNTHETIC_BACKUP.geojson", "{}",
                   mtime=time.time() - 86_400)
    fresh = _write(tmp_path / "flood_mask.tif", "mask", mtime=time.time())

    run_started = time.time() - 60
    outputs = manifest.output_paths_from(tmp_path, since=run_started)
    assert fresh in outputs
    assert stale not in outputs

    excluded = manifest.stale_outputs_from(tmp_path, since=run_started)
    assert excluded == [stale]


def test_the_manifest_states_which_files_it_left_out(tmp_path):
    """A reader must be able to see that a leftover exists, and that it is not ours."""
    stale = _write(tmp_path / "flood_mask_real.tif", "0", mtime=time.time() - 86_400)
    fresh = _write(tmp_path / "flood_mask.tif", "mask", mtime=time.time())
    run_started = time.time() - 60

    payload = manifest.build_manifest(
        _meta(),
        inputs=[],
        outputs=manifest.output_paths_from(tmp_path, since=run_started),
        stale_outputs=manifest.stale_outputs_from(tmp_path, since=run_started),
        repo_root=tmp_path,
    )

    assert [entry["path"] for entry in payload["outputs"]] == [fresh.name]
    assert payload["stale_outputs_not_from_this_run"] == [stale.name]


def test_inventory_mode_still_sees_the_whole_directory(tmp_path):
    """`since=None` is the plain inventory callers and tests rely on."""
    _write(tmp_path / "old.tif", "x", mtime=time.time() - 86_400)
    _write(tmp_path / "new.tif", "x", mtime=time.time())
    assert sorted(p.name for p in manifest.output_paths_from(tmp_path)) == [
        "new.tif",
        "old.tif",
    ]


def test_manifest_without_the_freshness_report_omits_the_key(tmp_path):
    """Callers that did not ask for the distinction get no misleading empty list."""
    payload = manifest.build_manifest(_meta(), [], [], tmp_path)
    assert "stale_outputs_not_from_this_run" not in payload


def test_real_run_refuses_a_directory_stamped_as_demo(tmp_path):
    """Writing real results next to the fixture notice would mislabel them."""
    assert demo_notice_blocker(tmp_path) is None

    (tmp_path / DEMO_NOTICE_FILENAME).write_text("synthetic", encoding="utf-8")
    message = demo_notice_blocker(tmp_path)
    assert message is not None
    assert DEMO_NOTICE_FILENAME in message


def test_manifest_output_entries_carry_a_hash_of_the_file_on_disk(tmp_path):
    """The freshness filter must not stop hashing the files it does include."""
    fresh = _write(tmp_path / "damage_assessment.geojson", '{"features": []}',
                   mtime=time.time())
    run_started = time.time() - 60
    payload = manifest.build_manifest(
        _meta(),
        inputs=[],
        outputs=manifest.output_paths_from(tmp_path, since=run_started),
        repo_root=tmp_path,
    )
    entry = payload["outputs"][0]
    assert entry["path"] == fresh.name
    assert entry["sha256"] == manifest.sha256_file(fresh)
    assert json.loads(json.dumps(payload))  # serialisable, as write_manifest needs
