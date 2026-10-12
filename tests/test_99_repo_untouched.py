"""99. The test suite must never write into the repository's results/ or data/.

Runs last (filename ordering) and compares modification times against the
session start. Real artifacts and real downloads live in those directories; a
test that clobbers them destroys the only evidence the pipeline ever ran, and
would make a green suite meaningless.
"""

from __future__ import annotations

from pathlib import Path

PROTECTED = ("results", "data")


def test_results_and_data_were_not_modified_by_the_suite(repo_root: Path, session_start: float):
    touched = []
    for name in PROTECTED:
        base = repo_root / name
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if mtime >= session_start - 1.0:
                touched.append(f"{path} (mtime {mtime})")

    assert not touched, (
        "the test suite wrote into a protected directory:\n  "
        + "\n  ".join(touched[:10])
        + "\nEvery test must build its inputs under tmp_path and pass explicit "
        "--output paths; results/ and data/ hold the real artifacts."
    )
