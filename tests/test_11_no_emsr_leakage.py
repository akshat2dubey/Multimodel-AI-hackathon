"""11. Leakage guard: the running code must never read the reference maps.

Using the official EMS activation products as an INPUT is a disqualifying
violation — they may only be consulted by a human, for the comparison section.
This test therefore asserts that nothing under scripts/ or dashboard/ mentions
the reference product at all, so no path can quietly start depending on it.
"""

from __future__ import annotations

from pathlib import Path

SOURCE_SUFFIXES = {".py", ".html", ".js", ".json", ".txt", ".cfg", ".ini", ".toml"}
FORBIDDEN = "emsr"


def _source_files(scripts_dir: Path, dashboard_dir: Path):
    for base in (scripts_dir, dashboard_dir):
        for path in sorted(base.rglob("*")):
            if path.is_file() and path.suffix.lower() in SOURCE_SUFFIXES:
                if "__pycache__" in path.parts:
                    continue
                yield path


def test_no_source_file_references_the_reference_maps(scripts_dir, dashboard_dir):
    offenders = []
    for path in _source_files(scripts_dir, dashboard_dir):
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        if FORBIDDEN in text:
            for lineno, line in enumerate(text.splitlines(), start=1):
                if FORBIDDEN in line:
                    offenders.append(f"{path}:{lineno}: {line.strip()[:100]}")

    assert not offenders, (
        "the pipeline source references the Copernicus EMS reference product. "
        "It may be used for human comparison only, never as a program input "
        "(that would be disqualifying):\n  " + "\n  ".join(offenders)
    )


def test_no_docs_path_is_read_by_the_pipeline(scripts_dir, dashboard_dir):
    """No source file may hard-code a docs/ reference-map path either."""
    offenders = []
    for path in _source_files(scripts_dir, dashboard_dir):
        if path.suffix.lower() != ".py":
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            lower = line.lower()
            if "docs/" in lower and ("bulletin" in lower or "reference" in lower):
                offenders.append(f"{path}:{lineno}: {stripped[:100]}")

    assert not offenders, (
        "a script appears to read a reference-map document under docs/; that is "
        "checking material, not an input:\n  " + "\n  ".join(offenders)
    )
