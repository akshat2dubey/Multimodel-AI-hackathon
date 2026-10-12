"""T36 — dependency and outbound-network hygiene.

Two things are easy to lose under deadline pressure and both are checkable:
every fetch must be HTTPS (never ``file:``), and a declared dependency must be
one the code actually imports (and vice versa).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import net_safe  # noqa: E402


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "http://overpass-api.de/api/interpreter",
        "ftp://example.org/data.tif",
        "data:text/plain,x",
        "not a url",
    ],
)
def test_non_https_urls_are_refused(url):
    with pytest.raises(ValueError, match="non-HTTPS"):
        net_safe.checked_url(url)


def test_https_urls_are_allowed():
    assert net_safe.checked_url("https://api.ohsome.org/v1") == "https://api.ohsome.org/v1"


def test_no_module_calls_urllib_urlopen_directly():
    """Direct urlopen accepts file:/custom schemes; net_safe is the one entry point."""
    offenders = []
    for path in sorted(SCRIPTS.rglob("*.py")):
        if path.name == "net_safe.py" or "__pycache__" in path.parts:
            continue
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
        ):
            if "urllib.request.urlopen(" in line:
                offenders.append(f"{path.name}:{lineno}: {line.strip()[:80]}")
    assert not offenders, (
        "use `from net_safe import urlopen` so the HTTPS scheme check is applied:\n  "
        + "\n  ".join(offenders)
    )


def test_flask_cors_is_neither_declared_nor_imported():
    """It was a dependency nothing used; the dashboard is same-origin only."""
    requirements = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert not re.search(r"^\s*flask[-_]cors", requirements, re.MULTILINE)

    for path in list((REPO_ROOT / "dashboard").rglob("*.py")) + list(SCRIPTS.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        assert "flask_cors" not in text, f"{path} imports flask_cors"
