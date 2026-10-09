"""Load CDSE credentials from the project secret file into os.environ.

The cdse / CDSEClient libraries read CDSE_CLIENT_ID and
CDSE_CLIENT_SECRET directly from the process environment. This module
reads the local, git-ignored secret file once and injects those
variables so downstream scripts (download_sentinel.py,
download_sentinel2.py, and any future authenticated caller) can call
build_client() without either value ever appearing on a command line,
in a shell history entry, or in a log.

Usage:
    from cdse_env import load_cdse_env
    load_cdse_env()          # raises SystemExit if creds are missing

Or, from a shell, pre-load the variables for one invocation:
    python - c "import cdse_env; cdse_env.load_cdse_env()"
    python scripts/download_sentinel.py ...
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import NoReturn

SECRET_FILE = Path(__file__).resolve().parent.parent / ".secrets" / "cdse.env"


def _fail(msg: str) -> NoReturn:
    print(msg, file=sys.stderr)
    sys.exit(1)


def load_cdse_env() -> None:
    """Read .secrets/cdse.env and set CDSE_CLIENT_ID / CDSE_CLIENT_SECRET.

    Called more than once is safe (the file is re-read each time).
    Missing or empty values raise a non-zero exit with actionable text.
    """
    if not SECRET_FILE.is_file():
        _fail(
            "Copernicus credentials not found.\n"
            "  Expected: hackathon-trackb/.secrets/cdse.env\n"
            "  with two lines:\n"
            "    CDSE_CLIENT_ID=<id>\n"
            "    CDSE_CLIENT_SECRET=<secret>\n"
            "  (The secret is shown only once when you create the OAuth client.)\n"
        )

    values: dict[str, str] = {}
    for raw in SECRET_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            _fail(f"Malformed line in {SECRET_FILE}: {raw!r}")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            _fail(f"Empty key in {SECRET_FILE}: {raw!r}")
        if key in values:
            _fail(f"Duplicate key in {SECRET_FILE}: {key}")
        values[key] = value

    for env_name in ("CDSE_CLIENT_ID", "CDSE_CLIENT_SECRET"):
        if env_name not in values:
            _fail(
                f"{env_name} missing from {SECRET_FILE}.\n"
                "  Re-create the OAuth client — the client secret is shown only once."
            )
        if not values[env_name]:
            _fail(
                f"{env_name} is empty in {SECRET_FILE}.\n"
                "  Re-create the OAuth client — the client secret is shown only once."
            )
        # Prefer an already-set env var (e.g. from the shell) over the file,
        # so an explicit runtime override wins without surprising behaviour.
        if env_name not in os.environ or not os.environ[env_name]:
            os.environ[env_name] = values[env_name]
        elif not values[env_name]:
            # Only the file is set and it is empty — clear any leftover env.
            os.environ.pop(env_name, None)
            _fail(f"{env_name} empty in {SECRET_FILE}; nothing to use.")

    # Defensive: re-verify after a potential override so we never start a
    # download with blank credentials.
    missing = [k for k in ("CDSE_CLIENT_ID", "CDSE_CLIENT_SECRET")
               if not os.environ.get(k)]
    if missing:
        _fail(
            f"Credentials missing after load: {', '.join(missing)}.\n"
            "  Set them in your shell, or fix .secrets/cdse.env."
        )


if __name__ == "__main__":
    load_cdse_env()
    # Success path: report status only. Never print the values.
    print("CDSE credentials loaded from .secrets/cdse.env")
