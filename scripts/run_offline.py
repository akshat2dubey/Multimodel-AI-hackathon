#!/usr/bin/env python3
"""Compatibility entry point for the offline synthetic demo."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from demo.run_offline import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
