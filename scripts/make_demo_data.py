#!/usr/bin/env python3
"""Compatibility entry point for the synthetic demo-data generator."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from demo.make_demo_data import main  # noqa: E402

if __name__ == "__main__":
    main()
