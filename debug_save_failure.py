#!/usr/bin/env python3
"""Script to inspect and debug damage_assessment.py save_failure_to_geojson behavior."""

import inspect
import json
import scripts.damage_assessment as da


def main():
    # Get the source code of save_failure_to_geojson
    src = inspect.getsource(da.save_failure_to_geojson)
    lines = src.split('\n')
    for i, line in enumerate(lines, 1):
        print(f"{i:3d}: {line}")


if __name__ == "__main__":
    main()
