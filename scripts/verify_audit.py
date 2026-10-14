#!/usr/bin/env python3
"""Verify the append-only dashboard audit hash chain."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def canonical(entry: dict) -> str:
    return json.dumps(entry, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def verify(path: Path) -> tuple[bool, int]:
    previous = "0" * 64
    count = 0
    if not path.exists():
        return True, 0
    for count, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            entry = json.loads(line)
            actual = entry.pop("entry_hash")
            if entry.get("prev_hash") != previous:
                return False, count
            expected = hashlib.sha256((previous + canonical(entry)).encode("utf-8")).hexdigest()
            if actual != expected:
                return False, count
            previous = actual
        except (json.JSONDecodeError, KeyError, TypeError):
            return False, count
    return True, count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=ROOT / "logs" / "audit.jsonl")
    args = parser.parse_args()
    valid, count = verify(args.path)
    if valid:
        print(f"Audit chain valid ({count} entries)")
        return 0
    print(f"Audit chain broken at entry {count}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
