"""Move project pipeline files to or from the configured private S3 bucket."""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.object_storage import ObjectStore, ObjectStoreError, PERSISTENT_ROOTS  # noqa: E402


class _Progress:
    def __init__(self):
        self.lock = threading.Lock()
        self.bytes = 0
        self.next_notice = 128 * 1024 * 1024

    def __call__(self, chunk: int) -> None:
        with self.lock:
            self.bytes += chunk
            if self.bytes >= self.next_notice:
                print(f"Transferred {self.bytes // (1024 * 1024)} MiB", flush=True)
                self.next_notice = self.bytes + 128 * 1024 * 1024


def _load_local_env() -> None:
    """Read simple KEY=value entries without printing or replacing shell values."""
    path = ROOT / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key.strip(), value)


def main() -> int:
    action = sys.argv[1] if len(sys.argv) == 2 else ""
    if action not in ("push", "restore"):
        print("Usage: python scripts/persist.py push|restore", file=sys.stderr)
        return 2
    try:
        _load_local_env()
        store = ObjectStore.from_env(ROOT, required=True)
        store.progress_callback = _Progress()
        count = (store.upload_roots(list(PERSISTENT_ROOTS)) if action == "push"
                 else store.sync_roots(list(PERSISTENT_ROOTS)))
    except ObjectStoreError as exc:
        print(f"Persistent storage operation failed: {exc}", file=sys.stderr)
        return 1
    print(f"{action}: completed for {count} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
