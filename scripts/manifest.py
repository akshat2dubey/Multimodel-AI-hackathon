#!/usr/bin/env python3
"""
Provenance manifest and the hard data rules (T34).

The hackathon's rules are a disqualification boundary, not a style guide: only
Sentinel-1/2, Copernicus DEM, a pre-event OSM snapshot and the listed training
data may enter the pipeline. Copernicus EMS/UNOSAT damage maps may *check*
results but must never be inputs.

Two things live here:

* ``rules_check`` — machine-evaluated booleans over the run's own metadata and
  the files it actually used. ``False`` means a rule was broken; ``None`` means
  the rule could not be evaluated from what the run recorded (e.g. the relative
  orbit was not reported), which is recorded honestly rather than as a pass.
* ``build_manifest`` / ``write_manifest`` — one JSON file per run listing every
  input and output with its SHA-256, the model or method used, and the rule
  results, so a judge (or we) can re-derive what produced a result.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Forbidden input sources: a damage map produced after the event, or a service
# that publishes one. Matching is on the path/name of an input, because that is
# where such a file would appear.
FORBIDDEN_INPUT_PATTERN = re.compile(
    r"(emsr|ems_rapid|unosat|copernicus[_ -]?ems|damage[_ -]?map|flood[_ -]?map[_ -]?official)",
    re.IGNORECASE,
)

# Kept short on purpose: these are the versions that change what a raster or a
# float means, and they are what a re-run needs to reproduce the numbers.
TRACKED_PACKAGES = ("numpy", "rasterio", "geopandas", "scipy", "torch", "flask")

CHUNK = 1024 * 1024


class RuleViolation(RuntimeError):
    """A hard data rule failed: the run must not continue."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(repo_root: Path) -> str | None:
    """The commit the run was executed from, or None when unavailable."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def _versions() -> dict:
    import importlib.metadata

    found = {"python": platform.python_version()}
    for name in TRACKED_PACKAGES:
        try:
            found[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            found[name] = None
    return found


def _relative_orbit(sidecar: Path) -> int | None:
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    orbit = payload.get("relative_orbit")
    return int(orbit) if isinstance(orbit, (int, float)) else None


def rules_check(run_meta: dict, input_paths: list[Path] | None = None,
                sentinel_dir: Path | None = None) -> dict:
    """Evaluate the hard data rules. Values are True / False / None (unknown)."""
    checks: dict[str, bool | None] = {}

    snapshot = str(run_meta.get("osm_snapshot_date") or "")
    event = str(run_meta.get("event_date") or "")
    # ISO dates compare lexicographically; anything else is not comparable.
    checks["osm_snapshot_before_event"] = (
        snapshot < event if snapshot and event else None
    )

    before = str(run_meta.get("before_date") or "")
    after = str(run_meta.get("after_date") or "")
    # Inclusive on the event day: a scene acquired later on the flood day still
    # shows the flood, so requiring a strictly later "after" date would refuse a
    # legitimate pair.
    checks["acquisitions_bracket_event"] = (
        before <= event <= after if before and after and event else None
    )

    paths = [Path(p) for p in (input_paths or [])]
    if paths:
        offenders = [p.name for p in paths if FORBIDDEN_INPUT_PATTERN.search(str(p))]
        checks["no_forbidden_input_sources"] = not offenders
    else:
        checks["no_forbidden_input_sources"] = None

    if sentinel_dir is not None:
        before_orbit = _relative_orbit(sentinel_dir / "before" / "S1_PROCESSED.json")
        after_orbit = _relative_orbit(sentinel_dir / "after" / "S1_PROCESSED.json")
        # Same relative orbit and same flight direction is what makes a
        # before/after pair comparable at all.
        checks["sentinel_orbit_same_for_pair"] = (
            before_orbit == after_orbit
            if before_orbit is not None and after_orbit is not None else None
        )
    else:
        checks["sentinel_orbit_same_for_pair"] = None

    return checks


def enforce_rules(checks: dict) -> None:
    """Stop the run on a broken rule; report which rules stayed unknown."""
    failed = [name for name, ok in checks.items() if ok is False]
    unknown = [name for name, ok in checks.items() if ok is None]
    if failed:
        raise RuleViolation(
            "hard data rule(s) violated: " + ", ".join(failed)
            + "\nOnly Sentinel-1/2, Copernicus DEM, a pre-event OSM snapshot and "
            "the listed training datasets may be inputs; EMS/UNOSAT damage maps "
            "may only be used to check results."
        )
    if unknown:
        print("  ℹ rules not evaluable yet (recorded as null): " + ", ".join(unknown))


def describe_rule_failure(checks: dict) -> str | None:
    failed = [name for name, ok in checks.items() if ok is False]
    if not failed:
        return None
    return ("hard data rule(s) failed at the end of the run: "
            + ", ".join(failed)
            + " — see results/manifest.json")


def _entries(paths: list[Path], base: Path) -> list[dict]:
    """Hash each existing file, recording paths relative to the repo root."""
    entries = []
    for path in sorted({Path(p) for p in paths}):
        if not path.is_file():
            continue
        try:
            relative = str(path.resolve().relative_to(base.resolve()))
        except ValueError:
            relative = path.name          # outside the repo: name only
        entries.append({
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return entries


def build_manifest(
    run_meta: dict,
    inputs: list[Path],
    outputs: list[Path],
    repo_root: Path,
    method: str | None = None,
    model: Path | None = None,
    sentinel_dir: Path | None = None,
) -> dict:
    checks = rules_check(run_meta, inputs, sentinel_dir)
    model_entry = None
    if model is not None and Path(model).is_file():
        sidecar = Path(str(model) + ".sha256")
        model_entry = {
            "path": Path(model).name,
            "sha256": sha256_file(Path(model)),
            "verified_against_sidecar": (
                sidecar.is_file()
                and sidecar.read_text(encoding="utf-8").strip().lower()
                == sha256_file(Path(model))
            ),
        }
    return {
        "run_id": run_meta.get("run_id"),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repo_commit": git_commit(repo_root),
        "parameters": run_meta,
        "method": method or run_meta.get("method") or "unknown",
        "model": model_entry,
        "inputs": _entries(inputs, repo_root),
        "outputs": _entries(outputs, repo_root),
        "versions": _versions(),
        "rules_check": checks,
    }


def write_manifest(results_dir: Path, manifest: dict) -> Path:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def input_paths_from(data_dir: Path) -> list[Path]:
    """Every file the pipeline consumed, in the layout run_pipeline creates."""
    candidates: list[Path] = []
    for sub in ("sentinel1/before", "sentinel1/after", "sentinel2/before",
                "sentinel2/after", "osm", "dem"):
        directory = Path(data_dir) / sub
        if directory.is_dir():
            candidates += [p for p in directory.rglob("*") if p.is_file()]
    return candidates


def output_paths_from(results_dir: Path) -> list[Path]:
    directory = Path(results_dir)
    if not directory.is_dir():
        return []
    return [p for p in directory.rglob("*")
            if p.is_file() and p.name != "manifest.json"]


if __name__ == "__main__":      # a small inspector, useful in the Q&A
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "results/manifest.json")
    if not target.is_file():
        print(f"no manifest at {target}")
        raise SystemExit(1)
    payload = json.loads(target.read_text(encoding="utf-8"))
    print(f"run {payload.get('run_id')} @ {payload.get('created_utc')}")
    print(f"commit {payload.get('repo_commit')}  method {payload.get('method')}")
    print("rules_check:")
    for name, ok in (payload.get("rules_check") or {}).items():
        print(f"  {name}: {ok}")
