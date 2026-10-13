"""One place for a run's case-study metadata: area, AOI and dates.

Why this module exists
----------------------
The area name, the bounding box, the acquisition dates, the OSM snapshot date
and the event date used to be literals in a dozen files — and they disagreed:

  * ``config/presets.py`` said before/after = 2026-08-14 / 2026-08-28,
  * the README said 2026-08-24 / 2026-08-31,
  * the Sentinel-1 pair actually downloaded (orbit 19) is
    2026-08-24 / 2026-09-05 (see ``data/sentinel1/*/PRODUCT_INFO.json`` and
    ``docs/EMSR927_COMPARISON.md``),
  * ``generate_report.py`` hard-coded the corridor name into its header and the
    OSM snapshot date into its data label.

A report whose header names a corridor and a date that the data does not
support is worse than no report. So the flow is now:

    CLI flags (+ an optional ``--preset``)  --resolve_run_meta-->  run_meta dict
    run_meta dict                           --write_run_meta-->   <results>/run_meta.json
    <results>/run_meta.json                 --load_run_meta-->    the same dict

Every stage reads that one dict, so the report header, the damage metadata and
the cut-off metadata cannot drift apart again. ``--bbox`` is the primary AOI
input; center+extent is only a fallback for when no box is known.

Nothing here is secret, and no coordinate is invented: the case-study AOI comes
from a preset (see config/presets.py) or from the operator's ``--bbox``.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

from config.presets import DEFAULT_EXTENT, DEFAULT_PRESET, PRESETS

# Written next to the artifacts it describes (results/run_meta.json), so a stage
# can find it from its own primary input without being told a second path.
RUN_META_FILENAME = "run_meta.json"
RUN_META_SCHEMA = "jalnetra.run_meta/1"

# The date fields that must parse as YYYY-MM-DD.
DATE_FIELDS = ("event_date", "before_date", "after_date", "osm_snapshot_date")

# The order of the four numbers in a --bbox string, stated once.
BBOX_ORDER = "min_lon,min_lat,max_lon,max_lat"


def parse_bbox(text: str) -> tuple[float, float, float, float]:
    """Parse ``min_lon,min_lat,max_lon,max_lat`` into four floats.

    Fails loudly with the expected order rather than accepting a silently
    transposed box: a lat/lon swap yields an empty download and an empty map,
    which is far harder to notice than a parse error.
    """
    parts = [p.strip() for p in str(text).split(",")]
    if len(parts) != 4:
        raise ValueError(
            f"--bbox must be {BBOX_ORDER} (4 comma-separated numbers), got {text!r}"
        )
    try:
        min_lon, min_lat, max_lon, max_lat = (float(p) for p in parts)
    except ValueError as exc:
        raise ValueError(
            f"--bbox must be {BBOX_ORDER} (4 comma-separated numbers), got {text!r}"
        ) from exc
    if (
        not all(math.isfinite(value) for value in (min_lon, min_lat, max_lon, max_lat))
        or not (-180 <= min_lon < max_lon <= 180)
        or not (-90 <= min_lat < max_lat <= 90)
    ):
        raise ValueError(
            f"--bbox must contain valid longitude/latitude bounds with min < max, "
            f"got {text!r} ({BBOX_ORDER})"
        )
    return (min_lon, min_lat, max_lon, max_lat)


def bbox_from_center(center_lat: float, center_lon: float, extent: float) -> tuple:
    """The fallback AOI: a square of half-width ``extent`` degrees."""
    return (
        round(center_lon - extent, 6),
        round(center_lat - extent, 6),
        round(center_lon + extent, 6),
        round(center_lat + extent, 6),
    )


def center_from_bbox(bbox) -> tuple[float, float]:
    """The box's midpoint, used when only a bbox is known."""
    min_lon, min_lat, max_lon, max_lat = bbox
    return (round((min_lat + max_lat) / 2, 6), round((min_lon + max_lon) / 2, 6))


def resolve_bbox(
    bbox: str | tuple | list | None,
    center_lat: float | None = None,
    center_lon: float | None = None,
    extent: float | None = None,
) -> tuple[float, float, float, float]:
    """A download stage's AOI: ``--bbox`` primary, center+extent the fallback.

    Shared by all four download scripts so they agree on the order of the four
    numbers, the fallback half-width and the error text. Raises ValueError (with
    guidance) when neither input is usable.
    """
    if bbox:
        return parse_bbox(bbox)
    if center_lat is None or center_lon is None:
        raise ValueError(
            "no AOI: pass --bbox (preferred, " + BBOX_ORDER + ") or both "
            "--center-lat and --center-lon (with an optional --extent)."
        )
    return bbox_from_center(center_lat, center_lon, DEFAULT_EXTENT if extent is None else extent)


def preset_values(preset: str | None = DEFAULT_PRESET) -> dict:
    """The named preset's raw values (a copy), or an empty dict for no preset."""
    if preset is None:
        return {}
    if preset not in PRESETS:
        raise ValueError(
            f"unknown --preset {preset!r}; available: {', '.join(sorted(PRESETS))}"
        )
    return dict(PRESETS[preset])


def resolve_run_meta(
    *,
    preset: str | None = None,
    area: str | None = None,
    bbox: str | tuple | list | None = None,
    center_lat: float | None = None,
    center_lon: float | None = None,
    extent: float | None = None,
    event_date: str | None = None,
    before_date: str | None = None,
    after_date: str | None = None,
    osm_snapshot_date: str | None = None,
    confirm_aoi: bool = False,
) -> dict:
    """CLI arguments (over) preset defaults -> the one run_meta dict.

    An explicit argument always wins over the preset. ``--bbox`` is the primary
    AOI input: when it is absent the AOI is derived from center+extent and the
    run is marked ``aoi_confirmed: false`` so every artifact it produces says
    out loud that the AOI was not confirmed.
    """
    # No preset means no preset defaults: every value must be given explicitly,
    # so an arbitrary area cannot silently inherit the case study's dates.
    base = preset_values(preset)

    def pick(cli, key):
        if cli is not None:
            return cli
        return base.get(key)

    meta: dict = {
        "schema": RUN_META_SCHEMA,
        "preset": preset,
        "area": pick(area, "area"),
        "event_date": pick(event_date, "event_date"),
        "before_date": pick(before_date, "before_date"),
        "after_date": pick(after_date, "after_date"),
        "osm_snapshot_date": pick(osm_snapshot_date, "osm_snapshot_date"),
    }

    # ── AOI: --bbox primary, center+extent fallback ──────────────────────────
    cli_bbox = parse_bbox(bbox) if bbox else None
    preset_bbox = base.get("bbox")
    if cli_bbox is not None:
        box, source = cli_bbox, "cli"
    elif preset_bbox:
        box, source = parse_bbox(",".join(str(v) for v in preset_bbox)), "preset"
    else:
        box, source = None, "center_extent_fallback"

    # A center is only "given" when the operator passed it on the CLI. The
    # preset's center belongs to the preset's FALLBACK box, so pairing it with a
    # different box would reintroduce exactly the drift this module removes.
    center_given = center_lat is not None or center_lon is not None
    if center_given and (center_lat is None or center_lon is None):
        raise ValueError(
            "--center-lat and --center-lon must be given together (a center needs "
            "both). Prefer --bbox, the primary AOI input."
        )

    picked_extent = pick(extent, "extent")

    if box is None:
        fallback_lat = center_lat if center_given else base.get("center_lat")
        fallback_lon = center_lon if center_given else base.get("center_lon")
        if fallback_lat is None or fallback_lon is None:
            raise ValueError(
                "no AOI: pass --bbox OR --center-lat/--center-lon (with an "
                "optional --extent). A run cannot guess where to look."
            )
        if picked_extent is None:
            picked_extent = 0.3
        box = bbox_from_center(fallback_lat, fallback_lon, picked_extent)
        picked_center_lat, picked_center_lon = fallback_lat, fallback_lon
    elif center_given:
        # The operator's center wins over the box midpoint, but a center outside
        # its own box is almost always a lat/lon swap — refuse rather than
        # download one place and report another.
        picked_center_lat, picked_center_lon = center_lat, center_lon
        if not (box[0] <= center_lon <= box[2] and box[1] <= center_lat <= box[3]):
            raise ValueError(
                f"--center-lat/--center-lon ({center_lat}, {center_lon}) is outside "
                f"--bbox {box}; check for a latitude/longitude swap."
            )
    else:
        # flood_path.py and the dashboard both need a center; derive it from the
        # box so every stage points at the same place.
        picked_center_lat, picked_center_lon = center_from_bbox(box)

    meta["bbox"] = [float(v) for v in box]
    meta["bbox_source"] = source
    meta["center_lat"] = float(picked_center_lat)
    meta["center_lon"] = float(picked_center_lon)
    if picked_extent is not None:
        meta["extent"] = float(picked_extent)

    # A supplied box does not prove that it describes the intended event. Require
    # explicit confirmation for CLI overrides; a source-backed preset can carry
    # its own confirmation. A derived fallback is never "confirmed".
    meta["aoi_confirmed"] = bool(
        source == "preset" and base.get("aoi_confirmed", False)
    )

    # ── Required + consistent values ─────────────────────────────────────────
    missing = [k for k in ("area", *DATE_FIELDS) if not meta.get(k)]
    if missing:
        raise ValueError(
            "missing case-study value(s): " + ", ".join(missing)
            + ". Pass them explicitly or pick a --preset "
            f"({', '.join(sorted(PRESETS))})."
        )
    for key in DATE_FIELDS:
        try:
            datetime.strptime(meta[key], "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError(f"{key} must be YYYY-MM-DD, got {meta[key]!r}") from exc

    # The rules requirement, enforced in one place: the OSM snapshot must be
    # pre-event. Post-event OSM edits as input is a disqualification.
    if meta["osm_snapshot_date"] >= meta["event_date"]:
        raise ValueError(
            f"OSM snapshot {meta['osm_snapshot_date']} is not before the event "
            f"{meta['event_date']} — using post-event OSM edits as input is "
            "prohibited by the hackathon rules."
        )
    meta["generated_at"] = datetime.now(tz=timezone.utc).isoformat()
    return meta


def write_run_meta(results_dir: Path | str, meta: dict) -> Path:
    """Write ``meta`` to ``<results_dir>/run_meta.json`` and return that path."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / RUN_META_FILENAME
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def load_run_meta(path_or_dir: Path | str) -> dict | None:
    """Read run_meta.json from an explicit file, a directory, or an artifact path.

    Accepting an artifact path is what lets a stage find the metadata from the
    ``--flood-mask`` it was already given, so no caller has to be told a second
    path. Returns None when there is no run_meta.json to read.
    """
    p = Path(path_or_dir)
    if p.is_dir():
        candidates = [p / RUN_META_FILENAME]
    elif p.name == RUN_META_FILENAME:
        candidates = [p]
    else:
        # An artifact path (e.g. flood_mask.tif). Read the metadata BESIDE it,
        # never the artifact itself: a raster is not UTF-8, and trying to parse
        # it here turned a valid run into a UnicodeDecodeError.
        candidates = [p.parent / RUN_META_FILENAME]
    for candidate in candidates:
        if candidate.is_file():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                print(f"⚠️ run_meta unreadable at {candidate}: {exc}", file=sys.stderr)
                return None
    return None


def run_meta_for_script(explicit: Path | str | None, artifact: Path | str | None) -> tuple[dict, Path | None]:
    """The run_meta a standalone stage should use.

    Order: an explicit ``--run-meta``, else the run_meta.json beside the stage's
    primary input, else the default preset's values (with a warning). Returns
    ``(meta, source_path)``; ``source_path`` is None for the preset fallback so a
    caller can say where the numbers came from.
    """
    if explicit is not None:
        meta = load_run_meta(explicit)
        if meta is None:
            raise FileNotFoundError(
                f"--run-meta {explicit} not found. Point it at the run_meta.json "
                "the pipeline wrote into results/."
            )
        return meta, Path(explicit)
    if artifact is not None:
        meta = load_run_meta(artifact)
        if meta is not None:
            return meta, Path(artifact)
    print(
        "⚠️ No run_meta.json found beside the input — falling back to the "
        f"'{DEFAULT_PRESET}' preset values in config/presets.py. Run the "
        "pipeline (or pass --run-meta) for values that match the actual data.",
        file=sys.stderr,
    )
    return resolve_run_meta(preset=DEFAULT_PRESET), None


def describe(meta: dict) -> str:
    """A one-line summary of a run_meta dict, for logs and report footers."""
    bbox = meta.get("bbox")
    box = (
        f"{bbox[0]:.4f},{bbox[1]:.4f},{bbox[2]:.4f},{bbox[3]:.4f}"
        if bbox else "unset"
    )
    confirmed = "confirmed" if meta.get("aoi_confirmed") else "UNCONFIRMED (fallback AOI)"
    return (
        f"{meta.get('area')} | bbox {box} [{meta.get('bbox_source')}, {confirmed}] | "
        f"before {meta.get('before_date')} / after {meta.get('after_date')} | "
        f"event {meta.get('event_date')} | OSM snapshot {meta.get('osm_snapshot_date')}"
    )


def add_run_meta_argument(parser) -> None:
    """Add the shared ``--run-meta`` flag to a stage's parser."""
    parser.add_argument(
        "--run-meta",
        type=Path,
        default=None,
        help=(
            "Path to the run_meta.json written by run_pipeline.py (area, bbox, "
            f"dates). Defaults to the {RUN_META_FILENAME} beside the primary input."
        ),
    )
