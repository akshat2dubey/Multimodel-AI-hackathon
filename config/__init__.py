"""Project configuration package.

``config/presets.py``  — paths + the case-study PRESETS registry (the one file to
                        edit for a different corridor/date).
``config/run_meta.py`` — resolves CLI overrides + preset into the run's metadata
                        (area, bbox, dates) and reads/writes results/run_meta.json.
"""

from .presets import (  # noqa: F401
    AFTER_DATE,
    AREA,
    BEFORE_DATE,
    CENTER_LAT,
    CENTER_LON,
    DATA_DEMO_DIR,
    DATA_DEMO_NAME,
    DATA_DIR,
    DEFAULT_EXTENT,
    DEFAULT_PRESET,
    DEMO_NOTICE_FILENAME,
    EVENT_DATE,
    FIXTURE_EXTENT,
    LOGS_DIR,
    MODELS_DIR,
    OSM_SNAPSHOT_DATE,
    PRESETS,
    PROJECT_ROOT,
    REPORT_DIR,
    RESULTS_DEMO_DIR,
    RESULTS_DEMO_NAME,
    RESULTS_DIR,
    RESULTS_NAME,
    TRISHULI,
    case_study,
    is_demo_dir,
)
from .run_meta import (  # noqa: F401
    RUN_META_FILENAME,
    RUN_META_SCHEMA,
    add_run_meta_argument,
    bbox_from_center,
    center_from_bbox,
    describe,
    load_run_meta,
    parse_bbox,
    preset_values,
    resolve_bbox,
    resolve_run_meta,
    run_meta_for_script,
    write_run_meta,
)
