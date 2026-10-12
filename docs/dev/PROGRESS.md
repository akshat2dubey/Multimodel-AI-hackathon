# Jalnetra — Execution Progress Tracker

**Repository**: `akshat2dubey/Multimodel-AI-hackathon`
**Plan**: `docs/dev/MASTER_PLAN_v2.txt` (extracted from JALNETRA_MASTER_EXECUTION_PLAN_v2.pdf)
**Audit date**: 2026-10-12

> **GIT POLICY — READ FIRST.** Do **not** push to the remote until the human explicitly
> says so. All work stays local (commits are fine, pushes are not). This was re-confirmed
> by the human on 2026-10-12.

---

## Phase 0 — GATE AUDIT (verified 2026-10-12)

Each row was checked against the code and by running the check, not by assuming.

| Old prompt | Task | Gate check | Verdict | Evidence |
|---|---|---|---|---|
| #1 | T07 | Sentinel-1 preprocessing bridge | **PARTIAL** | `tests/test_18_preprocess_s1.py` passes (mocked Process API → 4 GeoTIFFs + sidecar). On disk the 4 tifs exist and are pixel-aligned (640×640, EPSG:4326, identical bounds, float32) **but their bounds are 78.45–78.95°E / 28.05–28.55°N — the old, wrong AOI**, not the case-study corridor. No `s1_meta.json` / `S1_PROCESSED.json` sidecar exists on disk (only `PRODUCT_INFO.json`). |
| #2 | T03 | One results/data path + demo separation | **PASS (1 deviation)** | `tests/test_15_demo_separation.py` → 10 passed. `scripts/run_offline.py` ran end-to-end and refused to touch `results/`. Deviation: `make_demo_data.py` and `run_offline.py` still live in `scripts/`, not `demo/` as T03 specifies. |
| #3 | T04 | run_meta.json + zero hardcoded area/date/coords | **PARTIAL** | `tests/test_14_no_hardcoded_presets.py` + `tests/test_17_run_meta.py` → 13 passed; no banned coordinate/date literal appears in `scripts/` or `dashboard/` code. But the plan's stricter grep still finds `Trishuli`/`Bhote` in `scripts/make_demo_data.py`, `scripts/run_offline.py`, `scripts/run_pipeline.py` docstrings, and `28.3`/`85.7` in `dashboard/templates/demo.html`. |
| #4 | T08 | Change mask = newly flooded, not "after water" | **PASS** | `tests/test_02_change_mask.py` |
| #5 | T10 | Baseline fallback, no synthetic training, model contract | **PASS** | `tests/test_05_no_synthetic_training.py`; `run_pipeline.py` contains no training call (comments only) |
| #6 | T11 | Performance of damage + cut-off | **PASS** | `test_5000_features_on_a_3000x3000_mask_finish_within_budget` → **7.20 s** (budget 20 s) |
| #7 | T13 | Cut-off logic rewrite | **PASS** | `tests/test_08_cutoff_logic.py` |
| #8 | T12 | Empty-data crash fixes | **PASS** | `tests/test_07_empty_inputs.py` |
| #9 | T09 | Inference bug fixes | **PASS** | `tests/test_03_tiling_coverage.py`, `tests/test_04_pixel_area.py` |
| #10 | T16 | Area/date form + background job | **PASS** (fixed 2026-10-12) | `POST /api/runs` spawns `run_pipeline.py` in the background and answers **202 + run_id**; `GET /api/run-status` returns state/step/progress/redacted log tail; a second POST while running → **409**; `POST /api/cancel` stops the job. `tests/test_19_run_job.py` (9 tests). UI verified live: bbox form, draw-AOI-on-map, 2 s polling. |

### Extra gate checks (all cheap ones, done)

| Check | Verdict | Evidence |
|---|---|---|
| `pytest -q` is green | **PASS** | **82 passed in 69.66 s** (includes the `slow` marker) |
| One end-to-end run on toy/mocked input completes | **PASS** | `python scripts/run_offline.py` → exit 0, 5/5 stages OK, artifacts in `results_demo/` |
| No banned literal outside `config/` and `docs/` (shipped test) | **PASS** | `test_14` passes; stricter plan wording is PARTIAL (see T04) |
| No script silently creates synthetic data; `run_pipeline.py` never trains | **PASS** | generator is explicit, confined to `*_demo/`, banner printed; no training call in the orchestrator |
| Every script valid on empty inputs (exit 0) | **PASS** | `tests/test_07_empty_inputs.py` |
| T06 sub-check: no wildcard CORS | **PASS** (fixed 2026-10-12) | `CORS(app)` removed. Live probe: `Origin: http://evil.example` → **no** `Access-Control-Allow-Origin` on `/api/status`, `/api/cutoffs`, `/api/damage`, `/api/flood-mask`. Guarded by `test_api_grants_no_cross_origin_read_access`. |
| T06 sub-check: loopback bind + debug off | **PASS** | `test_server_binds_loopback_with_debug_off_by_default` passes |
| T18 sub-check: token on state-changing routes | **PARTIAL** | No/wrong token → 403 (correct, fails closed). But Origin/Host allowlist is absent and the token is env-only (`JALNETRA_PIPELINE_TOKEN`), not generated at startup and injected into the page as T18 requires. |

### Gate verdict summary

**8 PASS, 1 PARTIAL-with-fix-needed (T07), 2 PARTIAL (T03, T04), 1 OPEN (T18 Origin/token).**

Gate fixes completed 2026-10-12: **T06** (wildcard CORS removed, loopback/debug defaults) and
**T16** (background job + status route + single-job lock), both with tests and a live check.
Still open: **T18** (Origin/Host allowlist + startup-generated token), **T07** (re-run
preprocessing on the confirmed AOI once H1/H2 land), **T03/T04** (move the demo scripts to
`demo/`, strip the remaining case-study literals from docstrings and `demo.html`).

---

## Execution checklist (plan order — top to bottom)

### Phase 0 — Gate audit
- [x] **G0** Gate audit run and recorded above (2 tasks failed, 3 partial — fixes queued below)
- [ ] **T07** Sentinel-1 preprocessing bridge — finish: re-process on the confirmed AOI + write the sidecar
- [ ] **T03** One results/data path + demo separation — move `make_demo_data.py`/`run_offline.py` → `demo/`
- [ ] **T04** run_meta.json + zero hardcoded area/date/coords — strip remaining docstrings/`demo.html` literals

### Phase 1 — Remaining fixes
- [ ] **T01** Repo hygiene + secret scan
- [ ] **T02** Test scaffold (already present — extend only)
- [ ] **T14** Damage labels + accuracy
- [ ] **T15** Cut-off output = real GeoJSON, no path leak
- [ ] **T06** Secure defaults on dashboard (host/debug/CORS)
- [ ] **T18** Token / same-origin / no open CORS
- [ ] **T33** Live-run robustness (cache, retries, timings)
- [ ] **T31** AI evaluation (held-out, IoU/F1, baseline)
- [ ] **T32** EMSR927 comparison (check-only)

### Phase 2 — Standout features
- [ ] **T29** Flood-path validation + click start (bonus)
- [ ] **T23** Rescue triage + "repair this link"
- [ ] **T24** Confidence score + blind-spot map
- [ ] **T25** Debris vs water classes
- [ ] **T26** Grounded sitrep with number ledger
- [ ] **T27** Acquisition timeline "why this pair"
- [ ] **T28** Dashboard usability pack
- [ ] **T30** OPTIONAL: Sentinel-2 confirmation layer

### Phase 3 — Security
- [ ] **T05** Secrets management + redaction
- [ ] **T17** Input validation allowlist
- [ ] **T19** Safe errors + logging
- [ ] **T20** Rate limiting + resource limits
- [ ] **T21** XSS-safe rendering + security headers
- [ ] **T22** Safe file handling + payload size
- [ ] **T36** Dependency hygiene
- [ ] **T34** Provenance manifest + rule checks
- [ ] **T35** Safe deserialization / model supply chain
- [ ] **T37** CI security pipeline
- [ ] **T39** Tamper-evident audit log + kill switch + incident checklist
- [ ] **T38** Container sandbox (optional)

### Phase 4 — Docs & deliverables
- [ ] **T40** Attribution everywhere
- [ ] **T41** Reproducibility + cached case-study pack
- [ ] **T43** Make every document truthful
- [ ] **T42** One-page sitrep generic
- [ ] **T44** 6-page report + limitations + threat model
- [ ] **T45** Judge Q&A sheet
- [ ] **T46** 3-minute demo script

---

## Decisions & unverified items

- **H1 (Case-study bbox)**: **still pending human input.** The preset runs on the
  centre+extent fallback and stamps everything `aoi_confirmed: false`. Must not be guessed.
  Blocks: T04 finalisation, T07 re-run, T31, T32.
- **H2 (Copernicus credentials)**: `.secrets/cdse.env` exists locally (git-ignored). Not
  validated live in this audit.
- **H3 (Training data + GPU)**: pending. No trained model exists. `models/` holds only
  `flood_unet_smoke.pt` — a 2-epoch synthetic smoke checkpoint (Dice 0.44) that detects
  nothing on the real scene; it must never be presented as a result.
- **H4 (Nepali review)**: pending — ship English-only unless it arrives.
- **H5 (History rewrite / rotation)**: not needed yet; nothing to rewrite as far as this audit found.
- **Real-run state**: `results/` is stale relative to the committed code — it predates the
  `config/run_meta.py` refactor and contains no `run_meta.json`, `water_after.tif` or
  `valid_mask.tif`. `results/flood_mask_real.tif` is **all zeros** and
  `results/change_detection.json` reports 0.0 ha, so no real flood mask exists yet.
- **Synthetic-data notice conflict**: `data/osm/meta.json` is a stale synthetic leftover
  (says 96 buildings, `synthetic: true`) while the real snapshot metadata is
  `data/osm/osm_metadata.json` (3,755 buildings). Its `ohsome_total_check` is `0`.
- **T11 choice**: performance gate kept at the plan's 20 s budget; measured 7.20 s.
- **T16 design**: the job's state is mirrored to `results/run_status.json` (written atomically)
  so it survives a page reload and a crash leaves evidence; the process handle lives outside
  that dict so the JSON payload stays serialisable and path-free. `POST /api/run-pipeline` is
  kept as an alias of `/api/runs` so the existing UI and tests keep working.
- **T06 decision**: `JALNETRA_HOST` / `JALNETRA_DEBUG` are the documented knobs; the older
  `HOST` / `FLASK_DEBUG` names are still honoured so existing commands do not break.
