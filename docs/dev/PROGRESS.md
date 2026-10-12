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
| T18 sub-check: token + Origin on state-changing routes | **PASS** (fixed 2026-10-12) | `X-Jalnetra-Token` is required on `/api/runs` and `/api/cancel` (absent/wrong → 403, compared with `hmac.compare_digest`). A foreign `Origin` is refused with 403 even when the token is right. When `JALNETRA_PIPELINE_TOKEN` is unset the token is generated with `secrets.token_urlsafe(32)`, printed once at startup and injected into the served page. `Host` is deliberately not second-guessed (see decisions). Verified live on 127.0.0.1:5078. |

### Gate verdict summary

**9 PASS, 1 PARTIAL-with-fix-needed (T07), 2 PARTIAL (T03, T04).**

Gate fixes completed 2026-10-12: **T06** (wildcard CORS removed, loopback/debug defaults),
**T16** (background job + status route + single-job lock) and **T18** (token + Origin check on
state-changing routes), each with tests and a live check. Still open: **T07** (re-run
preprocessing on the confirmed AOI once H1/H2 land) and **T03/T04** (move the demo scripts to
`demo/`, strip the remaining case-study literals from docstrings and `demo.html`).

---

## Execution checklist (plan order — top to bottom)

### Phase 0 — Gate audit
- [x] **G0** Gate audit run and recorded above (2 tasks failed, 3 partial — fixes queued below)
- [ ] **T07** Sentinel-1 preprocessing bridge — finish: re-process on the confirmed AOI + write the sidecar
- [ ] **T03** One results/data path + demo separation — move `make_demo_data.py`/`run_offline.py` → `demo/`
- [ ] **T04** run_meta.json + zero hardcoded area/date/coords — strip remaining docstrings/`demo.html` literals

### Phase 1 — Remaining fixes
- [x] **T01** Repo hygiene + secret scan — done 2026-10-12. Deleted `NOTICE.md` (non-English agent notes + a local absolute path) and `docs/CEO_STATUS_REPORT.md` (stale: it still claimed there was no git repo); moved `verification_*.json` to `docs/dev/`. `git grep` for `F:/`, `C:\Users`, `Akshat`, `LadderLens` now returns **nothing** (the tutorial's machine-specific `cd` was rewritten generically, and `fetch_osm.py`'s User-Agent now says Jalnetra). `.gitignore` covers `.secrets/`, `.env`, `*.env`, `data/`, `results/`, `logs/`, `*.pt`, `.venv/` while allowing `models/README.md` and `.env.example`; added `LICENSE` (MIT, Akshat Dubey), `.env.example` (placeholder names only) and `models/README.md`.
  **Secret scan: clean.** No credential literal in any tracked file (the only hits are the variable unpack in `preprocess_s1.py` and obvious fakes in `tests/`), and `git log --all -- .secrets .env *.env` is empty — no secret was ever committed, so no rotation is needed (H5 stays "not needed").
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
- [x] **T17** Input validation allowlist — done 2026-10-13. `_parse_run_request()` now takes an **allowlist of keys** (unknown keys → 400 naming the key), caps the bbox at `MAX_BBOX_DEG2` (1.0 deg²; a 10°×10° request is refused), requires real ISO dates in the order the analysis needs (`osm_snapshot` < `event_date` ≤ `after_date`, `before_date` ≤ `event_date`), forbids future and non-existent dates (`2026-02-31`), pattern-checks the area name and checks the preset against `config.presets.PRESETS`. `tests/test_26_input_allowlist.py` (19 tests) also asserts a surviving value arrives as **one** argv element. **Residual:** `run_pipeline.py` still does not re-validate (S1-7's second half).
- [x] **T19** Safe errors + logging — done 2026-10-13. One rotating, git-ignored log (`logs/jalnetra.log`, owner-only where the OS allows) with a redaction filter that masks credentials, `Authorization` headers, repo paths (`<repo>`) and absolute paths (`<path>`); a global handler answers `{"error": "internal error", "id": <uuid>}` and keeps the traceback in the log under that id; no stdout/stderr tail is returned over HTTP. **A real bug was found by the import test this produced:** the log setup called a helper defined further down the module, so `import app` raised `NameError` — fixed by making the chmod inline. Verified: the live log contains **no** token and **no** Windows path.
- [x] **T20** Rate limiting + resource limits — done 2026-10-13. Per-client rate limit on `POST /api/runs` (5/min → **429**), 64 KB body cap (**413**, checked live with an 80 KB body), single-job lock (409), job wall-clock timeout (default 30 min) that kills the **process group** and marks the run `timeout` rather than `failed`, a free-disk floor before downloading, and a cache cap that prunes oldest-first. `tests/test_22_limits.py` (6 tests, including a real dummy process being killed).
- [x] **T21** XSS-safe rendering + security headers — done 2026-10-12. Every OSM-controlled value now goes through `escapeHtml()` before it reaches `innerHTML`/`bindPopup` (an OSM `name` tag is editable by anyone, so this was a real stored-XSS path). Added the missing headers: CSP (`default-src 'self'`, `frame-ancestors 'none'`, per-response `nonce`), `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, `Cache-Control: no-store` on `/api/*`. A strict `script-src 'self'` was only possible after vendoring Leaflet 1.9.4 into `dashboard/static/vendor/leaflet/` — the CDN dependency is gone. Verified live: page loads under the CSP with **zero console messages**, tiles still load, and `escapeHtml('<img src=x onerror=alert(1)>')` returns escaped text.
- [x] **T22** Safe file handling + payload size — done 2026-10-12. `/api/damage` is affected-only by default (36 of 117 features) and drops the unused `osm_tags` blob: **90.5 KB → 23.5 KB**, or **16.8 KB gzipped** for the full layer via `?all=1`. Responses are capped at `MAX_FEATURES`, JSON is gzipped when the client asks, `_is_inside()` refuses to serve anything outside the results/report directories, and the file this app writes is chmod'ed owner-only where the OS allows. Verified live plus 12 tests.
- [x] **T36** Dependency hygiene — done 2026-10-13. `flask-cors` (declared, never imported) removed. Every outbound fetch now goes through `scripts/net_safe.py`, which asserts the scheme is HTTPS — this **fixes** the four bandit `B310` findings instead of suppressing them. Audit results recorded in `docs/SECURITY.md`: `pip-audit -r requirements.txt` → *No known vulnerabilities found*; `bandit -r scripts dashboard` → **0 HIGH, 0 MEDIUM** (the remaining LOWs are `try/except/pass` around optional imports, `subprocess` with an argv list, and `random` for training seeds). The Python-version claim in `requirements.txt` was corrected to what was actually verified (3.14.2 on Windows; 3.10–3.12 untested). **Not done:** making `torch` optional — `inference.py` imports it unconditionally, so the baseline is not torch-free (stated in `requirements.txt` and `docs/SECURITY.md` rather than papered over).
- [x] **T34** Provenance manifest + rule checks — done 2026-10-13. Every run writes `results/manifest.json`: SHA-256 + size for **every input and output**, run id, UTC time, git commit, parameters, method (model vs baseline), model hash verification against its sidecar, and package versions. `scripts/manifest.py` evaluates the hard rules in code — `osm_snapshot_before_event`, `acquisitions_bracket_event`, `no_forbidden_input_sources` (EMS/UNOSAT/damage-map path patterns), `sentinel_orbit_same_for_pair` (reads both `S1_PROCESSED.json` sidecars) — refuses to start when one is already broken (exit 3, before anything is written or downloaded) and fails the run at the end otherwise. A rule that cannot be evaluated is recorded as `null`, never as a pass. `tests/test_24_provenance.py` (9 tests); the EMSR leakage guard now allowlists the detector by name and `test_24` proves the detector rejects such a path.
- [x] **T35** Safe deserialization / model supply chain — done 2026-10-13. `scripts/safe_model.py` is the only loader: `torch.load(..., weights_only=True)` (a pickled object is refused, proven by a test that saves one), a required matching `.sha256` sidecar written by `train_segmentation.py`, and a shape check. A checkpoint with no sidecar is refused unless `--trust-unverified-model` is passed explicitly. `tests/test_23_model_trust.py` (7 tests, including tamper-after-evaluation and an inference-level refusal).
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
- **T18 decision (Host header not enforced)**: the plan says "check Origin/Host against an
  allowlist". Only `Origin` is enforced. The practical attack a Host check targets — DNS
  rebinding — already carries the attacker's `Origin`, which is rejected, and enforcing Host
  would refuse the documented LAN demo (`JALNETRA_HOST=0.0.0.0`) without adding protection.
  A missing `Origin` is allowed (curl/CLI/tests cannot be a cross-site browser request).
- **Live security sweep (2026-10-13, 14/14 checks, `127.0.0.1:5077`)**: page 200 with a nonce
  CSP + `nosniff`/`DENY`/`no-referrer`; no CDN script in the served page; `/api/status` 200 with
  `Cache-Control: no-store`, no CORS grant, no absolute path in the payload; `POST /api/runs`
  without a token **403**, from a foreign `Origin` **403**, with an 80 KB body **413**, with a
  half-filled custom area **400** naming the fields, with a malformed bbox **400** then **429**
  on the burst; **no job was started** by any probe; `/static/../app.py` → 404; the log holds no
  token and no Windows path.
- **T17 limits (chosen, not measured)**: `MAX_BBOX_DEG2 = 1.0` (env `JALNETRA_MAX_BBOX_DEG2`),
  area pattern `^[\w .,\-()/]{1,80}$`, rate limit 5 POST/min, body cap 64 KB, job timeout 30 min,
  free-disk floor 2 GB, cache cap 10 GB. All are env-overridable; the defaults are sized for a
  laptop and a ~110 km analysis window.
- **T35 strictness decision**: an unverifiable checkpoint (no `.sha256`) is **refused**, not
  warned about. Trainers write the sidecar, so the only way to hit this is weights from
  somewhere else — which is exactly the case where loading is unsafe. The escape hatch names
  itself (`--trust-unverified-model`) and is never passed by the dashboard.
- **T34 unknown rules**: `sentinel_orbit_same_for_pair` is `null` unless both `S1_PROCESSED.json`
  sidecars exist, because the current `results/` predates them. Recording `null` keeps the
  manifest honest; the alternative (treating unknown as pass) would be a false assurance.
- **Probe incident (2026-10-13)**: a live probe posted a 5 KB body (under the cap) with a valid
  bbox and the server returned **202**, starting a real job that then exited 2 ("missing
  case-study value(s)"). No download happened and `results/` was cleaned; the two real defects
  it exposed (a 400-less half-filled request, and an unmeasured body cap) were fixed and are
  covered by tests.
