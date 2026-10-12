# Jalnetra — Full Audit (frontend ↔ backend, security, features)

**Date:** 2026-10-12 · **Audited against:** working tree + live dashboard on `127.0.0.1:5078`
**Method:** every claim below was produced by running something (HTTP probe, `git`, `grep` on the
current source, or reading the file) — not by reading the plan and assuming. Where an artifact is
stale, that is called out, because the plan's problem statements were written against older code.

---

## 1. Push state

| Ref | Local | Remote | Verdict |
|---|---|---|---|
| `cutoff-damage-refactor` | `2a668c6` | `2a668c6` | ✅ pushed |
| `main` | `2b2f629` (stale locally) | `2a668c6` | ✅ pushed by the human; local ref just needs a `fetch` |
| commits not on the remote | — | — | ✅ none (`git log origin/cutoff-damage-refactor..HEAD` is empty) |

Published this session: `T06` (99b1e12), `T16` (21a2ce0), `T18` (32ed0aa), `T01` (2a668c6).

**Still local-only (not mine, never committed):** ` M scripts/cutoff_analysis.py`,
` M scripts/damage_assessment.py`, ` M scripts/preprocess_s1.py`, ` M scripts/train_segmentation.py`,
`?? tests/test_18_preprocess_s1.py`, plus scratch files (`debug_save_failure.py`,
`pdf_extract.txt`, `pdf_clean.txt`, `jalnetra_rename.sh`).

---

## 2. Backend ↔ frontend contract — **everything the UI calls works**

All routes probed live against a running server.

| Route | Status | Shape returned | Consumed by the UI? | Verdict |
|---|---|---|---|---|
| `/` | 200 | HTML 32.7 KB | — | ✅ renders, run panel present |
| `/api/status` | 200 | json, 16 keys (`area, bbox, bbox_source, aoi_confirmed, flood_stats, damage_summary, cutoff_summary, demo_data, data_source, before/after/event_date, center_lat/lon`) | `updateStats()`, `applyRunBbox()` | ✅ keys match |
| `/api/flood-mask` | 200 | FeatureCollection, 14.5 KB | `addFloodLayer()` reads `.features` | ✅ |
| `/api/damage` | 200 | FeatureCollection + metadata, 90 KB, **117 features** | `addDamageLayers()` reads `.features` | ⚠️ works, but returns **all** features incl. 81 `UNAFFECTED` (T22) |
| `/api/cutoffs` | 200 | `{cutoff_settlements, cutoff_villages, flooded_roads, summary, village_count, note}` | `addCutoffLayer()` reads `cutoff_villages` → falls back to `cutoff_settlements` | ✅ |
| `/api/flood-path` | 200 | FeatureCollection + metadata, 343 KB | `addFloodPathLayer()` reads `.features` | ✅ |
| `/api/situation-report` | 200 | HTML 13.9 KB | **nothing** — the button opens `/report` instead | ⚠️ dead endpoint |
| `/report` | 200 | HTML 10.7 KB | "View Situation Report" button | ✅ |
| `/demo` | 200 | HTML 7.1 KB | "Demo Mode" button | ✅ |
| `/api/run-status` | 200 | `{state, step, progress, run_id, started_at, log_tail}` | `pollRunStatus()` every 2 s | ✅ (T16, built this session) |
| `POST /api/runs` | **403** without token | — | `startRun()` | ✅ gated (T18) |
| `POST /api/cancel` | **403** without token | — | cancel button | ✅ gated |
| `POST /api/run-pipeline` | **403** without token | — | — | ✅ legacy alias |

**No frontend call is broken.** Every URL the templates fetch exists and answers. Empty-results
behaviour (the judge's first load, before any run) is covered by `tests/test_12_dashboard_contract.py`
— all 9 GET routes return 200 on an empty results dir.

### The one systemic frontend problem: the served artifacts are **stale**

`results/` was written on **Oct 8**, before the `config/run_meta.py` refactor. It contains no
`run_meta.json`, `water_after.tif` or `valid_mask.tif`, and `/api/status` falls back to the preset
because there is no run metadata to read. So the dashboard currently renders numbers that the
current code would not produce — with the synthetic banner correctly showing. **A fresh real run is
what makes the UI honest**, not more UI work.

---

## 3. What is broken / what needs fixing

| # | Problem | Evidence | Plan task |
|---|---|---|---|
| 1 | **The cut-off artifact on disk is not GeoJSON and leaks absolute local paths** — `analysis_timestamp` is the CWD, `flood_mask` is an absolute path | `results/cutoff_settlements.geojson` → `analysis_timestamp = "F:\\Akshat\\edc ideas\\LadderLens\\MultiModel\\hackathon-trackb"`, `flood_mask = "F:\\...\\results\\flood_mask.tif"`, no `type` key | ⚠️ **stale artifact, code already fixed** — current `cutoff_analysis.py:873-879` writes `"type": "FeatureCollection"`, an ISO-8601 UTC timestamp and the relative name `flood_mask.tif`. Re-running fixes it; T15 needs no code change |
| 2 | **Damage labels overclaim and bridges are judged on the wrong pixels** | `damage_assessment.py:268-359` still emits `DESTROYED` / `SEVERELY_DAMAGED`; no `approach`/`abutment`/`deck` logic exists anywhere in the file | **T14 (open)** |
| 3 | **No shared flood-fraction threshold** | `grep FLOOD_FRACTION_THRESHOLD` over `config/`, `damage_assessment.py`, `cutoff_analysis.py` → no match; damage and cut-off each keep their own number | **T14 (open)** |
| 4 | **`/api/damage` ships everything** — 90 KB, 69 % of it `UNAFFECTED` | live response: 117 features, 81 `UNAFFECTED` | **T22 (open)** |
| 5 | **No security headers at all** | live `GET /` response headers: `Server, Date, Content-Type, Content-Length, Connection` — no CSP, `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`, no `no-store` on `/api/*` | **T21 (open)** |
| 6 | **Stored-XSS surface via OSM tags** — village names and feature names go into `innerHTML`/`bindPopup` unescaped | `index.html:627,649,666-668,679,703` (`${v.name}`, `${f.properties.name}`); OSM `name`/`amenity` are attacker-editable | **T21 (open)** |
| 7 | **`torch.load` without `weights_only=True`** — a hostile checkpoint is code execution | `inference.py:87`, `train_segmentation.py:789` | **T35 (open)** |
| 8 | **Unused dependency still pinned** | `requirements.txt:27 flask-cors>=6.0` while `dashboard/app.py` no longer imports it (removed in T06) | **T36 (open)** |
| 9 | **No logging, no provenance manifest, no rate limit** | no `import logging` anywhere in `scripts/` or `dashboard/`; no `results/manifest.json`; no `MAX_CONTENT_LENGTH` or rate limit | **T19, T34, T20 (open)** |
| 10 | **Nothing prevents a forbidden input from being used** | no manifest `rules_check`, no leakage test against `docs/EMSR927*` | **T34, T32 (open)** |
| 11 | `main` is stale locally (cosmetic) | local `main 2b2f629`, `origin/main 2a668c6` | `git fetch` |

---

## 4. Security status

| Task | Requirement | State | Evidence |
|---|---|---|---|
| T06 | loopback bind, debug off, no wildcard CORS | ✅ **done** | live: binds `127.0.0.1`, `debug=False`, no `Access-Control-Allow-Origin` for a foreign Origin |
| T18 | token + Origin on state-changing routes | ✅ **done** | 403 for absent/wrong token and for a foreign Origin; live-tested |
| T39 (part) | kill switch | ✅ **done** | `POST /api/cancel`, token-gated, kills the job |
| T05 | secrets loaded in one place | 🟡 partial | `cdse_env.py` + `.secrets/cdse.env` exist, but `redact()` only lives in the dashboard — the pipeline scripts have no central redaction |
| T17 | shared input validation allowlist | 🟡 partial | `_parse_run_request()` guards the dashboard (bbox/date/length); `run_pipeline.py` validates nothing of its own, so a CLI user bypasses it |
| T19 | structured errors + rotating log | ❌ missing | no logging module anywhere |
| T20 | rate limit, body cap, job timeout | ❌ missing | none found; only the single-job lock exists |
| T21 | XSS-safe rendering + headers | ❌ missing | see §3 rows 5–6 |
| T22 | safe file handling, compact JSON | ❌ missing | see §3 row 4 |
| T34 | provenance manifest + rule checks | ❌ missing | no `manifest.json` |
| T35 | safe deserialization | ❌ missing | see §3 row 7 |
| T36 | dependency hygiene | ❌ missing | see §3 row 8 |
| T37 | CI pipeline | ❌ missing | no `.github/workflows/` |
| T39 | tamper-evident audit log | ❌ missing | no `logs/audit.jsonl`, no `verify_audit.py` |

**Answer to "is the main security feature implemented?"** The two that stop *remote* misuse are done
and verified: the dashboard is loopback-only with the debugger off, and every state-changing route
needs a token **and** a loopback Origin. That was the RCE + cross-site-read exposure the status
report flagged, and it is closed. What is **not** done is the second layer: security headers, XSS-safe
rendering, input validation at the pipeline boundary, safe deserialization, and the provenance/audit
trail that also protects against *disqualification* (T34). So: **exposure closed, hardening open.**

---

## 5. Feature inventory

**Present and working:** the 5-stage pipeline; the flood mask (`flood_mask.tif`, newly-flooded only,
with `water_after`/`valid_mask`); damage overlay; cut-off reachability with a pre-event baseline and
`no_reference_facility` handling (T13 — genuinely done, `cutoff_analysis.py:724-746`); the DEM
flood-path bonus; the situation report; the Flask/Leaflet dashboard including the run panel built
this session.

**Not implemented (all nine plan scripts are absent):**

| Feature | Script | Status |
|---|---|---|
| T23 rescue triage + "repair this link" | `triage.py` | missing |
| T24 confidence score + blind-spot map | `confidence.py` | missing |
| T25 debris vs water classes | `flood_classes.tif` | missing |
| T26 grounded sitrep with number ledger | `number_ledger.py` | missing |
| T27 acquisition timeline "why this pair" | `list_acquisitions.py` | missing |
| T28 dashboard usability pack | — | missing |
| T29 flood-path validation + click start | — | partial (flood path exists; validation/click-start not) |
| T30 Sentinel-2 confirmation layer | — | missing |
| T31 held-out AI evaluation | `evaluate.py` | missing |
| T32 EMSR927 comparison (check-only) | `compare_emsr927.py` | missing |
| T17 shared validator | `validation.py` | missing |
| T10 named baseline module | `baseline_change.py` | missing as a module, but the baseline **exists** inside `inference.py --baseline` and the gate test passes |

---

## 6. Priority order from this audit

1. **Re-run the real pipeline into `results/`** — it is blocked on the AOI (H1) and a model (H3),
   but a baseline run with the current code would immediately fix the stale-artifact problems
   (#1) and make the dashboard honest.
2. **T14** — damage labels + bridge approaches + one shared threshold (30 % of the rubric).
3. **T31** — held-out evaluation, so the AI claim has a number behind it (25 %).
4. **T21 + T22 + T19** — headers, escaping, affected-only responses, structured errors.
5. **T34** — provenance manifest with rule checks; it protects against disqualification.
6. **T20/T35/T36/T37** — the remaining hardening, cheap and quick.
7. **Phase 2 features** (T23/T24/T26 are the highest-value three) only after 1–5.
