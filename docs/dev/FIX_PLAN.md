# Jalnetra — Fix Register, ranked worst → least

Severity is judged by what actually loses the submission: **showing wrong data**, **being
exploitable**, **disqualification**, then **missing rubric points**, then polish.
Every row cites the evidence from `docs/dev/AUDIT.md`. Plan task ids are from the master plan.

**GIT POLICY: local commits only. Nothing is pushed without the human saying so.**

---

## S0 — Critical: the product shows data it did not compute

| # | Fix | Evidence | Task | Blocked on |
|---|---|---|---|---|
| **S0-1** | `results/` is stale (Oct 8, pre-`run_meta`) and `flood_mask_real.tif` is all zeros, so every headline number the dashboard and report render is old synthetic data. Needs a real run with the current code. | `results/` has no `run_meta.json`/`water_after.tif`/`valid_mask.tif`; `/api/status` falls back to the preset; raster min/max = 0/0 | T03/T04/T08/T10 | **H1 (AOI), H3 (model)** — a baseline run unblocks it today |

## S1 — Exploitable security holes

| # | Fix | Evidence | Task | Status |
|---|---|---|---|---|
| **S1-1** | **Stored XSS**: OSM `name`/`amenity` flow into `innerHTML`/`bindPopup` unescaped at 5 sites → arbitrary script in a judge's browser when they click a village. | `index.html:627,649,666-668,679,703` | **T21** | ✅ closed (escape helper + strict nonce CSP, Leaflet vendored) |
| **S1-2** | **No security headers at all** — no CSP, `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`; `/api/*` is cacheable. | live response headers on `GET /` | **T21** | ✅ closed (live-verified) |
| **S1-3** | **`torch.load` without `weights_only=True`** — a hostile checkpoint is arbitrary code execution. | `inference.py:87`, `train_segmentation.py:789` | **T35** | ✅ closed (`safe_model.py`: `weights_only` + `.sha256` sidecar) |
| **S1-4** | **`/api/damage` ships the whole layer** (90 KB, 69 % `UNAFFECTED`) and no file route uses a path allowlist — a payload and information-disclosure problem. | live response: 117 features, 81 `UNAFFECTED` | **T22** | ✅ closed (90.5 → 23.5 KB, 16.8 KB gzipped, path containment) |
| **S1-5** | **No safe errors / logging** — no rotating log, no error ids, so an exception path can leak internals with no record. | no `import logging` in `scripts/` or `dashboard/` | **T19** | ✅ closed (redacted rotating log + error ids); found/fixed an import-time `NameError` |
| **S1-6** | **No rate limit, no body cap, no job timeout** — the run route can be hammered or left running forever. | no `MAX_CONTENT_LENGTH`, no rate dict; only the single-job lock | **T20** | ✅ closed (429 / 413 / timeout / disk + cache caps) |
| **S1-7** | **Validation only at the dashboard** — `run_pipeline.py` validates nothing, so the CLI (the documented way to run) bypasses the guard. | `_parse_run_request()` in `app.py`; no validator in `run_pipeline.py` | **T17** | ⚠️ **half closed** — the request allowlist is in; the CLI path still does not re-validate |

> **Security sufficiency answer (after S1):** the *exposure* class is closed and live-verified —
> loopback bind, debug off, no wildcard CORS, token + loopback-Origin on every state-changing
> route, kill switch, and now the hardening layer above: escaping + CSP, bounded requests and
> jobs, redacted logs, HTTPS-only fetches, safe model loading, hashed provenance.
> Verdict for a **loopback, single-operator prototype: sufficient**, with four named residual
> gaps: (1) the CLI does not re-validate input (S1-7 half), (2) no hash-chained audit log (T39),
> (3) no CI running these checks on push (T37), (4) no container sandbox and no egress filtering
> (T38). Audit evidence: `pip-audit` clean, bandit 0 HIGH/0 MEDIUM, 14/14 live checks — all
> recorded in `docs/SECURITY.md`.

## S2 — Disqualification risk (rules compliance)

| # | Fix | Evidence | Task |
|---|---|---|---|
| **S2-1** | No provenance manifest and no coded `rules_check`, so nothing proves EMSR927 / post-event OSM were never used as input. | no `results/manifest.json` | **T34** | ✅ closed (hashes + 4 rule checks, refusal before download) |
| **S2-2** | No leakage test forbidding `scripts/`/`dashboard/` from touching `docs/EMSR927*`, and no EMSR927 comparison at all. | no `compare_emsr927.py`; no such test | **T32** |
| **S2-3** | Attribution strings are not verified in the sitrep/footer by a test. | no attribution test | **T40** |

## S3 — Correctness of the headline mapping (30 % + 25 % of the score)

| # | Fix | Evidence | Task |
|---|---|---|---|
| **S3-1** | Damage still emits `DESTROYED` / `SEVERELY_DAMAGED` from SAR (cannot see structural destruction — an honesty violation), bridges are still judged on deck pixels over the river, and damage/cut-off use **different** flood thresholds. | `damage_assessment.py:268-359`; no `approach`/`abutment`/`deck` logic; no `FLOOD_FRACTION_THRESHOLD` anywhere | **T14** |
| **S3-2** | No held-out evaluation, so the AI component (25 %) has no IoU/F1 number and no baseline comparison. | no `evaluate.py`, no `eval_metrics.json` | **T31** |
| **S3-3** | No debris class even though the brief asks for "flooded **and debris-covered** areas". | no `flood_classes.tif` / `debris_mask.tif` | **T25** |
| **S3-4** | No confidence score or blind-spot map, so the system cannot say where it cannot see (steep terrain, radar shadow, nodata). | no `confidence.py` / `blind_spots.tif` | **T24** |

## S4 — Missing standout features (the differentiators)

| # | Fix | Task |
|---|---|---|
| **S4-1** | Rescue triage + "repair this link first" ranked repair list (cut-off is 20 %). | **T23** |
| **S4-2** | Grounded sitrep with a number ledger, so every printed figure traces to a file. | **T26** |
| **S4-3** | Acquisition timeline "why this pair" (same-orbit justification, live). | **T27** |
| **S4-4** | Dashboard usability pack: before/after swipe, layer toggles, exports, ranked table. | **T28** |
| **S4-5** | Flood-path validation + click-to-start from the map (bonus). | **T29** |
| **S4-6** | Optional Sentinel-2 confirmation layer. | **T30** |

## S5 — Robustness and engineering hygiene

| # | Fix | Task |
|---|---|---|
| **S5-1** | Download cache keyed by bbox+date, Overpass retries/backoff, `--fast` mode, per-step timings (live judging safety). | **T33** |
| **S5-2** | Pin dependencies, drop the now-unused `flask-cors`, make `torch` optional so the baseline runs without it. | **T36** |
| **S5-3** | CI: pytest + bandit + pip-audit + gitleaks on push, README badge. | **T37** |
| **S5-4** | Tamper-evident `logs/audit.jsonl` + `verify_audit.py` (kill switch already done). | **T39** |
| **S5-5** | Move the demo scripts to `demo/` and strip the last case-study literals (gate partials T03/T04). | T03, T04 |
| **S5-6** | Container sandbox (optional). | **T38** |

## S6 — Deliverables

T42 one-page sitrep (single page, generic) · T43 make every document truthful · T44 six-page report
with limitations + threat model · T45 judge Q&A · T46 three-minute demo script · T41 reproducibility
package + cached case-study result pack.

---

## Execution order I am following (worst first, unblocked work only)

1. ~~**S1-1/S1-2 → T21** XSS escaping + security headers (+ vendor Leaflet so a strict CSP works)~~ ✅ **done**
2. ~~**S1-4 → T22** affected-only responses, path allowlist, compact JSON~~ ✅ **done**
3. ~~**S1-5 → T19** logging + safe error handlers~~ ✅ **done**
4. ~~**S1-6 → T20** rate limit, body cap, job timeout~~ ✅ **done**
5. ~~**S1-3 → T35** `weights_only=True` + checkpoint hash~~ ✅ **done**
6. ~~**S1-7 → T17** request allowlist~~ ✅ **done** (CLI re-validation still open)
7. ~~**S2-1 → T34** provenance manifest + rule checks~~ ✅ **done**
8. ~~**S5-2 → T36** dependency hygiene~~ ✅ **done** (`torch` optionality declined, stated)
9. **S3-1 → T14** damage labels, bridge approaches, one shared threshold  ← **next**
10. **S3-2 → T31** held-out evaluation  ← needs H3 (training data + GPU)
11. **S5-3 → T37** CI so these checks run on every push

S0-1 stays blocked on H1/H3 and is called out in every report until it is unblocked:
`results/` still holds the Oct-8 artifacts, so the dashboard keeps rendering numbers the current
code would not produce. Nothing in the security block changes that.
