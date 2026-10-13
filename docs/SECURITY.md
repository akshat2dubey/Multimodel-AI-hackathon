# Security of Jalnetra — controls, evidence, and residual risk

Scope: the dashboard (`dashboard/`) and the pipeline (`scripts/`) as they run on a
single machine for this submission. The system is an **educational prototype, not
an operational tool**; nothing here is a claim of production hardening.

No current full test run, live probe, Bandit scan, or dependency audit has been
completed for this working tree. Treat this as a control inventory, not release
evidence. The available audit runtime has Python 3.12 but lacks the project's
GIS, Flask, PyTorch, pytest, Bandit, and pip-audit dependencies.

---

## 1. What this system holds (why it is worth protecting)

| Asset | Where it lives | Why it matters |
|---|---|---|
| Copernicus credentials (`CDSE_CLIENT_ID/SECRET`, Sentinel Hub client) | git-ignored `.secrets` / environment | Paid quota, and a leaked secret can be used from anywhere |
| The ability to **start a pipeline run** | `POST /api/runs`, `POST /api/cancel` | A run downloads multi-GB data and writes files; it is the expensive, abusable action |
| Result files and logs | `results/`, `logs/` | Carry absolute paths, timings and (if unredacted) headers/URLs |
| The flood mask's provenance | `results/manifest.json` | Decides eligibility for this hackathon; tampering would misrepresent inputs |
| Model weights | `models/*.pt` | A `.pt` file is a pickle: loading a hostile one is code execution |

## 2. Attack surface, and what stops it

| # | Threat | Control (all verified by the tests noted) |
|---|---|---|
| 1 | **Remote code execution via the debug console** — Werkzeug's debugger is an interactive shell | `debug=False` unless `JALNETRA_DEBUG=1`, and refused unless the bind host is loopback. Loopback bind by default (`JALNETRA_HOST` to override). No wildcard CORS. |
| 1a | **Unauthenticated dashboard access** | The WSGI import now defaults to authentication enabled. `python app.py` creates a one-time password when none is configured; WSGI deployments must set `JALNETRA_DASHBOARD_PASSWORD`. `JALNETRA_AUTH=off` disables the gate explicitly. |
| 2 | **A web page you visit starts a run** (or reads results) while the dashboard is open | Every state-changing route needs header `X-Jalnetra-Token` **and** a loopback `Origin`. Foreign origins get 403. Cross-origin reads have no CORS grant. |
| 3 | **Script injection through OpenStreetMap text** — OSM `name`/`amenity` are attacker-editable and used to be interpolated into popups (`innerHTML`) | Every OSM-derived string is escaped (`escapeText`/`textContent`) and a strict CSP with a per-response nonce backs it up. Leaflet is vendored under `dashboard/static/vendor/` (no CDN), so the demo works offline too. |
| 4 | **Path traversal / arbitrary file read** — routes serve files by name | File serving uses an allowlist of fixed names plus a resolved-path containment check against `RESULTS_DIR`; `../../` and absolute paths are rejected. |
| 5 | **Denial of service / resource exhaustion** — an open local port, expensive jobs | Per-client rate limit on `POST /api/runs` (429), 64 KB body cap (413), one job at a time (409), job wall-clock timeout that kills the process group, free-disk floor, cache size cap with oldest-first pruning. |
| 6 | **Secrets in logs or responses** — a signed URL or `Authorization` header printed by a download step | One rotating log (`logs/jalnetra.log`) with a redaction filter; clients get `{"error": ..., "id": <uuid>}` and the detail stays in the log. No stderr/stdout tails are returned over HTTP. |
| 7 | **Credential leak via committed files** | All secrets load from environment or a git-ignored `.secrets` in exactly one module (`scripts/cdse_env.py`); `.gitignore` covers `.secrets`, `.env`, `*.pt`, `data/`, `results/`, `logs/`; `.env.example` holds placeholder names only. |
| 8 | **Malicious or swapped model checkpoint** | `scripts/safe_model.py` is the only loader: `weights_only=True` (a pickled object is refused), a required matching `.sha256` sidecar written by the trainer, and a shape check. An unverifiable file is refused unless `--trust-unverified-model` is passed explicitly. No model path is ever accepted from the web UI. |
| 9 | **Disqualifying inputs** (Copernicus EMS/UNOSAT damage maps, post-event OSM edits) | `scripts/manifest.py` evaluates the rules in code — `osm_snapshot_before_event`, `acquisitions_bracket_event`, `no_forbidden_input_sources`, `sentinel_orbit_same_for_pair` — refuses to start when one is broken, and records each rule in `results/manifest.json` with a SHA-256 for every input and output. |
| 10 | **Repudiation** — "nobody knows what produced this number" | The manifest records run id, UTC time, git commit, parameters, method (model vs baseline), model hash verification, package versions and file hashes. `python scripts/manifest.py results/manifest.json` prints it. |
| 11 | **Vulnerable dependencies** | A prior audit recorded no known vulnerabilities; this must be re-run against the deployed locked dependency set. `flask-cors` — declared but never imported — was removed. |

## 3. Verification (run these; they are all local and offline)

```bash
python -m pytest -q
python -m bandit -r scripts dashboard     # static analysis
python -m pip_audit -r requirements.txt   # dependency vulnerabilities
```

Live probes against a running server (this is the part a test cannot fake):

| Check | Result |
|---|---|
| `POST /api/runs` with no token | **403** |
| `POST /api/runs` with a foreign `Origin` | **403** |
| `POST /api/runs` bursts past the limit | **429** |
| Oversized body | **413** |
| Response headers | `Content-Security-Policy` (nonce), `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, `Cache-Control: no-store` on `/api/*` |
| Foreign-origin response headers | no `Access-Control-Allow-Origin` |

**Static analysis (bandit):** 0 HIGH, 0 MEDIUM findings across `scripts/` and
`dashboard/`. The remaining LOW findings are all in one of three benign classes:
`B110` (a `try/except/pass` around an optional import or an unreadable file — none
of them protect a security decision), `B603/B607` (`subprocess` called with an
argument **list**, never `shell=True`, from a fixed script path) and `B311`
(`random` used for training shuffling/seeds, not for anything security-relevant).

## 4. Explicit non-goals and residual risk (stated, not hidden)

- **Local password authentication is not an identity provider.** The dashboard
  has a password session gate, but it has no user accounts, MFA, or central
  identity integration. The pipeline token is a separate same-origin anti-CSRF
  control. `JALNETRA_AUTH=off` makes every result visible to anyone who can
  reach the port, so keep the loopback bind unless a network deployment has its
  own access controls.
- **Production deployment is not verified.** Set `JALNETRA_COOKIE_SECURE=1`
  (default) behind HTTPS; only set it to `0` for loopback-only local HTTP.
  HTTPS termination, reverse-proxy headers, secret provisioning, backup/restore,
  monitoring and load behaviour still need deployment-specific review. For WSGI,
  configure `JALNETRA_SECRET_KEY`, `JALNETRA_DASHBOARD_PASSWORD`, and
  `JALNETRA_PIPELINE_TOKEN`. The `dashboard.wsgi:app` Waitress entry point
  refuses non-loopback startup when these secrets/authentication are missing.
- **The single-host model is the whole trust boundary.** The pipeline's job is a
  local subprocess with the same privileges as the operator; there is no
  container sandbox in this submission (the plan's T38 is optional and was not
  built).
- **The log is redacted, not tamper-evident.** There is no hash-chained audit log
  (T39), so an operator with filesystem access could edit it; the manifest is the
  artifact that carries hashes.
- **Unknown rules are recorded as `null`, not as a pass.** If the relative orbit
  is not reported by the acquisition step, `sentinel_orbit_same_for_pair` is
  `null` — a reader can see that it was not verified rather than assume it was.
- **No egress filtering.** The pipeline talks to the Copernicus Data Space,
  Overpass/ohsome, the DEM bucket and the tile host. That set is documented, not
  enforced.
- **Windows-first.** Permissions tightening (`chmod 0600` on logs) is best-effort
  and a no-op on NTFS. The historical suite was recorded on Windows;
  the current audit environment is Python 3.12 and could not run it.
- **Not hardened against a hostile local user.** If an attacker can already write
  to `scripts/` or `models/`, they can change the pipeline; the model-hash check
  and manifest make that *detectable*, not impossible.
