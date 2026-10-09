# FloodMapper — Complete Tutorial, Status & Security Baseline

**Project:** Bhote Koshi–Trishuli flood mapping pipeline (Track B hackathon entry)
**Last audited:** 2026‑10‑09
**Audit basis:** filesystem inspection, not memory. Every number below was read off disk.

---

## 0. The 30‑second version

You have a **working, end‑to‑end flood‑mapping pipeline** for a Himalayan river corridor.
It downloads a satellite digital elevation model, traces where water would flow, overlays
OpenStreetMap infrastructure, and serves the results through a live web dashboard.

**Two outputs are real. The rest are labelled synthetic placeholders** that exist so the
code, the dashboard and the report could all be built and tested before the satellite
imagery arrives. The whole system is honest about which is which — there is a
`DEMO_DATA_NOTICE.txt` and every synthetic artifact carries `"synthetic": true` in its own
metadata.

**What it is waiting on:** a free Copernicus account. That single 15‑minute signup is the
gate on the entire remote‑sensing half.

---

## 1. Run it right now (the tutorial proper)

### 1.1 Open the folder

Your shell is at `C:\Users\Admin`. The project is on `F:`. PowerShell needs quotes around
the path because it contains spaces:

```powershell
cd "F:/Akshat/edc ideas/LadderLens/MultiModel/hackathon-trackb"
```

That was the exact error in your terminal — unquoted spaces. Forward slashes work fine in
PowerShell.

### 1.2 Check Python & dependencies

```powershell
python --version
python -c "import flask, rasterio, numpy, geopandas; print('deps OK')"
```

`requirements.txt` covers: Flask, rasterio, numpy, geopandas, shapely, scipy, torch.
If something is missing: `pip install -r requirements.txt`.

### 1.3 Start the dashboard (no internet, no account needed)

```powershell
python dashboard/app.py
```

Then open **http://localhost:5000**.

There are **10 routes**. All 9 GET routes return HTTP 200; the tenth is POST‑only.
Verified against a live server on 2026‑10‑09:

| Route | What it shows | Status | Size / time |
|---|---|---|---|
| `/` | Map UI: flood layer, roads, buildings, bridges, flood path | mixed | 23,363 B / 0.04 s |
| `/api/status` | Headline stats: flood, damage, cut‑offs | synthetic | 1,029 B / 0.04 s |
| `/api/flood-mask` | Flood extent polygons | synthetic | 14,530 B / 0.01 s |
| `/api/damage` | Damaged buildings / bridges | synthetic | 90,360 B / 0.01 s |
| `/api/cutoffs` | Cut‑off settlements | synthetic | 40,254 B / 0.00 s |
| `/api/flood-path` | The real DEM flow trace | **REAL** | 343,114 B / 0.04 s |
| `/api/situation-report` | Situation report as JSON | synthetic | 13,946 B / 0.08 s |
| `/report` | Rendered situation report | synthetic | 10,716 B / 0.01 s |
| `/demo` | Static sample view | synthetic | 7,187 B / 0.03 s |
| `/api/run-pipeline` | POST only (GET → **405**) — re‑runs the pipeline | — | 153 B |

> **`/api/change-detection` does not exist** — it returns **404**. Change‑detection
> figures are nested inside `/api/status` under `flood_stats` instead. Earlier notes
> listing that endpoint were wrong; corrected here after testing.

### 1.4 In PowerShell, test a route without a browser

```powershell
Invoke-RestMethod http://localhost:5000/api/flood-path | Select-Object -First 1
```

(`Invoke-RestMethod` is the PowerShell way; `curl` on Windows PowerShell is an alias for
`Invoke-WebRequest` and behaves differently — that trips people up.)

### 1.5 Re‑run the pipeline from the terminal

```powershell
python scripts/run_pipeline.py `
  --area "Bhote Koshi - Trishuli Corridor" `
  --center-lat 28.3 `
  --center-lon 85.7 `
  --before-date 2026-08-14 `
  --after-date 2026-08-28 `
  --osm-snapshot 2026-07-27 `
  --skip-training
```

> **Note:** the longitude must be **85.7**, not 78.7. 78.7°E is the Indo‑Gangetic plain
> (~150–250 m elevation, no glacier, no valley) and was an early documentation error that
> has been corrected everywhere. The dashboard's own pipeline trigger was still using
> 78.7 and was fixed during this audit.

### 1.6 Regenerate just the flood path from the real DEM

```powershell
python scripts/flood_path.py --pit-radius 3 --max-breaches 5000
```

Use `--pit-radius 0` to get the original strict‑D8 behaviour (53‑pixel trace, for
comparison).

---

## 2. The pipeline, stage by stage

```
                        ┌─────────────────────────────────────┐
                        │  Copernicus Data Space (needs acct) │
                        └──────────────┬──────────────────────┘
                                       │
   ┌───────────────────────────────────┼────────────────────────────┐
   ▼                                   ▼                            ▼
Sentinel-1 GRD          Sentinel-2 L2A               Copernicus DEM GLO-30
download_sentinel.py    download_sentinel2.py        download_dem.py
   │  SYNTHETIC             │  SYNTHETIC                │  ✅ REAL (no account)
   ▼                        ▼                           ▼
inference.py  ──► flood_mask.tif  ──┐                  flood_path.py
   │  (needs trained model)         │                   │  ✅ REAL
   ▼                                ▼                   ▼
change_detection.json      damage_assessment.py    flood_path.geojson
   │  SYNTHETIC              │  SYNTHETIC
   ▼                          ▼
cutoff_analysis.py  ──►  cutoff_settlements.geojson  (SYNTHETIC)
   │
   ▼
generate_report.py ──► report/situation_report.html  (SYNTHETIC)
   │
   ▼
dashboard/app.py ──► web UI  (serves all of the above)
```

### 2.1 Script inventory

| Script | Purpose | Status |
|---|---|---|
| `download_dem.py` | Pulls Copernicus GLO‑30 1° tiles from the public AWS S3 bucket, mosaics them | ✅ **Works, real data on disk** |
| `download_sentinel.py` | Searches CDSE for a same‑orbit‑track Sentinel‑1 GRD pair, downloads it | ⛔ Needs CDSE account |
| `download_sentinel2.py` | Downloads Sentinel‑2 L2A bands B02/B03/B04/B08/B11/B12 + SCL | ⛔ Needs CDSE account |
| `fetch_osm.py` | Pulls OSM roads/buildings/settlements/hospitals/bridges | ⚠️ Fixture on disk |
| `make_demo_data.py` | Fabricates the synthetic GeoTIFFs & OSM | ✅ Works (that's the point) |
| `train_segmentation.py` | U‑Net / ResNet‑U‑Net training on Kuro Siwo | ⛔ Needs dataset + layout decision |
| `inference.py` | Tiled model inference → flood mask, change detection | ⛔ Needs trained model |
| `damage_assessment.py` | 3×3 building cluster rule vs flood mask | ✅ Works on synthetic |
| `cutoff_analysis.py` | Road‑network reachability → cut‑off settlements | ✅ Works on synthetic |
| `flood_path.py` | D8 steepest‑descent flow trace + sink breaching | ✅ **Works on real DEM** |
| `generate_report.py` | Renders `situation_report.html` with embedded maps | ✅ Works on synthetic |
| `run_pipeline.py` | Orchestrates all of the above | ✅ Works |
| `run_offline.py` | Pipeline variant that skips all downloads | ✅ Works |
| `dashboard/app.py` | Flask API + web UI | ✅ Works (see §8) |

---

## 3. What is REAL vs SYNTHETIC — the honest ledger

### ✅ REAL — verified from disk during this audit

| Artifact | Evidence |
|---|---|
| `data/dem/dem.tif` | **258 MB**, `PRODUCT_INFO.json` has `"synthetic": false` |
| `data/dem/tiles/` | **6 real GLO‑30 tiles**, 38–45 MB each, N27–N28 / E084–E086 |
| DEM bounds | 83.9999–87.0000 °E, 27.0001–29.0001 °N |
| DEM source | `https://copernicus-dem-30m.s3.amazonaws.com` (public, no account) |
| `results/flood_path.geojson` | 804 KB, 9,171 vertices, traced from the real DEM |
| Flood path metadata | `"synthetic": false`, attributed to Copernicus WorldDEM‑30 |

**Flood path specifics:** starts ~28.35 °N / 85.75 °E at ~5,431 m, descends to 234 m,
**251 km** of channel, **5,961 m** total drop, **1,201 sinks breached**, `cap_reached: false`.

### ⛔ SYNTHETIC — must be replaced before submission

| Artifact | Real? | Why it's fake |
|---|---|---|
| `data/sentinel1/{before,after}/` | ❌ | Demo fixture GeoTIFFs (VV/VH), not real SAFE products |
| `data/sentinel2/{before,after}/` | ❌ | Demo fixture bands + SCL |
| `data/osm/*.geojson` | ❌ | Fabricated geometry, not a real OSM snapshot |
| `results/flood_mask.tif` | ❌ | From fixture, not real inference |
| `results/flood_mask_before.tif` | ❌ | Same |
| `results/flood_mask.geojson` | ❌ | Derived from synthetic mask |
| `results/change_detection.json` | ❌ | `"synthetic": true` asserted in the file itself |
| `results/damage_assessment.geojson` | ❌ | Fixture ∩ fixture |
| `results/cutoff_settlements.geojson` | ❌ | Fixture road graph |
| `report/situation_report.html` | ❌ | Generated from all of the above |
| `models/flood_unet.pt` | ❌ | **`models/` is empty — no trained model exists** |
| `data/kuro_siwo/` | ❌ | Not present |

**Numbers that are fake and must never be quoted as findings:**
6,740.79 ha total water · 5,278.36 ha flood extent · 78.3 % flood coverage ·
68 cut‑off settlements · 70.8 % cut‑off rate · 4 severely damaged buildings ·
3 damaged bridges · 76.57 m resolution · 640×640 px extent.

All of those are properties of the fabricated fixture.

---

## 4. How much is done

| Track | Progress | Detail |
|---|---|---|
| **Elevation / terrain** | **~95 %** | Real DEM, real flow trace, sink breaching, docs |
| **Dashboard & API** | **~90 %** | All 10 routes live; map UI complete; needs hardening |
| **Report generation** | **~85 %** | Template done, renders correctly; content is synthetic |
| **Documentation** | **~85 %** | README + report.md accurate, caveats written in |
| **OSM / infrastructure** | **~60 %** | Code works; still pointed at the fixture |
| **Sentinel ingestion** | **~50 %** | Search/download code written, never executed for real |
| **ML model** | **~35 %** | Training code complete; no data, no model, no inference run |
| **Deliverables (video, portal)** | **0 %** | Entirely yours |

**Overall: roughly 65 % of a submission.** That 65 % is solid and honest. The missing 35 %
is a single blocked dependency plus one deliverable only you can produce.

---

## 5. How polished is it

| Component | Polish | Notes |
|---|---|---|
| `flood_path.py` | ★★★★★ | Real data, verified, backward‑compatible, well documented |
| `download_dem.py` | ★★★★☆ | Robust: validates tile size, detects S3 error pages, mosaics cleanly |
| `dashboard/app.py` | ★★★☆☆ | Functional & complete, but **insecure by default** (§8) |
| Map UI (`index.html`) | ★★★★☆ | 655 lines, layered map, working |
| `train_segmentation.py` | ★★★★☆ | 821 lines, U‑Net + ResNet‑U‑Net, Dice/BCE‑Dice, augmentation — but unproven |
| `inference.py` | ★★★☆☆ | Written, never executed against a real model |
| `download_sentinel.py` | ★★★☆☆ | Same‑orbit‑track pairing logic is the hard part and it's correct |
| `generate_report.py` | ★★★☆☆ | Works; output is only as real as its inputs |
| Docs | ★★★★☆ | Honest, precise, caveats stated |
| Repo hygiene | ★★☆☆☆ | **Not a git repo; no `.gitignore`; stray debug logs** |

**The code is more polished than the data.** That is the correct order to build in, and it
means the moment real imagery lands, the system should light up.

---

## 6. What is left

### Blocking
1. **Copernicus account + OAuth client** → unblocks Sentinel‑1, Sentinel‑2, and therefore
   the flood mask, damage assessment and cut‑off analysis.

### Your decisions
2. Which Kuro Siwo source (GeoTIFF recommended — simplest mapping to the expected layout).
3. Confirm 28.3 °N / 85.7 °E is genuinely your case‑study location.
4. Authorise the GB‑scale Kuro Siwo download.

### Work I can do once unblocked
5. Wire the `.env` loader so `CDSE_CLIENT_ID` / `CDSE_CLIENT_SECRET` never touch a command
   line or a log.
6. Build the Kuro Siwo → `images/` + `masks/` converter.
7. Train the model and run real inference.
8. Re‑run damage / cut‑off analysis against real OSM.
9. Regenerate the report with real numbers.

### Yours alone
10. The 3‑minute demo video.
11. Hackathon portal submission.
12. Publishing the repo (nothing has been pushed).

### Housekeeping
13. `dashboard/tmp_check.log`, `tmp_d.log`, `tmp_dash.log`, `tmp_dash2.log`, `tmp_f.log` —
    four stray debug logs left over from dashboard testing.
14. `results/flood_path_SYNTHETIC_BACKUP.geojson` — superseded backup of the old
    synthetic path.

---

## 7. Security & safety baseline (phase 2 starts here)

### Current posture: **publish‑ready for a laptop demo, NOT safe to expose**

The dashboard is built as a local demo tool. It has **zero** authentication, authorization,
origin control or rate limiting. That would be acceptable if it only ever listened on
`127.0.0.1` — but as shipped it binds `0.0.0.0` (S2), so it is already exposed to your
local network *right now*, and materially worse on venue Wi‑Fi or in any deployment.

Anyone who can reach the port can (a) read every result, and (b) with debug on, get a
Python console on your machine.

**In short: fix S1–S3 before you demo this anywhere but your own machine.**

### Findings

| # | Finding | Location | Severity | Why it matters |
|---|---|---|---|---|
| S1 | `debug` defaults to **on** (`FLASK_DEBUG` defaults to `"1"`) | [app.py:381](../dashboard/app.py#L381) | **Critical if exposed** | The Werkzeug debugger console permits **arbitrary code execution** for anyone who can reach a traceback page. This is not theoretical. |
| S2 | Binds `host="0.0.0.0"` | [app.py:386](../dashboard/app.py#L386) | **High** | Confirmed in the live log: `Running on all addresses (0.0.0.0)` and it announced itself on `10.52.131.43`. Anyone on the same Wi‑Fi can reach it. Combined with S1 this is remote code execution. |
| S3 | `CORS(app)` with no origin restriction | [app.py:27](../dashboard/app.py#L27) | **High** | **Verified live:** a request with `Origin: https://evil.example.com` came back with `Access-Control-Allow-Origin: https://evil.example.com` — the origin is *reflected*, not allowlisted. Any website a victim browses can read every API response. |
| S4 | `POST /api/run-pipeline` is unauthenticated and spawns a subprocess (300 s timeout) | [app.py:344](../dashboard/app.py#L344) | **Medium** | Not command injection — the argument list is fixed and no user input reaches it — but it is an unauthenticated action trigger, so it is a trivial resource‑exhaustion / denial‑of‑service vector. **Verified live:** the CORS preflight answered `Access-Control-Allow-Methods: ... POST ...` for a foreign origin, so a hostile page can fire it cross‑site. |
| S5 | No `.gitignore` | repo root | **High** | Only a matter of time before a 258 MB DEM, a 145 MB dataset, or `.secrets/cdse.env` gets committed. |
| S6 | Not a git repository at all | repo root | **Medium** | No change history, no way to revert a bad edit, no diff to review. |
| S7 | Pipeline stdout/stderr echoed to the HTTP client | [app.py](../dashboard/app.py) | **Low** | If the pipeline ever prints a path, token or key, it leaks to the caller. |
| S8 | No rate limiting or auth on any `/api/*` route | [app.py](../dashboard/app.py) | **Medium** | All data is world‑readable to anyone who can reach the port. |
| S9 | `CDSE_CLIENT_ID` / `CDSE_CLIENT_SECRET` read straight from environment; no `.env` loader, no `.secrets/` convention yet | `scripts/download_*.py` | **Medium** | Secrets end up on command lines and in shell history. |

**The fix pattern for S1–S3 is small and worth doing before any demo on shared Wi‑Fi:**

```
host="127.0.0.1"          # not 0.0.0.0
debug=False               # explicit, not env-defaulted to true
CORS(app, origins=["http://localhost:5000"])   # or drop CORS entirely
```

### Hardening backlog for phase 2

**Tier 1 — do before any shared‑network demo**
- Bind to `127.0.0.1`; make `0.0.0.0` an explicit opt‑in flag.
- Default `debug=False`; require an explicit env var to enable it.
- Restrict CORS to known origins, or remove it (same‑origin serving needs none).
- Add `.gitignore` (`data/`, `results/`, `models/`, `*.tif`, `.secrets/`, `__pycache__/`,
  `tmp_*.log`).
- `git init` and make the first commit so changes are reviewable.

**Tier 2 — before any public deployment**
- Move secrets into `.secrets/cdse.env` + a loader; add `.secrets/` to `.gitignore`.
- Secret scanning (gitleaks / trufflehog) over the tree before publishing.
- Rate‑limit `/api/*`; put `/api/run-pipeline` behind a token or turn it off in production.
- Stop echoing subprocess output to clients; log to a file instead.
- Add `Content-Security-Policy`, `X-Content-Type-Options`, `X-Frame-Options` headers.
- Serve over HTTPS if it ever leaves localhost.
- Dependency audit: `pip-audit` over `requirements.txt`, pin the versions.

**Tier 3 — data integrity & honesty (fits "safety" as much as "security")**
- A guard that **refuses to render** `report/situation_report.html` if any upstream input
  is synthetic, unless explicitly overridden. The project's biggest reputational risk is
  not an attacker — it's a synthetic number being quoted as a real measurement on stage.
- Keep the `"synthetic": true` marker propagated through every derived artifact.
- Verify the licence/attribution lines in report.md §7 survive into whatever is submitted.

**Not a concern right now:** SQL injection, XSS via user content, CSRF on state‑changing
endpoints (there is no user input and no persistent state), and supply‑chain risk beyond
the requirement pins. The attack surface is small — it's just wide open.

---

## 8. Known bugs and rough edges

| Item | Status |
|---|---|
| `flood_path.py` start‑coordinate unpacking bug | ✅ Fixed (`start_row, start_col = rowcol(...)`) |
| Sink breaching (`--pit-radius`, `--max-breaches`) | ✅ Implemented |
| Backward compatibility (`--pit-radius 0`) | ✅ Reproduces the original 53‑pixel trace |
| Docs claiming 78.7 °E | ✅ Corrected to 85.7 °E |
| **Dashboard pipeline trigger using 78.7 °E** | ✅ **Fixed during this audit** |
| Flood path stops at the southern DEM edge, not a confluence | ⚠️ Documented limitation |
| Flood path intersects no settlements | ⚠️ Expected — OSM layer is still the fixture |
| 1,201 sink breaches required | ⚠️ Documented; it's a *least‑obstruction drainage route*, not a surveyed channel |
| `train_segmentation.py` expects a dataset layout the repo loader doesn't produce | ⚠️ Needs the converter |

---

## 9. The critical path in one line

**Create the Copernicus account, put the credentials in `hackathon-trackb/.secrets/cdse.env`,
and tell me.** Everything else downstream I can take from there — and phase‑2 hardening can
run in parallel with it.
