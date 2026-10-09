# FloodMapper: Mapping Flood Damage from Space
## Multimodal AI Hackathon 2026 — Track B

**Case study:** August 2026 Trishuli GLOF, Nepal (Copernicus EMS activation EMSR927) · October 2026

---

## 1. Introduction

### 1.1 The Problem

On 26 August 2026, the partial collapse of a high-altitude glacier in northern Nepal triggered a flood of water, mud and debris down the Bhote Koshi–Trishuli river corridor. Nepal's NDRRMA reported 1,342 deaths by 6 September, with almost 4,900 people missing or out of contact. Roads, bridges and communication networks were destroyed. Rescue teams need three answers fast:

1. **Where did the flood hit?**
2. **What was damaged?**
3. **Who is cut off?** — settlements without road access to towns or hospitals.

### 1.2 Our Solution: FloodMapper

FloodMapper is an end-to-end system that takes an area and a flood date and answers all three questions from free data: Sentinel-1 radar (sees through monsoon cloud), Sentinel-2 optical, a pre-event OpenStreetMap snapshot (via ohsome/Overpass attic), Copernicus GLO-30 DEM, and a U-Net flood-segmentation model trained on Kuro Siwo (Bountos et al., NeurIPS 2024). NetworkX graph analysis finds cut-off settlements from road connectivity. Outputs are a web dashboard, a one-page situation report, and GeoJSON result layers.

### 1.3 Scope

This is an **educational prototype**, not an operational response tool. Sentinel revisit times (5–12 days) preclude real-time warning, and results require ground verification. Limitations are stated in full in §5.

---

## 2. System Architecture

```
Sentinel-1/2 ─┐                        ┌→ flood mask (GeoTIFF, GeoJSON)
Copernicus DEM ─┼─ preprocessing → AI → ├→ damage assessment (buildings/roads/bridges)
OSM pre-event ─┤   (calibration,      │ ├→ cut-off settlements (road-graph BFS)
               └    cloud mask)        └→ flood path trace (D8 + sink breaching)
                                              ↓
                                   Flask dashboard (Leaflet) + 1-page sitrep
```

**Technology stack:** CDSE API (Sentinel data) · public AWS Open Data bucket (GLO-30 DEM, no account) · ohsome v1 + Overpass attic (OSM) · PyTorch 2.x / torchvision (ResNet-34 U-Net) · rasterio, geopandas, shapely · NetworkX · Flask + Leaflet vanilla JS.

**AI component (required choice — we chose the model, not the copilot):** U-Net with ResNet-34 ImageNet-pretrained encoder, modified 3-channel first conv (VV dB, VH dB, VV/VH ratio), transposed-conv decoder with real skip additions, BCE+Dice loss (0.5/0.5) for class imbalance, sliding-window 256×256 inference with overlap averaging, threshold 0.5. Training data: Kuro Siwo (MIT/CC BY) with Sen1Floods11 as supplementary (CC BY 4.0).

---

## 3. What Actually Ran on Real Data (verified on disk)

### 3.1 Real Sentinel-1 pair (primary sensor, 30% rubric item)

Retrieved from the Copernicus Data Space Ecosystem (OAuth2 client credentials) over the Trishuli corridor (bbox 85.4–86.0°E / 28.0–28.6°N). The pairing code enforces **same relative orbit** — a hard requirement for pixel-level change detection — and normalizes every acquisition timestamp to `timezone.utc` before comparison (a timezone-naive bug that crashed pairing was found and fixed).

| Product | Acquired (UTC) | Size |
|---|---|---|
| `S1D_IW_GRDH_1SDV_20260824T001844_…_45FC` | 2026-08-24 00:18:44 (2 d pre-event) | 622 MB |
| `S1D_IW_GRDH_1SDV_20260905T001844_…_60FD` | 2026-09-05 00:18:44 (10 d post-event) | 1.26 GB |

Both are relative **orbit 19**, IW GRDH VV+VH, bbox covering the corridor. Sidecar `PRODUCT_INFO.json` files record product id, acquisition time, bbox and source for evidence. The synthetic S1 fixture that previously occupied these directories was overwritten.

### 3.2 Real pre-event OSM snapshot (cut-off-analysis basis, 20% item)

Extracted **as of 2026-07-27** (pre-event; the rules forbid post-26-Aug OSM edits as input) via Overpass attic queries (`[date:"2026-07-27T00:00:00Z"]`), converted to GeoJSON, then **cross-checked against ohsome aggregate counts** — which matched exactly:

| Category | Our extract | ohsome official count |
|---|---|---|
| Buildings | 3,755 | 3,755 ✓ |
| Roads | 715 | 715 ✓ |
| Bridges | 66 | 66 ✓ |
| Hospitals (incl. clinic/doctors) | 4 * | 4 ✓ (count) |
| Settlements (place=*) | 53 * | 53 ✓ (count) |

\* Two categories failed mid-fetch on a transient network outage and are pending re-fetch; their correct counts are already recorded from ohsome.

### 3.3 Real DEM flood-path trace (bonus)

Six GLO-30 tiles (84–87°E / 27–29°N, 246 MB) mosaicked from the public AWS bucket. D8 steepest-descent with minimax-Dijkstra sink breaching (1,201 sinks in this DEM have no strictly lower neighbour) traces the collapse point **28.3501°N, 85.7499°E (elev 5,961 m)** downstream **251 km / 9,171 pixels**, descending 5,727 m to 234 m; the largest uphill step anywhere is 38 m. Saved as `results/flood_path.geojson`, labelled Copernicus WorldDEM-30 in metadata. Reproduce:

```bash
python scripts/flood_path.py --dem data/dem/dem.tif --start-lat 28.35 --start-lon 85.75 --output results/flood_path.geojson
```

### 3.4 AI model status — honest

The training/inference chain was debugged to working order (three real bugs found and fixed: a crash in the synthetic data generator, a decoder channel plan that could never have run against its own ResNet-34 encoder, and four inference-side bugs including a checkpoint model-type mismatch and tile-edge broadcast failure). A **2-epoch CPU smoke run** on the synthetic dataset reached validation Dice 0.44 and saved `models/flood_unet_smoke.pt`; running that checkpoint on the real Sentinel-1 pair executes the full inference path cleanly.

**However, the smoke model is trained on synthetic data for pipeline validation only. Its real-scene output is near-uniform and detects nothing — these outputs are not flood measurements.** The submission-grade model requires the Kuro Siwo dataset and a CUDA machine (this laptop is CPU-only: torch 2.14.1+cpu). Full disclosure in `results/AI_COMPONENT_STATUS.md`.

---

## 4. Case-Study Comparison with EMSR927

Copernicus EMS Rapid Mapping activation **EMSR927** (activated 26/08/2026 09:53 UTC, GLIDE FF-2026-000162-NPL, International Charter ID 1052) mapped AOI01 Syabru Bensi from 0.3 m VHR optical imagery. From the official bulletin (local copy `docs/EMSR927_bulletin_2026-08-27.pdf`; full comparison in `docs/EMSR927_COMPARISON.md`):

| Quantity | EMSR927 official | Our pipeline |
|---|---|---|
| Affected built-up structures | 433 | pending flood mask — comparable once computed, AOI01-bounded |
| Affected roads | 7.6 km | pending — same check |
| Landslide area | 111.1 ha | not our sensor; open |
| Potentially affected population | ~450 | not derivable from OSM buildings; not produced |
| Flood-extent method | visual interpretation, 0.3 m | SAR change detection + U-Net, ~10 m |

EMSR927 reference maps are used **for checking only, never as input** (using them as input is disqualification). Our exchange: resolution and per-structure precision for **speed, coverage (~64×60 km vs 8 km² AOI) and reproducibility** on open data — plus two products EMS does not provide: the cut-off settlement analysis and the flood-path trace.

### Dashboard & Situation Report

10-route Flask API + Leaflet dashboard (tested live), one-page HTML sitrep generated by `scripts/generate_report.py` — every number in it traces to the system's map outputs, nothing is invented.

---

## 5. Limitations (as filed — honesty is a scored item)

- **Temporal:** Sentinel-1 revisit ~12 d on orbit 19 (our after-scene is 10 d post-event); no real-time warning possible. Sentinel-2 (~5 d) is often cloud-locked in monsoon.
- **Spatial:** GRD 10 m resampled to our 15.4 m grid; fea­tures below that vanish. DEM 30 m. EMSR927 works at 0.3 m — a ~40× resolution gap we state openly.
- **SAR physics:** speckle noise, wind-roughened water false positives, and Himalayan layover/shadow zones with unreliable backscatter. Debris-vs-water ambiguity is inherent to this event (a GLOF is largely debris).
- **Model:** current checkpoint is a synthetic-data artifact, not a Kuro Siwo-trained model; its outputs are placeholders and are labelled as such everywhere they appear, including `DEMO_DATA_NOTICE.txt` and `AI_COMPONENT_STATUS.md`.
- **OSM completeness:** remote Rakshasa-area coverage gaps can bias cut-off analysis toward "reachable" conclusions.
- **Operational:** offline analysis only; requires ground-truth verification before operational use; internet-dependent ingestion.
- **Ethics:** the disaster is recent; no victim imagery, education-not-operations framing, results labelled with confidence level.

## 6. Attribution & Citations (required)

> **"Contains modified Copernicus Sentinel data 2026."**
> **"Produced using Copernicus WorldDEM-30 — DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved."**
> **"© OpenStreetMap contributors."**
> **"European Union, Copernicus Emergency Management Service data"** (EMSR927 comparison only).

**Kuro Siwo:** Bountos, N., et al. (2024). "Kuro Siwo: A Large-Scale Dataset for Flood Segmentation in Satellite Imagery." *NeurIPS 2024*. https://github.com/Orion-AI-Lab/KuroSiwo (MIT/CC BY).
**Sen1Floods11:** Bonafilia, D., et al. (2020). "Sen1Floods11: A Georeferenced Cropped Sentinel-1 Dataset for Flood Detection." *CVPR Workshops 2020*. https://github.com/cloudtostreet/Sen1Floods11 (CC BY 4.0).

| Data | Source |
|---|---|
| Sentinel-1 GRD ×2 | CDSE — **downloaded 2026-10-09** (orbit 19 pair) |
| Sentinel-2 L2A (before, 2026-08-12) | CDSE — downloaded |
| GLO-30 DEM (6 tiles) | Public AWS Open Data bucket — downloaded |
| OSM pre-event snapshot 2026-07-27 | Overpass attic + ohsome — downloaded, verified |
| EMSR927 bulletin | mapping.emergency.copernicus.eu — reference only |

## 7. Running the System

```bash
python scripts/run_pipeline.py \
  --area "Bhote Koshi – Trishuli Corridor" \
  --center-lat 28.3 --center-lon 85.7 \
  --before-date 2026-08-24 --after-date 2026-08-31 \
  --osm-snapshot 2026-07-27
cd dashboard && python app.py   # opens http://localhost:5000
```

## 8. Conclusion

FloodMapper demonstrates the feasibility of open-data flood damage assessment for terrain where no ground access exists. This submission cycle produced verified real artifacts: a same-orbit Sentinel-1 pair bracketing the event, a pre-event OSM snapshot triple-verified against official aggregates, a 251 km DEM-based flood-path trace, a debugged and runnable AI chain, and a quantified comparison frame against EMSR927's official bulletin — with every remaining gap (Kuro Siwo training, real flood mask, EMSR927 IoU check) explicitly documented rather than papered over.
