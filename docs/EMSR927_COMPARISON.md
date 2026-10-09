# Case-Study Comparison — Our Pipeline vs Copernicus EMSR927

**Event:** Flash flood / Glacial Lake Outburst Flood (GLOF), Rasuwa District, Nepal
**Event date:** 26 August 2026 (night, UTC) — GLIDE `FF-2026-000162-NPL`
**Activation:** EMSR927, activated 26/08/2026 09:53 UTC (Int. Charter Act. ID 1052)
**Reference source:** Copernicus EMS Rapid Mapping bulletin EMSR927-AOI01 (Syabru Bensi), product version 1 — local copy [EMSR927_bulletin_2026-08-27.pdf](EMSR927_bulletin_2026-08-27.pdf), fetched 2026-10-09 from the public rasuwa-flood-bulletin mirror.

## What the official EMSR927 AOI01 product reports

From the bulletin (first Dossier product, published 27/08/2026):

| Quantity | EMSR927 value |
|---|---|
| Affected built-up structures | **433 No.** |
| Affected roads | **7.6 km** |
| Landslide area | **111.1 ha** |
| Potentially affected population | **~ 450** |
| Pre-event image | Pléiades Neo — CNES (29/11/2025, 0.3 m) |
| Post-event image | WorldView-3 — Vantor (27/08/2026, 0.3 m) |
| Thematic layer method | Visual interpretation of post-event VHR imagery |

## What our pipeline works with (for honest comparison)

| Input | Our data | Notes |
|---|---|---|
| Sentinel-1 pair | before 2026-08-24 / after 2026-09-05 (GRD, orbit 19) — real download | 2 days before vs 10 days after the event |
| Sentinel-2 before | 2026-08-12 S2C-L2A | cloud-limited |
| OSM pre-event snapshot | 2026-07-27 (buildings 3,755 / roads 715 / bridges 66) | real Overpass-attic extract, verified against ohsome |
| DEM | Copernicus GLO-30 | real |
| Flood mask | NOT yet reliable | AI model still on synthetic smoke weights (`models/flood_unet_smoke.pt`, Dice 0.44) |

## Quantitative comparison — what is comparable today

1. **Building damage.** EMSR927 counts 433 affected built-up structures inside
   its AOI01 (~8 km² of the valley). Our damage assessment runs on the full
   3,755-footprint OSM building set across the wider 64 km×64 km corridor, and
   currently over synthetic flood geometry, so **no comparable number exists
   yet**. Once the real flood mask lands, the correct cross-check is:
   ratio of our affected footprint count within AOI01's bounds to their 433.
2. **Roads affected.** EMSR927: 7.6 km of road damaged in AOI01. Our pipeline
   reports flooded road *segment counts and lengths* from OSM ways; the same
   AOI01-bounded comparison applies, and 7.6 km is the verification target.
3. **Flood extent.** EMSR927 maps flood extent visually at 0.3 m; our
   Sentinel-1 GRD chain works at ~7.4 m (GRD) resampled to 15.4 m in the
   640-px study grid — a resolution gap of ~40×, which we state openly rather
   than hiding.
4. **Population at risk.** EMSR927: ~450 potentially affected. Not derivable
   from our building-only OSM snapshot; we do not produce this number and do
   not pretend to.

## Differences in method, stated plainly

| Aspect | Copernicus EMSR927 | Our pipeline |
|---|---|---|
| Imagery | VHR optical (0.3 m) visual interpretation | Sentinel-1 SAR change detection + U-Net |
| Timing | D+1 product | 12-day revisit limited (Sentinel-1 orbit 19) |
| Automation | Analyst-driven | Fully automated, reproducible pipeline |
| Coverage | AOI01 ≈ 8 km² detail tiles | ~64 km × 60 km corridor |
| Cost | Manned activation | Free/open data + open model weights |

**Key takeaway for the demo:** our pipeline trades resolution and precision
for **speed, coverage and reproducibility** on open data — it produces
cut-off-settlement analysis (not in the EMS product) and a flood-path trace
(not in the EMS product) over a much wider area, while the EMS product
remains the ground truth for structure-level damage counts within its small AOI.
