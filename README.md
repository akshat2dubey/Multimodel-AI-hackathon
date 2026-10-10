# Jalnetra — Flood Damage Mapping from Space

Jalnetra is a flood-damage mapping tool for the Himalayas. It takes **an area** and **a flood date** and uses free satellite data to answer the three questions rescue teams care about:

1. **Where did the flood hit?** — a map of flooded and debris-covered areas, made by comparing before/after satellite images.
2. **What was damaged?** — an estimate of which buildings, roads, and bridges were hit, from OpenStreetMap overlaid on that flood map.
3. **Who is cut off?** — which settlements no longer have a road connection to the nearest town or hospital, found by running reachability over the road network.

It is built for the Multimodal AI Hackathon 2026, Track B (Mapping Flood Damage from Space), and uses Sentinel-1 radar (which sees through monsoon clouds), Sentinel-2 optical when the sky is clear, a Copernicus digital elevation model for flood-path tracing, and a pre-event OpenStreetMap snapshot. There is also a bonus flood-path tracer that, given any upstream point, traces the likely flow down the valley and lists the settlements along it.

## The real event behind the project

On **26 August 2026**, a partial collapse of a high-altitude glacier in northern Nepal sent a flood of water, mud, and debris down the **Bhote Koshi–Trishuli river corridor**. Nepal's National Disaster Risk Reduction and Management Authority reported **1,342 deaths** by 6 September, with almost **4,900 people** still missing or out of contact (as reported by the Copernicus Emergency Management Service). Jalnetra's case study is that event.

## What Jalnetra actually does

Jalnetra is a pipeline with a live web dashboard and a generated one-page situation report. The pipeline runs end-to-end from raw satellite data for any area and date — the idea is that a judge can point it at a corridor and a date on the day, and it produces the map, the damage numbers, the cut-off list, and the report from that data.

- **Sentinel-1 radar** is the backbone: it sees through monsoon clouds, and for change detection it is compared against a before image from the **same orbit track** — different tracks view the terrain from different angles and cannot be compared pixel by pixel.
- **Sentinel-2 optical** is used when the sky is clear; it is cloud-dependent, so it is optional rather than required.
- **OpenStreetMap** provides the buildings, roads, bridges, and hospitals as vector data to overlay on the flood map. The snapshot must be **pre-event** (on or before 26 August 2026; the latest available at the time of building this is 27 July 2026). OSM edits made after the event are for checking only, never as input.
- **Copernicus DEM** is used for the flood-path bonus. It comes from a public AWS Open Data bucket with no account needed.
- The **AI component** is a flood segmentation model (U-Net with a ResNet-34 encoder) trained on the listed training data, or a situation-report copilot — choose one. In either case, every number in a generated report must come from the system's maps, never invented by the language model.

## What's real right now

The code, the dashboard, the report generator, and the pipeline orchestration all exist and run. The terrain pipeline is fully real: a **258 MB Copernicus digital elevation model** of the Trishuli corridor was downloaded from the public AWS Open Data bucket (6 tiles, 38–45 MB each, covering 84–87°E × 27–29°N), and a **flood-flow trace** was computed from it — about **251 km** of channel, **5,961 m** of descent, written as a verified GeoJSON the dashboard serves.

Everything from the satellite imagery onward is **synthetic demo data** right now. The flood mask, the damage numbers, the cut-off list, and the report itself are all marked synthetic in their own metadata, and there is a `DEMO_DATA_NOTICE.txt` that calls this out plainly. They exist so the code, dashboard, and report could be built and exercised before the real Sentinel imagery and the real pre-event OSM snapshot arrive. Until that data lands and the model runs against it, none of those numbers should be quoted as a measurement of the actual Nepal event.

## How to run it

```bash
# Install dependencies
python -m pip install -r requirements.txt

# Start the dashboard locally
cd dashboard
python app.py
```

Then open the dashboard in a browser. That is the live, visual face of Jalnetra: the flood layer, roads, buildings, bridges, cut-off markers, and the traced flood path.

To run the full pipeline from raw satellite data:

```bash
python scripts/run_pipeline.py \
  --area "Bhote Koshi corridor" \
  --center-lat 28.3 \
  --center-lon 85.7 \
  --before-date 2026-08-24 \
  --after-date 2026-08-31 \
  --osm-snapshot 2026-07-27
```

That one command runs the whole chain: satellite ingestion → terrain analysis → infrastructure overlay → damage and cut-off analysis → report generation → the dashboard.

## Copernicus credentials

Sentinel-1 and Sentinel-2 come from the Copernicus Data Space Ecosystem via an OAuth client-credentials grant. It is free and takes about 15 minutes. Only the account holder can do it, and the secret should stay out of chat and command lines.

1. Register free at [dataspace.copernicus.eu](https://dataspace.copernicus.eu).
2. Create an OAuth client under Identity & Access Management.
3. Copy the client ID and secret (shown only once).
4. Save them to a git-ignored `.secrets/cdse.env` file.

The helper `scripts/cdse_env.py` reads that file so the secret never appears on a command line or in a log. Without valid credentials, the Sentinel download scripts exit non-zero — they no longer write placeholders.

## The flags

| Flag | Meaning |
|---|---|
| `--area` | Human-readable area description |
| `--center-lat` / `--center-lon` | Center of the area of interest |
| `--before-date` | Pre-event date (YYYY-MM-DD) |
| `--after-date` | Post-event date (YYYY-MM-DD) |
| `--osm-snapshot` | OpenStreetMap snapshot date (YYYY-MM-DD), before the event |

For the Trishuli case study:

| Item | Value |
|---|---|
| Center latitude | **28.3°N** |
| Center longitude | **85.7°E** |
| Before-date | **2026-08-24** |
| After-date | **2026-08-31** |
| OSM snapshot date | **2026-07-27** |

The OSM snapshot date must be before 26 August 2026. The latest available pre-event snapshot at the time of building this is 27 July 2026.

## Scripts

| Path | What it does |
|---|---|
| `scripts/run_pipeline.py` | One-command orchestrator for the full pipeline |
| `scripts/download_sentinel.py` | Sentinel-1 radar before/after download |
| `scripts/download_sentinel2.py` | Sentinel-2 optical before/after download |
| `scripts/fetch_osm.py` | Pre-event OSM snapshot via the ohsome API |
| `scripts/download_dem.py` | Copernicus DEM download (no account needed) |
| `scripts/train_segmentation.py` | Flood segmentation model training |
| `scripts/inference.py` | Model inference → flood mask GeoTIFF |
| `scripts/damage_assessment.py` | Flood mask + OSM → damage assessment |
| `scripts/cutoff_analysis.py` | Road-network reachability → cut-off settlements |
| `scripts/flood_path.py` | DEM-based flood-path tracing (bonus) |
| `scripts/generate_report.py` | Generates the one-page situation report |
| `scripts/cdse_env.py` | Loads Copernicus creds from git-ignored `.secrets/cdse.env` |
| `dashboard/app.py` | Live web dashboard (Flask) |

## Datasets and licenses

**Allowed as inputs:** Sentinel-1 and Sentinel-2 (Copernicus Data Space, free registration; before/after images); OpenStreetMap pre-event snapshot via the ohsome API (buildings, roads, bridges; ODbL; snapshot before 26 Aug 2026); **Kuro Siwo** (recommended training data; [github.com/Orion-AI-Lab/KuroSiwo](https://github.com/Orion-AI-Lab/KuroSiwo); MIT License per the NeurIPS 2024 paper, repo README also says CC BY; cite Bountos et al., 2024); **Sen1Floods11** (optional training data; [github.com/cloudtostreet/Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11); original repo has no license file, mirrors say CC BY 4.0; fine for non-commercial educational use; cite Bonafilia et al., CVPR Workshops 2020); and Copernicus DEM for flood-path tracing (Copernicus / AWS Open Data).

**Not allowed as inputs:** Copernicus EMS, UNOSAT, or any other published damage maps; and OSM edits made after the event. These may only be used to check your results, never as input. Breaking that rule leads to disqualification.

The official reference to check against is **Copernicus EMSR927** at [mapping.emergency.copernicus.eu](https://mapping.emergency.copernicus.eu), free of charge; credit _"European Union, Copernicus Emergency Management Service data."_ Used for checking only.

## Limitations

- **Sentinel satellites revisit an area only every few days.** They cannot warn of a sudden glacier collapse minutes ahead. This system is for response, not warning.
- **Radar change detection needs same-orbit-track before/after images.** Different tracks view the terrain from different angles and cannot be compared pixel by pixel. If the available acquisitions do not line up, the comparison degrades.
- **Optical imagery is cloud-dependent.** In a monsoon-season Himalayan flood, Sentinel-2 can be largely blank, which is why Sentinel-1 radar is the backbone, not the nice-to-have.
- **The flood mask, damage numbers, and cut-off list are only as good as the model and the imagery.** Until real Sentinel imagery downloads and a trained model runs against it, the headline outputs are synthetic demo data.
- **OSM is incomplete and unevenly distributed**, especially for high-altitude settlements. A cut-off analysis is only as good as the road network in the snapshot.
- **This is an educational prototype, not an operational tool.** Treat the topic with respect: do not use images of victims in demos or reports. The disaster is recent and many families are still affected.

## Required attribution

Every submission using this pipeline has to carry these attributions:

- _"Contains modified Copernicus Sentinel data 2026."_
- _"Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved."_
- _"© OpenStreetMap contributors."_
- A citation of whatever training dataset you used (Bountos et al., 2024 for Kuro Siwo; Bonafilia et al., 2020 for Sen1Floods11).

## Deliverables

- A map dashboard, plus a one-page situation report generated by the system.
- A GitHub repository with a README explaining how to run the system.
- A report (max 6 pages), including a limitations section explaining what the system cannot do.
- A 3-minute demo video.
- The required data attributions and training-dataset citations.
- Case-study results compared with the EMSR927 reference maps.

*Jalnetra — Flood Damage Mapping from Space. Track B, Multimodal AI Hackathon 2026.*