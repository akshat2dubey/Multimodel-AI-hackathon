# Jalnetra — Flood Damage Mapping from Space

Jalnetra is a research prototype for exploring flood-damage mapping from satellite data. Its current outputs are **not validated for emergency response or other operational decisions**. It takes an area and dates, then produces an experimental flood-change map, an OpenStreetMap overlay, a road-network reachability analysis, and a situation report.

1. **Where did the flood hit?** — a map of flooded and debris-covered areas, made by comparing before/after satellite images.
2. **What was damaged?** — an estimate of which buildings, roads, and bridges were hit, from OpenStreetMap overlaid on that flood map.
3. **Who is cut off?** — which settlements no longer have a road connection to the nearest town or hospital, found by running reachability over the road network.

It is built for the Multimodal AI Hackathon 2026, Track B (Mapping Flood Damage from Space), and uses Sentinel-1 radar (which sees through monsoon clouds), Sentinel-2 optical when the sky is clear, a Copernicus digital elevation model for flood-path tracing, and a pre-event OpenStreetMap snapshot. There is also a bonus flood-path tracer that, given any upstream point, traces the likely flow down the valley and lists the settlements along it.

Jalnetra includes a web dashboard and a generated situation report. The pipeline can fetch and process inputs when credentials and dependencies are available, but a successful run does not establish scientific accuracy. Verify AOI and input coverage and independently evaluate the change-detection outputs before using any results.

- **Sentinel-1 radar** is the backbone: it sees through monsoon clouds, and for change detection it is compared against a before image from the **same orbit track** — different tracks view the terrain from different angles and cannot be compared pixel by pixel.
- **Sentinel-2 optical** is used when the sky is clear; it is cloud-dependent, so it is optional rather than required.
- **OpenStreetMap** provides the buildings, roads, bridges, and hospitals as vector data to overlay on the flood map. The snapshot must be **pre-event** (on or before 26 August 2026; the latest available at the time of building this is 27 July 2026). OSM edits made after the event are for checking only, never as input.
- **Copernicus DEM** is used for the flood-path bonus. It comes from a public AWS Open Data bucket with no account needed.
- The **AI component** is a flood segmentation model (U-Net with a ResNet-34 encoder) trained on the listed training data, or a situation-report copilot — choose one. In either case, every number in a generated report must come from the system's maps, never invented by the language model.

The case study is the **26 August 2026 Rasuwa flood** in Nepal. Incident context and impact figures change as authorities update them; this repository does not use casualty estimates as model inputs or validated outputs.

The code, the dashboard, the report generator, and the pipeline orchestration all exist and run. The terrain pipeline is fully real: a **258 MB Copernicus digital elevation model** of the Trishuli corridor was downloaded from the public AWS Open Data bucket (6 tiles, 38–45 MB each, covering 84–87°E × 27–29°N), and a **flood-flow trace** was computed from it — about **251 km** of channel, **5,961 m** of descent, written as a verified GeoJSON the dashboard serves.

The checkout contains real Sentinel-1 scenes, DEM tiles, and a pre-event OSM extract, but they are cropped to a wider, different area than the EMSR927 AOI01 preset. The current flood model is only a two-epoch synthetic smoke model; its own status note says it detects no meaningful flood signal on the available scenes. Existing results also have an old run manifest. Treat all current maps, damage counts, and cutoff counts as illustrative; they are not measurements of the event.

**Release status: not production-ready.** The configured AOI still needs independent source verification, imagery and OSM must be freshly acquired for matching bounds, and the detection method must be validated against genuine labeled observations with documented accuracy and uncertainty. This prototype is unsuitable for rescue dispatch, safety decisions, or public impact reporting.

## How to run it

```bash
# Use a clean virtual environment with a supported Python and install dependencies
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt

# Start the dashboard locally (the local launcher generates a one-time password)
cd dashboard
python app.py
```

Then open the dashboard in a browser. For local HTTP, the loopback launcher selects a non-Secure session cookie. Use a strong `JALNETRA_DASHBOARD_PASSWORD` even for a shared demo; do not bind beyond loopback without the public deployment secrets.

### Hosted prototype deployment

Antideploy deployment uses Waitress and the platform port through the project
`Dockerfile`. Pipeline inputs and artifacts are stored in the app's private
S3-compatible bucket, not only in the disposable container. See
[`docs/ANTIDEPLOY.md`](docs/ANTIDEPLOY.md) for required secrets, initial
migration and deployment steps. Keep the three `JALNETRA_*` credentials in the
host's secret manager. Complete the checks in [`docs/SECURITY.md`](docs/SECURITY.md)
and scientific validation before handling real users or operational data.

To run the full pipeline from raw satellite data:

```bash
python scripts/run_pipeline.py --preset trishuli
```

That one command runs the whole chain: satellite ingestion → terrain analysis → infrastructure overlay → damage and cut-off analysis → report generation → the dashboard.

To point it at a **different** area, give the bounding box — the primary AOI input — and the dates:

```bash
python scripts/run_pipeline.py \
  --area "Some Other Corridor" \
  --bbox <min_lon>,<min_lat>,<max_lon>,<max_lat> \
  --before-date <YYYY-MM-DD> \
  --after-date <YYYY-MM-DD> \
  --event-date <YYYY-MM-DD> \
  --osm-snapshot <YYYY-MM-DD>
```

`--center-lat`/`--center-lon` with `--extent` is a fallback for when no bounding box is known; every download stage takes `--bbox` first.

### The run's metadata is resolved once

Whatever the inputs, they are merged (CLI over preset) into **one file, `results/run_meta.json`**, holding the area name, the bbox, the acquisition dates, the event date and the OSM snapshot date:

```json
{
  "area": "...",
  "bbox": [min_lon, min_lat, max_lon, max_lat],
  "bbox_source": "cli | preset | center_extent_fallback",
  "aoi_confirmed": false,
  "before_date": "...", "after_date": "...",
  "event_date": "...", "osm_snapshot_date": "..."
}
```

Every downstream stage — damage assessment, cut-off analysis, the situation report, the dashboard — reads *that* file, so the mask, the damage numbers and the report header cannot disagree about the area or the dates. Before this existed, the preset said `2026-08-14/08-28`, the README said `2026-08-24/08-31`, and the Sentinel-1 pair actually on disk is `2026-08-24/09-05`.

`aoi_confirmed` is `false` whenever the AOI came from the center+extent fallback rather than a real `--bbox`; a report from such a run prints an "AREA OF INTEREST NOT CONFIRMED" banner instead of presenting a provisional extent as settled.

`run_pipeline.py` **never trains.** Training needs real labelled data, so it is a separate, explicit step. If `models/flood_unet.pt` is missing the pipeline says so and falls back to the **log-ratio change-detection baseline** (a VV backscatter drop between the two acquisitions), which still produces a real flood mask from the real imagery. It is a physical threshold, not a segmentation model — to use the network, train it first:

```bash
python scripts/train_segmentation.py --dataset ./data/kuro_siwo --epochs 50
```

### Training and the flood mask

The model is a U-Net (optionally with a ResNet-34 encoder). Three things about it are easy to get wrong, so they are pinned down:

- **One input scaling, shared.** [`scripts/s1_normalization.py`](scripts/s1_normalization.py) is the only place that maps Sentinel-1 dB to the network's 0..1 input range. Training and inference both call it, and the window it used is written into the checkpoint's config, so inference re-applies the *same* scaling the weights were trained against instead of re-deriving one.
- **The mask is the change, not the river.** `flood_mask.tif` is `after AND NOT before` — newly flooded water only. The permanent river and lakes are subtracted using the pre-event water extent, so a riverside building is not reported as destroyed merely for being next to a river. `water_after.tif` holds all post-event water (river included) for comparison, and `flood_mask_before.tif` holds the permanent-water mask.
- **Speckle is not a flood.** Small connected blobs are dropped from the delivered mask, because a couple of dark radar pixels would otherwise become a "flooded" building or a severed road downstream.

Every run therefore writes four rasters. `damage_assessment.py` and `cutoff_analysis.py` read `flood_mask.tif` and need no change.

**To point it at another corridor or date, add a preset to [`config/presets.py`](config/presets.py), or pass `--bbox` and the dates directly.** Configuration lives in [`config/`](config/): `presets.py` holds the case-study values and the paths, and [`config/run_meta.py`](config/run_meta.py) resolves them (CLI over preset), writes `results/run_meta.json` and reads it back in any stage. `scripts/` and `dashboard/` must not restate the coordinates, dates or snapshot date — a test scans both trees, docstrings included, for stray case-study literals.

## Tests

```bash
python -m pytest -q              # full logic test suite
python -m pytest -q -m "not slow"  # skip the full-scene-sized performance test
```

`tests/` holds **logic tests on generated toy data**: a 200×200 raster on a ~20 m grid and a handful of OSM-style features, all built in `tmp_path`. They check that the code consuming the data behaves as documented — that the delivered flood mask is *new* water and not the permanent river, that tiling covers every pixel, that hectares come from the raster transform, that a partly flooded road does not sever a whole valley, that empty inputs produce empty (not crashed) outputs, and that the dashboard degrades gracefully. They also pin down the training contract (an empty dataset fails, `--smoke-test` is opt-in and names its checkpoint `*_smoke.pt`), that training and inference scale inputs with the same dB → 0..1 window, that inference with no checkpoint still writes a real mask via the log-ratio baseline, and that the case-study values are resolved into one `results/run_meta.json` that every stage reads (including that no coordinate or date is hardcoded in `scripts/` or `dashboard/`).

They are not science tests: a green run does **not** show the flood map is correct. No network, no Copernicus account, no GPU, and nothing is written into `results/` or `data/` (a test asserts that).

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
| `--preset` | Named case-study preset (available: `trishuli`). Supplies defaults; any explicit flag overrides it |
| `--bbox` | **Primary** area of interest: `min_lon,min_lat,max_lon,max_lat` |
| `--center-lat` / `--center-lon` / `--extent` | Fallback AOI, used only when no `--bbox` is given |
| `--area` | Human-readable area description |
| `--before-date` / `--after-date` | The two Sentinel-1 acquisition dates (YYYY-MM-DD) |
| `--event-date` | The flood event date (YYYY-MM-DD) |
| `--osm-snapshot` | OpenStreetMap snapshot date (YYYY-MM-DD), before the event |

### The `trishuli` preset

| Item | Value |
|---|---|
| Before-date | **2026-08-24** — the pre-event acquisition actually downloaded (orbit 19) |
| After-date | **2026-09-05** — the post-event acquisition actually downloaded |
| Event date | **2026-08-26** |
| OSM snapshot date | **2026-07-27** |
| AOI (bbox) | **provisional** — copied from the linked EMSR927 AOI01 resource; independently verify before use |

The acquisition dates are the Sentinel-1 pair on disk (`data/sentinel1/*/PRODUCT_INFO.json`); the event itself is 26 August 2026 (GLIDE `FF-2026-000162-NPL`, EMSR927 activated 26/08/2026). The rules require the OSM snapshot to precede the event, which is enforced in `config/run_meta.py`.

The preset stores provisional EMSR927 AOI01 bounds and leaves `aoi_confirmed` false by default. The checked-in input sidecars still describe the older `[85.4, 28.0, 86.0, 28.6]` extent, so they do not match this AOI. The pipeline will fetch new inputs for its selected bbox; do not reuse the old outputs as evidence. Use `--confirm-aoi` only after independently checking the boundary against the authoritative source.

## Scripts

| Path | What it does |
|---|---|
| `scripts/run_pipeline.py` | One-command orchestrator for the full pipeline |
| `scripts/download_sentinel.py` | Sentinel-1 radar before/after download |
| `scripts/download_sentinel2.py` | Sentinel-2 optical before/after download |
| `scripts/fetch_osm.py` | Pre-event OSM snapshot via the ohsome API |
| `scripts/download_dem.py` | Copernicus DEM download (no account needed) |
| `scripts/train_segmentation.py` | Trains the flood segmentation model on a **real** dataset (Kuro Siwo / Sen1Floods11 layout). Refuses to run on an empty dataset; `--smoke-test` generates synthetic tiles and writes a clearly named `*_smoke.pt` |
| `scripts/inference.py` | Tiled inference → `flood_mask.tif` (newly flooded only) + `water_after.tif`, `flood_mask_before.tif` (permanent water), `valid_mask.tif`. Falls back to the log-ratio baseline automatically when no checkpoint exists |
| `scripts/s1_normalization.py` | The one dB → 0..1 input scaling, shared by training and inference |
| `scripts/mask_postprocess.py` | Speckle control — drops connected flood blobs below a pixel floor |
| `scripts/preprocess_s1.py` | Sentinel-1 calibration / speckle filter → dB GeoTIFFs + sidecar |
| `scripts/damage_assessment.py` | Flood mask + OSM → damage assessment |
| `scripts/cutoff_analysis.py` | Road-network reachability → cut-off settlements |
| `scripts/flood_path.py` | DEM-based flood-path tracing (bonus) |
| `scripts/generate_report.py` | Generates the one-page situation report |
| `scripts/cdse_env.py` | Loads Copernicus creds from git-ignored `.secrets/cdse.env` |
| `scripts/run_offline.py` | Runs every stage on the synthetic fixture (no account needed) |
| `config/presets.py` | **All paths + the case-study PRESETS — the one file to edit** |
| `config/run_meta.py` | Resolves CLI + preset into the run's metadata and reads/writes `results/run_meta.json` |
| `dashboard/app.py` | Live web dashboard (Flask) |
| `tests/` | pytest logic tests on generated toy data (no network/GPU) |

## The synthetic fixture (no account needed)

```bash
python scripts/run_offline.py                 # every stage, on fabricated data

# Look at it — the dashboard shows a DEMO DATA banner when serving the fixture.
cd dashboard
# PowerShell: $env:JALNETRA_RESULTS_DIR="results_demo"; python app.py
# macOS/Linux: JALNETRA_RESULTS_DIR=results_demo python app.py
```

A demo run keeps its own trees: generated inputs in `data_demo/`, generated
artifacts in `results_demo/`, and the report it writes in `results_demo/` too.
`data/`, `results/` and `report/` hold the real downloads, the real pipeline
output and the published report — the generator refuses to write into them
(and refuses to overwrite any file it did not create) unless `--force` is
given, so a demo cannot quietly destroy the only real data in the repo. The
dashboard tells the two apart by which directory it was pointed at, never by
trusting a flag inside the data, and shows a banner when that directory is the
fixture.

## Datasets and licenses

**Allowed as inputs:** Sentinel-1 and Sentinel-2 (Copernicus Data Space, free registration; before/after images); OpenStreetMap pre-event snapshot via the ohsome API (buildings, roads, bridges; ODbL; snapshot before 26 Aug 2026); **Kuro Siwo** (recommended training data; [github.com/Orion-AI-Lab/KuroSiwo](https://github.com/Orion-AI-Lab/KuroSiwo); MIT License per the NeurIPS 2024 paper, repo README also says CC BY; cite Bountos et al., 2024); **Sen1Floods11** (optional training data; [github.com/cloudtostreet/Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11); original repo has no license file, mirrors say CC BY 4.0; fine for non-commercial educational use; cite Bonafilia et al., CVPR Workshops 2020); and Copernicus DEM for flood-path tracing (Copernicus / AWS Open Data).

**Not allowed as inputs:** Copernicus EMS, UNOSAT, or any other published damage maps; and OSM edits made after the event. These may only be used to check your results, never as input. Breaking that rule leads to disqualification.

The official reference to check against is **Copernicus EMSR927** at [mapping.emergency.copernicus.eu](https://mapping.emergency.copernicus.eu), free of charge; credit _"European Union, Copernicus Emergency Management Service data."_ Used for checking only.

## Limitations

- **Not production-ready for operational response.** The checked-in Sentinel-1 and OSM bounds do not match the provisional AOI; the smoke model was trained on synthetic data and is not validated. Current layers and derived counts are illustrative only.
- **Sentinel satellites revisit an area only every few days.** They cannot warn of a sudden glacier collapse minutes ahead. This system is for response, not warning.
- **Radar change detection needs same-orbit-track before/after images.** Different tracks view the terrain from different angles and cannot be compared pixel by pixel. If the available acquisitions do not line up, the comparison degrades.
- **Optical imagery is cloud-dependent.** In a monsoon-season Himalayan flood, Sentinel-2 can be largely blank, which is why Sentinel-1 radar is the backbone, not the nice-to-have.
- **The flood mask, damage numbers, and cut-off list are only as good as the model and the imagery.** A log-ratio baseline is available, but it is experimental and has not been independently evaluated for this event.
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
