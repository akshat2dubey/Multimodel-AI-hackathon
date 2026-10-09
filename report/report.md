# FloodMapper: Mapping Flood Damage from Space
## Multimodal AI Hackathon 2026 — Track B

**Team:** FloodMapper Team
**Date:** October 2026
**Case Study:** August 2026 Trishuli Glacier Flood, Nepal (Copernicus EMS Activation EMSR927)

---

## 1. Introduction

### 1.1 The Problem

On 26 August 2026, the partial collapse of a high-altitude glacier in northern Nepal triggered an avalanche of ice and rock that sent a flood of water, mud, and debris down the Bhote Koshi–Trishuli river corridor. Nepal's National Disaster Risk Reduction and Management Authority (NDRRMA) reported 1,342 deaths by 6 September, with almost 4,900 people still missing or out of contact. In disasters like this, ground sensors are swept away, roads and bridges are destroyed, and communication networks fail. Rescuers urgently need to know:

1. **Where did the flood hit?** — Which areas were inundated?
2. **What was damaged?** — Which buildings, roads, and bridges were hit?
3. **Who is cut off?** — Which settlements no longer have road access to towns or hospitals?

### 1.2 Our Solution: FloodMapper

FloodMapper is an end-to-end system that takes an area and a flood date and uses free satellite imagery to answer all three questions. It combines:

- **Sentinel-1 radar imagery** (sees through monsoon clouds) for before/after change detection
- **Sentinel-2 optical imagery** (high-resolution detail when sky is clear)
- **OpenStreetMap pre-event data** (buildings, roads, bridges, hospitals)
- **Copernicus DEM** (elevation data for flood path tracing)
- **A U-Net deep learning model** trained on the Kuro Siwo flood segmentation dataset for water/debris detection
- **NetworkX graph analysis** for cut-off settlement detection via road network connectivity

### 1.3 Scope and Limitations

This system is an **educational prototype**, not an operational disaster response tool. It demonstrates the feasibility of using open satellite data and AI to rapidly assess flood damage in remote areas. It is not designed for real-time early warning (Sentinel satellites revisit areas every few days), and its results must be verified by ground teams before use in rescue operations.

---

## 2. System Architecture

### 2.1 Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      FloodMapper Pipeline                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  STEP 1: DATA INGESTION                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │Sentinel-1│  │Sentinel-2│  │ Copernicus│  │ OpenStreetMap │  │
│  │  (radar) │  │ (optical)│  │    DEM    │  │ (pre-event)   │  │
│  └──────────┘  └──────────┘  └──────────┘  └───────────────┘  │
│       │             │              │              │             │
│       ▼             ▼              ▼              ▼             │
│  STEP 2: PREPROCESSING                                          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ • S-1: Radiometric calibration, speckle filtering        │   │
│  │ • S-2: Cloud masking (SCL band), surface reflectance     │   │
│  │ • OSM: GeoJSON extraction via ohsome API                 │   │
│  │ • DEM: Tile extraction, nodata handling                   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  STEP 3: AI — FLOOD SEGMENTATION (U-Net)                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Input:  Sentinel-1 VV+VV bands (or S-2 multi-spectral)  │   │
│  │  Model:  U-Net with ResNet-34 encoder (ImageNet pretrained) │
│  │  Output: Flood probability map (0-1) per pixel            │   │
│  │  Threshold: 0.5 → binary flood mask                      │   │
│  │  Trained on: Kuro Siwo dataset (Bountos et al., 2024)    │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  STEP 4: DAMAGE ASSESSMENT                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  • Rasterize OSM buildings onto flood mask grid           │   │
│  │  • Compute overlap ratio for each building                │   │
│  │  • Classify: DESTROYED / SEVERE / PARTIAL / UNAFFECTED   │   │
│  │  • Same for roads (SEGMENT_CUT / AFFECTED) and bridges   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  STEP 5: CUT-OFF ANALYSIS                                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  • Build road network graph from OSM road segments        │   │
│  │  • Remove edges where road intersects flood mask          │   │
│  │  • BFS from hospital/town nodes → find unreachable        │   │
│  │  • Settlements not reachable = cut off                    │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  STEP 6: FLOOD PATH TRACING (BONUS)                            │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  • D8 steepest-descent flow routing on Copernicus DEM    │   │
│  │  • Start from upstream point (glacier collapse area)      │   │
│  │  • Trace path downstream to valley floor                  │   │
│  │  • Intersect with OSM settlements along path              │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  STEP 7: OUTPUTS                                                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  • GeoJSON flood mask (GeoTIFF)                          │   │
│  │  • GeoJSON damage assessment (buildings, roads, bridges)  │   │
│  │  • GeoJSON cut-off settlements with nearest town info     │   │
│  │  • GeoJSON flood path with settlements along it           │   │
│  │  • HTML situation report (1-pager for rescue teams)        │   │
│  │  • Web dashboard (interactive map, layers, queries)       │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Technology Stack

| Component | Technology |
|-----------|-----------|
| Satellite data access | Copernicus Data Space Ecosystem (CDSE) API, `cdse` Python library |
| DEM data | Copernicus WorldDEM-30 (GLO-30) via the public Copernicus DEM AWS Open Data bucket — no account required |
| OSM data | ohsome API v1 (`api.ohsome.org/v1`) |
| AI/ML framework | PyTorch 2.x, torchvision (ResNet-34 backbone) |
| GIS processing | rasterio, geopandas, shapely, GDAL |
| Graph analysis | NetworkX |
| Dashboard backend | Flask + Flask-CORS |
| Dashboard frontend | Leaflet.js (mapping), vanilla JS |
| Report generation | Python string templating → HTML |
| Training data | Kuro Siwo (Orion-AI-Lab, NeurIPS 2024), MIT/CC BY license |

---

## 3. AI Component: Flood Segmentation Model

### 3.1 Model Architecture

We chose the **U-Net with ResNet-34 encoder** architecture for flood segmentation. This is a well-established encoder-decoder convolutional neural network designed for biomedical image segmentation, adapted here for satellite imagery.

**Encoder (contracting path):**
- ResNet-34 backbone pre-trained on ImageNet (transfer learning)
- Modified first convolution layer to accept Sentinel-1 input channels (VV, VH, VV/VH ratio = 3 channels) instead of 3-channel RGB
- Produces hierarchical feature maps at 1/4, 1/8, 1/16, and 1/32 resolution

**Decoder (expanding path):**
- Transposed convolution upsampling layers
- Skip connections from encoder feature maps at each resolution level
- Final 1×1 convolution produces single-channel flood probability map
- Sigmoid activation: output in [0, 1] representing flood probability

**Input composition:**
- Sentinel-1 VV polarization (backscatter in dB)
- Sentinel-1 VH polarization (backscatter in dB)
- VV/VH ratio (difference in dB, useful for distinguishing water from land)

**Output:**
- Per-pixel flood probability map at 10m resolution (same as Sentinel-1 GRD)
- Thresholded at 0.5 for binary flood mask

### 3.2 Training Data

**Primary dataset: Kuro Siwo** (Bountos et al., NeurIPS 2024)
- Large-scale dataset for flood segmentation in satellite imagery
- Contains Sentinel-1 and Sentinel-2 imagery with flood annotations
- MIT/CC BY license — free use with credit
- Download: `https://github.com/Orion-AI-Lab/KuroSiwo`

**Supplementary dataset: Sen1Floods11** (Bonafilia et al., CVPR Workshops 2020)
- 11 flood events with Sentinel-1 imagery and water labels
- CC BY 4.0 license — free for non-commercial educational use
- Download: `https://github.com/cloudtostreet/Sen1Floods11`

### 3.3 Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | U-Net + ResNet-34 encoder |
| Input channels | 3 (VV dB, VH dB, VV/VH ratio) |
| Output | 1 channel (flood probability) |
| Image size | 256×256 pixels (2.56 km × 2.56 km at 10m) |
| Batch size | 16 |
| Optimizer | Adam (lr = 1e-4) |
| Loss function | BCE + Dice loss (weighted 0.5/0.5) |
| Epochs | 50 (with early stopping, patience = 5) |
| Data augmentation | Random horizontal/vertical flip, 90° rotations, random crop/pad |
| Train/val split | 80/20 |
| Hardware | CUDA GPU (when available) |

### 3.4 Loss Function

The combined **BCE + Dice loss** handles the class imbalance inherent in flood mapping (flooded pixels are a small fraction of total pixels):

$$\mathcal{L} = 0.5 \cdot \mathcal{L}_{BCE} + 0.5 \cdot \mathcal{L}_{Dice}$$

Where:
- $\mathcal{L}_{BCE} = -\frac{1}{N}\sum(y\log(\hat{y}) + (1-y)\log(1-\hat{y}))$
- $\mathcal{L}_{Dice} = 1 - \frac{2\sum y\hat{y} + \epsilon}{\sum y + \sum \hat{y} + \epsilon}$

### 3.5 Inference Pipeline

1. Load before/after Sentinel-1 products
2. Radiometric calibration: convert digital numbers to backscatter coefficient σ° in dB
   - $σ° = 10 \cdot \log_{10}(DN^2) + calibration\_constant$
3. Speckle filtering: 3×3 median filter (or Lee filter for production)
4. Create input tensor: [VV_dB, VH_dB, VV_dB - VH_dB] per pixel
5. Slide window inference (256×256 tiles, 64px overlap for seamless stitching)
6. Sigmoid activation → probability map
7. Threshold at 0.5 → binary flood mask
8. Save as GeoTIFF with original georeferencing

### 3.6 Expected Performance

On held-out test scenes from diverse geographies:
- **Expected IoU (Intersection over Union):** 0.70–0.85 for clear flood events
- **Expected Dice coefficient:** 0.80–0.90
- Primary error modes:
  - False positives: Wet soil, shadows, smooth surfaces mimicking water
  - False negatives: Undercut riverbanks, narrow water bodies below resolution
  - Domain gap: Model trained on non-Himalayan data may underperform on Himalayan topography

---

## 4. Case Study: August 2026 Trishuli Flood

### 4.1 Study Area

- **Location:** Bhote Koshi–Trishuli river corridor, Nepal
- **Center coordinates:** Approximately 28.3°N, 85.7°E (the Bhote Koshi–Trishuli corridor itself; an earlier draft of this document carried 78.7°E, which is in the Indo-Gangetic plain and is not the case-study area)
- **Area extent:** ~30 km × 30 km for the Sentinel imagery footprint; the Copernicus DEM downloaded for flood-path tracing covers a wider box (84.0–87.0°E / 27.0–29.0°N) so the traced path is not cut off at the imagery edge
- **Event date:** 26 August 2026
- **Primary imagery:** Sentinel-1 GRD (Descending, IW mode, VV+VH polarization)
- **Before image:** ~12 days before event (same relative orbit for valid comparison)
- **After image:** ~2 days after event
- **OSM snapshot:** 27 July 2026 (pre-event, latest available before flood)

### 4.2 Sentinel-1 Data Selection

Critical consideration for change detection: **images must be from the same orbit track**. Sentinel-1 has a 12-day repeat cycle; for a given area, the same ground track is imaged roughly every 12 days at the same local time. Images from different tracks view the terrain from different incidence angles and **cannot be compared pixel-by-pixel**.

Our pipeline searches for products matching:
- Same relative orbit number
- Same orbit direction (descending for this area)
- Same swath mode (IW — Interferometric Wide)
- Before date: as close as possible to event date, before the event
- After date: as close as possible to event date, after the event

### 4.3 Flood Extent Results

*(These results would be populated by running the pipeline on actual downloaded data. For the hackathon submission, we show the pipeline structure and placeholder results.)*

| Metric | Value |
|--------|-------|
| Area analyzed | ~900 km² (30km × 30km) |
| Resolution | 10m (Sentinel-1 GRD) |
| Total pixels | ~9,000,000 |
| Flood probability threshold | 0.5 |
| **Flooded area (estimated)** | *See results after pipeline run* |
| **Flood percentage (estimated)** | *See results after pipeline run* |

### 4.4 Damage Assessment Results

| Feature Type | Total | Affected | Destroyed/Severe | Details |
|--------------|-------|----------|-----------------|---------|
| Buildings | *See results* | *See results* | *See results* | From OSM pre-event snapshot |
| Road segments | *See results* | *See results* | *See results* | Road segments with >50% flood overlap |
| Bridges | *See results* | *See results* | *See results* | Bridges with >30% flood overlap |

### 4.5 Cut-off Settlements

The cut-off analysis identifies settlements with no road connection to hospitals or major towns after removing flooded road segments.

| Metric | Value |
|--------|-------|
| Total settlements analyzed | *See results* |
| Total hospitals in network | *See results* |
| Reachable settlements | *See results* |
| **Cut-off settlements** | *See results* |
| Cut-off percentage | *See results* |
| Flooded road segments | *See results* |

**Top priority rescue targets** (cut-off settlements closest to damaged road segments):

*Results populated after pipeline run.*

### 4.6 Flood Path Tracing (Bonus)

Using Copernicus WorldDEM-30 elevation data downloaded from the public Copernicus DEM AWS Open Data bucket (no account required), the D8 steepest-descent algorithm traces the likely flood path from the glacier collapse area downstream through the valley. Settlements along this path are at highest risk.

The real DEM for the case-study corridor (6 Copernicus GLO-30 tiles covering 84.0–87.0°E / 27.0–29.0°N, mosaicked to `data/dem/dem.tif`, 246 MB) was downloaded from the public AWS Open Data bucket and used to run the tracer. Elevation over that area ranges from 69 m to 8,738 m. Starting at the glacier-collapse point 28.3501°N, 85.7499°E (elevation 5,961 m), the steepest-descent trace follows the valley downstream for 9,171 pixels — about 251 km — descending 5,727 m to 234 m at 27.0029°N, 86.7538°E, where it runs off the southern edge of the downloaded DEM. The largest single uphill step anywhere on the path is only 38 m, so the route is a near-monotonic descent. The result is saved as `results/flood_path.geojson` and labelled in its metadata as Copernicus WorldDEM-30 (not synthetic).

Strict D8 flow routing cannot follow this route unaided: the DEM contains 1,257,211 pixels with no strictly lower neighbour (flat terrain and local sinks), at which a strict walk stops dead. The tracer therefore breaches every sink along its lowest surrounding saddle (a minimax-Dijkstra "least-obstruction" escape) before carrying on. 1,201 sinks were breached for this trace. Those hops are short — median 5 pixels (~155 m), 90th percentile 11 pixels (~340 m), and only 1.2 % of hops exceed 775 m — so the path is best read as a least-obstruction drainage route through DEM noise and flats, not a surveyed river channel. Breaching is on by default (`--pit-radius 200`); `--pit-radius 0` disables it and reproduces strict D8, which stops at the first sink.

No settlements were intersected along this traced segment because no real OSM settlement layer for the Trishuli corridor is in place yet (the OSM data in `data/osm/` is the synthetic demo fixture). Once real OSM pre-event data is fetched for the area, the tracer will annotate the path with the settlements it passes.

**Reproduce this trace:**

```bash
python scripts/flood_path.py \
  --dem data/dem/dem.tif \
  --start-lat 28.35 --start-lon 85.75 \
  --output results/flood_path.geojson
```

`--pit-radius` (default 200 px) controls sink breaching and `--max-breaches` (default 5000) caps it; both are recorded in the output's metadata. Note that `run_pipeline.py` derives its start point as `center_lat + 0.1`, so a full-pipeline run starts at 28.4°N, 85.7°E rather than exactly the point used above.

---

## 5. Dashboard and Situation Report

### 5.1 Interactive Dashboard

The FloodMapper dashboard (`dashboard/app.py`) provides:

- **Interactive map** (Leaflet.js) centered on the study area with:
  - Flood extent overlay (semi-transparent blue polygon from flood mask)
  - OSM buildings colored by damage level (red = destroyed, orange = severe, yellow = partial, green = unaffected)
  - OSM roads with cut segments highlighted in red dashed lines
  - Bridge markers (red = damaged, green = intact)
  - Cut-off settlement markers with click-to-view details
  - Flood path overlay (blue dashed line, bonus feature)
- **Statistics panel** showing key metrics:
  - Flood extent (hectares)
  - Buildings affected (count and percentage)
  - Roads cut (count)
  - Cut-off settlements (count and percentage)
- **Layer toggle** buttons to show/hide each data layer
- **Cut-off settlements list** with nearest reachable town and distance
- **"Run Pipeline" button** to trigger the full analysis
- **"View Report" button** to open the situation report

### 5.2 Situation Report (1-pager)

The auto-generated situation report (`report/situation_report.html`) is a printable 1-page HTML document designed for rescue teams. It contains:

1. **Header:** Event name, date, area, data sources, generation timestamp
2. **Alert box:** Number of cut-off settlements prominently displayed
3. **Statistics cards:** Flood extent, buildings affected, roads cut, bridges damaged
4. **Cut-off settlements table:** Ranked list with coordinates, nearest town, distance
5. **Damage breakdown:** Detailed tables by damage level
6. **Footer:** All required data attributions and citations

**Every number in the report comes from the system's maps.** No numbers are invented or estimated by language models.

### 5.3 Comparison with EMSR927 Reference Maps

The Copernicus Emergency Management Service activation EMSR927 provides official flood extent maps for this event. Our system's results should be compared against these reference maps:

- **Copernicus EMS EMSR927 portal:** https://mapping.emergency.copernicus.eu/activation-details?activationId=EMSR927
- **Comparison metrics:**
  - Intersection over Union (IoU) between our flood mask and EMS reference
  - Precision: fraction of our detected flood that matches EMS reference
  - Recall: fraction of EMS reference flood that we detected
- **Important:** EMSR927 maps are used for **checking only** — they are NOT used as input to our system. Using them as input would be disqualification.

---

## 6. Limitations

### 6.1 Temporal Limitations

| Limitation | Impact |
|------------|--------|
| **Sentinel-1 revisit time: ~6-12 days** | Cannot provide real-time monitoring. First available after-image may be several days post-event. Cannot warn of sudden glacier collapse minutes ahead. |
| **Sentinel-2 revisit time: ~5 days (A+B constellation)** | Optical imagery even less frequent. In monsoon season, cloud-free scenes are rare — may not have any usable optical image. |
| **OSM snapshot date: 27 July 2026** | OSM data may not reflect the most current mapping. Buildings/roads mapped after this date are not included. |

### 6.2 Spatial Resolution Limitations

| Data | Resolution | Limitation |
|------|-----------|------------|
| Sentinel-1 GRD | 10m | Cannot detect damage to small structures (<10m), narrow footpaths, or individual trees/vehicles |
| Sentinel-2 L2A | 10m (visible/NIR), 20m (SWIR) | Same resolution constraints as Sentinel-1 |
| Copernicus WorldDEM-30 | 30m | Flow path tracing limited to 30m accuracy; cannot capture small gullies or micro-topography |

### 6.3 Data Quality Limitations

| Issue | Impact |
|-------|--------|
| **SAR speckle noise** | Salt-and-pepper noise in Sentinel-1 images. Mitigated by median filtering but can cause false flood detections at edges. |
| **Radar backscatter ambiguity** | Changes in backscatter can be caused by factors other than flooding: vegetation growth, soil moisture changes, wind-roughened water surfaces. False positives possible. |
| **Layover and shadowing** | In steep Himalayan terrain, radar signals can be distorted (layover: mountain tops appear "laid over" toward sensor; shadowing: areas behind peaks receive no signal). These areas have unreliable flood detection. |
| **Cloud cover (optical)** | Monsoon season over Nepal means Sentinel-2 optical images are frequently cloud-obscured. System primarily relies on Sentinel-1 radar which penetrates clouds. |
| **OSM completeness** | Remote Himalayan areas may have incomplete OSM coverage. Missing roads/buildings lead to false conclusions about connectivity or damage. |

### 6.4 Model Limitations

| Limitation | Impact |
|------------|--------|
| **Training data domain gap** | Kuro Siwo and Sen1Floods11 contain flood events from around the world, but may under-represent Himalayan topography, glacial lake outburst floods, and monsoon conditions. Model performance in Nepal may differ from training performance. |
| **Debris vs. water discrimination** | The model is trained to detect "water or debris" but may not distinguish between floodwater, mudflows, and wet ground. This is a known ambiguity in SAR-based flood mapping. |
| **No hydraulic modeling** | The flood path tracing uses a simplified D8 flow model on a DEM. It does not account for hydraulic factors like flood volume, flow velocity, channel capacity, or levees. It represents the *potential* flow path, not necessarily the *actual* inundation extent. |

### 6.5 Operational Limitations

- **Not real-time:** This is an offline analysis system. It cannot trigger alerts when a flood is happening.
- **Not certified:** This is an educational prototype. It has not been validated for operational disaster response.
- **Verification required:** Results should be verified by ground teams or higher-resolution commercial imagery before directing rescue operations.
- **Internet dependency:** Requires internet access to download satellite data and OSM data. In a real disaster, local caching or pre-downloaded data would be needed.

### 6.6 Ethical Considerations

This disaster is recent (August 2026), and many families are still affected. We treat the topic with respect:
- No images of victims are used in demos or reports
- The system is presented as an educational prototype, not as an operational tool that could be relied upon for life-or-death decisions
- Results are clearly labeled with their limitations
- Proper attribution is given to all data sources

---

## 7. Data Attribution and Citations

### Required Attribution (must appear in every submission):

> - **"Contains modified Copernicus Sentinel data 2026."**
> - **"Produced using Copernicus WorldDEM-30 — DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved."**
> - **"© OpenStreetMap contributors."**
> - **"European Union, Copernicus Emergency Management Service data"** (for EMSR927 comparison only)

### Training Dataset Citations:

**Kuro Siwo (primary training data):**
> Bountos, N., et al. (2024). "Kuro Siwo: A Large-Scale Dataset for Flood Segmentation in Satellite Imagery." *Advances in Neural Information Processing Systems (NeurIPS)*, 2024.
> Dataset available at: https://github.com/Orion-AI-Lab/KuroSiwo
> License: MIT / CC BY (per NeurIPS 2024 paper and repository README)

**Sen1Floods11 (supplementary training data, optional):**
> Bonafilia, D., Tellman, B., Anderson, T., Issenberg, S., Kugler, T., & Sullivan, J. (2020). "Sen1Floods11: A Georeferenced Cropped Sentinel-1 Dataset for Flood Detection." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPR Workshops)*, 2020.
> Dataset available at: https://github.com/cloudtostreet/Sen1Floods11
> License: CC BY 4.0 (per mirrors; original repository has no license file)

### Data Sources:

| Data | Source | URL |
|------|--------|-----|
| Sentinel-1 GRD | Copernicus Data Space Ecosystem (CDSE, OAuth2 client credentials) | https://dataspace.copernicus.eu |  **NOT YET DOWNLOADED** — `data/sentinel1/` is empty. Requires a free CDSE account (`CDSE_CLIENT_ID` / `CDSE_CLIENT_SECRET`). Same-orbit-track before/after pair required for valid change detection. |
| Sentinel-2 L2A | Copernicus Data Space Ecosystem (CDSE, OAuth2 client credentials) | https://dataspace.copernicus.eu |  **NOT YET DOWNLOADED** — `data/sentinel2/` is empty. Requires a free CDSE account (`CDSE_CLIENT_ID` / `CDSE_CLIENT_SECRET`). |
| Copernicus WorldDEM-30 (GLO-30) | Public Copernicus DEM AWS Open Data bucket (anonymous, no account) — downloaded to `data/dem/dem.tif` | https://copernicus-dem-30m.s3.amazonaws.com |
| OpenStreetMap (pre-event) | ohsome API / Overpass | https://api.ohsome.org/v1 |  **NOT YET FETCHED for the case-study area** — the `data/osm/` files on disk are the synthetic demo fixture, not a real pre-event snapshot. A real snapshot dated on/before 2026-07-27 for the Trishuli corridor is required before damage/cutoff results are real. |
| Copernicus EMS EMSR927 (reference only) | Copernicus Emergency Management Service | https://mapping.emergency.copernicus.eu |

### Software Dependencies:

- PyTorch (https://pytorch.org/) — BSD license
- torchvision (https://pytorch.org/vision/) — BSD license
- Rasterio (https://github.com/rasterio/rasterio) — BSD license
- GeoPandas (https://geopandas.org/) — BSD license
- NetworkX (https://networkx.org/) — BSD license
- Flask (https://flask.palletsprojects.com/) — BSD license
- Leaflet (https://leafletjs.com/) — BSD license

---

## 8. Running the System

### 8.1 Prerequisites

- Python 3.9+
- Copernicus Data Space Ecosystem account (free registration at https://dataspace.copernicus.eu)
- Internet connection for data download

### 8.2 Setup

```bash
# Clone and set up environment
cd hackathon-trackb
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 8.3 Run Full Pipeline

```bash
# NOTE: this example still uses the old 78.7E coordinates (Indo-Gangetic plain).
# For the actual Trishuli case study, use --center-lon 85.7 (and the real Trishuli DEM
# already in data/dem/dem.tif). The pipeline's DEM has already been switched to the
# real Trishuli GLO-30, so the flood-path tracer demonstrably works on the right geography.
python scripts/run_pipeline.py \
  --area "Bhote Koshi – Trishuli River Corridor, Nepal" \
  --center-lat 28.3 \
  --center-lon 85.7 \
  --before-date 2026-08-14 \
  --after-date 2026-08-28 \
  --osm-snapshot 2026-07-27
```

### 8.4 Run Individual Steps

```bash
# NOTE: these examples still use the old 78.7E coordinates. For the actual Trishuli
# case study, use --center-lon 85.7 for the Sentinel downloads and the OSM fetch.
# The pipeline's DEM is already the real Trishuli GLO-30 at data/dem/dem.tif.

# Download satellite data (needs Copernicus account: CDSE_CLIENT_ID / CDSE_CLIENT_SECRET)
python scripts/download_sentinel.py --center-lat 28.3 --center-lon 85.7 --before-date 2026-08-14 --after-date 2026-08-28 --output data/sentinel1

# Fetch OSM pre-event data (snapshot dated on/before 2026-07-27)
python scripts/fetch_osm.py --center-lat 28.3 --center-lon 85.7 --snapshot-date 2026-07-27 --output data/osm

# Train segmentation model (needs Kuro Siwo dataset in data/kuro_siwo/, see Section 3.2)
python scripts/train_segmentation.py --dataset data/kuro_siwo --output models/flood_unet.pt --epochs 50 --img-size 256

# Run inference (needs real Sentinel-1 before/after in data/sentinel1/before + data/sentinel1/after)
python scripts/inference.py --before-dir data/sentinel1/before --after-dir data/sentinel1/after --model models/flood_unet.pt --output results/flood_mask.tif

# Assess damage (needs real flood_mask.tif + real OSM buildings/roads/bridges)
python scripts/damage_assessment.py --flood-mask results/flood_mask.tif --osm-buildings data/osm/buildings.geojson --osm-roads data/osm/roads.geojson --osm-bridges data/osm/bridges.geojson --output results/damage_assessment.geojson

# Cut-off analysis (needs real roads + buildings + flood_mask + hospitals)
python scripts/cutoff_analysis.py --roads data/osm/roads.geojson --buildings data/osm/buildings.geojson --flood-mask results/flood_mask.tif --hospitals data/osm/hospitals.geojson --output results/cutoff_settlements.geojson

# Generate situation report (needs REAL flood_mask.tif + damage_assessment.geojson + cutoff_settlements.geojson — all currently synthetic)
python scripts/generate_report.py --area "Bhote Koshi – Trishuli Corridor" --before-date 2026-08-14 --after-date 2026-08-28 --flood-mask results/flood_mask.tif --damage results/damage_assessment.geojson --cutoffs results/cutoff_settlements.geojson --output report/situation_report.html
```

### 8.5 Start Dashboard

```bash
cd dashboard
python app.py
# Open http://localhost:5000
```

### 8.6 Demo Mode

The dashboard includes a demo mode (`/demo`) that shows sample data visualization without requiring real satellite data to be downloaded.

---

## 9. Deliverables Checklist

| Deliverable | Status |
|-------------|--------|
| Map dashboard (web UI with flood overlay, OSM features, cut-off markers) | ✅ Implemented (`dashboard/app.py`) |
| One-page situation report generated by the system | ✅ Implemented (`scripts/generate_report.py`) |
| GitHub repository with README | ✅ Implemented (`README.md`) |
| Report (max 6 pages) including limitations section | ✅ This document |
| 3-minute demo video | ⏳ To be recorded |
| Flood path tracing (bonus feature) with real Copernicus DEM | ✅ Implemented (`scripts/flood_path.py`); real GLO-30 DEM downloaded and traced (results/flood_path.geojson) |
| Required data attributions and training dataset citations | ✅ Included in Sections 7 and 1.3 |
| Case-study results compared with EMSR927 reference maps | ⏳ To be filled after pipeline run |
| Copernicus DEM (real GLO-30) for the Trishuli corridor | ✅ Downloaded from AWS Open Data bucket; 6 tiles mosaicked to data/dem/dem.tif; 251 km flood path traced |

---

## 10. Conclusion

FloodMapper demonstrates that free satellite data, combined with modern AI techniques, can provide rapid flood damage assessment for remote areas where ground access is impossible. The system answers the three critical questions for rescue teams — where the flood hit, what was damaged, and who is cut off — using only openly available data.

The key innovation is the end-to-end pipeline that takes raw satellite imagery and produces actionable information for rescue teams: a map dashboard, a situation report, and a prioritized list of cut-off settlements. The U-Net segmentation model provides automated flood detection that can be applied to any area with Sentinel-1 coverage, and the graph-based cut-off analysis provides a systematic method for identifying communities that need emergency road clearance.

However, the limitations are significant and must be clearly communicated. The system cannot warn of floods in real-time, its spatial resolution is limited to 10 meters, its accuracy in steep Himalayan terrain is uncertain, and its results require ground verification. These limitations are inherent to the data sources and should not be overlooked in any operational deployment.

---

*Generated by FloodMapper — Multimodal AI Hackathon 2026, Track B*
*Contains modified Copernicus Sentinel data 2026.*
