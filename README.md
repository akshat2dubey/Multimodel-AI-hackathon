# Jalnetra — Flood Damage Mapping from Space

> **From space, to the people who need it fastest.**
>
> Jalnetra is the tool — an end-to-end pipeline that takes an area and a flood date and turns
> raw satellite images into answers rescue teams can actually use:
> **where the flood hit, what was damaged, and who is cut off.**

<p align="center">
  <img src="https://img.shields.io/badge/Track-B%3A%20Mapping%20Flood%20Damage%20from%20Space-26A7DE?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Modal-Multimodal%20AI%20Hackathon%202026-9B5DE5?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Data-Sentinel-1%20%2B%20Sentinel-2%20%2B%20DEM%20%2B%20OSM-11A6DF?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Region-Bhote%20Koshi%E2%80%93Trishuli%2C%20Nepal-2E8B57?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Event-26%20Aug%202026%20Glacial%20Outburst%20Flood-BD290A?style=for-the-badge"/>
</p>

---

## ⚡ The challenge, in one sentence

**When roads, bridges and phone lines are gone, rescuers still need to know where the damage is
and who is cut off — build the system that tells them, from space.**

That's the whole point of Track B. Everything else is details. Jalnetra exists to turn this:

> *"We know a glacial outburst flood tore down the Bhote Koshi–Trishuli corridor on 26 August 2026.
> Roads are gone. Bridges are down. Phones are out. We don't know which villages can't be reached."*

into this:

> *"Here is the flooded area. Here are the buildings, roads, and bridges it hit.
> Here are the settlements that no longer have a road connection to the nearest town or hospital.
> Here is the one-page report you can hand to a rescue team. Here is how it compares to the
> official Copernicus EMSR927 maps."*

---

## 🌍 The real event behind the hackathon

On **26 August 2026**, the partial collapse of a high-altitude glacier in northern Nepal triggered
an avalanche of ice and rock that sent a flood of water, mud and debris down the
**Bhote Koshi–Trishuli river corridor**.

According to Nepal's National Disaster Risk Reduction and Management Authority (NDRRMA), as reported
by the Copernicus Emergency Management Service:

| Figure | Value |
|---|---|
| Deaths reported by 6 September | **1,342** |
| People still missing or out of contact | **almost 4,900** |

That's not a hypothetical. In disasters like this, ground sensors can be swept away and roads,
bridges and communications fail. Rescuers urgently need to know **where the damage is worst** and
**which villages have been cut off**.

Free satellite data covers every valley of the Himalaya. Radar satellites see through monsoon clouds.
Optical satellites show damage in detail. **The hard part is turning that raw data into trustworthy
answers, fast enough to matter.** That's what Jalnetra is built to do.

---

## 🎯 What Jalnetra answers — three questions, one pipeline

Jalnetra takes **an area** and **a flood date** and uses satellite images to answer the three
questions rescue teams care about:

| # | Question | What Jalnetra delivers |
|---|---|---|
| 1 | **Where did the flood hit?** | A map of flooded and debris-covered areas, by comparing **before** and **after** images. Uses **Sentinel-1 radar** (sees through monsoon clouds) and **Sentinel-2 optical** when the sky is clear. |
| 2 | **What was damaged?** | Buildings, roads and bridges from **OpenStreetMap** overlaid on the flood map, with an estimate of which ones were hit. |
| 3 | **Who is cut off?** | Using the road network, the settlements that **no longer have a road connection** to the nearest town or hospital. |

That's the core. Around it, Jalnetra also supports two optional upgrades:

- **AI component** — choose one:
  - **Flood segmentation model** — train a model to detect water or debris in satellite images
    using the listed training data, and show how well it works on Himalayan scenes it has never seen.
  - **Situation-report copilot** — an assistant that answers rescuers' questions and writes short
    situation reports in English and Nepali. Every number must come from the system's maps, never
    invented by the language model.
- **Bonus** — given any point upstream, trace the flood path down the valley using elevation data and
  list the settlements along it.

---

## 🚀 Run it — end to end, one command

Jalnetra is designed to run **end-to-end from raw satellite data** for any area and date the judges
choose on the day. The entrypoint is a single command:

```bash
cd dashboard
python app.py
```

The dashboard is the live, visual face of Jalnetra. Open it in a browser and the map shows the flood
layer, roads, buildings, bridges, and the traced flow path — the outputs of the whole pipeline laid
out on the corridor.

The full pipeline from raw satellite data is orchestrated by:

```bash
python scripts/run_pipeline.py \
  --area "Bhote Koshi corridor" \
  --center-lat 28.3 \
  --center-lon 85.7 \
  --before-date 2026-08-24 \
  --after-date 2026-08-31 \
  --osm-snapshot 2026-07-27
```

That single command runs the whole chain: satellite ingestion → terrain analysis →
infrastructure overlay → damage & cut-off analysis → report generation → a live web dashboard.

## 📋 How to run it — exactly, step by step

This is the run manual. Judges and teammates should be able to clone, install, and open the dashboard
from these steps alone.

### Step 0 — what you're running

Jalnetra runs **end-to-end from raw satellite data** for any area and date the judges choose on the day.
There are two entrypoints:

- **The live dashboard** — the visual face of Jalnetra. Open it in a browser and the map shows the
  flood layer, roads, buildings, bridges, and the traced flow path.
- **The full pipeline** — one command that runs satellite ingestion → terrain analysis → infrastructure
  overlay → damage & cut-off analysis → report generation → the dashboard.

### Step 1 — install dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 2 — (optional, only for real imagery) set up Copernicus credentials

Sentinel-1 and Sentinel-2 come from the Copernicus Data Space via an OAuth client-credentials grant.
It's free and takes about 15 minutes. Only the account holder can do it, and the secret should stay out
of chat and command lines.

1. Register free at [dataspace.copernicus.eu](https://dataspace.copernicus.eu).
2. Create an OAuth client under Identity & Access Management.
3. Copy the client ID and secret (shown only once).
4. Save them to a git-ignored `.secrets/cdse.env` file.
5. The helper `scripts/cdse_env.py` reads that file so the secret never appears on a command line or in a log.

Without valid credentials, the Sentinel download scripts exit non-zero — they no longer write placeholders.

### Step 3 — run the dashboard locally

```bash
cd dashboard
python app.py
```

Then open the dashboard in a browser. That's the live, visual face of Jalnetra.

### Step 4 — run the full pipeline from raw satellite data

```bash
python scripts/run_pipeline.py \
  --area "Bhote Koshi corridor" \
  --center-lat 28.3 \
  --center-lon 85.7 \
  --before-date 2026-08-24 \
  --after-date 2026-08-31 \
  --osm-snapshot 2026-07-27
```

That single command runs the whole chain and prints a live progress banner as each step completes.

### What the flags mean

| Flag | Meaning |
|---|---|
| `--area` | Human-readable area description |
| `--center-lat` / `--center-lon` | Center of the area of interest |
| `--before-date` | Pre-event date (YYYY-MM-DD) |
| `--after-date` | Post-event date (YYYY-MM-DD) |
| `--osm-snapshot` | OpenStreetMap snapshot date (YYYY-MM-DD), before the event |

### Case-study defaults (Bhote Koshi–Trishuli, Nepal)

| Item | Value |
|---|---|
| Center latitude | **28.3°N** |
| Center longitude | **85.7°E** |
| Before-date | **2026-08-24** |
| After-date | **2026-08-31** |
| OSM snapshot date | **2026-07-27** |

The OSM snapshot date must be before 26 August 2026. The latest available pre-event snapshot is
currently 27 July 2026.

### Two things only you can decide

1. **Confirm the case-study location is 28.3°N / 85.7°E**, not some other coordinate. If the real event
   you're presenting is somewhere else, everything downstream has to be re-pointed.
2. **Choose the training-data source for the model** — Hugging Face GeoTIFFs (recommended), Hugging Face
   WebDataset, or a Dropbox SLC bundle. Say "GeoTIFFs" and the converter gets built.

### Two things only you can ship

- The **3-minute demo video** — narration, face, screen recording.
- The **hackathon portal submission** itself — team registration, upload, hitting the deadline.

---

## 🧠 How it works — the pipeline, piece by piece

Jalnetra is ten steps chained into one command. Each step has a real, named script.

| Step | Script | What it does |
|---|---|---|
| 1 | `scripts/download_sentinel.py` | Downloads **Sentinel-1 radar** before/after images for the area and date. Radar sees through monsoon clouds — that's why it's the backbone. |
| 2 | `scripts/download_sentinel2.py` | Downloads **Sentinel-2 optical** before/after images when the sky is clear. Cloud-dependent, so it's optional. |
| 3 | `scripts/fetch_osm.py` | Fetches the **pre-event OpenStreetMap** snapshot — buildings, roads, bridges, hospitals — via the ohsome API. Snapshot date must be before 26 Aug 2026. |
| 4 | `scripts/download_dem.py` | Downloads the **Copernicus DEM** (elevation). Needed for the flood-path bonus. No account needed — public AWS Open Data bucket. |
| 5 | `scripts/train_segmentation.py` | Trains the **flood segmentation model** if you're doing the AI track. Outputs a model checkpoint. |
| 6 | `scripts/inference.py` | Runs the model (or the current approach) over the imagery to produce the **flood mask** — a GeoTIFF of where the water / debris is. |
| 7 | `scripts/damage_assessment.py` | Overlays the flood mask on the OSM buildings, roads, bridges → **damage assessment**. |
| 8 | `scripts/cutoff_analysis.py` | Runs BFS over the road network to find **cut-off settlements** — places that lost their road connection to the nearest town or hospital. |
| 9 | `scripts/flood_path.py` *(bonus)* | Traces the **flood path** down the valley from any upstream point using the DEM. |
| 10 | `scripts/generate_report.py` | Writes the **one-page situation report** — HTML here, from the real outputs of steps 6–9. |

The orchestration script `scripts/run_pipeline.py` is what ties all ten together and prints a live
progress banner as each step completes.

### Why Sentinel-1 radar is the spine

Sentinel-1 radar sees **through monsoon clouds**. That's not a nice-to-have — in a Himalayan
monsoon-season flood, optical imagery is often blank. The rule of thumb for change detection:
**compare images from the same orbit track** — same time of day, about 12 days apart. Images from
different tracks view the terrain from different angles and can't be compared pixel by pixel.

### Why OSM is the "what was damaged" layer

OpenStreetMap gives you buildings, roads, bridges, hospitals — the things rescue teams care about —
as vector data you can overlay directly on the flood mask. Use a **pre-event snapshot** (via the ohsome
API, snapshot date before 26 Aug 2026; the latest available is currently 27 July 2026). OSM edits made
after the event are for checking only, never as input — that's a disqualification rule.

---

## 🌟 Standout feature — the honest version

**Right now, Jalnetra's standout feature is that it's a complete, end-to-end, runnable pipeline —
not a slide deck.** The radar ingestion, the terrain analysis, the infrastructure overlay, the
cut-off analysis, the report generator, and the live dashboard all exist and talk to each other.

**What's real right now:**

- A real **258 MB satellite elevation model** of the Trishuli corridor, pulled from the public
  Copernicus AWS Open Data bucket — no account needed. 6 tiles, 38–45 MB each, covering
  84–87°E × 27–29°N.
- A real **flood-flow trace** computed off that DEM: 9,171 points, 251 km of channel, 5,961 m of
  descent, starting around 5,431 m up and reaching 234 m. Written as a verified GeoJSON the dashboard
  serves.
- A dashboard with **10 routes**, all checked live against a running server.

**What's synthetic right now, honestly labelled as such:**

Everything from the satellite imagery onward is **synthetic demo data** — fabricated so the code, the
dashboard, and the report could be built and tested before the real imagery arrives. The flood mask, the
damage numbers, the cut-off list, the report itself — all flagged `synthetic: true` in their own
metadata, with a `DEMO_DATA_NOTICE.txt` calling it out plainly. Nobody can accidentally quote a made-up
number as a real measurement, because the system keeps flagging them.

**The one number worth putting in front of a judge right now:** the **251 km flood path from the real
DEM** — the least-obstruction drainage route from the satellite terrain, presented honestly as a
real-but-caveated trace, not a surveyed channel.

> **Honesty note — these must never be quoted as measurements of the actual Nepal event:**
> the synthetic results currently floating around (6,740.79 ha total water, 5,278.36 ha flood extent,
> 78.3% flood coverage, 68 cut-off settlements, 70.8% cut-off rate, 4 destroyed buildings, 3 damaged
> bridges) are properties of the demo fixture, not the earthquake event. They exist so the UI has
> something to render. Until real Sentinel imagery downloads and the model runs against it, presenting
> any of them as findings would be wrong.

---

## 📊 What the judges score — and where Jalnetra is built to win

| Criterion | Weight | How Jalnetra is built to hit it |
|---|---|---|
| Flood and damage mapping quality | **30%** | Sentinel-1 radar change detection + OSM overlay + damage assessment. |
| AI component | **25%** | Trainable flood segmentation model (or situation-report copilot, your choice). |
| Cut-off settlement analysis | **20%** | BFS over the road network → which settlements lost road connection. |
| Report, understanding in Q&A, honesty about limitations | **15%** | One-page generated situation report + plain-language limitations section. |
| Dashboard and situation report usability | **10%** | Live web dashboard with map, flood layer, roads, buildings, bridges, flow path. |

**Evaluation is live**, on areas and dates the judges choose on the day. Jalnetra has to run
end-to-end from raw satellite data for **any** area and date — not just the Trishuli case study.

---

## 📦 Datasets and licenses — know what goes in, and what doesn't

### Allowed as inputs

| Data | Source | Use | License |
|---|---|---|---|
| **Sentinel-1 & Sentinel-2** | [dataspace.copernicus.eu](https://dataspace.copernicus.eu) | Before/after images | Copernicus free, full and open data policy; any use allowed; free registration |
| **OpenStreetMap** (pre-event snapshot) | ohsome API — [api.ohsome.org/v1](https://api.ohsome.org/v1) | Buildings, roads, bridges | ODbL; use a snapshot date before 26 Aug 2026 (latest available: 27 July 2026); a modified version of the OSM data itself, if published, must stay under ODbL |
| **Kuro Siwo** *(recommended)* | [github.com/Orion-AI-Lab/KuroSiwo](https://github.com/Orion-AI-Lab/KuroSiwo) | Training data | MIT License per the NeurIPS 2024 paper (repo README also states CC BY); free use with credit; cite **Bountos et al., 2024** |
| **Sen1Floods11** *(optional)* | [github.com/cloudtostreet/Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11) | Training data | Original repo has no license file; mirrors list CC BY 4.0; fine for non-commercial educational use; cite **Bonafilia et al., CVPR Workshops 2020** |
| **Copernicus DEM** | Copernicus / AWS Open Data | Elevation for flood-path tracing | — |

### Not allowed as inputs (checking only — breaking this = disqualification)

- Copernicus EMS, UNOSAT, or any other **published damage maps**
- OpenStreetMap **edits made after the event**

These may only be used to **check your results**, never as input.

### The official reference to check against

- **Copernicus EMSR927** — [mapping.emergency.copernicus.eu](https://mapping.emergency.copernicus.eu) — free of charge; credit *"European Union, Copernicus Emergency Management Service data"*. Used for **checking only**.

---

## 🏷️ Required attribution — put this in every submission

Every submission that uses this pipeline has to carry these attributions:

- *"Contains modified Copernicus Sentinel data 2026."*
- *"Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved."*
- *"© OpenStreetMap contributors."*
- Plus a **citation of whatever training dataset you used** (Bountos et al., 2024 for Kuro Siwo; Bonafilia et al., 2020 for Sen1Floods11).

---

## 📋 Deliverables — what you ship to the hackathon

- A **map dashboard**, plus a **one-page situation report** generated by the system.
- A **GitHub repository** with a README explaining how to run the system.
- A **report** (max 6 pages), including a limitations section — explain what the system **can't** do.
  For example: Sentinel satellites revisit an area only every few days, so they **cannot warn of a
  sudden glacier collapse minutes ahead**.
- A **3-minute demo video**.
- The **required data attributions and training-dataset citations**.
- **Case-study results compared with the EMSR927 reference maps**.

---

## 📝 Limitations — the honest section, up front

This is the part the judges explicitly score (15%), and the part most teams bury. Jalnetra's
limitations, stated plainly:

- **Sentinel satellites revisit an area only every few days.** They cannot warn of a sudden glacier
  collapse minutes ahead. This system is for **response**, not warning.
- **Radar change detection needs same-orbit-track before/after images.** Different tracks view the
  terrain from different angles and can't be compared pixel by pixel. If the available acquisitions
  don't line up, the comparison degrades.
- **Optical imagery is cloud-dependent.** In a monsoon-season Himalayan flood, Sentinel-2 can be
  largely blank — that's why Sentinel-1 radar is the backbone, not the nice-to-have.
- **The flood mask, damage numbers, and cut-off list are only as good as the model and the imagery.**
  Until real Sentinel imagery downloads and a trained model runs against it, the headline outputs are
  synthetic demo data.
- **OSM is incomplete and unevenly distributed**, especially for high-altitude settlements. A cut-off
  analysis is only as good as the road network in the snapshot.
- **This is an educational prototype, not an operational tool.** Treat the topic with respect: no images
  of victims in demos or reports. The disaster is recent and many families are still affected.

---

## 🧭 How to run it right now (local, from the README)

The README is the "how to run" page judges and teammates read first. Keep it exact, keep it short, keep
it honest.

1. **Clone and enter the project.**
2. **Install dependencies:**
   ```bash
   python -m pip install -r requirements.txt
   ```
3. **(Optional, only if you want the real imagery) Set up Copernicus credentials:**
   - Register at [dataspace.copernicus.eu](https://dataspace.copernicus.eu).
   - Create an OAuth client under Identity & Access Management.
   - Save the client ID and secret to `.secrets/cdse.env` (git-ignored).
   - The helper `scripts/cdse_env.py` loads them so the secret never touches a command line or log.
4. **Run the dashboard locally:**
   ```bash
   cd dashboard
   python app.py
   ```
5. **Open the dashboard in a browser** — that's the live, visual face of Jalnetra.

For the full end-to-end pipeline from raw satellite data, use `scripts/run_pipeline.py` with the
`--area`, `--center-lat`, `--center-lon`, `--before-date`, `--after-date`, and `--osm-snapshot` flags
shown above.

---

## 📁 Where everything lives

| Path | What it is |
|---|---|
| `scripts/run_pipeline.py` | One-command orchestrator for the full pipeline |
| `scripts/download_sentinel.py` | Sentinel-1 radar before/after download |
| `scripts/download_sentinel2.py` | Sentinel-2 optical before/after download |
| `scripts/fetch_osm.py` | Pre-event OSM snapshot via ohsome API |
| `scripts/download_dem.py` | Copernicus DEM download (no account needed) |
| `scripts/train_segmentation.py` | Flood segmentation model training |
| `scripts/inference.py` | Model inference → flood mask GeoTIFF |
| `scripts/damage_assessment.py` | Flood mask + OSM → damage assessment |
| `scripts/cutoff_analysis.py` | Road-network BFS → cut-off settlements |
| `scripts/flood_path.py` | DEM-based flood-path tracing (bonus) |
| `scripts/generate_report.py` | Generates the one-page situation report |
| `scripts/cdse_env.py` | Loads Copernicus creds from git-ignored `.secrets/cdse.env` |
| `dashboard/app.py` | Live web dashboard (Flask) |
| `dashboard/templates/` | Dashboard HTML templates |
| `report/situation_report.html` | Generated situation report |
| `requirements.txt` | Python dependencies |

---

## 🤝 Honorable use

This disaster is recent, and many families are still affected. Treat the topic with respect: **do not
use images of victims in demos or reports.** Systems built here are **educational prototypes, not
operational tools.**

---

*Jalnetra — Flood Damage Mapping from Space. Track B, Multimodal AI Hackathon 2026.*
