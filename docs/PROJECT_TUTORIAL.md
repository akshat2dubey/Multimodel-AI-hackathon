# Jalnetra — current run guide and status

**Updated:** 15 October 2026. This replaces the older project audit and its stale completion estimates.

Jalnetra is a research prototype for flood-change mapping from Sentinel-1 and overlaying the estimated mask with OpenStreetMap infrastructure. It is not an operational emergency-response tool.

## Case-study status

The confirmed study area is Copernicus EMSR927 AOI01, Syapru Besi, Rasuwa District, Nepal: `[85.117358, 27.972395, 85.581515, 28.077922]`. Before/after Sentinel-1 scenes from 24 August and 5 September 2026 share relative orbit 19 and have AOI-matched preprocessing sidecars. Pre-event OSM layers for 27 July were acquired for the same bounds through the Overpass attic fallback. The ohsome v2 extraction key still receives an access-denied response; the fallback handled a truncated mirror response and retrieved the required layers.

| Status | Evidence |
|---|---|
| REAL INPUTS | Sentinel-1 pair and sidecars; dated OSM (29,601 buildings, 1,879 roads, 74 bridges, 6 hospitals, 85 settlements); Copernicus DEM. |
| BASELINE RUN | `results/manifest.json` records all four provenance rules as true. The log-ratio baseline reports 16.72 ha newly flooded, 5 buildings with flood overlap, 1 road touched (0 fully cut), 0 bridges with flood overlap, and 0 of 6 mapped settlements cut off. |
| NOT VALIDATED | No event-trained model, held-out accuracy evaluation, uncertainty assessment, or spatial comparison with EMSR927 has been completed. Baseline outputs are experimental estimates. |
| OPTIONAL INPUT NOT AVAILABLE | No Sentinel-2 scenes met the cloud limit for this run. |
| HUMAN DELIVERABLES | The 3-minute demo video and portal submission still need to be produced by the project owner. |

The building extract is three features below the available ohsome aggregate count. The settlement network is incomplete; a zero cut-off count must not be read as proof that every community is connected. Full results are in `results/`, with the generated report in `report/situation_report.html` and the concise run report in `report/report.md`.

## Run the case study again

From the repository root, install dependencies in a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Then run the confirmed preset:

```powershell
python scripts/run_pipeline.py --preset trishuli --confirm-aoi
```

The command downloads/checks Sentinel-1, attempts optional Sentinel-2, fetches dated OSM (using the exact AOI/date cache when present), fetches the DEM, preprocesses Sentinel-1 with sidecars, runs the baseline if no verified trained checkpoint exists, builds impact/connectivity layers, writes the report, and validates provenance. It replaces generated files under `results/` and `report/`.

To open the local dashboard:

```powershell
cd dashboard
python app.py
```

The normal results dashboard reads `../results` by default. The offline demo uses synthetic data in separate `data_demo/` and `results_demo/` folders:

```powershell
cd ..
python demo/run_offline.py
```

Do not mix demo artifacts with `results/` or use demo numbers as case-study results.

## Credentials and data

- CDSE OAuth credentials go in the ignored `.secrets/cdse.env`; never paste secrets into chat or commit them.
- The ohsome key goes in `.secrets/ohsome.env` as `OHSOME_API_KEY=actual_key`. Current v2 extraction access is denied for this key, but the tested Overpass attic fallback can fetch historical geometry. Do not use a current OSM snapshot as a substitute for pre-event data.
- The before/after pair must bracket the event and use the same relative orbit. OSM input must be dated before 26 August 2026. EMSR927 reference products are for checking only, never model input.

## Remaining work

- Independently evaluate the log-ratio baseline with suitable held-out flood labels and document its limitations/uncertainty.
- Compare only spatially overlapping AOI01 outputs against the EMSR927 reference; the comparison is not done yet.
- Finish the six-page report, judge Q&A, and 3-minute demonstration deliverables after the evaluation claims are settled.
- Run the final test/security workflow in the submission environment and confirm the workflow passes. The local end-to-end run and manifest do not replace the test suite or external scientific review.

## Attribution

_Contains modified Copernicus Sentinel data 2026._

_Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved._

_© OpenStreetMap contributors._

When EMSR927 figures are used: _European Union, Copernicus Emergency Management Service data._
