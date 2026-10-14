# EMSR927 reference — comparison not yet performed

**Status: NOT DONE.** This file records the independent reference for a future evaluation. A real-input log-ratio baseline has now run for AOI01, but it has not been independently validated or spatially compared with EMSR927.

The reference is Copernicus EMS Rapid Mapping activation EMSR927, AOI01 (Syapru Besi, Rasuwa). It may be used only to evaluate results after the pipeline run and must never be used as an input. Cite the official EMSR927 product and use its published attribution when reporting any reference figures.

## Current evidence

| Component | Status |
|---|---|
| AOI01 boundary | Confirmed against the Copernicus activation catalogue. |
| Sentinel-1 before/after inputs | Real orbit-19 scenes reprocessed for AOI01; metadata sidecars are present. |
| Historical OSM for matching AOI | Acquired for 2026-07-27 via Overpass attic fallback and exact AOI/date cache. Counts: 29,601 buildings, 1,879 roads, 74 bridges, 6 hospitals, and 85 settlements. Building count is 3 below the ohsome aggregate. |
| AOI-specific event outputs | Pipeline completed with the log-ratio baseline. See `results/manifest.json` and `report/situation_report.html`; outputs are experimental estimates, not validated event measurements. |
| Held-out accuracy evaluation | Not done; no suitable labeled event dataset and evaluation metrics are available. |

## Next comparison steps

Compare only spatially overlapping outputs with the EMSR927 AOI01 reference, state the resolution and timing differences, and report the method and limitations. Do not fill in comparison metrics until they are computed from those artifacts.

**Required reference credit:** _European Union, Copernicus Emergency Management Service data._
