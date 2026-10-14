# Jalnetra — EMSR927 AOI01 preliminary run

**Status:** Experimental baseline output. This is not a validated damage assessment and is not suitable for emergency decisions.

## Study area and inputs

The run covers Copernicus EMSR927 AOI01 at Syapru Besi, Rasuwa District, Nepal, with bounds `[85.117358, 27.972395, 85.581515, 28.077922]`. The event date is 26 August 2026.

The pipeline used Sentinel-1 GRD acquisitions from 24 August and 5 September 2026 on relative orbit 19. Both were locally preprocessed to the AOI with VV/VH rasters and `S1_PROCESSED.json` sidecars. Historical OSM for 27 July 2026 was fetched through the Overpass attic fallback and cached for this AOI/date. The extract contains 29,601 buildings, 1,879 roads, 74 bridges, 6 hospitals, and 85 settlements. The building count is three below the available ohsome aggregate count. No suitable cloud-limited Sentinel-2 scenes were available.

## Method

No event-trained segmentation model is available. The run used the Sentinel-1 log-ratio baseline (VV backscatter change threshold), removed small speckle components, and excluded steep slopes above 35 degrees. Asset results mean spatial overlap with the estimated flood mask; they do not mean structural damage. Cut-off analysis tests connectivity on the mapped OSM road network.

## Current outputs

| Measure | Run output |
|---|---:|
| Newly flooded mask | 186 pixels; 16.72 ha |
| Buildings with major flood overlap | 5 of 29,601 |
| Roads with any flood overlap | 1 of 1,879 |
| OSM road ways fully cut | 0 |
| Flood-intersected graph edges | 7 |
| Bridges with flood overlap | 0 of 74 |
| Settlements marked cut off | 0 of 6 analyzed |

These figures are outputs of the selected threshold and available inputs. In particular, a zero cut-off result is not evidence that all communities are accessible: the OSM network and six settlement points may be incomplete, and the flood mask has not been checked against independent labels.

The full artifacts and provenance are in `results/manifest.json`; the generated one-page output is `report/situation_report.html`. The manifest reports `osm_snapshot_before_event`, `acquisitions_bracket_event`, `no_forbidden_input_sources`, and `sentinel_orbit_same_for_pair` as true. The EMSR927 comparison and held-out accuracy evaluation remain undone.

## Limitations

- The log-ratio mask has not been independently evaluated for this event; all dependent counts inherit that uncertainty.
- Sentinel-1 scenes are two days before and ten days after the event, so changes in that interval are not uniquely attributable to the flood.
- OSM coverage can be incomplete. The building extract differs from the available aggregate count by three features.
- The system estimates water/flood overlap; it does not infer structural collapse, population affected, or safe routes.
- This research prototype is not an operational emergency-response tool.

## Required attribution

_Contains modified Copernicus Sentinel data 2026._

_Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved._

_© OpenStreetMap contributors._

**Reference credit when EMSR927 figures are used:** _European Union, Copernicus Emergency Management Service data._
