# Production readiness

**Status: not ready for production or operational emergency response.** This is
an engineering and research prototype. A deployable website would not make its
flood or damage measurements reliable.

## Local changes made for this readiness pass

- Demo paths are prevented from overlapping or replacing real inputs/results.
- CLI AOI bounds are checked for finite values and legal longitude/latitude
  ranges. The preset boundary remains explicitly provisional and cannot be
  promoted by a generic confirmation checkbox.
- The pipeline now stops after a failed required Sentinel-1 download and
  validates freshly fetched OSM bounds, snapshot date, and required categories
  before analysis.
- Dashboard provenance also compares both Sentinel sidecars and OSM metadata
  to run metadata, checks required rule results, and requires the manifest's
  code revision to match the running checkout.
- Dashboard authentication now defaults on at WSGI import. Session cookies are
  Secure by default; built-in and WSGI startup refuse non-loopback binding
  unless authentication and all required secrets are configured.
- Added `dashboard.wsgi:app` and Waitress for a production WSGI entry point.
- README and security notes now describe the evidence honestly and identify the
  current system as non-operational.

## Release blockers

1. **AOI and coverage:** independently verify the EMSR927 AOI01 bounds against
   the authoritative source. The existing Sentinel-1 processed scenes and OSM
   snapshot cover `[85.4, 28.0, 86.0, 28.6]`, not the provisional target AOI.
   Fresh acquisitions and an OSM extract for matching bounds are required.
2. **Scientific validation:** the only available U-Net checkpoint is a two
   epoch synthetic smoke model and detects no meaningful flood signal on the
   available scenes. Train/evaluate a chosen method on suitable labeled data,
   report held-out performance and uncertainty, and compare with independent
   reference data under the applicable input rules. Until then, the baseline
   and derived damage/cutoff figures are illustrative only.
3. **Runtime verification:** install dependencies in a clean supported
   environment; run the full pytest suite, security scanner, dependency audit,
   and a real AOI end-to-end run. These checks could not run in the current
   environment: only NumPy is installed, and network access to package indexes
   is blocked.
4. **Hosting:** choose and validate a host, HTTPS termination, reverse-proxy
   behavior, managed secrets, persistent storage, backups, monitoring, resource
   limits, and recovery. A Waitress WSGI entry point is supplied, but hosting
   configuration and production smoke tests remain deployment-specific.

## Checks completed here

- Python `compileall` on `config/`, `dashboard/`, `scripts/`, and `tests/`.
- `git diff --check`.
- Parsed the test tree: 33 files and 175 test functions (not executed).
- Read local run metadata, Sentinel-1 sidecars, OSM metadata, manifest, and
  model status without changing the ignored real input/output directories.

No files were pushed to GitHub. Reassess readiness after all four release
blockers have evidence of completion.

## Operator actions for a hosted prototype

For a deadline demo (not operational use), the operator must:

1. Choose a host with a persistent disk and long-running Python processes. The
   deployment process needs the repository after the user chooses when to push.
2. Provision HTTPS and persistent storage for the checkout's `data/`, `results/`,
   `results_demo/`, `data_demo/`, `logs/`, and `report/` paths. Keep the app bound
   to loopback behind the host's reverse proxy.
3. Add these to the host's secret manager: a random `JALNETRA_SECRET_KEY`, a
   strong `JALNETRA_DASHBOARD_PASSWORD`, a random `JALNETRA_PIPELINE_TOKEN`,
   `JALNETRA_AUTH=on`, and `JALNETRA_COOKIE_SECURE=1`. Never commit them or send
   them in chat.
4. If running live downloads, configure CDSE OAuth credentials in the secret
   store / `.secrets/cdse.env` and confirm account access. Offline fixture demos
   need no CDSE credentials.
5. Run the offline fixture on the host, open the `results_demo` dashboard, and
   confirm the synthetic-data warning, authentication, logs, and restart behavior.
6. Present every map/count as experimental. Do not enable operational use based
   on this smoke check.

For a production emergency-response claim, additionally have a domain/event
owner verify the AOI against the authoritative EMSR927 geometry, acquire scenes
and pre-event OSM for that exact AOI, and provide genuine labeled reference data
and an evaluator who can review held-out accuracy and uncertainty. Then run the
full tests/scans and real AOI validation on the final host. Those are not
substitutable by the deployment operator's checkbox or by synthetic training.
