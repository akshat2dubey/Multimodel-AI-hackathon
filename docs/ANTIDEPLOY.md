# Antideploy deployment and durable files

The Antideploy app is `jalnetra.antideploy.app`. Its private S3-compatible
bucket is the source of truth for pipeline inputs and artifacts. The container
uses local disk only as working space while a pipeline stage runs; completed
and stable files are copied to the bucket, and previous versions are retained
under `history/` when a path is updated. A run is reported as complete only
after its data and artifacts have been saved.

## Environment

Antideploy supplies these bucket values to the running app; do not copy them to
the browser or commit them:

- `AWS_ENDPOINT_URL_S3`
- `AWS_REGION`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `BUCKET_NAME` (the client also accepts `S3_BUCKET` or `AWS_S3_BUCKET`)
- `APP_URL` (used to allow same-origin dashboard actions on the public hostname)

The app requires these application secrets:

- `JALNETRA_SECRET_KEY` — session signing key
- `JALNETRA_PIPELINE_TOKEN` — API token for pipeline and cancel actions

Set the two `JALNETRA_*` values in Antideploy’s secret manager. Antideploy's
Postgres database provides durable self-service accounts; its `DATABASE_URL`
is injected into the app automatically. Users create an account at
`/register`; passwords must be at least 12 characters. There is no email
verification or password recovery yet. The
`Dockerfile` binds Waitress to the platform `PORT`, enables secure cookies and
fails startup if the storage credentials are absent. `boto3` is declared in
`requirements.txt`. The small run-status record lives in the same private bucket.

## Initial migration

The existing checkout contains about 3.2 GB under `data/`. From the project
root, install the storage client and run the migration command:

```powershell
python -m pip install boto3
python scripts/persist.py push
```

The command reads the ignored local `.env` created from Antideploy’s storage
environment endpoint and prints only the number of files transferred. It
streams files; it does not load whole rasters into memory. It migrates `data/`,
`data_demo/`, `results/`, `results_demo/`, `report/`, and `models/`. Cache files,
logs, hidden files, and the smoke-only model checkpoint are excluded. It never
deletes remote objects, and a remote/local conflict stops the operation rather
than overwriting either side.

For a local recovery into a fresh checkout, place the storage variables in the
process environment or in the ignored `.env`, then run:

```powershell
python scripts/persist.py restore
```

The dashboard restores served results and reports before reading them. It
restores the Sentinel sidecars used for provenance checks and refreshes the
pipeline input/model trees before a run. Pipeline files are checkpointed while
a job runs and uploaded again before its final status is written. `data/cache/`
and application logs are temporary and are not archived.

## Deploy

1. Confirm the private bucket variables and Postgres database are enabled on
   the Antideploy app, then set the two `JALNETRA_*` application secrets above.
2. Deploy the full project directory. Exclude `.git`, `.env`, `.secrets`, local
   `data/`, `results/`, logs, and model weights from the source archive; the
   Docker ignore rules exclude those from the image context too.
3. Check deployment logs and open `https://jalnetra.antideploy.app/register` to
   create an account.
4. Run the persistence tests with `python -m pytest tests/test_object_storage.py
   tests/test_30_login.py -q`. These exercise a fresh instance, report restore,
   conflict handling, failure handling, and authentication.

When running the CLI pipeline directly on a workstation, run
`python scripts/persist.py push` after it finishes so those outputs reach the
bucket. The dashboard-managed runner performs this persistence itself.
