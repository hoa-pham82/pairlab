# Versioning

## Model versioning

Every pipeline run registers a new version of the model `meta_label` in the
MLflow model registry, together with:

- the hyperparameters and the feature list;
- validation metrics;
- the data version it was trained on (`data_version` parameter).

Two aliases point at versions: `production` (what `signal_api` serves) and
`challenger` (a newer model that did not beat production). `signal_api` loads
`models:/meta_label@production` when `MLFLOW_TRACKING_URI` is set.

## Data versioning (incremental)

Each run saves the training table to a Delta table,
`s3://delta-lake/ml/training_frame`, through `ml/versioning.py`.

- Rows are keyed by `pair_date_id`.
- The new table is compared with the stored one. Only new, changed or removed
  rows are written; untouched data files are not rewritten.
- If nothing changed, no new version is created.
- Any old version can be read back with
  `load_training_frame_version(uri, version)` (Delta "time travel").

So a second training run with 30 more rows stores those 30 rows, not a second
full copy. The run's MLflow record holds the version number and how many rows
were inserted, updated and deleted.

**Tests:** `tests/unit/test_versioning.py` — 9 tests: first snapshot, identical
frame, row order, new rows only (and that no existing file is removed), changed
row, removed row, reading old versions.

**Result on the Compose stack:** the table exists at version 0 with 301 rows in
one data file; a second pipeline run on unchanged data created no new version.
Proof: [`pngs/phase3_mlflow_versioning.txt`](pngs/phase3_mlflow_versioning.txt).
A second version with only changed rows will appear once the data is
regenerated.

**Downloading models from the laptop:** set
`MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD=false`. Otherwise the MLflow client is
given storage links that only resolve inside the Docker network and hangs.
`signal_api` sets this itself.
