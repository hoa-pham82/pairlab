# Data lineage (DataHub)

Status: emitter written and tested offline. DataHub itself has **not** been started
yet (it does not fit in memory next to the running stack), so there are no UI
screenshots yet. See "Still to do".

## What is published

Lineage answers "where did this table come from, and what breaks if I change it?".
We publish three kinds of things to DataHub:

| DataHub entity | In this project | Count |
|---|---|---|
| Dataset | a table or file set (S3 raw files, Delta tables, Postgres tables, Feast online views) | 19 |
| Data flow | an Airflow DAG | 4 |
| Data job | one task in a DAG, with the datasets it reads and writes | 10 |

Pipelines covered: `dp1_ingest_bronze`, `dp2_silver_gold`, `dp3_features`,
`feast_materialize`.

```
s3 vendor-raw/daily_bars ──dp1.ingest──▶ delta bronze/daily_bars ──dp2.ingest──▶ delta silver/daily_bars
                                         delta bronze/rejected_daily_bars        delta gold/dim_symbol
s3 vendor-raw/symbol_master.parquet ─────────────────────────────▶               delta gold/fact_daily_bar ──dp3.ingest──▶ delta gold/feat_symbol_daily
                                                                                 (+ Postgres copies)                       delta gold/feat_pair_daily
                                                                                                                           delta gold/obt_pair_backtest_input
                                                                                                                           (+ Postgres copies) ──feast.materialize──▶ Feast online views
```

Lineage is at task level: a task's outputs are shown as coming from all of its
inputs. Inside `dp2.ingest` and `dp3.ingest` the Postgres tables are really copies
of the Delta tables; that inner step is not shown.

## Files

| File | Role |
|---|---|
| `platform/datahub/lineage.yaml` | The spec: datasets, DAGs, tasks, and each task's inputs/outputs. Edit this to add a pipeline. |
| `platform/datahub/lineage_spec.py` | Loads the spec and rejects inconsistent ones. |
| `platform/datahub/emit_lineage.py` | Converts the spec to DataHub proposals and sends them. |
| `tests/unit/test_lineage_spec.py` | Validation rules, plus checks that the spec matches the DAG and Spark sources. |
| `tests/unit/test_datahub_emit.py` | Checks the proposals (identifiers and edges). |

The lineage is declared by hand, not captured from Airflow at run time. The tests
keep it honest: they fail if a DAG id or task id in the spec is not in the DAG
file, or a dataset name does not appear in the pipeline code.

## How to run

The DataHub client is not a project dependency: it would downgrade `pydantic`
in the shared environment. `uv` installs it into a throwaway environment from the
header of the script.

```bash
# Print what would be sent, send nothing
uv run platform/datahub/emit_lineage.py --dry-run

# Send to a running DataHub (default http://localhost:8080, or DATAHUB_GMS_URL)
uv run platform/datahub/emit_lineage.py

# Emitter tests (skipped by the normal `uv run pytest`)
uv run --no-project --with "acryl-datahub[datahub-rest]" --with pyyaml --with pytest \
    pytest tests/unit/test_datahub_emit.py -o addopts=
```

Proof so far: `docs/pngs/datahub_lineage_dryrun.json` (43 proposals from a dry run).

## Still to do

- Start DataHub in an agreed window with other containers stopped, emit, and take
  the UI screenshots for DP1, DP2, DP3 into `docs/pngs/`.
- Validation and data contracts on DataHub (the other half of the rubric item):
  `platform/contracts/` is still empty, so nothing is published for it.
- Lineage for the RAG ingest pipeline once its DAG exists.
