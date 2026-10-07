# Processing jobs (Spark and Flink)

Three Spark batch jobs turn the vendor's raw daily bars into feature tables.
One Flink job turns the tick stream into 1-minute bars.

```
MinIO vendor-raw/ ──DP1──► Delta bronze ──DP2──► Delta silver ──► Delta gold (dim, fact)
                                                                        │
                                                                       DP3
                                                                        ▼
                                                     Delta gold (features, OBT) ──► Postgres mirror

Redpanda ticks.raw ──Flink──► Redpanda bars.1m
                         └──► Postgres silver.stg_bars_1m
```

How the jobs are scheduled is in [orchestration.md](orchestration.md). The
injected data problems they handle are described in [generator.md](generator.md).

---

## 1. Spark

All jobs live in `platform/spark/jobs/` and run on the Compose Spark cluster
(one master, one worker with 2 cores and 2 GB). S3 credentials come from the
`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` environment variables.

### 1.1 DP1 — ingest to bronze

| | Baseline | Optimized |
|---|---|---|
| File | `dp1_ingest_bronze.py` | `dp1_ingest_bronze_optimized.py` |
| Used by Airflow | No (for comparison only) | Yes |
| Adaptive Query Execution | Off | On, with partition coalescing and skew-join handling |
| Schema evolution | Plain `read.parquet`, no schema merge | `mergeSchema=true`, then one common schema |
| Duplicates | Kept | Removed |
| Bad rows | Kept | Moved to a quarantine table |
| Distinct counts | Exact `COUNT(DISTINCT)` | `approx_count_distinct` |
| Output layout | One unpartitioned Delta table | Delta table partitioned by `dt` |

Both read the same input: `s3a://vendor-raw/daily_bars/schema_version=1|2|3/`.
Both write `s3a://delta-lake/bronze/daily_bars/`, so running the baseline
overwrites the optimized output; run the optimized job last.

**Steps of the optimized job, in order**

| Step | Problem it handles | What the code does |
|---|---|---|
| 1. Read | Schema evolution | `mergeSchema=true` gives the union of the three file schemas. `basePath` makes `schema_version` a column |
| 2. Normalize | Schema evolution (v3 renamed `volume` to `vol`) | `volume = coalesce(volume, vol)`, then drops `vol`. Adds `adj_close` and `source_exchange` if the files do not have them |
| 3. Quarantine | Bad rows | Rows with a null `symbol` or `ts`, a negative `volume`, or `high < low` go to `s3a://delta-lake/bronze/rejected_daily_bars/` |
| 4. Count | High cardinality | `approx_count_distinct` (HyperLogLog) on `symbol` and `ts` |
| 5. Salt and repartition | Skew | Adds a random `_salt` in 0–7 and repartitions into 8 partitions |
| 6. Dedup | Duplicates | `row_number()` over `(symbol, ts)`, keeping the highest `schema_version` |
| 7. Write | – | Delta, partitioned by `dt` |
| 8. Report | – | Appends `(job, rows_out, status)` to Postgres `platform.job_status`; the Airflow `validate` task reads it |

**Known limits of DP1**

- **The salt does not yet fix skew.** Step 6 shuffles again by `(symbol, ts)`
  and drops `_salt`, so the even partitions from step 5 are not kept. The daily
  bars also have no row skew: every symbol has about one row per day. The skew
  the generator creates is in trading volume and in the tick stream. A proper
  demonstration joins ticks to `dim_symbol` with and without salting
  (PLAN.md §5.3).
- **Cardinality is measured on small columns.** `symbol` has 20 values. The
  high-cardinality column in this domain is `trade_id` in the tick stream.
- **To check:** `dt` is built with `to_date(ts)`, and `ts` arrives as a
  nanosecond integer (`spark.sql.legacy.parquet.nanosAsLong=true`). Confirm in
  the Delta table that `dt` holds real dates.

> **TODO (Spark UI screenshots):** for the baseline and the optimized run, the
> Stages page showing task time and shuffle size per task (max against
> median). Add one sentence under each picture saying what it shows.
>
> **TODO (numbers):** rows read, rows quarantined, rows after dedup, and run
> time for both jobs, copied from the job output.

### 1.2 DP2 — silver and gold

File: `dp2_silver_gold.py`.

| Output | How it is built |
|---|---|
| Delta `silver/daily_bars` | Bronze, deduplicated again on `(symbol, ts)`; partitioned by `dt` |
| Delta `gold/dim_symbol` | One row per symbol with `sector`, `valid_from_ts`, `valid_to_ts`, `is_current` |
| Delta `gold/fact_daily_bar` | Silver plus `daily_return` and `log_return` (`lag()` over each symbol); partitioned by `dt` |
| Postgres `silver.stg_daily_bars`, `gold.dim_symbol`, `gold.fact_daily_bar` | JDBC copy of the three Delta tables |

- **Returns.** The first bar of each symbol has no previous close, so its
  returns are `NULL`, not zero.
- **Postgres copy.** Written with `mode=overwrite` and `truncate=true`. Spark
  then empties the table and reloads it; it does not drop it, so the primary
  keys and indexes from `init.sql` stay.

**SCD2 on `dim_symbol`: what works and what does not**

The job reads the symbol master (`vendor-raw/symbol_master.parquet`), keeps
the rows marked `is_current`, and merges them into the dimension:

```sql
MERGE INTO dim_symbol AS target USING updates AS src
ON target.symbol = src.symbol AND target.is_current = true
WHEN MATCHED AND target.sector != src.sector THEN
    UPDATE SET target.is_current = false, target.valid_to_ts = src.valid_from_ts
WHEN NOT MATCHED THEN INSERT *
```

- A new symbol is inserted as a current row. ✔
- When a sector changes, the old row is closed. ✔
- The new version of that symbol is **not inserted in the same run**, because
  the symbol matched. It is inserted on the next run.
- `valid_from_ts` is the time the job ran, not the date the change took
  effect, and only the master's current row is read. So on a first load the
  dimension holds CVX as `Energy-Alt` only; the earlier `Energy` period is not
  recorded.

The fix is described in [review-2026-10-07.md](review-2026-10-07.md), item 4.

### 1.3 DP3 — feature tables

File: `dp3_features.py`. Input: Delta `gold/fact_daily_bar`.

| Output | Content |
|---|---|
| Delta `gold/feat_symbol_daily` (partitioned by `symbol`) | `vol_21d`, `vol_63d`, `ret_21d`, `ret_63d`, `atr_14d` |
| Delta `gold/feat_pair_daily` | `hedge_ratio`, `spread`, `spread_vol`, `zscore`, `correlation_60d` for each configured pair |
| Delta `gold/obt_pair_backtest_input` | Pair features joined with open, high, low and volume of both legs |
| Postgres `gold.feat_symbol_daily`, `gold.feat_pair_daily`, `gold.obt_pair_backtest_input` | JDBC copy, `truncate=true` |

- **Windows are row-based** (`rowsBetween(-N, 0)`), so a missing trading day
  does not shrink the window.
- **Hedge ratio:** `corr(a, b) × std(a) / std(b)` over 60 bars, which equals
  the OLS slope.
- **Z-score:** `(spread − 30-bar mean) / 30-bar std`. It is `NULL` when the
  standard deviation is zero.
- **Feast columns:** every feature row gets `event_timestamp` (from `ts`) and
  `created` (job time).
- **Pairs** come from the `--pairs` argument, which Airflow fills from the
  variable `coint_pairs`.
- The job drops the two `feast.*` views before the copy and recreates them
  after it.

### 1.4 Storage optimization

File: `optimize_delta.py`. Runs `OPTIMIZE … ZORDER BY` and prints the number of
files and the size before and after.

| Table | Partitioned by | Z-ordered by |
|---|---|---|
| `gold/fact_daily_bar` | `dt` | `symbol` |
| `gold/feat_symbol_daily` | `symbol` | `ts` |
| `gold/obt_pair_backtest_input` | – | `symbol_a` |

See [data_storage.md](data_storage.md).

> **TODO (numbers):** the "Before / After" lines printed by the job for each
> table.

---

## 2. Flink

File: `platform/flink/sql/ticks_to_bars_1m.sql`, submitted with
`platform/flink/jobs/submit_ticks_to_bars.sh`.

| | |
|---|---|
| Source | Redpanda topic `ticks.raw`, JSON: `symbol`, `price`, `quantity`, `trade_id`, `event_time` (epoch milliseconds) |
| Sinks | Redpanda topic `bars.1m`; Postgres `silver.stg_bars_1m` (JDBC) |
| Cluster | One JobManager, one TaskManager, 4 slots, default parallelism 2, checkpoint every 30 s |
| Connectors | Kafka, JDBC and the Postgres driver, downloaded by `platform/flink/download_jars.sh` and copied into `/opt/flink/lib` at container start |

| Streaming problem | How the job handles it | Code |
|---|---|---|
| Late arrival | Event-time watermark 30 seconds behind the newest tick. The generator delays ticks by at most 30 s, so they still land in the right minute | `WATERMARK FOR event_ts AS event_ts - INTERVAL '30' SECOND` |
| Duplicates | Keeps the first copy of each `(symbol, trade_id)` | `ROW_NUMBER() OVER (PARTITION BY symbol, trade_id ORDER BY proctime) = 1` |
| Burst | The topic has 4 partitions, so a burst is spread over the task slots and waits in the broker until Flink catches up | Topic settings in `compose.yml` |

**Window processing**

```sql
SELECT symbol, window_start, window_end,
       FIRST_VALUE(price) AS `open`, MAX(price) AS high,
       MIN(price) AS low, LAST_VALUE(price) AS `close`,
       SUM(quantity) AS volume, COUNT(*) AS tick_count
FROM TABLE(TUMBLE(TABLE ticks_deduped, DESCRIPTOR(event_ts), INTERVAL '1' MINUTE))
GROUP BY symbol, window_start, window_end;
```

A tumbling window cuts time into fixed, non-overlapping one-minute buckets. A
bucket closes when the watermark passes its end.

**Known limits**

- A tick more than 30 s late is dropped. Flink SQL cannot send late rows to a
  separate output; that needs the DataStream API.
- The two `INSERT` statements run as two separate Flink jobs that each read
  the topic.
- The dedup keeps its state for ever (no state TTL).
- There is no baseline job (processing time, no watermark, no dedup) to
  compare bar counts with, and no measurement of the burst handling.

> **TODO (Flink UI screenshots):** the running job graph, and the watermark
> and back-pressure view of the window operator during a burst.
>
> **TODO (numbers):** ticks produced, bars written to `silver.stg_bars_1m`,
> and ticks dropped as late.
