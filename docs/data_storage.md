# Data Storage

## Storage layers

```
Redpanda (Kafka)          — streaming ingest
  ↓ Flink
LocalStack / MinIO (S3)   — object store for raw Parquet and Delta Lake files
  ↓ Spark
Postgres                  — relational mirror of every layer (SQL-queryable)
  ↓ Feast (offline → online)
Redis                     — online feature store for low-latency serving
```

---

## S3 (LocalStack / MinIO)

| Bucket | Content | Format |
|--------|---------|--------|
| `vendor-raw/daily_bars/dt=*/` | Generator output (OHLCV) | Parquet, partitioned by date |
| `vendor-raw/symbol_master.parquet` | 20-symbol master file | Parquet |
| `delta-lake/bronze/daily_bars/` | Post-DP1 normalised bars | Delta Lake |
| `delta-lake/silver/daily_bars/` | Deduped bars | Delta Lake |
| `delta-lake/gold/dim_symbol/` | SCD2 dimension | Delta Lake |
| `delta-lake/gold/fact_daily_bar/` | OHLCV + returns | Delta Lake |
| `delta-lake/gold/feat_symbol_daily/` | Symbol rolling features | Delta Lake |
| `delta-lake/gold/feat_pair_daily/` | Pair spread features | Delta Lake |
| `delta-lake/gold/obt_pair_backtest_input/` | One Big Table for backtester | Delta Lake |

### Why S3 before Postgres?

S3 (object storage) is the "source of truth" for the batch pipeline because:
1. Replayability — raw data is never mutated; any job can re-read from S3 from the beginning.
2. Scale — Parquet + columnar compression handles datasets far larger than Postgres can absorb efficiently.
3. Time travel — Delta Lake's transaction log keeps every version; you can roll back to any prior state.

Postgres holds a mirror so analysts can run ad-hoc SQL without a Spark session.

---

## Delta Lake

Delta Lake adds ACID transactions, schema enforcement, and time travel on top of Parquet:

- **MERGE INTO** — upsert without reading and rewriting the whole table.
- **OPTIMIZE + ZORDER** — compacts many small files into ~128 MB files and sorts rows by the queried column so Parquet statistics can skip irrelevant row groups.
- **Schema evolution** — `mergeSchema=true` accepts new columns; `overwriteSchema=true` allows a full schema replacement.

---

## Postgres schemas

| Schema | Purpose |
|--------|---------|
| `vendor` | Raw symbol master from generator |
| `bronze` | Raw daily bars (pre-dedup) |
| `silver` | Deduped bars; 1-minute streaming bars |
| `gold` | Dimensions, facts, features, OBT |
| `feast` | Views pointing at gold tables; used by Feast offline store |

---

## Redis (online store)

Feast materializes the latest feature row per entity key from Postgres into Redis.
Redis provides sub-millisecond feature lookup at inference time — Postgres would be 10–100× slower for point lookups at scale.
