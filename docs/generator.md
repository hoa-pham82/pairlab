# Data generator

Real market data is too clean to show how a pipeline copes with bad input. The
generator (`generator/`) produces synthetic daily prices and a tick stream,
and injects the data problems the pipelines must handle. Everything is driven
by a YAML configuration and a seed, so a run can be repeated exactly.

```
generator/configs/*.yaml
        │
        ▼
offline/generate.py ── daily bars, symbol master ──┬─► data/generated/*.parquet      (generate_all.py)
                                                   └─► MinIO vendor-raw/, Postgres   (ingest_to_platform.py)
streaming/tick_producer.py ── ticks ───────────────┬─► data/generated/ticks.jsonl    (generate_all.py)
                                                   └─► Redpanda topic ticks.raw      (produce_to_kafka)
```

---

## 1. What is generated

| Dataset | Content | How it is made |
|---|---|---|
| Daily bars | `symbol`, `ts`, `open`, `high`, `low`, `close`, `volume` | A common market factor plus each symbol's own random walk |
| Cointegrated pairs | KO/PEP, XOM/CVX, JPM/BAC, GS/MS | The second symbol follows the first; the gap between them is a mean-reverting (Ornstein–Uhlenbeck) process |
| Symbol master | `symbol`, `sector`, `listed_at`, `delisted_at`, `valid_from`, `valid_to`, `is_current` | One row per symbol; two rows for a symbol whose sector changes |
| Ticks | `trade_id`, `symbol`, `price`, `quantity`, `event_time` (epoch milliseconds) | A random walk per symbol, sampled every 10–500 ms |

Delisting: each symbol that is not part of a pair may stop trading in the
second half of the period (`delist_rate` per year). Its bars end there. This
is what the point-in-time universe and the SCD2 dimension are for.

---

## 2. Configuration

File: `generator/configs/default.yaml`. The same keys are fields of
`GeneratorConfig` in `generator/offline/generate.py`.

| Key | Default | Meaning |
|---|---|---|
| `seed` | 42 | Same seed, same data |
| `n_symbols` | 20 | Number of symbols |
| `n_cointegrated_pairs` | 4 | Pairs built to move together |
| `start`, `days` | 2018-01-02, 2200 | First date and number of trading days (about 8.7 years) |
| `ou_theta`, `ou_sigma` | 0.10, 0.005 | Speed and noise of the pair gap's return to its mean |
| `gbm_drift`, `gbm_vol` | 0.00005, 0.012 | Daily drift and volatility of prices |
| `delist_rate` | 0.03 | Share of non-pair symbols delisted per year |
| `skew_zipf_exponent` | 1.2 | How uneven trading volume is across symbols |
| `duplicate_rate` | 0.02 | Share of bars written twice |
| `schema_evolution`, `schema_change_date` | true, 2020-06-01 | Whether, and from when, the vendor changes the file schema |
| `sector_change_date`, `sector_change_symbols`, `sector_change_new_value` | 2022-06-01, CVX, Energy-Alt | A sector reclassification, for the SCD2 dimension |
| `burst_probability` | 0.05 | Chance that a tick starts a burst of 5–20 ticks |
| `late_arrival_probability`, `late_arrival_max_seconds` | 0.03, 30 | Share of ticks with an event time 1–30 s in the past |
| `streaming_duplicate_rate` | 0.015 | Share of ticks that repeat a recent `trade_id` |
| `regime_shift_date`, `regime_shift_multiplier` | unset, 5.0 | Data drift: see section 4 |

| Config file | Use |
|---|---|
| `default.yaml` | The dataset the platform and the ML model run on; every problem switched on |
| `drift.yaml` | 10 symbols, 504 days, pairs break on 2022-01-03 |
| `e2e-small.yaml` | 6 symbols, 252 days, no injected problems; for a quick end-to-end check |

---

## 3. Simulated data problems

### Offline

| Problem (rubric) | In this domain | How it is simulated | Code |
|---|---|---|---|
| Skew | A few large stocks do most of the trading | Volume per symbol follows a Zipf distribution with exponent `skew_zipf_exponent`; the same weights choose which symbol a tick belongs to | `generate_bars` in `generator/offline/generate.py` |
| High cardinality | Every trade has its own ID; N symbols give N²/2 candidate pairs | `trade_id` is a UUID per tick | `generator/streaming/tick_producer.py` |
| Schema evolution | The vendor changes its file format | Three file versions with different columns, see below | `_split_versions` in `generator/scripts/ingest_to_platform.py` |
| Another problem: duplicates | The vendor resends rows | `duplicate_rate` of the rows in each file are copied | `inject_duplicates` in `generator/problems/inject.py` |

The three schema versions, each written as its own Parquet file under
`vendor-raw/daily_bars/schema_version=N/`:

| Version | Rows | Columns |
|---|---|---|
| v1 | Before `schema_change_date` | `symbol`, `ts`, `open`, `high`, `low`, `close`, `volume` |
| v2 | On or after the change date, about 95% | v1 plus `adj_close` |
| v3 (breaking) | On or after the change date, about 5% | v2 plus `source_exchange`, and `volume` renamed to `vol` |

The 5% share of v3 is a constant in the script (`_V3_FRACTION`), not a
configuration key.

### Streaming

| Problem (rubric) | In this domain | How it is simulated |
|---|---|---|
| Burst | Market open, or a news event | With probability `burst_probability` a tick becomes 5–20 ticks at the same instant |
| Late arrival | A slow venue or network | With probability `late_arrival_probability` the tick's `event_time` is set 1 to `late_arrival_max_seconds` seconds earlier than the time it is sent |
| Another problem: duplicates | A replay after a reconnect | With probability `streaming_duplicate_rate` the tick reuses one of the last 100 `trade_id` values |

### Where the generated data is stored

| Destination | What | Stands for |
|---|---|---|
| MinIO `s3://vendor-raw/daily_bars/schema_version=N/part-00000.parquet` | Daily bars, three schema versions | Files delivered by a data vendor |
| MinIO `s3://vendor-raw/symbol_master.parquet` | Symbol master with sector history | A reference file from another department |
| MinIO `s3://vendor-raw/sector_changes.json` | The list of sector changes | Documentation of the change |
| Postgres `vendor.symbols` | Current row per symbol | A table owned by another department |
| Redpanda topic `ticks.raw` | Ticks as JSON | A live market-data feed |

DP1 then pulls the files from `vendor-raw` into the bronze zone
([processing_jobs.md](processing_jobs.md)).

---

## 4. Data drift

`generator/configs/drift.yaml` sets `regime_shift_date` and
`regime_shift_multiplier`. From that date the noise of every pair's gap is
multiplied (by 5 in the file). The two prices stop moving together, which is
the event the drift checks (`regime_api`, the `ml_drift` DAG) must detect.

The label table and the merge of labels with features are described in
[ml.md](ml.md) §1, with the proof in
[`pngs/phase3_label_merge.txt`](pngs/phase3_label_merge.txt).

> **TODO (screenshot):** `drift.yaml` next to the `regime_api` answer for one
> pair before and after `regime_shift_date`.

---

## 5. Running it

```bash
# Local files only (no Docker): bars, symbol master, ticks, quality summary
uv run python generator/scripts/generate_all.py \
    --config generator/configs/default.yaml --output data/generated

# Into the platform: MinIO vendor-raw/ and Postgres vendor.symbols
# (needs the core profile; AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY set to the MinIO credentials)
uv run python generator/scripts/ingest_to_platform.py \
    --config generator/configs/default.yaml
```

Ticks go to Redpanda through `produce_to_kafka` in
`generator/streaming/tick_producer.py`. No script calls it yet; run it from
Python (needs the `stream` profile):

```python
from generator.streaming.tick_producer import produce_to_kafka
produce_to_kafka(["KO", "PEP", "XOM", "CVX"], n_ticks=10_000,
                 bootstrap_servers="localhost:19092")
```

After the last tick it sends one tick per symbol dated three minutes ahead, so
the Flink watermark moves on and the last one-minute windows close.

---

## 6. Data characteristics

From `data/generated/quality_summary.json`, written by `generate_all.py` with
`default.yaml` (seed 42).

### Offline

| Measure | Value |
|---|---|
| Symbols | 20 |
| Bars before duplicates | 41,452 (fewer than 20 × 2200 because delisted symbols stop early) |
| Duplicate bars injected | 829 (2.0%) |
| Bars written | 42,281 |
| Bars with no `adj_close` (schema v1, dated before 2020-06-01) | 12,843 |
| Cointegrated pairs | KO/PEP, XOM/CVX, JPM/BAC, GS/MS |
| Format and size on disk | Parquet; `daily_bars.parquet` 2.4 MB, `symbol_master.parquet` 5 KB |

**Skew: share of total trading volume, top five symbols**

| Symbol | Share |
|---|---|
| KO | 30.5% |
| PEP | 13.9% |
| XOM | 8.9% |
| CVX | 6.5% |
| JPM | 5.2% |
| Top five of 20 symbols | 65.0% |

### Streaming

| Measure | Value |
|---|---|
| Ticks | 10,000 over 10 symbols |
| Ticks sent inside a burst | 4,219 (42.2%) |
| Late ticks | 291 (2.9%) |
| Duplicate `trade_id` | 136 (1.4%) |
| Format and size on disk | JSON lines; `ticks.jsonl` 1.6 MB |

Bursts are 5% of the *events* but 42% of the *ticks*, because one burst emits
5 to 20 ticks.

> **TODO (screenshot):** the terminal output of `generate_all.py`, which prints
> the row counts and the top-five volume share.
>
> **TODO (numbers still to measure):**
> - Cardinality: `approx_count_distinct(trade_id)` on the tick stream and on
>   `symbol` (the summary has no distinct counts yet).
> - Duplicate rate **after** dedup, from the DP1 job output (rows read against
>   rows written to bronze).
> - Null counts per schema version in the bronze Delta table.
> - Tick counts by symbol, to show the Zipf skew in the stream.

---

## 7. Known gaps

- **Sector change is missing from the summary.** `quality_summary.json` shows
  `"sector_changes": []` although `default.yaml` reclassifies CVX.
  `generate_all.py` filters the YAML keys with `hasattr(GeneratorConfig, k)`,
  which is false for the list field `sector_change_symbols`, so that key is
  dropped. `ingest_to_platform.py` filters with `__dataclass_fields__` and
  keeps it.
- **`generate_all.py` does not pass the streaming settings.** It calls
  `generate_ticks` without the burst, late-arrival and duplicate rates, so the
  function's own defaults apply. They equal the values in `default.yaml`, but
  a config that changes them (such as `e2e-small.yaml`, which sets them to 0)
  has no effect on the tick file.
- **The local file and the platform files differ in how v3 is built.**
  `generate_all.py` writes one Parquet file in which 0.5% of the later rows
  have `vol` instead of `volume`. `ingest_to_platform.py` writes three
  separate files with about 5% of the later rows as v3. The second is the
  realistic one and is what DP1 reads.
