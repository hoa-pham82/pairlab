# ML — meta-labeling model

The model does not pick trades. The pairs strategy proposes an entry; the model
answers one question: *is this entry likely to work?* This is called
meta-labeling.

## 1. Label table

**Table:** `gold.label_pair_reversion(pair_date_id, label)` — two columns, as the
rubric asks.

| Column | Meaning |
|---|---|
| `pair_date_id` | Pair and entry date, e.g. `KO__PEP|2020-03-02` |
| `label` | `1` = the spread reverted, `0` = it stopped out or timed out |

**How a label is decided** (`ml/labels.py`):

1. An *entry* is a day where the absolute z-score moves above 2.0.
2. Look at the next 20 trading days.
3. If the absolute z-score falls below 0.5 first, the label is `1`.
4. If it rises above 4.0 first, or neither happens in 20 days, the label is `0`.
5. Entries in the last 20 days of data with no outcome yet are dropped, because
   their result is not known.

The thresholds are the strategy's own entry, exit and stop-loss levels
(PLAN.md §4.1).

**Run it:**

```
uv run python -m ml.build_labels
```

**Result on the default dataset:** 301 entries across 4 pairs, 58% labelled `1`.

**Proof:** [`pngs/phase3_label_merge.txt`](pngs/phase3_label_merge.txt) shows the
table definition, the class counts, a sample of labels merged with
`gold.feat_pair_daily`, and that all 301 labels match a feature row.

**Tests:** `tests/unit/test_labels.py` — 32 tests. Equivalence partitions and
boundary values are documented at the top of the file; a Hypothesis test checks
that adding later data never changes a label that was already decided.

## 2. Training dataset

`ml/dataset.py` turns the label table into a training table.

1. `entity_frame` splits `pair_date_id` into the Feast entity key
   (`symbol_pair`) and the entry date.
2. `load_training_frame` asks Feast for the pair features (`zscore`,
   `hedge_ratio`, `spread_vol`, `correlation_60d`) as of each entry date.
3. `purged_time_split` puts the latest 25% of signals in validation and removes
   training rows whose 20-day label window reaches the validation period.

On the default dataset: 301 rows, no nulls; 224 training rows (2018-02 to
2024-04) and 76 validation rows (2024-05 to 2026-05). One row is purged.

## 3. Notebook

[`notebooks/ml.ipynb`](../notebooks/ml.ipynb) runs the five rubric steps and
is saved with its outputs.

| Step | What it does | Code |
|---|---|---|
| 1. Load | Reads the label table, joins features through Feast | `load_training_frame` |
| 2. Split | Time-based split with a purge gap | `purged_time_split` |
| 3. Train | LightGBM classifier, fixed seed | `train_model` |
| 4. Evaluate | AUC, precision, recall, accuracy, take rate | `evaluate` |
| 5. Save | Writes `models/meta_label.joblib` and reloads it | `save_model` |

Re-run it:

```
cd notebooks && uv run jupyter nbconvert --to notebook --execute --inplace ml.ipynb
```

LightGBM on macOS needs the OpenMP library: `brew install libomp`.

**Result:** validation AUC 0.38, precision 0.47 against a baseline of 0.50 for
taking every signal. The model has no skill on this dataset; an AUC below 0.5
on 76 validation rows means it is slightly worse than a coin flip here, which
is within what chance produces at this sample size. That is expected: the
generated spreads are random by construction, so four features say little
about which signal will revert. The rubric asks for a working,
task-appropriate pipeline, not accuracy. More features (half-life,
cointegration p-value, volatility ratio) are the planned improvement.

**Tests:** `tests/unit/test_dataset.py` (14) and `tests/unit/test_train.py` (9).

## 4. Training pipeline

`ml/pipeline.py` runs the notebook's steps as one function, `run_pipeline`,
and records the result in MLflow.

| Step | What happens |
|---|---|
| 1. Version data | The training table is saved as the next version of a Delta table (see [versioning.md](versioning.md)) |
| 2. Split | Same purged time split as the notebook |
| 3. Train | LightGBM with fixed seed |
| 4. Evaluate | AUC, precision, recall, accuracy, take rate on validation |
| 5. Register | Parameters, metrics, the data version and the model go to MLflow; the model is registered as `meta_label` |

After registering, the new version gets an alias:

- `production` if there is no production model yet, or its validation AUC is
  at least the current production model's;
- `challenger` otherwise. The production model is left in place.

Run it (needs the `core` and `ml` Compose profiles):

```
uv run python -m ml.pipeline
```

**Tests:** `tests/unit/test_pipeline.py` — 11 tests against a local SQLite
MLflow store, covering logging, registration, incremental data versions, the
promotion boundary, and failing cleanly on too few rows.

**Result on the Compose stack:** two runs registered `meta_label` versions 3
and 4; `production` points at version 4; both record `data_version` 0. (Versions
1 and 2 were trained on the earlier dataset and removed after their files were
lost with the object store.) Proof:
[`pngs/phase3_mlflow_versioning.txt`](pngs/phase3_mlflow_versioning.txt).

**Status:** the Airflow DAG `ml_train` wraps these steps; it has not run yet.

## 5. Backtest with and without the filter

`ml/backtest_filter.py` runs the engine twice on the same bars: the plain pairs
strategy, and the same strategy wrapped in `MetaLabelFilter`
(`ml/filtered_strategy.py`), which drops an entry when the model's probability
is below 0.5.

- The filter looks up the features dated the same day as the signal, never a
  later day (tested).
- The filter only acts from 2024-05-20, the start of the model's validation
  period. Before that date both runs are identical, so the difference is
  out of sample.

```
uv run python -m ml.backtest_filter --out docs/pngs/phase3_backtest_filter.json
```

Result on the default dataset, 2018 to mid-2026, 4 pairs, $1M starting cash:

| Metric | Without filter | With filter |
|---|---|---|
| Sharpe ratio | 1.58 | 1.62 |
| Total P&L | $47,673 | $48,100 |
| Max drawdown | 0.22% | 0.22% |
| Leg trades (two per pair trade) | 516 | 496 |
| Hit rate | 52.9% | 53.2% |
| Profit factor | 1.17 | 1.18 |
| Entries scored by the filter | – | 61 (51 taken, 10 skipped) |

**Reading it:** the two runs are practically the same. The filter skipped 10
of 61 out-of-sample entries and P&L moved by $427, which is far inside the
noise for that few trades. This matches the model's validation AUC: it has no
skill on this data, so it cannot help. The machinery is in place; a better
model is what would make the filter matter.

Raw output: [`pngs/phase3_backtest_filter.json`](pngs/phase3_backtest_filter.json).

**Tests:** `tests/unit/test_filtered_strategy.py` (16) and
`tests/unit/test_backtest_filter.py` (9): a filter that accepts everything
reproduces the baseline exactly; one that rejects everything never trades; a
filter not yet active changes nothing.

**Note on the trade statistics:** an earlier run showed a profit factor of 0.51
next to a positive P&L. That was an engine bug (short legs were dropped when
matching entries to exits) and is fixed; the numbers above are from the fixed
engine.
