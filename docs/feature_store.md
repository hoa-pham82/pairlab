# Feature Store (Feast)

## What it is

Feast is an open-source feature store. It separates feature *computation* (Spark)
from feature *serving* (Redis), so:
- Training always reads features from the same offline source as serving (no skew).
- Online serving gets sub-millisecond latency from Redis.
- Feature definitions are version-controlled Python objects.

---

## Architecture in this project

```
Spark DP3
  → Postgres gold.feat_* tables
      → Feast offline store (Postgres feast schema views)
          → feast materialize → Redis online store
```

---

## Feature views

### `symbol_daily_fv`

Entity: `symbol` (ticker string, e.g. "KO")  
Source: `feast.feat_symbol_daily`  
TTL: **3 days** — applies to both the online store (Redis) and offline point-in-time joins. In offline joins, Feast drops a feature row if it is older than the TTL relative to the entity timestamp being joined. Daily features update on trading days; 3 days covers weekends (Sat/Sun) so Friday's features remain valid for Monday's entity rows. Longer wastes Redis memory on stale rows.

| Feature | Description |
|---------|-------------|
| vol_21d | 21-bar annualised volatility |
| vol_63d | 63-bar annualised volatility |
| ret_21d | 21-bar cumulative log return |
| ret_63d | 63-bar cumulative log return |
| atr_14d | 14-bar average true range |

### `pair_daily_fv`

Entity: `symbol_pair` (string "A__B", e.g. "KO__PEP")  
Source: `feast.feat_pair_daily` (adds `symbol_a || '__' || symbol_b` as entity key)  
TTL: **3 days** — same reasoning as `symbol_daily_fv`; TTL applies to both online lookups and offline point-in-time joins.

| Feature | Description |
|---------|-------------|
| hedge_ratio | OLS beta of the pair |
| zscore | Spread z-score (30-bar mean/std) |
| spread_vol | 30-bar spread standard deviation |
| correlation_60d | 60-bar return correlation |

### `bars_1m_fv`

Entity: `symbol`  
Source: `silver.stg_bars_1m` (`window_start` aliased to `event_timestamp`)  
TTL: **10 minutes** — 1-minute bars are produced continuously by Flink. Serving a bar older than 10 minutes would feed the model stale market state. Refresh by materializing after the Flink aggregation window closes.

| Feature | Description |
|---------|-------------|
| open | First trade price in the 1-min window |
| high | Highest trade price |
| low | Lowest trade price |
| close | Last trade price |
| volume | Total quantity traded |
| tick_count | Number of individual ticks in the window |

---

## Offline vs online store

| Aspect | Offline (Postgres) | Online (Redis) |
|--------|-------------------|----------------|
| Use case | Training, backtest | Real-time inference |
| Latency | Seconds (SQL scan) | < 1 ms (key lookup) |
| Data | Full history | Latest row per entity |
| Populated by | Spark DP3 writes | `feast materialize` |

---

## Commands

```bash
# Apply feature definitions (run once, or after any change to features.py)
feast -c platform/feature_repo apply

# Push features into Redis up to now
feast -c platform/feature_repo materialize-incremental $(date -u +%Y-%m-%dT%H:%M:%S)

# Full re-materialize (slow — use when Redis is empty or after schema change)
feast -c platform/feature_repo materialize 2018-01-01T00:00:00 2021-12-31T00:00:00
```

---

## Online lookup example

```python
from feast import FeatureStore

store = FeatureStore(repo_path="platform/feature_repo")
result = store.get_online_features(
    features=["symbol_daily_fv:vol_21d", "symbol_daily_fv:ret_21d"],
    entity_rows=[{"symbol": "KO"}, {"symbol": "PEP"}],
).to_dict()
# {'symbol': ['KO', 'PEP'], 'vol_21d': [0.224, 0.393], 'ret_21d': [-0.004, 0.042]}
```
