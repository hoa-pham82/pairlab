"""Feast feature definitions for pairlab.

Three feature views:
  symbol_daily_fv  — per-symbol rolling features (vol, ret, ATR)       TTL 3 days
  pair_daily_fv    — per-pair spread features (hedge_ratio, zscore …)   TTL 3 days
  bars_1m_fv       — latest 1-minute OHLCV bar per symbol               TTL 10 min

TTL reasoning:
  3 days: daily features update on trading days. The TTL must cover non-trading
  days (Sat/Sun) so Friday's features remain valid for Monday's signal lookup.
  Anything longer wastes Redis memory on stale rows.
  10 min: 1-minute bars are produced continuously by Flink. Serving a bar older
  than 10 minutes would feed the model data from a stale market state.

Materialize to Redis (online store) with:
    feast -c platform/feature_repo materialize-incremental $(date +%Y-%m-%dT%H:%M:%S)
"""

from datetime import timedelta

from feast import Entity, FeatureView, Field
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import (
    PostgreSQLSource,
)
from feast.types import Float64, Int64

# ── Entities ──────────────────────────────────────────────────────────────────

symbol = Entity(
    name="symbol",
    description="Ticker symbol (e.g. KO, PEP)",
)

symbol_pair = Entity(
    name="symbol_pair",
    description="Cointegrated pair key: symbol_a__symbol_b",
)

# ── Sources ───────────────────────────────────────────────────────────────────

symbol_daily_source = PostgreSQLSource(
    name="symbol_daily_source",
    query="SELECT * FROM feast.feat_symbol_daily",
    timestamp_field="event_timestamp",
)

pair_daily_source = PostgreSQLSource(
    name="pair_daily_source",
    query=(
        "SELECT *, symbol_a || '__' || symbol_b AS symbol_pair "
        "FROM feast.feat_pair_daily"
    ),
    timestamp_field="event_timestamp",
)

# Alias window_start → event_timestamp so Feast point-in-time joins work
bars_1m_source = PostgreSQLSource(
    name="bars_1m_source",
    query=(
        "SELECT symbol, window_start AS event_timestamp, "
        "open, high, low, close, volume, tick_count "
        "FROM silver.stg_bars_1m"
    ),
    timestamp_field="event_timestamp",
)

# ── Feature Views ─────────────────────────────────────────────────────────────

symbol_daily_fv = FeatureView(
    name="symbol_daily_fv",
    entities=[symbol],
    ttl=timedelta(days=3),
    schema=[
        Field(name="vol_21d",  dtype=Float64),
        Field(name="vol_63d",  dtype=Float64),
        Field(name="ret_21d",  dtype=Float64),
        Field(name="ret_63d",  dtype=Float64),
        Field(name="atr_14d",  dtype=Float64),
    ],
    source=symbol_daily_source,
    tags={"owner": "pairlab", "layer": "gold"},
)

pair_daily_fv = FeatureView(
    name="pair_daily_fv",
    entities=[symbol_pair],
    ttl=timedelta(days=3),
    schema=[
        Field(name="hedge_ratio",      dtype=Float64),
        Field(name="zscore",           dtype=Float64),
        Field(name="spread_vol",       dtype=Float64),
        Field(name="correlation_60d",  dtype=Float64),
    ],
    source=pair_daily_source,
    tags={"owner": "pairlab", "layer": "gold"},
)

bars_1m_fv = FeatureView(
    name="bars_1m_fv",
    entities=[symbol],
    ttl=timedelta(minutes=10),
    schema=[
        Field(name="open",       dtype=Float64),
        Field(name="high",       dtype=Float64),
        Field(name="low",        dtype=Float64),
        Field(name="close",      dtype=Float64),
        Field(name="volume",     dtype=Float64),
        Field(name="tick_count", dtype=Int64),
    ],
    source=bars_1m_source,
    tags={"owner": "pairlab", "layer": "silver"},
)
