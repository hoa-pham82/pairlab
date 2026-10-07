"""Tests for the pure DataFrame transforms inside the DP1, DP2 and DP3 Spark jobs."""

from __future__ import annotations

import math

import pytest

NS = 1_000_000_000
DAY_NS = 86_400 * NS

BAR_COLUMNS = "symbol string, ts long, open double, high double, low double, close double, volume double"


def _bars(spark, rows, schema=BAR_COLUMNS):
    return spark.createDataFrame(rows, schema)


def _fact(spark, closes_by_symbol: dict[str, list[float]]):
    """fact_daily_bar-shaped rows: one bar per day per symbol, with log returns."""
    rows = []
    for symbol, closes in closes_by_symbol.items():
        prev = None
        for day, close in enumerate(closes):
            log_ret = None if prev is None else math.log(close / prev)
            rows.append((symbol, day * DAY_NS, close, close + 1, close - 1, close, 1000.0, log_ret, "d"))
            prev = close
    return spark.createDataFrame(
        rows,
        "symbol string, ts long, open double, high double, low double, close double, "
        "volume double, log_return double, dt string",
    )


# ── DP1: schema normalization ──────────────────────────────────────────────────
# Partitions: v1 (no adj_close, no schema_version), v2 (adj_close present),
#             v3 (volume renamed to vol, so volume is null and vol carries the value).

class TestNormalizeSchema:
    def test_v1_gets_defaults(self, spark):
        from dp1_ingest_bronze_optimized import _normalize_schema

        row = _normalize_schema(_bars(spark, [("KO", 0, 1.0, 2.0, 0.5, 1.5, 10.0)])).first()
        assert row["adj_close"] == 1.5
        assert row["schema_version"] == 1
        assert row["source_exchange"] is None

    def test_v2_keeps_its_adj_close(self, spark):
        from dp1_ingest_bronze_optimized import _normalize_schema

        df = spark.createDataFrame(
            [("KO", 0, 1.0, 2.0, 0.5, 1.5, 10.0, 1.4, 2)],
            BAR_COLUMNS + ", adj_close double, schema_version int",
        )
        row = _normalize_schema(df).first()
        assert row["adj_close"] == 1.4
        assert row["schema_version"] == 2

    def test_v3_vol_becomes_volume(self, spark):
        from dp1_ingest_bronze_optimized import _normalize_schema

        df = spark.createDataFrame(
            [("KO", 0, 1.0, 2.0, 0.5, 1.5, None, 7.0)], BAR_COLUMNS + ", vol double"
        )
        out = _normalize_schema(df)
        assert "vol" not in out.columns
        assert out.first()["volume"] == 7.0


# ── DP1: validation split ──────────────────────────────────────────────────────
# Partitions: valid row; missing key (symbol, ts); bad volume; inverted range;
#             unknown value (null) — must go to quarantine, never vanish.
# Boundaries: volume 0 (valid) vs -1 (bad); high == low (valid) vs high < low (bad).

@pytest.mark.parametrize(
    ("row", "is_bad"),
    [
        (("KO", 0, 1.0, 2.0, 1.0, 1.5, 10.0), False),
        (("KO", 0, 1.0, 2.0, 1.0, 1.5, 0.0), False),
        (("KO", 0, 1.0, 2.0, 2.0, 2.0, 10.0), False),
        ((None, 0, 1.0, 2.0, 1.0, 1.5, 10.0), True),
        (("KO", None, 1.0, 2.0, 1.0, 1.5, 10.0), True),
        (("KO", 0, 1.0, 2.0, 1.0, 1.5, -1.0), True),
        (("KO", 0, 1.0, 1.0, 2.0, 1.5, 10.0), True),
        (("KO", 0, 1.0, 2.0, 1.0, 1.5, None), True),
        (("KO", 0, 1.0, None, 1.0, 1.5, 10.0), True),
    ],
    ids=["valid", "volume-0", "high-eq-low", "null-symbol", "null-ts", "volume-neg",
         "high-lt-low", "null-volume", "null-high"],
)
def test_split_bad_rows(spark, row, is_bad):
    from dp1_ingest_bronze_optimized import _split_bad_rows

    good, bad = _split_bad_rows(_bars(spark, [row]))
    assert (good.count(), bad.count()) == ((0, 1) if is_bad else (1, 0))


# ── DP1 / DP2: dedup on (symbol, ts), newest schema version wins ───────────────

DEDUP_SCHEMA = BAR_COLUMNS + ", schema_version int, ingested_at string"


@pytest.mark.parametrize("fn", ["dp1", "dp2"])
def test_dedup_keeps_newest_schema_version(spark, fn):
    from dp1_ingest_bronze_optimized import _dedup
    from dp2_silver_gold import silver_from_bronze

    dedup = {"dp1": _dedup, "dp2": silver_from_bronze}[fn]
    df = spark.createDataFrame(
        [
            ("KO", 0, 1.0, 2.0, 1.0, 1.0, 10.0, 1, "x"),
            ("KO", 0, 1.0, 2.0, 1.0, 2.0, 10.0, 2, "x"),
            ("KO", DAY_NS, 1.0, 2.0, 1.0, 3.0, 10.0, 1, "x"),
        ],
        DEDUP_SCHEMA,
    )
    out = {r["ts"]: r["close"] for r in dedup(df).collect()}
    assert out == {0: 2.0, DAY_NS: 3.0}


def test_silver_drops_ingest_time_and_stamps_silver_time(spark):
    from dp2_silver_gold import silver_from_bronze

    df = spark.createDataFrame([("KO", 0, 1.0, 2.0, 1.0, 1.0, 10.0, 1, "x")], DEDUP_SCHEMA)
    out = silver_from_bronze(df)
    assert "ingested_at" not in out.columns
    assert out.first()["silver_ts"] is not None


# ── DP2: fact returns ──────────────────────────────────────────────────────────

def test_fact_returns_are_per_symbol(spark):
    from dp2_silver_gold import fact_from_silver

    silver = spark.createDataFrame(
        [("KO", 0, 100.0), ("KO", DAY_NS, 110.0), ("PEP", 0, 50.0), ("PEP", DAY_NS, 25.0)],
        "symbol string, ts long, close double",
    )
    rows = {(r["symbol"], r["ts"]): r for r in fact_from_silver(silver).collect()}
    assert rows[("KO", 0)]["daily_return"] is None
    assert rows[("KO", DAY_NS)]["daily_return"] == pytest.approx(0.1)
    assert rows[("KO", DAY_NS)]["log_return"] == pytest.approx(math.log(1.1))
    assert rows[("PEP", DAY_NS)]["daily_return"] == pytest.approx(-0.5)


# ── DP3: feature tables ────────────────────────────────────────────────────────

def test_symbol_features_have_feast_timestamps(spark):
    from dp3_features import symbol_features

    out = symbol_features(_fact(spark, {"KO": [10.0, 11.0, 12.0]}))
    first = out.orderBy("ts").first()
    assert first["event_timestamp"].timestamp() == 0
    assert first["created"] is not None
    assert {"dt", "daily_return"}.isdisjoint(out.columns)


def test_pair_hedge_ratio_recovers_exact_ratio(spark):
    from dp3_features import pair_features

    closes_b = [10.0, 11.0, 13.0, 12.0, 15.0]
    fact = _fact(spark, {"A": [2 * c for c in closes_b], "B": closes_b})
    last = pair_features(fact, "A", "B").orderBy("ts").collect()[-1]
    assert last["hedge_ratio"] == pytest.approx(2.0)
    assert (last["symbol_a"], last["symbol_b"]) == ("A", "B")


def test_pair_zscore_is_null_when_spread_is_constant(spark):
    from dp3_features import pair_features

    closes_b = [10.0, 11.0, 13.0]
    fact = _fact(spark, {"A": [2 * c for c in closes_b], "B": closes_b})
    assert all(r["zscore"] is None for r in pair_features(fact, "A", "B").collect())


def test_pair_features_only_use_shared_timestamps(spark):
    from dp3_features import pair_features

    fact = _fact(spark, {"A": [1.0, 2.0, 3.0], "B": [1.0, 2.0]})
    assert pair_features(fact, "A", "B").count() == 2


def test_obt_has_one_row_per_pair_and_day_with_both_legs(spark):
    from dp3_features import obt_from, pair_features

    fact = _fact(spark, {"A": [1.0, 2.0, 4.0], "B": [1.0, 3.0, 2.0]})
    obt = obt_from(pair_features(fact, "A", "B"), fact)
    assert obt.count() == 3
    assert {"open_a", "open_b", "volume_a", "volume_b"} <= set(obt.columns)
    assert {"spread", "spread_mean"}.isdisjoint(obt.columns)
