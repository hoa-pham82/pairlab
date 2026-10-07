"""DP3 — compute offline feature tables from gold.fact_daily_bar.

Writes:
  Delta s3a://delta-lake/gold/feat_symbol_daily/
  Delta s3a://delta-lake/gold/feat_pair_daily/
  Delta s3a://delta-lake/gold/obt_pair_backtest_input/
  Postgres gold.feat_symbol_daily, gold.feat_pair_daily, gold.obt_pair_backtest_input

Features computed:
  Symbol-level : vol_21d, vol_63d, ret_21d, ret_63d, atr_14d
  Pair-level   : hedge_ratio (OLS slope), zscore (30d), spread_vol, correlation_60d
  Label        : did the spread revert within horizon_bars?

Run:
    spark-submit --master spark://spark-master:7077 \
        --packages io.delta:delta-spark_2.12:3.2.0,org.apache.hadoop:hadoop-aws:3.3.4,\
org.postgresql:postgresql:42.7.3 \
        dp3_features.py
"""

from __future__ import annotations

import argparse
import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    avg,
    col,
    corr,
    current_timestamp,
    from_unixtime,
    lit,
    sqrt,
    stddev,
    when,
)
from pyspark.sql.functions import (
    sum as spark_sum,
)
from pyspark.sql.window import Window


def build_spark(s3_endpoint: str) -> SparkSession:
    """Build SparkSession with Delta Lake and S3A."""
    return (
        SparkSession.builder.appName("DP3-Features")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.hadoop.fs.s3a.endpoint", s3_endpoint)
        .config("spark.hadoop.fs.s3a.access.key", os.environ.get("AWS_ACCESS_KEY_ID", "pairlabs3"))
        .config("spark.hadoop.fs.s3a.secret.key", os.environ.get("AWS_SECRET_ACCESS_KEY", "pairlabs3key"))
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )


# ─── Symbol features ─────────────────────────────────────────────────────────

def symbol_features(fact):
    """Rolling volatility, return and ATR features per symbol, with Feast timestamps."""
    w21 = Window.partitionBy("symbol").orderBy("ts").rowsBetween(-20, 0)
    w63 = Window.partitionBy("symbol").orderBy("ts").rowsBetween(-62, 0)
    w14 = Window.partitionBy("symbol").orderBy("ts").rowsBetween(-13, 0)
    return (
        fact
        .withColumn("vol_21d", stddev("log_return").over(w21) * sqrt(lit(252.0)))
        .withColumn("vol_63d", stddev("log_return").over(w63) * sqrt(lit(252.0)))
        .withColumn("ret_21d", spark_sum("log_return").over(w21))
        .withColumn("ret_63d", spark_sum("log_return").over(w63))
        # ATR approximation using (high-low) range
        .withColumn("atr_14d", avg(col("high") - col("low")).over(w14))
        # ts is Unix nanoseconds; divide by 1e9 to get seconds before casting
        .withColumn("event_timestamp", from_unixtime(col("ts").cast("double") / 1e9).cast("timestamp"))
        .withColumn("created", current_timestamp())
        .drop("daily_return", "dt")
    )


def build_feat_symbol(spark: SparkSession) -> None:
    """Compute symbol features from fact_daily_bar and write them to Delta."""
    fact = spark.read.format("delta").load("s3a://delta-lake/gold/fact_daily_bar/")
    feat = symbol_features(fact)

    feat.write.format("delta").mode("overwrite").partitionBy("symbol").save(
        "s3a://delta-lake/gold/feat_symbol_daily/"
    )
    print(f"feat_symbol_daily rows: {feat.count()}")


# ─── Pair features ───────────────────────────────────────────────────────────

def pair_features(fact, sym_a: str, sym_b: str):
    """Pair features for one pair, aligned on shared timestamps.

    hedge_ratio: OLS slope of close_a ~ close_b over trailing 60 bars
    zscore:      (spread - mean) / std over trailing 30 bars
    """
    a = fact.filter(col("symbol") == sym_a).select(
        col("ts"), col("close").alias("close_a"), col("log_return").alias("ret_a")
    )
    b = fact.filter(col("symbol") == sym_b).select(
        col("ts"), col("close").alias("close_b"), col("log_return").alias("ret_b")
    )
    w60 = Window.orderBy("ts").rowsBetween(-59, 0)
    w30 = Window.orderBy("ts").rowsBetween(-29, 0)
    return (
        a.join(b, "ts")
        # Hedge ratio: corr(close_a, close_b) * std(a)/std(b) ≈ OLS beta
        .withColumn(
            "_std_b", stddev("close_b").over(w60)
        )
        .withColumn(
            "hedge_ratio",
            # null when too few points (std_b null) or close_b is constant (std_b 0)
            when(col("_std_b").isNotNull() & (col("_std_b") != 0),
                corr("close_a", "close_b").over(w60)
                * stddev("close_a").over(w60)
                / col("_std_b")
            ),
        )
        .drop("_std_b")
        .withColumn("spread", col("close_a") - col("hedge_ratio") * col("close_b"))
        .withColumn("spread_mean", avg("spread").over(w30))
        .withColumn("_spread_vol", stddev("spread").over(w30))
        .withColumn("zscore",
            when(col("_spread_vol") != 0,
                (col("spread") - col("spread_mean")) / col("_spread_vol")
            )  # null when spread is constant — signals no information
        )
        .drop("_spread_vol")
        .withColumn("correlation_60d", corr("ret_a", "ret_b").over(w60))
        .withColumn("symbol_a", lit(sym_a))
        .withColumn("symbol_b", lit(sym_b))
        # ts is Unix nanoseconds; divide by 1e9 to get seconds before casting
        .withColumn("event_timestamp", from_unixtime(col("ts").cast("double") / 1e9).cast("timestamp"))
        .withColumn("created", current_timestamp())
    )


def build_feat_pair(spark: SparkSession, pairs: list[tuple[str, str]]) -> None:
    """Compute pair features for the configured cointegrated pairs and write them to Delta."""
    fact = spark.read.format("delta").load("s3a://delta-lake/gold/fact_daily_bar/")
    rows = [pair_features(fact, sym_a, sym_b) for sym_a, sym_b in pairs]

    if rows:
        from functools import reduce

        from pyspark.sql import DataFrame
        all_pairs = reduce(DataFrame.union, rows)
        all_pairs.write.format("delta").mode("overwrite").save(
            "s3a://delta-lake/gold/feat_pair_daily/"
        )
        print(f"feat_pair_daily rows: {all_pairs.count()}")


# ─── OBT ─────────────────────────────────────────────────────────────────────

def obt_from(pair, fact):
    """Join pair features with each leg's OHLCV into one row per (pair, ts)."""
    ohlcv_a = fact.select(
        col("symbol").alias("symbol_a"), col("ts"),
        col("open").alias("open_a"), col("high").alias("high_a"),
        col("low").alias("low_a"), col("volume").alias("volume_a"),
    )
    ohlcv_b = fact.select(
        col("symbol").alias("symbol_b"), col("ts"),
        col("open").alias("open_b"), col("high").alias("high_b"),
        col("low").alias("low_b"), col("volume").alias("volume_b"),
    )
    return (
        pair
        .join(ohlcv_a, ["ts", "symbol_a"])
        .join(ohlcv_b, ["ts", "symbol_b"])
        .drop("spread_mean", "spread")
    )


def build_obt(spark: SparkSession) -> None:
    """Build obt_pair_backtest_input from pair features and fact_daily_bar."""
    pair = spark.read.format("delta").load("s3a://delta-lake/gold/feat_pair_daily/")
    fact = spark.read.format("delta").load("s3a://delta-lake/gold/fact_daily_bar/")
    obt = obt_from(pair, fact)
    obt.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(
        "s3a://delta-lake/gold/obt_pair_backtest_input/"
    )
    print(f"obt_pair_backtest_input rows: {obt.count()}")


def run(s3_endpoint: str, pg_url: str, pairs: list[tuple[str, str]]) -> None:
    spark = build_spark(s3_endpoint)
    spark.sparkContext.setLogLevel("WARN")

    build_feat_symbol(spark)
    build_feat_pair(spark, pairs)
    build_obt(spark)

    pg_props = {"user": "pairlab", "password": "pairlab", "driver": "org.postgresql.Driver"}
    import psycopg2 as _pg
    # Parse host/port/db from jdbc:postgresql://host:port/db
    _stripped = pg_url.replace("jdbc:postgresql://", "")          # "postgres:5432/pairlab"
    _host_port, _db = _stripped.split("/", 1)                      # "postgres:5432", "pairlab"
    _host, _port = _host_port.split(":")                           # "postgres", "5432"
    _pg_dsn = f"host={_host} port={_port} dbname={_db} user=pairlab password=pairlab"

    # Drop Feast views that depend on gold.feat_* before overwrite, recreate after
    with _pg.connect(_pg_dsn) as _conn, _conn.cursor() as _cur:
        _cur.execute("DROP VIEW IF EXISTS feast.feat_symbol_daily CASCADE")
        _cur.execute("DROP VIEW IF EXISTS feast.feat_pair_daily CASCADE")
        _conn.commit()

    # truncate=true sends TRUNCATE instead of DROP+CREATE so indexes are preserved
    pg_props_trunc = {**pg_props, "truncate": "true"}
    for delta_path, pg_table in [
        ("s3a://delta-lake/gold/feat_symbol_daily/", "gold.feat_symbol_daily"),
        ("s3a://delta-lake/gold/feat_pair_daily/", "gold.feat_pair_daily"),
        ("s3a://delta-lake/gold/obt_pair_backtest_input/", "gold.obt_pair_backtest_input"),
    ]:
        df = spark.read.format("delta").load(delta_path)
        df.write.jdbc(url=pg_url, table=pg_table, mode="overwrite", properties=pg_props_trunc)
        print(f"Mirrored {pg_table} to Postgres")

    # Recreate Feast views pointing at the freshly written tables
    with _pg.connect(_pg_dsn) as _conn, _conn.cursor() as _cur:
        _cur.execute("""
            CREATE OR REPLACE VIEW feast.feat_symbol_daily AS
            SELECT * FROM gold.feat_symbol_daily
        """)
        _cur.execute("""
            CREATE OR REPLACE VIEW feast.feat_pair_daily AS
            SELECT * FROM gold.feat_pair_daily
        """)
        _conn.commit()
    print("Recreated feast.feat_symbol_daily and feast.feat_pair_daily views")

    spark.stop()
    print("=== DP3: done ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3-endpoint", default="http://minio:4566")
    parser.add_argument("--pg-url", default="jdbc:postgresql://postgres:5432/pairlab")
    # KO/PEP, XOM/CVX, JPM/BAC, GS/MS — match generator cointegrated pairs
    parser.add_argument("--pairs", default="KO:PEP,XOM:CVX,JPM:BAC,GS:MS")
    args = parser.parse_args()
    pairs = [tuple(p.split(":")) for p in args.pairs.split(",")]
    run(args.s3_endpoint, args.pg_url, pairs)
