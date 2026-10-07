"""DP2 — promote Delta bronze → silver (dedup + normalize) → gold (SCD2 dim + fact tables).

Writes:
  Delta s3a://delta-lake/silver/daily_bars/
  Delta s3a://delta-lake/gold/dim_symbol/
  Delta s3a://delta-lake/gold/fact_daily_bar/
  Postgres silver.stg_daily_bars, gold.dim_symbol, gold.fact_daily_bar (via JDBC)

Run:
    spark-submit --master spark://spark-master:7077 \
        --packages io.delta:delta-spark_2.12:3.2.0,org.apache.hadoop:hadoop-aws:3.3.4,\
org.postgresql:postgresql:42.7.3 \
        dp2_silver_gold.py
"""

from __future__ import annotations

import argparse
import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    current_timestamp,
    exp,
    lag,
    lit,
    log as spark_log,
    row_number,
    when,
)
from pyspark.sql.window import Window


def build_spark(s3_endpoint: str, pg_url: str) -> SparkSession:
    """Build SparkSession with Delta Lake, S3A, and JDBC for Postgres writes."""
    return (
        SparkSession.builder.appName("DP2-SilverGold")
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


# ─── Silver ──────────────────────────────────────────────────────────────────

def build_silver(spark: SparkSession) -> None:
    """Read bronze Delta, dedup on (symbol, ts), write silver."""
    bronze = spark.read.format("delta").load("s3a://delta-lake/bronze/daily_bars/")
    w = Window.partitionBy("symbol", "ts").orderBy(col("schema_version").desc())
    silver = (
        bronze
        .withColumn("_rn", row_number().over(w))
        .filter(col("_rn") == 1)
        .drop("_rn", "ingested_at")
        .withColumn("silver_ts", current_timestamp())
    )
    silver.write.format("delta").mode("overwrite").partitionBy("dt").save(
        "s3a://delta-lake/silver/daily_bars/"
    )
    print(f"Silver rows: {silver.count()}")


# ─── Gold dim_symbol (SCD2) ───────────────────────────────────────────────────

def _upsert_dim_symbol_scd2(spark: SparkSession, silver_df, s3_endpoint: str) -> None:
    """Merge new symbol attributes into dim_symbol as SCD2 using Spark SQL MERGE.

    Sector data comes from vendor-raw/symbol_master.parquet (not the bars).
    When a symbol's sector changes: close old row, insert new one.
    """
    dim_path = "s3a://delta-lake/gold/dim_symbol/"

    # Read only current rows from symbol master (SCD2 master has one historical row
    # per sector change; using all rows would give multiple source rows per symbol
    # and cause the MERGE to raise DELTA_MULTIPLE_SOURCE_ROW_MATCHING_TARGET_ROW).
    master = (
        spark.read.parquet("s3a://vendor-raw/symbol_master.parquet")
        .filter(col("is_current") == True)  # noqa: E712
        .select("symbol", "sector")
    )

    # One distinct symbol per current silver partition
    symbols = silver_df.select("symbol").distinct()

    latest = (
        symbols.join(master, "symbol", "left")
        .withColumn("valid_from_ts", current_timestamp())
        .withColumn("valid_to_ts", lit(None).cast("timestamp"))
        .withColumn("is_current", lit(True))
    )

    # Check whether the Delta table already exists
    try:
        spark.read.format("delta").load(dim_path).limit(1).collect()
        table_exists = True
    except Exception:
        table_exists = False

    if not table_exists:
        latest.write.format("delta").mode("overwrite").save(dim_path)
    else:
        latest.createOrReplaceTempView("_scd2_updates")
        spark.sql(f"""
            MERGE INTO delta.`{dim_path}` AS target
            USING _scd2_updates AS src
            ON target.symbol = src.symbol AND target.is_current = true
            WHEN MATCHED AND target.sector != src.sector THEN
                UPDATE SET target.is_current = false, target.valid_to_ts = src.valid_from_ts
            WHEN NOT MATCHED THEN INSERT *
        """)

    print(f"dim_symbol rows: {spark.read.format('delta').load(dim_path).count()}")


# ─── Gold fact_daily_bar ──────────────────────────────────────────────────────

def build_fact(spark: SparkSession, silver_df) -> None:
    """Compute fact_daily_bar: add daily_return and log_return columns."""
    w = Window.partitionBy("symbol").orderBy("ts")
    fact = (
        silver_df
        .withColumn("prev_close", lag("close", 1).over(w))
        .withColumn("daily_return", (col("close") - col("prev_close")) / col("prev_close"))
        .withColumn("log_return", spark_log(col("close") / col("prev_close")))
        .drop("prev_close", "silver_ts")
    )
    fact.write.format("delta").mode("overwrite").partitionBy("dt").save(
        "s3a://delta-lake/gold/fact_daily_bar/"
    )
    print(f"fact_daily_bar rows: {fact.count()}")


# ─── Postgres mirror ─────────────────────────────────────────────────────────

def _write_pg(df, table: str, pg_url: str, pg_props: dict) -> None:
    # truncate=true sends TRUNCATE instead of DROP+CREATE, preserving primary keys and indexes
    props = {**pg_props, "truncate": "true"}
    df.write.jdbc(url=pg_url, table=table, mode="overwrite", properties=props)
    print(f"Wrote {table} to Postgres")


def run(s3_endpoint: str, pg_url: str) -> None:
    spark = build_spark(s3_endpoint, pg_url)
    spark.sparkContext.setLogLevel("WARN")
    pg_props = {"user": "pairlab", "password": "pairlab", "driver": "org.postgresql.Driver"}

    build_silver(spark)
    silver_df = spark.read.format("delta").load("s3a://delta-lake/silver/daily_bars/")

    _upsert_dim_symbol_scd2(spark, silver_df, s3_endpoint)
    build_fact(spark, silver_df)

    dim_df = spark.read.format("delta").load("s3a://delta-lake/gold/dim_symbol/")
    _write_pg(dim_df, "gold.dim_symbol", pg_url, pg_props)

    fact_df = spark.read.format("delta").load("s3a://delta-lake/gold/fact_daily_bar/")
    _write_pg(fact_df, "gold.fact_daily_bar", pg_url, pg_props)

    silver_df2 = spark.read.format("delta").load("s3a://delta-lake/silver/daily_bars/")
    _write_pg(silver_df2, "silver.stg_daily_bars", pg_url, pg_props)

    spark.stop()
    print("=== DP2: done ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3-endpoint", default="http://minio:4566")
    parser.add_argument("--pg-url", default="jdbc:postgresql://postgres:5432/pairlab")
    args = parser.parse_args()
    run(args.s3_endpoint, args.pg_url)
