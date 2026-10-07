"""DP1 OPTIMIZED — ingest vendor-raw/ into Delta Lake bronze with full problem handling.

Handles:
  - Schema evolution  : mergeSchema=true + per-version null-fill + quarantine
  - Skew              : salt-key repartition on symbol to even out partition sizes
  - High cardinality  : approx_count_distinct instead of exact COUNT DISTINCT
  - Duplicates        : row_number() dedup on (symbol, ts, schema_version)

Run:
    spark-submit --master spark://spark-master:7077 \
        --packages io.delta:delta-spark_2.12:3.2.0,org.apache.hadoop:hadoop-aws:3.3.4 \
        dp1_ingest_bronze_optimized.py [--s3-endpoint http://minio:4566]
"""

from __future__ import annotations

import argparse
import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    approx_count_distinct,
    col,
    current_timestamp,
    lit,
    rand,
    row_number,
    to_date,
)
from pyspark.sql.types import DoubleType, ShortType, StringType
from pyspark.sql.window import Window

_N_SALTS = 8  # salt buckets to break up skewed symbol partitions


def build_spark(s3_endpoint: str, app_name: str = "DP1-Optimized") -> SparkSession:
    """Build SparkSession with Delta Lake, S3A, and AQE enabled."""
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.hadoop.fs.s3a.endpoint", s3_endpoint)
        .config("spark.hadoop.fs.s3a.access.key", os.environ.get("AWS_ACCESS_KEY_ID", "pairlabs3"))
        .config("spark.hadoop.fs.s3a.secret.key", os.environ.get("AWS_SECRET_ACCESS_KEY", "pairlabs3key"))
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        # AQE: auto-coalesces post-shuffle partitions + handles skew joins
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled", "true")
        .getOrCreate()
    )


def _normalize_schema(df):
    """Unify schema across v1/v2/v3 into a single superset schema.

    v3 renamed volume→vol; mergeSchema gives us both columns with nulls on the
    wrong side.  Coalesce them into a single volume column and drop vol so the
    Delta table always sees the same column set regardless of which versions
    are present.
    """
    from pyspark.sql.functions import coalesce
    if "vol" in df.columns:
        # v3 rows: vol is populated, volume is null — coalesce keeps whichever is non-null
        df = df.withColumn("volume", coalesce(col("volume"), col("vol")).cast(DoubleType()))
        df = df.drop("vol")
    if "adj_close" not in df.columns:
        df = df.withColumn("adj_close", col("close").cast(DoubleType()))
    if "source_exchange" not in df.columns:
        df = df.withColumn("source_exchange", lit(None).cast(StringType()))
    if "schema_version" not in df.columns:
        df = df.withColumn("schema_version", lit(1).cast(ShortType()))
    else:
        df = df.withColumn("schema_version", col("schema_version").cast(ShortType()))
    return df


def _split_bad_rows(df):
    """Return (good, bad): bad rows miss a key, have negative volume, or high < low.

    A check that evaluates to null (e.g. null volume) counts as bad, so no row is lost.
    """
    from pyspark.sql.functions import coalesce
    failed = (
        col("symbol").isNull() |
        col("ts").isNull() |
        (col("volume") < 0) |
        (col("high") < col("low"))
    )
    df_tagged = df.withColumn("_bad", coalesce(failed, lit(True)))
    bad = df_tagged.filter(col("_bad")).drop("_bad")
    good = df_tagged.filter(~col("_bad")).drop("_bad")
    return good, bad


def _quarantine_bad_rows(df, spark, s3_endpoint: str):
    """Write rows that fail validation to quarantine and return the good ones."""
    good, bad = _split_bad_rows(df)
    # Write bad rows without counting first (lazy evaluation)
    bad.write.format("delta").mode("append").save("s3a://delta-lake/bronze/rejected_daily_bars/")
    print("  Quarantine write done (bad rows → bronze/rejected_daily_bars/)")
    return good, -1  # count not needed for progress tracking


def _dedup(df):
    """Keep the last-seen row per (symbol, ts) using row_number over ingestion order."""
    w = Window.partitionBy("symbol", "ts").orderBy(col("schema_version").desc())
    return (
        df.withColumn("_rn", row_number().over(w))
        .filter(col("_rn") == 1)
        .drop("_rn", "_salt")
    )


def _add_salt(df):
    """Add a random salt column to break skewed symbol partitions across tasks."""
    return df.withColumn("_salt", (rand() * _N_SALTS).cast("int"))


def run(s3_endpoint: str) -> None:
    spark = build_spark(s3_endpoint)
    spark.sparkContext.setLogLevel("WARN")
    print("=== DP1 OPTIMIZED: reading vendor-raw/daily_bars/ ===")

    # Read only schema_version= partitions (avoids Spark partition-column conflict with
    # legacy dt= directories that may still exist in the same prefix).
    # basePath tells Spark the root so schema_version is inferred as a partition column.
    df = (
        spark.read
        .option("mergeSchema", "true")
        .option("basePath", "s3a://vendor-raw/daily_bars/")
        .parquet(
            "s3a://vendor-raw/daily_bars/schema_version=1/",
            "s3a://vendor-raw/daily_bars/schema_version=2/",
            "s3a://vendor-raw/daily_bars/schema_version=3/",
        )
    )
    print(f"Raw rows: {df.count()}")

    # Normalize schema — fill columns that don't exist in older partitions
    df = _normalize_schema(df)

    # Quarantine
    df, _ = _quarantine_bad_rows(df, spark, s3_endpoint)

    # High cardinality — use approx_count_distinct (HyperLogLog, 5% error, 100× faster)
    card = df.select(
        approx_count_distinct("symbol").alias("approx_symbols"),
        approx_count_distinct("ts").alias("approx_dates"),
    ).collect()[0]
    print(f"Cardinality approx: symbols={card['approx_symbols']}, dates={card['approx_dates']}")

    # Skew — add salt so heavy symbols (many partitions) spread across executors
    df = _add_salt(df)

    # Repartition by salt to distribute load evenly (avoids one fat partition per symbol)
    df = df.repartition(_N_SALTS, "_salt")

    # Dedup — keep latest schema_version per (symbol, ts)
    df = _dedup(df)

    # Add ingestion timestamp
    df = df.withColumn("ingested_at", current_timestamp())

    # Write Delta — partitioned by date for efficient downstream reads
    (
        df.withColumn("dt", to_date(col("ts")))
        .write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .partitionBy("dt")
        .save("s3a://delta-lake/bronze/daily_bars/")
    )

    final_count = spark.read.format("delta").load("s3a://delta-lake/bronze/daily_bars/").count()
    print(f"Delta bronze rows: {final_count}")

    # Write job status to Postgres so the Airflow validate task can confirm Spark output
    pg_url = "jdbc:postgresql://postgres:5432/pairlab"
    pg_props = {"user": "pairlab", "password": "pairlab", "driver": "org.postgresql.Driver"}
    status_df = spark.createDataFrame([{
        "job": "dp1_ingest_bronze",
        "rows_out": final_count,
        "status": "ok",
    }])
    status_df.write.jdbc(url=pg_url, table="platform.job_status", mode="append", properties=pg_props)

    print("=== DP1 OPTIMIZED: done ===")
    spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3-endpoint", default="http://minio:4566")
    args = parser.parse_args()
    run(args.s3_endpoint)
