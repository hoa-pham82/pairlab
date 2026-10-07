"""Delta Lake storage optimization: OPTIMIZE (compaction) + ZORDER by symbol.

Run after DP2/DP3 to compact many small Parquet files into fewer large ones,
and cluster rows by symbol so range queries (e.g. "all bars for KO") skip
irrelevant files (data skipping via file-level min/max statistics).

Run:
    spark-submit --master spark://spark-master:7077 \
        --packages io.delta:delta-spark_2.12:3.2.0,org.apache.hadoop:hadoop-aws:3.3.4 \
        optimize_delta.py
"""

from __future__ import annotations

import argparse
import time

from pyspark.sql import SparkSession


def build_spark(s3_endpoint: str) -> SparkSession:
    return (
        SparkSession.builder.appName("DeltaOptimize")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.hadoop.fs.s3a.endpoint", s3_endpoint)
        .config("spark.hadoop.fs.s3a.access.key", "test")
        .config("spark.hadoop.fs.s3a.secret.key", "test")
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        # Target 128 MB files after compaction
        .config("spark.databricks.delta.optimize.maxFileSize", str(128 * 1024 * 1024))
        .getOrCreate()
    )


def _file_stats(spark: SparkSession, path: str) -> dict:
    detail = spark.sql(f"DESCRIBE DETAIL delta.`{path}`").collect()[0]
    return {"numFiles": detail["numFiles"], "sizeInBytes": detail["sizeInBytes"]}


def optimize_table(spark: SparkSession, path: str, zorder_col: str) -> None:
    """Run OPTIMIZE + ZORDER via Spark SQL and print before/after file counts."""
    before = _file_stats(spark, path)
    print(f"\n  Before: {before['numFiles']} files, {before['sizeInBytes'] / 1024:.1f} KB")

    t0 = time.time()
    spark.sql(f"OPTIMIZE delta.`{path}` ZORDER BY ({zorder_col})")
    elapsed = time.time() - t0

    after = _file_stats(spark, path)
    print(f"  After:  {after['numFiles']} files, {after['sizeInBytes'] / 1024:.1f} KB  ({elapsed:.1f}s)")
    print(f"  Compacted {before['numFiles']} → {after['numFiles']} files")


def run(s3_endpoint: str) -> None:
    spark = build_spark(s3_endpoint)
    spark.sparkContext.setLogLevel("WARN")

    tables = [
        # fact_daily_bar: partitioned by dt → ZORDER by symbol (data col)
        ("s3a://delta-lake/gold/fact_daily_bar/", "symbol"),
        # feat_symbol_daily: partitioned by symbol → ZORDER by ts (data col)
        ("s3a://delta-lake/gold/feat_symbol_daily/", "ts"),
        # obt_pair_backtest_input: unpartitioned → ZORDER by symbol_a
        ("s3a://delta-lake/gold/obt_pair_backtest_input/", "symbol_a"),
    ]

    for path, zcol in tables:
        print(f"\n=== OPTIMIZE + ZORDER({zcol}): {path} ===")
        optimize_table(spark, path, zcol)

    spark.stop()
    print("\n=== Delta optimization: done ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3-endpoint", default="http://minio:4566")
    args = parser.parse_args()
    run(args.s3_endpoint)
