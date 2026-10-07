"""DP1 BASELINE — ingest raw Parquet from S3 vendor-raw/ into Delta Lake bronze zone.

Intentionally no skew handling, no schema merging, no dedup — used to show
what breaks before optimization (Spark UI screenshots show data skew + schema errors).

Run:
    spark-submit --master spark://spark-master:7077 \
        --packages io.delta:delta-spark_2.12:3.2.0,org.apache.hadoop:hadoop-aws:3.3.4 \
        dp1_ingest_bronze.py [--s3-endpoint http://minio:4566]
"""

from __future__ import annotations

import argparse
import logging

from pyspark.sql import SparkSession
from pyspark.sql.functions import lit

log = logging.getLogger(__name__)


def build_spark(s3_endpoint: str, app_name: str = "DP1-Baseline") -> SparkSession:
    """Build SparkSession with Delta Lake + S3A for LocalStack."""
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.hadoop.fs.s3a.endpoint", s3_endpoint)
        .config("spark.hadoop.fs.s3a.access.key", "test")
        .config("spark.hadoop.fs.s3a.secret.key", "test")
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        # Baseline: AQE off to show raw skew in Spark UI
        .config("spark.sql.adaptive.enabled", "false")
        .getOrCreate()
    )


def run(s3_endpoint: str) -> None:
    """Read all daily bar Parquet partitions and write to Delta bronze."""
    spark = build_spark(s3_endpoint)
    spark.sparkContext.setLogLevel("WARN")

    print("=== DP1 BASELINE: reading vendor-raw/daily_bars/ ===")

    # Baseline: read without mergeSchema — will fail/lose cols when schema evolves
    df = spark.read.parquet(f"s3a://vendor-raw/daily_bars/")

    print(f"Schema: {df.schema.simpleString()}")
    print(f"Row count (raw): {df.count()}")

    # Baseline: no dedup — duplicate rows land in Delta
    df_out = df.withColumn("ingested_at", lit(None).cast("timestamp"))

    # Baseline: write in a single shuffle (no partitioning → shows skew in Spark UI)
    df_out.write.format("delta").mode("overwrite").save("s3a://delta-lake/bronze/daily_bars/")

    print(f"Wrote {df_out.count()} rows to s3a://delta-lake/bronze/daily_bars/")

    # Cardinality check — baseline uses exact COUNT DISTINCT (expensive on high-cardinality cols)
    card = df.selectExpr("COUNT(DISTINCT symbol) as symbols").collect()[0]
    print(f"Cardinality (exact): symbols={card['symbols']}")

    spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3-endpoint", default="http://minio:4566")
    args = parser.parse_args()
    run(args.s3_endpoint)
