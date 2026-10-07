"""Generate synthetic data and push it to the core platform (S3 + Postgres).

Writes v1/v2/v3 Parquet files to MinIO vendor-raw/daily_bars/ with genuinely
different schemas so Spark's mergeSchema=true has real work to do:
  v1 (before schema_change_date): {symbol, ts, open, high, low, close, volume, schema_version=1}
  v2 (on/after schema_change_date, ~95% of rows): adds adj_close, schema_version=2
  v3 (on/after schema_change_date, ~5% of rows):  adds source_exchange,
      renames volume → vol (breaking), schema_version=3

Each version is written as a separate Parquet partition file.
inject_duplicates is applied per version to embed the duplicate-rate problem.

Removed: bronze.raw_daily_bars write (generator should not own platform tables;
DP1 validate now reads platform.job_status written by the Spark job).

Usage:
    uv run python generator/scripts/ingest_to_platform.py [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from generator.offline.generate import GeneratorConfig, generate_bars
from generator.problems.inject import inject_duplicates
from generator.sinks.postgres_sink import write_symbol_master

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Fraction of post-change rows that become v3 (breaking schema)
_V3_FRACTION = 0.05


def _s3_client(endpoint: str):
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id="test",
        aws_secret_access_key="test",
        region_name="us-east-1",
    )


def _upload_df(df: pd.DataFrame, s3, bucket: str, key: str) -> None:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
    log.info("  Uploaded s3://%s/%s (%d rows)", bucket, key, len(df))


def _split_versions(
    bars: pd.DataFrame,
    schema_change_date: str,
    duplicate_rate: float,
    seed: int,
    v3_fraction: float = _V3_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split bars into v1 / v2 / v3 DataFrames with genuinely different schemas."""
    rng = np.random.default_rng(seed)
    change_ts = pd.Timestamp(schema_change_date, tz="UTC")

    v1_mask = bars["ts"] < change_ts
    post_mask = ~v1_mask
    v3_sel = post_mask & (rng.random(len(bars)) < v3_fraction)
    v2_mask = post_mask & ~v3_sel

    # v1: base schema only
    v1 = bars[v1_mask].copy()
    v1["schema_version"] = 1
    v1 = inject_duplicates(v1, duplicate_rate, seed)

    # v2: base + adj_close
    v2 = bars[v2_mask].copy()
    v2["adj_close"] = v2["close"] * rng.uniform(0.995, 1.005, v2_mask.sum())
    v2["schema_version"] = 2
    v2 = inject_duplicates(v2, duplicate_rate, seed + 1)

    # v3: base + adj_close + source_exchange; volume renamed to vol (breaking change)
    v3 = bars[v3_sel].copy()
    v3["adj_close"] = v3["close"] * rng.uniform(0.995, 1.005, v3_sel.sum())
    v3["vol"] = v3["volume"]          # rename: Spark will see both columns
    v3 = v3.drop(columns=["volume"])  # drop original so schema really diverges
    v3["source_exchange"] = "NYSE"
    v3["schema_version"] = 3
    v3 = inject_duplicates(v3, duplicate_rate, seed + 2)

    return v1, v2, v3


def _load_config(path: str) -> GeneratorConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    fields = {k: v for k, v in data.items() if k in GeneratorConfig.__dataclass_fields__}
    return GeneratorConfig(**fields)


def main(config_path: str, pg_dsn: str, s3_endpoint: str) -> None:
    log.info("Loading config from %s", config_path)
    cfg = _load_config(config_path)

    log.info("Generating data (seed=%d, symbols=%d, days=%d)", cfg.seed, cfg.n_symbols, cfg.days)
    bars_df, master_df, pairs, sector_changes = generate_bars(cfg)
    log.info(
        "Generated %d bar rows, %d symbols, %d pairs, %d sector changes",
        len(bars_df), len(master_df["symbol"].unique()), len(pairs), len(sector_changes),
    )

    s3 = _s3_client(s3_endpoint)

    # ── S3: v1 / v2 / v3 Parquet files ──────────────────────────────────────
    log.info("Splitting bars into v1/v2/v3 schema versions")
    v1, v2, v3 = _split_versions(
        bars_df,
        cfg.schema_change_date,
        cfg.duplicate_rate,
        cfg.seed,
    )
    log.info("  v1=%d rows, v2=%d rows, v3=%d rows (before duplicates: %d total)",
             len(v1), len(v2), len(v3), len(bars_df))

    _upload_df(v1, s3, "vendor-raw", "daily_bars/schema_version=1/part-00000.parquet")
    _upload_df(v2, s3, "vendor-raw", "daily_bars/schema_version=2/part-00000.parquet")
    _upload_df(v3, s3, "vendor-raw", "daily_bars/schema_version=3/part-00000.parquet")

    # Symbol master
    buf = io.BytesIO()
    master_df.to_parquet(buf, index=False)
    buf.seek(0)
    s3.put_object(Bucket="vendor-raw", Key="symbol_master.parquet", Body=buf.getvalue())
    log.info("Uploaded symbol_master.parquet")

    # Sector changes (metadata for documentation / SCD2 exercise)
    if sector_changes:
        import json
        sc_buf = io.BytesIO(json.dumps(sector_changes, default=str).encode())
        s3.put_object(Bucket="vendor-raw", Key="sector_changes.json", Body=sc_buf.getvalue())
        log.info("Uploaded sector_changes.json: %s", sector_changes)

    # ── Postgres: vendor.symbols only ────────────────────────────────────────
    # Only the current/active row per symbol (is_current=True) goes to vendor.symbols.
    # The full SCD2 history is in gold.dim_symbol (written by DP2).
    current_master = master_df[master_df["is_current"]].drop(
        columns=["valid_from", "valid_to", "is_current"], errors="ignore"
    )
    log.info("Writing %d current symbols to vendor.symbols", len(current_master))
    write_symbol_master(current_master, dsn=pg_dsn)

    log.info(
        "Ingest complete — v1=%d v2=%d v3=%d rows → S3; %d symbols → Postgres",
        len(v1), len(v2), len(v3), len(current_master),
    )
    log.info("Cointegrated pairs: %s", pairs)
    if sector_changes:
        log.info("Sector changes: %s", sector_changes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="generator/configs/default.yaml")
    parser.add_argument("--pg-dsn", default="postgresql://pairlab:pairlab@localhost:5432/pairlab")
    parser.add_argument("--s3-endpoint", default="http://localhost:4566")
    args = parser.parse_args()
    main(args.config, args.pg_dsn, args.s3_endpoint)
