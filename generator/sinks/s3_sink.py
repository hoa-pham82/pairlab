"""Upload generated Parquet files to S3-compatible storage (LocalStack / MinIO)."""

from __future__ import annotations

import io
import logging
from pathlib import Path

import boto3
import pandas as pd

log = logging.getLogger(__name__)

_DEFAULT_ENDPOINT = "http://localhost:4566"
_DEFAULT_BUCKET = "vendor-raw"


def _client(endpoint: str, access_key: str, secret_key: str):
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="us-east-1",
    )


def upload_parquet(
    df: pd.DataFrame,
    key: str,
    bucket: str = _DEFAULT_BUCKET,
    endpoint: str = _DEFAULT_ENDPOINT,
    access_key: str = "test",
    secret_key: str = "test",
) -> str:
    """Upload a DataFrame as Parquet to S3. Returns the s3://bucket/key URI."""
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, engine="pyarrow")
    buf.seek(0)
    s3 = _client(endpoint, access_key, secret_key)
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
    uri = f"s3://{bucket}/{key}"
    log.info("Uploaded %d rows → %s", len(df), uri)
    return uri


def upload_parquet_partitioned(
    df: pd.DataFrame,
    prefix: str,
    partition_col: str = "dt",
    bucket: str = _DEFAULT_BUCKET,
    endpoint: str = _DEFAULT_ENDPOINT,
    access_key: str = "test",
    secret_key: str = "test",
) -> list[str]:
    """Upload a DataFrame partitioned by date column. Returns list of S3 URIs."""
    df = df.copy()
    if partition_col not in df.columns:
        df[partition_col] = pd.to_datetime(df["ts"]).dt.date.astype(str)

    uris = []
    for dt_val, group in df.groupby(partition_col):
        key = f"{prefix}/{partition_col}={dt_val}/data.parquet"
        uri = upload_parquet(group.drop(columns=[partition_col], errors="ignore"),
                             key, bucket, endpoint, access_key, secret_key)
        uris.append(uri)

    log.info("Uploaded %d partitions to s3://%s/%s/", len(uris), bucket, prefix)
    return uris


def upload_file(
    local_path: Path,
    key: str,
    bucket: str = _DEFAULT_BUCKET,
    endpoint: str = _DEFAULT_ENDPOINT,
    access_key: str = "test",
    secret_key: str = "test",
) -> str:
    """Upload any local file to S3. Returns the s3://bucket/key URI."""
    s3 = _client(endpoint, access_key, secret_key)
    s3.upload_file(str(local_path), bucket, key)
    uri = f"s3://{bucket}/{key}"
    log.info("Uploaded %s → %s", local_path.name, uri)
    return uri
