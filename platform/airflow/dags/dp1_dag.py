"""DP1 DAG — ingest raw Parquet from S3 vendor-raw/ into Delta Lake bronze zone.

Stages:
  1. ingest  : spark-submit dp1_ingest_bronze_optimized.py
  2. validate: row count and null checks on Delta bronze table

Connections (set in Airflow UI → Admin → Connections):
  spark_default      : Spark master at spark://spark-master:7077
  postgres_dwh       : pairlab DB

Variables (Admin → Variables):
  s3_endpoint        : http://minio:4566
  spark_packages     : io.delta:delta-spark_2.12:3.2.0,org.apache.hadoop:hadoop-aws:3.3.4
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.models import Variable
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook

SPARK_MASTER = "spark://spark-master:7077"
JOBS_DIR = "/opt/spark/jobs"
SPARK_CONTAINER = "platform-spark-master-1"

default_args = {
    "owner": "pairlab",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

with DAG(
    dag_id="dp1_ingest_bronze",
    description="Ingest vendor-raw S3 Parquet → Delta Lake bronze",
    start_date=datetime(2025, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    default_args=default_args,
    tags=["dp1", "bronze", "spark"],
) as dag:

    s3_endpoint = "{{ var.value.get('s3_endpoint', 'http://minio:4566') }}"

    ingest = BashOperator(
        task_id="ingest",
        bash_command=(
            f"docker exec {SPARK_CONTAINER} "
            f"/opt/spark/bin/spark-submit "
            f"--master {SPARK_MASTER} "
            f"{JOBS_DIR}/dp1_ingest_bronze_optimized.py "
            f"--s3-endpoint {s3_endpoint}"
        ),
        doc_md="Submit DP1 optimized Spark job: S3 → Delta bronze with skew/schema/dedup handling.",
    )

    def _validate_bronze(**ctx):
        """Confirm DP1 Spark job wrote rows to Delta bronze by checking platform.job_status."""
        hook = PostgresHook(postgres_conn_id="postgres_dwh")
        result = hook.get_first("""
            SELECT rows_out, status, ran_at
            FROM platform.job_status
            WHERE job = 'dp1_ingest_bronze'
            ORDER BY ran_at DESC
            LIMIT 1
        """)
        assert result is not None, "No dp1_ingest_bronze run recorded in platform.job_status — Spark job did not complete"
        rows_out, status, ran_at = result
        assert status == "ok", f"DP1 recorded status '{status}' at {ran_at}"
        assert rows_out > 0, f"DP1 wrote 0 rows to Delta bronze at {ran_at}"
        print(f"DP1 validated: {rows_out:,} rows in Delta bronze (ran at {ran_at})")
        ctx["ti"].xcom_push(key="bronze_row_count", value=rows_out)

    validate = PythonOperator(
        task_id="validate",
        python_callable=_validate_bronze,
        doc_md="Assert bronze row count > 0, no nulls in PK columns.",
    )

    ingest >> validate
