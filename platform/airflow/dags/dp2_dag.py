"""DP2 DAG — promote Delta bronze → silver (dedup) → gold (SCD2 dim + fact tables).

Stages:
  1. ingest  : spark-submit dp2_silver_gold.py
  2. validate: check dim_symbol SCD2 integrity + fact row count in Postgres

Depends on: dp1_ingest_bronze (must run first)
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.sensors.python import PythonSensor

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
    dag_id="dp2_silver_gold",
    description="Delta bronze → silver (dedup) → gold (SCD2 + fact_daily_bar)",
    start_date=datetime(2025, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    default_args=default_args,
    tags=["dp2", "silver", "gold", "spark", "scd2"],
) as dag:

    def _dp1_completed() -> bool:
        """Return True when DP1 wrote a successful row to platform.job_status in the last day."""
        hook = PostgresHook(postgres_conn_id="postgres_dwh")
        result = hook.get_first("""
            SELECT 1 FROM platform.job_status
            WHERE job = 'dp1_ingest_bronze'
              AND status = 'ok'
              AND ran_at > now() - INTERVAL '1 day'
        """)
        return result is not None

    wait_dp1 = PythonSensor(
        task_id="wait_dp1",
        python_callable=_dp1_completed,
        timeout=3600,
        poke_interval=30,
        mode="reschedule",
        doc_md="Wait for DP1 to write a successful row to platform.job_status (within last 24 h).",
    )

    s3_endpoint = "{{ var.value.get('s3_endpoint', 'http://minio:4566') }}"
    pg_url = "{{ var.value.get('pg_jdbc_url', 'jdbc:postgresql://postgres:5432/pairlab') }}"

    ingest = BashOperator(
        task_id="ingest",
        bash_command=(
            f"docker exec {SPARK_CONTAINER} "
            f"/opt/spark/bin/spark-submit "
            f"--master {SPARK_MASTER} "
            f"{JOBS_DIR}/dp2_silver_gold.py "
            f"--s3-endpoint {s3_endpoint} "
            f"--pg-url {pg_url}"
        ),
        doc_md="Promote bronze → silver (dedup) → gold (SCD2 dim_symbol + fact_daily_bar).",
    )

    def _validate_silver_gold(**ctx):
        hook = PostgresHook(postgres_conn_id="postgres_dwh")

        # Silver: must be non-empty
        silver_rows = hook.get_first("SELECT COUNT(*) FROM silver.stg_daily_bars")[0]
        assert silver_rows > 0, "silver.stg_daily_bars is empty — DP2 Spark job may have failed"

        # Silver: no duplicate (symbol, ts) pairs
        dupes = hook.get_first("""
            SELECT COUNT(*) FROM (
                SELECT symbol, ts, COUNT(*) c FROM silver.stg_daily_bars GROUP BY 1,2 HAVING COUNT(*) > 1
            ) t
        """)[0]
        assert dupes == 0, f"Duplicates in silver: {dupes}"

        # Gold dim_symbol: must be non-empty and have exactly one is_current row per symbol
        dim_rows = hook.get_first("SELECT COUNT(*) FROM gold.dim_symbol")[0]
        assert dim_rows > 0, "gold.dim_symbol is empty"
        bad_dim = hook.get_first("""
            SELECT COUNT(*) FROM (
                SELECT symbol, COUNT(*) c FROM gold.dim_symbol WHERE is_current GROUP BY 1 HAVING COUNT(*) > 1
            ) t
        """)[0]
        assert bad_dim == 0, f"SCD2 violation: multiple current rows for same symbol: {bad_dim}"

        # Gold fact: non-empty; no nulls in return columns except the first bar per symbol
        fact_rows = hook.get_first("SELECT COUNT(*) FROM gold.fact_daily_bar")[0]
        assert fact_rows > 0, "gold.fact_daily_bar is empty"
        fact_nulls = hook.get_first(
            "SELECT COUNT(*) FROM gold.fact_daily_bar WHERE log_return IS NULL AND ts != (SELECT MIN(ts) FROM gold.fact_daily_bar)"
        )[0]
        assert fact_nulls == 0, f"Unexpected nulls in log_return: {fact_nulls}"

        print(f"Validation passed — silver rows: {silver_rows}, dim rows: {dim_rows}, fact rows: {fact_rows}")

    validate = PythonOperator(
        task_id="validate",
        python_callable=_validate_silver_gold,
        doc_md="Assert no silver dupes, SCD2 integrity in dim_symbol, no unexpected nulls in fact.",
    )

    wait_dp1 >> ingest >> validate
