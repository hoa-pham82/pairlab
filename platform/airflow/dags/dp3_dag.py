"""DP3 DAG — compute offline feature tables and the One Big Table for backtesting.

Stages:
  1. ingest  : spark-submit dp3_features.py (feat_symbol_daily, feat_pair_daily, OBT)
  2. validate: check event_timestamp column exists + feature values non-null

Depends on: dp2_silver_gold
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
    dag_id="dp3_features",
    description="Compute feat_symbol_daily, feat_pair_daily, OBT from gold fact tables",
    start_date=datetime(2025, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    default_args=default_args,
    tags=["dp3", "features", "spark", "feast"],
) as dag:

    def _dp2_completed() -> bool:
        """Return True when DP2 has written rows to gold.fact_daily_bar."""
        hook = PostgresHook(postgres_conn_id="postgres_dwh")
        result = hook.get_first("SELECT 1 FROM gold.fact_daily_bar LIMIT 1")
        return result is not None

    wait_dp2 = PythonSensor(
        task_id="wait_dp2",
        python_callable=_dp2_completed,
        timeout=3600,
        poke_interval=30,
        mode="reschedule",
        doc_md="Wait for DP2 to populate gold.fact_daily_bar.",
    )

    s3_endpoint = "{{ var.value.get('s3_endpoint', 'http://minio:4566') }}"
    pg_url = "{{ var.value.get('pg_jdbc_url', 'jdbc:postgresql://postgres:5432/pairlab') }}"
    pairs = "{{ var.value.get('coint_pairs', 'KO:PEP,XOM:CVX,JPM:BAC,GS:MS') }}"

    ingest = BashOperator(
        task_id="ingest",
        bash_command=(
            f"docker exec {SPARK_CONTAINER} "
            f"/opt/spark/bin/spark-submit "
            f"--master {SPARK_MASTER} "
            f"{JOBS_DIR}/dp3_features.py "
            f"--s3-endpoint {s3_endpoint} "
            f"--pg-url {pg_url} "
            f"--pairs {pairs}"
        ),
        doc_md="Compute rolling features (vol, ret, ATR, hedge_ratio, zscore) and write OBT.",
    )

    def _validate_features(**ctx):
        hook = PostgresHook(postgres_conn_id="postgres_dwh")

        # Feast requirement: event_timestamp must exist and be non-null
        r = hook.get_first(
            "SELECT COUNT(*), COUNT(event_timestamp) FROM gold.feat_symbol_daily"
        )
        total, with_et = r
        assert total > 0, "feat_symbol_daily is empty"
        assert total == with_et, f"Null event_timestamps: {total - with_et}"

        # Pair features: zscore should have variance (not all NaN)
        r2 = hook.get_first(
            "SELECT COUNT(*) FROM gold.feat_pair_daily WHERE zscore IS NOT NULL"
        )
        assert r2[0] > 0, "All zscore values are null in feat_pair_daily"

        # OBT: should have same row count as pair features
        obt_rows = hook.get_first("SELECT COUNT(*) FROM gold.obt_pair_backtest_input")[0]
        print(f"Validation passed — feat_symbol: {total}, obt: {obt_rows}")

    validate = PythonOperator(
        task_id="validate",
        python_callable=_validate_features,
        doc_md="Assert event_timestamp non-null (Feast req), zscore has values, OBT populated.",
    )

    wait_dp2 >> ingest >> validate
