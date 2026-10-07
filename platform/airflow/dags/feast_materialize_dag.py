"""Feast materialize DAG — push latest gold features into Redis online store.

Runs after DP3 finishes to keep the online store fresh.
Uses `feast materialize-incremental` so only new rows since the last run are pushed.

Depends on: dp3_features
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.sensors.python import PythonSensor

FEAST_REPO = "/opt/airflow/feature_repo"
default_args = {
    "owner": "pairlab",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

with DAG(
    dag_id="feast_materialize",
    description="Materialize gold features from Postgres into Redis online store",
    start_date=datetime(2025, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    default_args=default_args,
    tags=["feast", "online-store", "redis"],
) as dag:

    def _dp3_completed() -> bool:
        """Return True when DP3 has written rows to gold.feat_pair_daily."""
        hook = PostgresHook(postgres_conn_id="postgres_dwh")
        result = hook.get_first("SELECT 1 FROM gold.feat_pair_daily LIMIT 1")
        return result is not None

    wait_dp3 = PythonSensor(
        task_id="wait_dp3",
        python_callable=_dp3_completed,
        timeout=3600,
        poke_interval=30,
        mode="reschedule",
        doc_md="Wait for DP3 to populate gold.feat_pair_daily.",
    )

    materialize = BashOperator(
        task_id="materialize",
        bash_command=(
            f"/opt/ml-venv/bin/feast -c {FEAST_REPO}"
            " materialize-incremental $(date -u +%Y-%m-%dT%H:%M:%S)"
        ),
        doc_md="Push features written since the last materialize run into Redis.",
    )

    wait_dp3 >> materialize
