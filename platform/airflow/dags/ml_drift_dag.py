"""ML drift DAG — daily drift check on the offline pair features.

Stages:
  1. check_drift     : PSI per pair from gold.feat_pair_daily, pushed to the
                       Prometheus Pushgateway (ml/drift.py). Last output line
                       is "drift" or "stable".
  2. drift_detected  : continue only when the check printed "drift"
  3. trigger_retrain : start the ml_train DAG

No ground-truth labels are needed: the check compares recent feature behaviour
with the year before it.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import ShortCircuitOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

ML_PYTHON = "/opt/ml-venv/bin/python"
WORKDIR = "/opt/airflow"
PUSHGATEWAY = "{{ var.value.get('pushgateway_url', 'http://pushgateway:9091') }}"

default_args = {
    "owner": "pairlab",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

with DAG(
    dag_id="ml_drift",
    description="Daily feature-drift check → Pushgateway → retrain on drift",
    start_date=datetime(2025, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    default_args=default_args,
    tags=["ml", "drift", "observability"],
) as dag:

    check_drift = BashOperator(
        task_id="check_drift",
        bash_command=f"{ML_PYTHON} -m ml.drift --pushgateway {PUSHGATEWAY}",
        cwd=WORKDIR,
        do_xcom_push=True,
        doc_md="Compute spread-change and z-score PSI; push gauges; print the verdict.",
    )

    drift_detected = ShortCircuitOperator(
        task_id="drift_detected",
        python_callable=lambda ti: ti.xcom_pull(task_ids="check_drift") == "drift",
        doc_md="Skip the retrain when the check printed 'stable'.",
    )

    trigger_retrain = TriggerDagRunOperator(
        task_id="trigger_retrain",
        trigger_dag_id="ml_train",
        doc_md="Start the training DAG.",
    )

    check_drift >> drift_detected >> trigger_retrain
