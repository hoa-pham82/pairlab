"""ML training DAG — rebuild labels, then train and register the meta-label model.

Stages (same steps as notebooks/ml.ipynb):
  1. build_labels : gold.feat_pair_daily → gold.label_pair_reversion
  2. train        : Feast features + labels → version data → split → train →
                    evaluate → register in MLflow (ml/pipeline.py)

Runs in the ML virtualenv of the Airflow image (/opt/ml-venv). Triggered
manually or by ml_drift when drift is detected.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

ML_PYTHON = "/opt/ml-venv/bin/python"
WORKDIR = "/opt/airflow"
FEAST_REPO = "/opt/airflow/feature_repo"

default_args = {
    "owner": "pairlab",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

with DAG(
    dag_id="ml_train",
    description="Build labels, train the meta-label model, register it in MLflow",
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,
    catchup=False,
    default_args=default_args,
    tags=["ml", "training", "mlflow"],
) as dag:

    build_labels = BashOperator(
        task_id="build_labels",
        bash_command=f"{ML_PYTHON} -m ml.build_labels",
        cwd=WORKDIR,
        doc_md="Write the two-column label table from pair z-scores.",
    )

    train = BashOperator(
        task_id="train",
        bash_command=f"{ML_PYTHON} -m ml.pipeline --feature-repo {FEAST_REPO}",
        cwd=WORKDIR,
        env={
            "AWS_ACCESS_KEY_ID": "{{ var.value.minio_access_key }}",
            "AWS_SECRET_ACCESS_KEY": "{{ var.value.minio_secret_key }}",
        },
        append_env=True,
        doc_md="Version the training data, train, evaluate, and register the model.",
    )

    build_labels >> train
