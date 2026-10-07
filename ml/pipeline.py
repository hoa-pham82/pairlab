"""Training pipeline: version data, split, train, evaluate, register in MLflow.

Usage:
    set -a; . ./.env; set +a      # object-store credentials
    uv run python -m ml.pipeline
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass, field

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from ml.dataset import PAIR_FEATURES, features_and_target, purged_time_split
from ml.train import DEFAULT_PARAMS, evaluate, train_model
from ml.versioning import DataVersion, snapshot_training_frame

PRODUCTION_ALIAS = "production"
CHALLENGER_ALIAS = "challenger"


@dataclass(frozen=True)
class PipelineConfig:
    """Where the pipeline tracks runs and stores data, and how it splits."""

    tracking_uri: str
    data_table_uri: str
    storage_options: dict[str, str] | None = None
    experiment: str = "meta_label"
    model_name: str = "meta_label"
    valid_fraction: float = 0.25
    horizon_bars: int = 20
    model_params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineResult:
    """What one pipeline run produced."""

    run_id: str
    model_version: str
    alias: str
    data_version: DataVersion
    metrics: dict[str, float]


def run_pipeline(frame: pd.DataFrame, cfg: PipelineConfig) -> PipelineResult:
    """Run the training steps on a labelled feature frame and register the model.

    The new model gets the ``production`` alias if there is none yet or its
    validation AUC is at least the current production model's; otherwise it
    gets ``challenger``.

    Raises:
        ValueError: if the split leaves no training or no validation rows.
    """
    data_version = snapshot_training_frame(frame, cfg.data_table_uri, cfg.storage_options)

    train, valid = purged_time_split(frame, cfg.valid_fraction, cfg.horizon_bars)
    if train.empty or valid.empty:
        raise ValueError(f"not enough rows to split: train={len(train)}, valid={len(valid)}")
    x_train, y_train = features_and_target(train)
    x_valid, y_valid = features_and_target(valid)

    model = train_model(x_train, y_train, cfg.model_params)
    metrics = evaluate(model, x_valid, y_valid)

    mlflow.set_tracking_uri(cfg.tracking_uri)
    mlflow.set_experiment(cfg.experiment)
    with mlflow.start_run() as run:
        mlflow.log_params({**DEFAULT_PARAMS, **cfg.model_params})
        mlflow.log_params(
            {
                "features": ",".join(PAIR_FEATURES),
                "valid_fraction": cfg.valid_fraction,
                "horizon_bars": cfg.horizon_bars,
                "data_table_uri": cfg.data_table_uri,
                "data_version": data_version.version,
            }
        )
        mlflow.log_metrics({k: v for k, v in metrics.items() if not math.isnan(v)})
        mlflow.log_metrics(
            {
                "train_rows": len(train),
                "valid_rows": len(valid),
                "data_rows_inserted": data_version.rows_inserted,
                "data_rows_updated": data_version.rows_updated,
                "data_rows_deleted": data_version.rows_deleted,
            }
        )
        info = mlflow.lightgbm.log_model(
            model, name="model", registered_model_name=cfg.model_name, input_example=x_valid.head(2)
        )

    version = str(info.registered_model_version)
    alias = _assign_alias(MlflowClient(cfg.tracking_uri), cfg.model_name, version, metrics["auc"])
    return PipelineResult(run.info.run_id, version, alias, data_version, metrics)


def _assign_alias(client: MlflowClient, name: str, version: str, auc: float) -> str:
    """Give the new version ``production`` if it is at least as good, else ``challenger``."""
    current_auc = _production_auc(client, name)
    promote = current_auc is None or (not math.isnan(auc) and auc >= current_auc)
    alias = PRODUCTION_ALIAS if promote else CHALLENGER_ALIAS
    client.set_registered_model_alias(name, alias, version)
    return alias


def _production_auc(client: MlflowClient, name: str) -> float | None:
    """Validation AUC of the current production model, or None if there is none."""
    try:
        current = client.get_model_version_by_alias(name, PRODUCTION_ALIAS)
    except mlflow.exceptions.MlflowException:
        return None
    return client.get_run(current.run_id).data.metrics.get("auc")


def _env(*names: str) -> str:
    """Return the first of the named environment variables that is set."""
    for name in names:
        if os.environ.get(name):
            return os.environ[name]
    raise SystemExit(f"set one of {', '.join(names)} (see .env.example)")


def load_frame_from_warehouse(dsn: str, feature_repo: str) -> pd.DataFrame:
    """Read the label table and join pair features through Feast."""
    import psycopg2
    from feast import FeatureStore

    from ml.dataset import load_training_frame
    from ml.labels import ID_COLUMN, LABEL_COLUMN

    with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(f"SELECT {ID_COLUMN}, {LABEL_COLUMN} FROM gold.label_pair_reversion")
        labels = pd.DataFrame(cur.fetchall(), columns=[ID_COLUMN, LABEL_COLUMN])
    return load_training_frame(FeatureStore(repo_path=feature_repo), labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dsn",
        default=os.environ.get(
            "POSTGRES_URL", "postgresql://pairlab:pairlab@localhost:5432/pairlab"
        ),
    )
    parser.add_argument("--feature-repo", default="platform/feature_repo")
    parser.add_argument(
        "--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")
    )
    parser.add_argument("--data-table-uri", default="s3://delta-lake/ml/training_frame")
    args = parser.parse_args()

    s3_options = {
        "AWS_ENDPOINT_URL": os.environ.get("S3_ENDPOINT", "http://localhost:4566"),
        "AWS_ACCESS_KEY_ID": _env("AWS_ACCESS_KEY_ID", "MINIO_ROOT_USER"),
        "AWS_SECRET_ACCESS_KEY": _env("AWS_SECRET_ACCESS_KEY", "MINIO_ROOT_PASSWORD"),
        "AWS_REGION": "us-east-1",
        "AWS_ALLOW_HTTP": "true",
        "AWS_S3_ALLOW_UNSAFE_RENAME": "true",
    }
    result = run_pipeline(
        load_frame_from_warehouse(args.dsn, args.feature_repo),
        PipelineConfig(
            tracking_uri=args.tracking_uri,
            data_table_uri=args.data_table_uri,
            storage_options=s3_options if args.data_table_uri.startswith("s3://") else None,
        ),
    )
    print(result)
