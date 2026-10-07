"""Tests for the training pipeline against a local SQLite MLflow store."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from ml.dataset import PAIR_FEATURES
from ml.pipeline import PipelineConfig, _assign_alias, run_pipeline
from ml.versioning import load_training_frame_version
from mlflow.tracking import MlflowClient


def _frame(n: int, seed: int = 0, signal: bool = True) -> pd.DataFrame:
    """Labelled features on consecutive business days; label follows zscore if ``signal``."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n)
    features = pd.DataFrame(rng.normal(size=(n, len(PAIR_FEATURES))), columns=PAIR_FEATURES)
    label = (features["zscore"] > 0) if signal else rng.integers(0, 2, n).astype(bool)
    return features.assign(
        pair_date_id=[f"KO__PEP|{d.date()}" for d in dates],
        label=label.astype(int),
        symbol_pair="KO__PEP",
        event_timestamp=dates,
    )


@pytest.fixture
def cfg(tmp_path, monkeypatch) -> PipelineConfig:
    monkeypatch.chdir(tmp_path)
    return PipelineConfig(
        tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
        data_table_uri=str(tmp_path / "training_frame"),
        horizon_bars=5,
        model_params={"n_estimators": 10},
    )


def test_run_logs_params_metrics_data_version_and_registers_model(cfg):
    result = run_pipeline(_frame(200), cfg)
    client = MlflowClient(cfg.tracking_uri)
    run = client.get_run(result.run_id)

    assert result.model_version == "1"
    assert result.alias == "production"
    assert result.metrics["auc"] > 0.9
    assert run.data.params["data_version"] == "0"
    assert run.data.params["n_estimators"] == "10"
    assert run.data.params["features"] == ",".join(PAIR_FEATURES)
    assert run.data.metrics["auc"] == pytest.approx(result.metrics["auc"])
    assert run.data.metrics["valid_rows"] == 50
    assert run.data.metrics["data_rows_inserted"] == 200
    assert str(client.get_model_version_by_alias("meta_label", "production").version) == "1"


def test_registered_model_reproduces_the_logged_predictions(cfg):
    import mlflow

    frame = _frame(200)
    run_pipeline(frame, cfg)
    mlflow.set_tracking_uri(cfg.tracking_uri)
    model = mlflow.lightgbm.load_model("models:/meta_label@production")
    prob = model.predict_proba(frame[PAIR_FEATURES])[:, 1]
    assert ((prob > 0.5) == (frame["zscore"] > 0)).mean() > 0.9


def test_second_run_versions_only_the_new_data(cfg):
    first = run_pipeline(_frame(200), cfg)
    second = run_pipeline(_frame(230), cfg)

    assert (first.data_version.version, second.data_version.version) == (0, 1)
    assert second.data_version.rows_inserted == 30
    assert second.model_version == "2"
    assert len(load_training_frame_version(cfg.data_table_uri, 0)) == 200
    assert len(load_training_frame_version(cfg.data_table_uri, 1)) == 230


def test_unchanged_data_keeps_the_data_version(cfg):
    run_pipeline(_frame(200), cfg)
    again = run_pipeline(_frame(200), cfg)
    assert again.data_version.version == 0
    assert not again.data_version.changed
    assert again.model_version == "2"


def test_worse_model_becomes_challenger_and_production_is_kept(cfg):
    run_pipeline(_frame(200), cfg)
    worse = run_pipeline(_frame(200, seed=1, signal=False), cfg)
    client = MlflowClient(cfg.tracking_uri)

    assert worse.alias == "challenger"
    assert str(client.get_model_version_by_alias("meta_label", "production").version) == "1"
    assert str(client.get_model_version_by_alias("meta_label", "challenger").version) == "2"


# Boundary values for promotion: new AUC just below, equal to, and just above
# the production AUC; plus an undefined (NaN) AUC, which never promotes.
@pytest.mark.parametrize(
    ("new_auc", "alias"),
    [(None, "production"), (1.0, "production"), (0.0, "challenger"), (float("nan"), "challenger")],
)
def test_promotion_rule(cfg, new_auc, alias):
    first = run_pipeline(_frame(200), cfg)
    client = MlflowClient(cfg.tracking_uri)
    auc = first.metrics["auc"] if new_auc is None else new_auc
    assert _assign_alias(client, "meta_label", "1", auc) == alias


@pytest.mark.parametrize("rows", [0, 3])
def test_too_few_rows_is_an_error_before_anything_is_logged(cfg, rows):
    with pytest.raises(ValueError, match="not enough rows"):
        run_pipeline(_frame(rows), cfg)
    assert MlflowClient(cfg.tracking_uri).search_registered_models() == []
