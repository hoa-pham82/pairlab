"""Tests for meta-label model training, evaluation, and persistence."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from ml.dataset import PAIR_FEATURES
from ml.train import evaluate, load_model, save_model, train_model


@pytest.fixture(scope="module")
def data() -> tuple[pd.DataFrame, pd.Series]:
    """200 rows where the label is 1 exactly when zscore is positive."""
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(size=(200, len(PAIR_FEATURES))), columns=PAIR_FEATURES)
    return x, (x["zscore"] > 0).astype(int)


@pytest.fixture(scope="module")
def model(data):
    return train_model(*data)


class FixedProbability:
    """Model stub that returns the same probability for every row."""

    def __init__(self, prob: float):
        self.prob = prob

    def predict_proba(self, x):
        return np.column_stack([np.full(len(x), 1 - self.prob), np.full(len(x), self.prob)])


def test_model_learns_a_separable_rule(model, data):
    assert evaluate(model, *data)["auc"] > 0.95


def test_training_is_deterministic(model, data):
    x, y = data
    again = train_model(x, y)
    np.testing.assert_array_equal(model.predict_proba(x), again.predict_proba(x))


def test_params_override_defaults(data):
    assert train_model(*data, params={"n_estimators": 3}).n_estimators == 3


# Boundary values for the decision threshold: probability just below, exactly
# at, and just above it. A signal is taken when probability >= threshold.
@pytest.mark.parametrize(
    ("prob", "take_rate"),
    [(0.49, 0.0), (0.5, 1.0), (0.51, 1.0)],
)
def test_threshold_boundary(prob, take_rate):
    x = pd.DataFrame({"zscore": [0.0, 0.0]})
    metrics = evaluate(FixedProbability(prob), x, pd.Series([0, 1]), threshold=0.5)
    assert metrics["take_rate"] == take_rate


def test_metrics_on_a_known_case():
    """Taking every signal when half are good: precision 0.5, recall 1."""
    x = pd.DataFrame({"zscore": [0.0] * 4})
    metrics = evaluate(FixedProbability(0.9), x, pd.Series([1, 0, 1, 0]))
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 1.0
    assert metrics["accuracy"] == 0.5


def test_single_class_target_gives_nan_auc_not_an_error():
    x = pd.DataFrame({"zscore": [0.0, 0.0]})
    metrics = evaluate(FixedProbability(0.1), x, pd.Series([1, 1]))
    assert math.isnan(metrics["auc"])
    assert metrics["precision"] == 0.0


def test_save_and_load_round_trip(model, data, tmp_path):
    x, _ = data
    path = save_model(model, tmp_path / "nested" / "model.joblib")
    assert path.exists()
    np.testing.assert_array_equal(load_model(path).predict_proba(x), model.predict_proba(x))
