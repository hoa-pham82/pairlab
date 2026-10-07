"""Score one feature row with a scikit-learn-style classifier."""

from __future__ import annotations

import pandas as pd

from ml.dataset import PAIR_FEATURES


def score_features(model, features: dict[str, float]) -> float:
    """Return the model's probability that the signal reverts."""
    row = pd.DataFrame([[features[name] for name in PAIR_FEATURES]], columns=PAIR_FEATURES)
    return float(model.predict_proba(row)[0, 1])


class ClassifierScorer:
    """Gives a classifier the ``predict_proba(features: dict) -> float`` interface."""

    def __init__(self, model) -> None:
        self._model = model

    def predict_proba(self, features: dict[str, float]) -> float:
        return score_features(self._model, features)
