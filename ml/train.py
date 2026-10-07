"""Train, evaluate, and save the meta-label classifier."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

DEFAULT_PARAMS: dict = {
    "n_estimators": 100,
    "learning_rate": 0.05,
    "num_leaves": 7,
    "min_child_samples": 10,
    "random_state": 42,
    "n_jobs": 1,
    "deterministic": True,
    "verbose": -1,
}


def train_model(x: pd.DataFrame, y: pd.Series, params: dict | None = None) -> LGBMClassifier:
    """Fit a LightGBM classifier; ``params`` override ``DEFAULT_PARAMS``."""
    model = LGBMClassifier(**{**DEFAULT_PARAMS, **(params or {})})
    model.fit(x, y)
    return model


def evaluate(model, x: pd.DataFrame, y: pd.Series, threshold: float = 0.5) -> dict[str, float]:
    """Score the model on held-out rows.

    Returns:
        ``auc`` (NaN when ``y`` has one class), ``precision``, ``recall``,
        ``accuracy``, and ``take_rate`` (share of signals the model would trade).
    """
    prob = model.predict_proba(x)[:, 1]
    take = (prob >= threshold).astype(int)
    return {
        "auc": float(roc_auc_score(y, prob)) if y.nunique() > 1 else float("nan"),
        "precision": float(precision_score(y, take, zero_division=0)),
        "recall": float(recall_score(y, take, zero_division=0)),
        "accuracy": float(accuracy_score(y, take)),
        "take_rate": float(take.mean()),
    }


def save_model(model, path: str | Path) -> Path:
    """Write the model to ``path`` with joblib and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: str | Path):
    """Load a model written by ``save_model``."""
    return joblib.load(path)
