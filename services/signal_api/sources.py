"""Where the signal API gets features and its model."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Protocol

import joblib
from ml.dataset import ENTITY_COLUMN, PAIR_FEATURE_VIEW, PAIR_FEATURES
from ml.scoring import score_features


class FeatureSource(Protocol):
    """Looks up the latest features for a pair."""

    def get_features(self, pair_id: str) -> dict[str, float] | None:
        """Return the feature values, or None if the pair has none stored."""


class Model(Protocol):
    """Scores one feature row."""

    version: str

    def predict_proba(self, features: dict[str, float]) -> float:
        """Return the probability that the signal reverts."""


class FeastFeatureSource:
    """Reads pair features from the Feast online store."""

    def __init__(self, store) -> None:
        self._store = store
        self._refs = [f"{PAIR_FEATURE_VIEW}:{name}" for name in PAIR_FEATURES]

    def get_features(self, pair_id: str) -> dict[str, float] | None:
        row = self._store.get_online_features(
            features=self._refs, entity_rows=[{ENTITY_COLUMN: pair_id}]
        ).to_dict()
        values = {name: row[name][0] for name in PAIR_FEATURES}
        if any(value is None for value in values.values()):
            return None
        return {name: float(value) for name, value in values.items()}


class _ClassifierModel:
    """Scores a feature dict with a scikit-learn-style classifier."""

    version: str
    _model: object

    @property
    def estimator(self) -> object:
        """The underlying fitted classifier."""
        return self._model

    def predict_proba(self, features: dict[str, float]) -> float:
        return score_features(self._model, features)


class JoblibModel(_ClassifierModel):
    """A classifier loaded from a joblib file; versioned by file content unless told otherwise."""

    def __init__(self, path: str | Path, version: str | None = None) -> None:
        data = Path(path).read_bytes()
        self.version = version or f"file-{hashlib.sha256(data).hexdigest()[:12]}"
        self._model = joblib.load(path)


class MlflowModel(_ClassifierModel):
    """The registry model version that currently holds ``alias``."""

    def __init__(self, tracking_uri: str, name: str, alias: str = "production") -> None:
        import mlflow
        from mlflow.tracking import MlflowClient

        # Presigned links name the object store's in-network host; fetch via the server.
        os.environ.setdefault("MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD", "false")
        mlflow.set_tracking_uri(tracking_uri)
        registered = MlflowClient(tracking_uri).get_model_version_by_alias(name, alias)
        self.version = f"{name}-v{registered.version}"
        self._model = mlflow.lightgbm.load_model(f"models:/{name}@{alias}")
