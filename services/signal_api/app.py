"""FastAPI app for the signal API."""

from __future__ import annotations

import logging
import math
import os

from fastapi import FastAPI, HTTPException, Response, status
from fastapi.concurrency import run_in_threadpool
from prometheus_client import Counter

from services.ab import CHALLENGER, CHAMPION, assign_variant
from services.common import HttpMetrics, setup_tracing
from services.signal_api.schemas import Health, SignalRequest, SignalResponse
from services.signal_api.sources import FeatureSource, Model

DEFAULT_THRESHOLD = 0.5
log = logging.getLogger(__name__)


def create_app(
    features: FeatureSource | None,
    model: Model | None,
    threshold: float = DEFAULT_THRESHOLD,
    challenger: Model | None = None,
    challenger_share: float = 0.5,
) -> FastAPI:
    """Build the app around a feature source and a model (either may be missing).

    With a ``challenger`` model, ``challenger_share`` of pairs (chosen by a
    stable hash of the pair ID) are scored by it; the rest by ``model``.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    assign_variant("", challenger_share)  # validates the share

    setup_tracing("signal_api")
    app = FastAPI(title="pairlab signal API", version="0.1.0")
    metrics = HttpMetrics("signal_api")
    metrics.install(app)
    decisions = Counter(
        "signal_api_decisions_total",
        "Scored signals by decision, model version and A/B variant",
        ["take_trade", "model_version", "variant"],
        registry=metrics.registry,
    )

    @app.get("/healthz", response_model=Health)
    async def healthz() -> Health:
        return Health(status="ok")

    @app.get("/readyz", response_model=Health)
    async def readyz(response: Response) -> Health:
        if features is None or model is None:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return Health(status="not ready")
        return Health(status="ready")

    @app.post("/signal", response_model=SignalResponse)
    async def signal(request: SignalRequest) -> SignalResponse:
        if features is None or model is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "model or features not loaded")

        values = await run_in_threadpool(features.get_features, request.pair_id)
        if values is None:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND, f"no online features for {request.pair_id}"
            )
        if not all(math.isfinite(v) for v in values.values()):
            raise HTTPException(422, f"non-finite features for {request.pair_id}")

        variant = CHAMPION
        if challenger is not None:
            variant = assign_variant(request.pair_id, challenger_share)
        scorer = challenger if variant == CHALLENGER else model
        prob = await run_in_threadpool(scorer.predict_proba, values)
        take_trade = prob >= threshold
        decisions.labels(str(take_trade).lower(), scorer.version, variant).inc()
        return SignalResponse(
            pair_id=request.pair_id,
            prob=prob,
            take_trade=take_trade,
            threshold=threshold,
            model_version=scorer.version,
            variant=variant,
            features=values,
        )

    return app


def build_default_app() -> FastAPI:
    """App wired to Feast and the saved model, from environment variables.

    ``FEAST_REPO`` (default ``platform/feature_repo``), ``TAKE_THRESHOLD``
    (default 0.5), and the model settings read by ``_load_model``. A missing
    model or feature store leaves the app running but not ready.
    """
    from services.vault_secrets import load_secrets
    load_secrets(["secret/data/minio", "secret/data/postgres"])

    from feast import FeatureStore

    from services.signal_api.sources import FeastFeatureSource

    try:
        features = FeastFeatureSource(
            FeatureStore(repo_path=os.environ.get("FEAST_REPO", "platform/feature_repo"))
        )
    except Exception:
        log.exception("feature store unavailable; /readyz will report not ready")
        features = None
    try:
        model = _load_model()
    except Exception:
        log.exception("model unavailable; /readyz will report not ready")
        model = None
    return create_app(
        features,
        model,
        float(os.environ.get("TAKE_THRESHOLD", DEFAULT_THRESHOLD)),
        challenger=_load_challenger(),
        challenger_share=float(os.environ.get("CHALLENGER_SHARE", "0.5")),
    )


def _load_challenger() -> Model | None:
    """Load the registry model at ``CHALLENGER_ALIAS`` (default ``challenger``), if any."""
    from services.signal_api.sources import MlflowModel

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        return None
    try:
        return MlflowModel(
            tracking_uri,
            os.environ.get("MODEL_NAME", "meta_label"),
            os.environ.get("CHALLENGER_ALIAS", "challenger"),
        )
    except Exception:
        log.info("no challenger model in the registry; serving the champion only")
        return None


def _load_model() -> Model:
    """Load from the MLflow registry if ``MLFLOW_TRACKING_URI`` is set, else from a file.

    Registry: ``MODEL_NAME`` (default ``meta_label``) at ``MODEL_ALIAS`` (default
    ``production``). File: ``MODEL_PATH`` (default ``models/meta_label.joblib``).
    """
    from services.signal_api.sources import JoblibModel, MlflowModel

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        return MlflowModel(
            tracking_uri,
            os.environ.get("MODEL_NAME", "meta_label"),
            os.environ.get("MODEL_ALIAS", "production"),
        )
    return JoblibModel(os.environ.get("MODEL_PATH", "models/meta_label.joblib"))
