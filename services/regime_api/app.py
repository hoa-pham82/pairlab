"""FastAPI app for the regime (drift detection) API."""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException, Response, status
from fastapi.concurrency import run_in_threadpool
from prometheus_client import Counter, Gauge

from services.common import HttpMetrics
from services.regime_api.drift import assess_pair
from services.regime_api.schemas import Health, RegimeRequest, RegimeResponse
from services.regime_api.sources import PriceSource

REFERENCE_BARS = 252
CURRENT_BARS = 63
DEFAULT_DSN = "postgresql://pairlab:pairlab@localhost:5432/pairlab"


def create_app(prices: PriceSource) -> FastAPI:
    """Build the app around a price source."""
    app = FastAPI(title="pairlab regime API", version="0.1.0")
    metrics = HttpMetrics("regime_api")
    metrics.install(app)
    regimes = Counter(
        "regime_api_assessments_total",
        "Assessments by resulting regime",
        ["regime"],
        registry=metrics.registry,
    )
    psi_gauge = Gauge(
        "regime_api_spread_psi", "Latest spread PSI per pair", ["pair_id"],
        registry=metrics.registry,
    )

    @app.get("/healthz", response_model=Health)
    async def healthz() -> Health:
        return Health(status="ok")

    @app.get("/readyz", response_model=Health)
    async def readyz(response: Response) -> Health:
        if not await run_in_threadpool(prices.ping):
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return Health(status="not ready")
        return Health(status="ready")

    @app.post("/regime", response_model=RegimeResponse)
    async def regime(request: RegimeRequest) -> RegimeResponse:
        closes = await run_in_threadpool(
            prices.get_closes, request.pair_id, REFERENCE_BARS + CURRENT_BARS
        )
        if closes is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"no prices for {request.pair_id}")
        try:
            result = await run_in_threadpool(
                assess_pair, closes[0], closes[1], REFERENCE_BARS, CURRENT_BARS
            )
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

        regimes.labels(result.regime.value).inc()
        psi_gauge.labels(request.pair_id).set(result.psi)
        return RegimeResponse(
            pair_id=request.pair_id,
            regime=result.regime,
            coint_pvalue=result.coint_pvalue,
            psi=result.psi,
            bars_used=result.bars_used,
        )

    return app


def build_default_app() -> FastAPI:
    """App wired to the warehouse, using ``POSTGRES_URL`` if set."""
    from services.regime_api.sources import PostgresPriceSource

    return create_app(PostgresPriceSource(os.environ.get("POSTGRES_URL", DEFAULT_DSN)))
