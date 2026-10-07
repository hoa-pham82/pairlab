"""Pieces shared by the web APIs: pair-ID validation and Prometheus metrics."""

from __future__ import annotations

import time
from typing import Annotated

from fastapi import FastAPI, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import StringConstraints

PAIR_ID_PATTERN = r"^[A-Z][A-Z0-9.]{0,9}__[A-Z][A-Z0-9.]{0,9}$"
PairId = Annotated[str, StringConstraints(pattern=PAIR_ID_PATTERN)]


class HttpMetrics:
    """Request count and latency for one app, on its own Prometheus registry."""

    def __init__(self, service: str) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            f"{service}_requests_total",
            "HTTP requests by route and status code",
            ["route", "status"],
            registry=self.registry,
        )
        self.latency = Histogram(
            f"{service}_request_seconds",
            "HTTP request latency by route",
            ["route"],
            registry=self.registry,
        )

    def install(self, app: FastAPI) -> None:
        """Record every request and expose ``GET /metrics``."""

        @app.middleware("http")
        async def record(request: Request, call_next):
            started = time.perf_counter()
            status = 500
            try:
                response = await call_next(request)
                status = response.status_code
                return response
            finally:
                route = request.scope.get("route")
                name = getattr(route, "path", "unmatched")
                self.requests.labels(name, str(status)).inc()
                self.latency.labels(name).observe(time.perf_counter() - started)

        @app.get("/metrics", include_in_schema=False)
        async def metrics() -> Response:
            return Response(generate_latest(self.registry), media_type=CONTENT_TYPE_LATEST)
