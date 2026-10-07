"""Pieces shared by the web APIs: pair-ID validation, Prometheus metrics, and OTel tracing."""

from __future__ import annotations

import os
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


def setup_tracing(service_name: str) -> None:
    """Initialise OpenTelemetry tracing and instrument FastAPI.

    Sends spans to the OTLP HTTP endpoint (Tempo in Docker, or OTEL_EXPORTER_OTLP_ENDPOINT).
    No-ops gracefully if the exporter cannot connect.
    """
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        return  # OTel not installed; skip silently

    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        return  # no endpoint configured; skip tracing (e.g. unit tests on host)
    resource = Resource.create({SERVICE_NAME: service_name})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces"))
    )
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor().instrument()
