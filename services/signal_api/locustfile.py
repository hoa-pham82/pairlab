"""Locust load test for the signal API.

Usage:
    uv run uvicorn services.signal_api.app:build_default_app --factory --port 8000
    uv run locust -f services/signal_api/locustfile.py --host http://localhost:8000 \
        --headless -u 50 -r 10 -t 30s --html docs/pngs/locust_signal_api.html
"""

from __future__ import annotations

import random

from locust import HttpUser, between, task

PAIRS = ["KO__PEP", "XOM__CVX", "JPM__BAC", "GS__MS"]


class SignalUser(HttpUser):
    """Mostly scores known pairs; sometimes asks for an unknown one or checks health."""

    wait_time = between(0.01, 0.05)

    @task(20)
    def score_known_pair(self) -> None:
        self.client.post("/signal", json={"pair_id": random.choice(PAIRS)})

    @task(1)
    def score_unknown_pair(self) -> None:
        with self.client.post(
            "/signal", json={"pair_id": "NOPE__X"}, catch_response=True
        ) as response:
            if response.status_code == 404:
                response.success()

    @task(1)
    def readiness(self) -> None:
        self.client.get("/readyz")
