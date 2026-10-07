"""Locust load test for the regime API (used to demonstrate autoscaling).

Usage:
    uv run locust -f services/regime_api/locustfile.py --host http://localhost:30081 \
        --headless -u 60 -r 20 -t 2m
"""

from __future__ import annotations

import random

from locust import HttpUser, between, task

PAIRS = ["KO__PEP", "XOM__CVX", "JPM__BAC", "GS__MS"]


class RegimeUser(HttpUser):
    """Asks for the regime of a known pair."""

    wait_time = between(0.05, 0.2)

    @task
    def regime(self) -> None:
        self.client.post("/regime", json={"pair_id": random.choice(PAIRS)})
