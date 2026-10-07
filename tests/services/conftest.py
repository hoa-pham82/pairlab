"""Fixtures for the web API tests: in-memory fakes instead of Feast, Redis, and Postgres."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient
from ml.dataset import PAIR_FEATURES
from services.regime_api.app import CURRENT_BARS, REFERENCE_BARS
from services.regime_api.app import create_app as create_regime_app
from services.signal_api.app import create_app as create_signal_app

FEATURES = {"zscore": 2.3, "hedge_ratio": 0.9, "spread_vol": 0.04, "correlation_60d": 0.6}


class FakeFeatureSource:
    """Feature lookup backed by a dict."""

    def __init__(self, rows: dict[str, dict[str, float]]):
        self.rows = rows
        self.calls: list[str] = []

    def get_features(self, pair_id):
        self.calls.append(pair_id)
        return self.rows.get(pair_id)


class FakeModel:
    """Deterministic model: probability is a fixed value, or a function of zscore."""

    version = "test-1"

    def __init__(self, prob: float | None = None):
        self.prob = prob

    def predict_proba(self, features):
        if self.prob is not None:
            return self.prob
        return float(1 / (1 + np.exp(-np.clip(features["zscore"], -50, 50))))


class FakePriceSource:
    """Price lookup backed by a dict of (close_a, close_b) arrays."""

    def __init__(self, rows: dict[str, tuple[np.ndarray, np.ndarray]], reachable: bool = True):
        self.rows = rows
        self.reachable = reachable

    def get_closes(self, pair_id, n_bars):
        closes = self.rows.get(pair_id)
        return None if closes is None else (closes[0][-n_bars:], closes[1][-n_bars:])

    def ping(self):
        return self.reachable


def make_pair(n: int, seed: int = 0, break_after: int | None = None):
    """Cointegrated closes; from ``break_after`` on, leg B follows its own random walk."""
    rng = np.random.default_rng(seed)
    log_a = np.log(100) + np.cumsum(rng.normal(0, 0.02, n))
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = 0.7 * spread[t - 1] + rng.normal(0, 0.004)
    log_b = log_a - spread
    if break_after is not None:
        log_b[break_after:] += np.cumsum(rng.normal(0, 0.03, n - break_after))
    return np.exp(log_a), np.exp(log_b)


@pytest.fixture
def feature_source() -> FakeFeatureSource:
    return FakeFeatureSource({"KO__PEP": dict(FEATURES)})


@pytest.fixture
def signal_client(feature_source) -> TestClient:
    return TestClient(create_signal_app(feature_source, FakeModel()))


@pytest.fixture
def history_bars() -> int:
    return REFERENCE_BARS + CURRENT_BARS


@pytest.fixture
def regime_client(history_bars) -> TestClient:
    prices = FakePriceSource(
        {
            "KO__PEP": make_pair(history_bars),
            "XOM__CVX": make_pair(history_bars, seed=1, break_after=REFERENCE_BARS),
            "GS__MS": make_pair(history_bars - 1),
        }
    )
    return TestClient(create_regime_app(prices))


assert set(FEATURES) == set(PAIR_FEATURES)
