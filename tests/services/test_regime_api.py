"""Tests for the regime API and its drift statistics."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from services.regime_api import sources
from services.regime_api.app import CURRENT_BARS, REFERENCE_BARS, build_default_app, create_app
from services.regime_api.drift import (
    Regime,
    RegimeThresholds,
    assess_pair,
    classify,
    population_stability_index,
)
from services.regime_api.sources import PostgresPriceSource

from tests.services.conftest import FakePriceSource, make_pair


class TestPsi:
    def test_identical_samples_give_zero(self):
        x = np.random.default_rng(0).normal(size=500)
        assert population_stability_index(x, x) == pytest.approx(0.0, abs=1e-12)

    def test_same_distribution_is_small_and_shifted_is_large(self):
        rng = np.random.default_rng(0)
        reference = rng.normal(size=5000)
        same = population_stability_index(reference, rng.normal(size=5000))
        shifted = population_stability_index(reference, rng.normal(loc=2.0, size=5000))
        wider = population_stability_index(reference, rng.normal(scale=5.0, size=5000))
        assert same < 0.02
        assert shifted > 1.0
        assert wider > 0.5

    def test_constant_reference_does_not_divide_by_zero(self):
        psi = population_stability_index(np.ones(50), np.array([1.0, 2.0, 3.0]))
        assert np.isfinite(psi) and psi > 0

    def test_current_outside_reference_range_is_finite(self):
        psi = population_stability_index(np.arange(100.0), np.full(20, 1e9))
        assert np.isfinite(psi) and psi > 1

    @pytest.mark.parametrize(
        ("reference", "current", "bins"),
        [([], [1.0], 10), ([1.0], [], 10), ([1.0, 2.0], [1.0], 1)],
    )
    def test_invalid_input_rejected(self, reference, current, bins):
        with pytest.raises(ValueError):
            population_stability_index(np.array(reference), np.array(current), bins)

    @settings(
        max_examples=100, deadline=None, suppress_health_check=[HealthCheck.differing_executors]
    )
    @given(
        reference=st.lists(st.floats(-1e3, 1e3, allow_nan=False), min_size=1, max_size=60),
        current=st.lists(st.floats(-1e3, 1e3, allow_nan=False), min_size=1, max_size=60),
        bins=st.integers(2, 12),
    )
    def test_psi_is_never_negative_and_is_repeatable(self, reference, current, bins):
        first = population_stability_index(np.array(reference), np.array(current), bins)
        again = population_stability_index(np.array(reference), np.array(current), bins)
        assert first >= -1e-12
        assert first == again


# ---------------------------------------------------------------------------
# Equivalence partitions for classify (defaults: p 0.05 / 0.10, PSI 0.20 / 0.40):
#   stable   — both statistics at or below their "shifting" cut-off
#   shifting — either above "shifting", neither above "broken"
#   broken   — either above "broken"
# Boundary values: each cut-off at exactly / just above.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("pvalue", "psi", "expected"),
    [
        (0.0, 0.0, Regime.STABLE),
        (0.05, 0.20, Regime.STABLE),
        (0.0501, 0.0, Regime.SHIFTING),
        (0.0, 0.2001, Regime.SHIFTING),
        (0.10, 0.40, Regime.SHIFTING),
        (0.1001, 0.0, Regime.BROKEN),
        (0.0, 0.4001, Regime.BROKEN),
        (1.0, 9.0, Regime.BROKEN),
    ],
)
def test_classify_boundaries(pvalue, psi, expected):
    assert classify(pvalue, psi) is expected


def test_classify_uses_custom_thresholds():
    strict = RegimeThresholds(pvalue_shifting=0.01, pvalue_broken=0.02)
    assert classify(0.03, 0.0, strict) is Regime.BROKEN


class TestAssessPair:
    N = REFERENCE_BARS + CURRENT_BARS

    def test_cointegrated_pair_is_not_broken(self):
        result = assess_pair(*make_pair(self.N))
        assert result.coint_pvalue < 0.05
        assert result.regime is not Regime.BROKEN
        assert result.bars_used == self.N

    def test_pair_that_decouples_is_broken(self):
        result = assess_pair(*make_pair(self.N, seed=1, break_after=REFERENCE_BARS))
        assert result.regime is Regime.BROKEN
        assert result.psi > 0.40

    def test_only_the_latest_bars_are_used(self):
        a, b = make_pair(self.N + 200, seed=2)
        assert assess_pair(a, b) == assess_pair(a[200:], b[200:])

    # Boundary: exactly enough history vs one bar short.
    @pytest.mark.parametrize(("bars", "ok"), [(N, True), (N - 1, False), (0, False)])
    def test_history_length_boundary(self, bars, ok):
        a, b = make_pair(self.N)
        if ok:
            assess_pair(a[:bars], b[:bars])
        else:
            with pytest.raises(ValueError, match="need"):
                assess_pair(a[:bars], b[:bars])

    @pytest.mark.parametrize("bad", [np.nan, np.inf, 0.0, -1.0])
    def test_bad_prices_rejected(self, bad):
        a, b = make_pair(self.N)
        a[10] = bad
        with pytest.raises(ValueError, match="prices"):
            assess_pair(a, b)

    def test_mismatched_lengths_rejected(self):
        a, b = make_pair(self.N)
        with pytest.raises(ValueError, match="same length"):
            assess_pair(a, b[:-1])


class TestRegimeEndpoint:
    def test_stable_pair(self, regime_client, history_bars):
        body = regime_client.post("/regime", json={"pair_id": "KO__PEP"}).json()
        assert body["pair_id"] == "KO__PEP"
        assert body["regime"] in {"stable", "shifting"}
        assert 0.0 <= body["coint_pvalue"] <= 1.0
        assert body["bars_used"] == history_bars

    def test_broken_pair(self, regime_client):
        body = regime_client.post("/regime", json={"pair_id": "XOM__CVX"}).json()
        assert body["regime"] == "broken"

    def test_unknown_pair_is_404(self, regime_client):
        assert regime_client.post("/regime", json={"pair_id": "NOPE__X"}).status_code == 404

    def test_too_little_history_is_422(self, regime_client):
        response = regime_client.post("/regime", json={"pair_id": "GS__MS"})
        assert response.status_code == 422
        assert "need 315 bars, got 314" in response.json()["detail"]

    @pytest.mark.parametrize("payload", [{"pair_id": "ko__pep"}, {"pair_id": "KO_PEP"}, {}])
    def test_invalid_requests_are_422(self, regime_client, payload):
        assert regime_client.post("/regime", json=payload).status_code == 422

    def test_repeated_requests_give_identical_responses(self, regime_client):
        first = regime_client.post("/regime", json={"pair_id": "KO__PEP"}).json()
        assert all(
            regime_client.post("/regime", json={"pair_id": "KO__PEP"}).json() == first
            for _ in range(3)
        )

    def test_metrics_record_regime_and_psi(self, regime_client):
        regime_client.post("/regime", json={"pair_id": "XOM__CVX"})
        text = regime_client.get("/metrics").text
        assert 'regime_api_assessments_total{regime="broken"} 1.0' in text
        assert 'regime_api_spread_psi{pair_id="XOM__CVX"}' in text
        assert 'regime_api_requests_total{route="/regime",status="200"} 1.0' in text


class TestHealth:
    def test_healthz(self, regime_client):
        assert regime_client.get("/healthz").json() == {"status": "ok"}

    @pytest.mark.parametrize(("reachable", "code"), [(True, 200), (False, 503)])
    def test_readyz_follows_the_price_source(self, reachable, code):
        client = TestClient(create_app(FakePriceSource({}, reachable=reachable)))
        assert client.get("/readyz").status_code == code


class FakeConnection:
    """Minimal psycopg2 connection stand-in returning canned rows."""

    def __init__(self, rows):
        self.rows, self.executed = rows, []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self):
        return self

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchall(self):
        return self.rows


class TestPostgresPriceSource:
    def test_reads_closes_for_the_pair(self, monkeypatch):
        conn = FakeConnection([(10.0, 20.0), (11.0, 21.0)])
        monkeypatch.setattr(sources.psycopg2, "connect", lambda *a, **k: conn)
        close_a, close_b = PostgresPriceSource("dsn").get_closes("KO__PEP", 2)
        assert close_a.tolist() == [10.0, 11.0]
        assert close_b.tolist() == [20.0, 21.0]
        assert conn.executed[0][1] == ("KO", "PEP", 2)

    def test_unknown_pair_gives_none(self, monkeypatch):
        monkeypatch.setattr(sources.psycopg2, "connect", lambda *a, **k: FakeConnection([]))
        assert PostgresPriceSource("dsn").get_closes("NOPE__X", 5) is None

    def test_ping(self, monkeypatch):
        monkeypatch.setattr(sources.psycopg2, "connect", lambda *a, **k: FakeConnection([]))
        assert PostgresPriceSource("dsn").ping() is True

        def refuse(*args, **kwargs):
            raise sources.psycopg2.OperationalError("down")

        monkeypatch.setattr(sources.psycopg2, "connect", refuse)
        assert PostgresPriceSource("dsn").ping() is False


def test_default_app_uses_postgres_url(monkeypatch):
    monkeypatch.setenv("POSTGRES_URL", "postgresql://nobody@127.0.0.1:1/none")
    client = TestClient(build_default_app())
    assert client.get("/healthz").status_code == 200
    assert client.get("/readyz").status_code == 503
