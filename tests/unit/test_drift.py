"""Tests for the feature-drift check used by the drift DAG."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from ml import drift
from ml.drift import DriftReport, drift_report, push_report

REF, CUR = 252, 63


def _pair(symbol_a: str, symbol_b: str, n: int, seed: int, break_after: int | None = None):
    """Feature rows for a cointegrated pair; leg B decouples from ``break_after`` on."""
    rng = np.random.default_rng(seed)
    log_a = np.log(100) + np.cumsum(rng.normal(0, 0.02, n))
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = 0.7 * spread[t - 1] + rng.normal(0, 0.004)
    log_b = log_a - spread
    if break_after is not None:
        log_b[break_after:] += np.cumsum(rng.normal(0, 0.03, n - break_after))
    return pd.DataFrame(
        {
            "symbol_a": symbol_a,
            "symbol_b": symbol_b,
            "event_timestamp": pd.bdate_range("2023-01-02", periods=n),
            "close_a": np.exp(log_a),
            "close_b": np.exp(log_b),
            "zscore": rng.normal(size=n),
        }
    )


def test_stable_pair_is_not_flagged():
    report = drift_report(_pair("KO", "PEP", REF + CUR, seed=0))
    assert report.spread_psi["KO__PEP"] < 0.40
    assert not report.drifted


def test_decoupled_pair_is_flagged_and_named():
    frame = pd.concat(
        [_pair("KO", "PEP", REF + CUR, 0), _pair("XOM", "CVX", REF + CUR, 1, break_after=REF)]
    )
    report = drift_report(frame)
    assert report.drifted_pairs == ["XOM__CVX"]
    assert report.drifted


# Boundary: exactly enough history is assessed; one bar short is skipped.
@pytest.mark.parametrize(("bars", "assessed"), [(REF + CUR, True), (REF + CUR - 1, False)])
def test_history_length_boundary(bars, assessed):
    report = drift_report(_pair("KO", "PEP", bars, seed=0))
    assert ("KO__PEP" in report.spread_psi) is assessed


def test_only_the_latest_bars_are_used():
    long = _pair("KO", "PEP", REF + CUR + 100, seed=2)
    assert drift_report(long).spread_psi == drift_report(long.tail(REF + CUR)).spread_psi


def test_no_assessable_pairs_gives_empty_stable_report():
    report = drift_report(_pair("KO", "PEP", 10, seed=0))
    assert report.spread_psi == {}
    assert np.isnan(report.zscore_psi)
    assert not report.drifted


def test_nan_zscores_are_ignored():
    frame = _pair("KO", "PEP", REF + CUR, seed=0)
    frame.loc[:40, "zscore"] = np.nan
    assert np.isfinite(drift_report(frame).zscore_psi)


# Boundary values for the verdict: PSI exactly at the threshold is not drift.
@pytest.mark.parametrize(("psi", "drifted"), [(0.39, False), (0.40, False), (0.41, True)])
def test_threshold_boundary(psi, drifted):
    assert DriftReport({"KO__PEP": psi}, 0.0, threshold=0.40).drifted is drifted


def test_push_report_sends_gauges_to_the_gateway(monkeypatch):
    sent = {}

    def fake_push(gateway, job, registry):
        from prometheus_client import generate_latest

        sent.update(gateway=gateway, job=job, text=generate_latest(registry).decode())

    monkeypatch.setattr("prometheus_client.push_to_gateway", fake_push)
    push_report(DriftReport({"KO__PEP": 0.5}, 0.12, threshold=0.40), "http://gw:9091")

    assert sent["gateway"] == "http://gw:9091"
    assert sent["job"] == "pairlab_drift"
    assert 'pairlab_spread_change_psi{pair_id="KO__PEP"} 0.5' in sent["text"]
    assert "pairlab_zscore_psi 0.12" in sent["text"]
    assert "pairlab_drift_detected 1.0" in sent["text"]


def test_push_report_omits_undefined_zscore_psi(monkeypatch):
    sent = {}

    def fake_push(gateway, job, registry):
        from prometheus_client import generate_latest

        sent["text"] = generate_latest(registry).decode()

    monkeypatch.setattr("prometheus_client.push_to_gateway", fake_push)
    push_report(DriftReport({}, float("nan"), threshold=0.40), "http://gw:9091")
    assert "pairlab_zscore_psi" not in sent["text"]
    assert "pairlab_drift_detected 0.0" in sent["text"]


def test_read_pair_features_returns_named_columns(monkeypatch):
    class Cursor:
        description = [type("C", (), {"name": n})() for n in ("symbol_a", "zscore")]

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def execute(self, sql):
            self.sql = sql

        def fetchall(self):
            return [("KO", 1.5)]

    class Conn(Cursor):
        def cursor(self):
            return Cursor()

    import psycopg2

    monkeypatch.setattr(psycopg2, "connect", lambda dsn: Conn())
    frame = drift.read_pair_features("dsn")
    assert frame.to_dict("records") == [{"symbol_a": "KO", "zscore": 1.5}]
