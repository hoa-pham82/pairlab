"""Tests for the with-vs-without-filter backtest comparison."""

from __future__ import annotations

from datetime import UTC, date, datetime

import numpy as np
import pandas as pd
import pytest
from ml.backtest_filter import compare_with_filter, read_bars, read_features
from ml.dataset import PAIR_FEATURES
from ml.scoring import ClassifierScorer, score_features

from pairlab.config import BacktestConfig, StrategyConfig

PAIRS = [("AAA", "BBB")]
START = pd.Timestamp("2021-01-04", tz="UTC")


@pytest.fixture(scope="module")
def market() -> tuple[pd.DataFrame, pd.DataFrame]:
    """600 days of a tightly cointegrated pair, with a feature row for every day."""
    rng = np.random.default_rng(3)
    n = 600
    days = pd.bdate_range(START, periods=n)
    log_a = np.log(100) + np.cumsum(rng.normal(0, 0.01, n))
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = 0.8 * spread[t - 1] + rng.normal(0, 0.01)
    closes = {"AAA": np.exp(log_a), "BBB": np.exp(log_a - spread)}
    bars = pd.concat(
        pd.DataFrame(
            {"symbol": s, "ts": days, "open": c, "high": c, "low": c, "close": c, "volume": 1e6}
        )
        for s, c in closes.items()
    ).reset_index(drop=True)
    features = pd.DataFrame(
        {"symbol_a": "AAA", "symbol_b": "BBB", "event_timestamp": days.tz_localize(None)}
    ).assign(**dict.fromkeys(PAIR_FEATURES, 1.0))
    return bars, features


def _cfg() -> BacktestConfig:
    return BacktestConfig(
        start_date=date(2021, 1, 1),
        end_date=date(2023, 12, 31),
        strategy=StrategyConfig(formation_window=120, zscore_window=20),
    )


class Fixed:
    def __init__(self, prob: float):
        self.prob = prob

    def predict_proba(self, features):
        return self.prob


def _run(market, prob, active_from=None):
    bars, features = market
    active_from = active_from or START.to_pydatetime()
    return compare_with_filter(bars, features, Fixed(prob), _cfg(), PAIRS, active_from)


def test_baseline_trades(market):
    assert _run(market, 1.0)["baseline"]["n_trades"] > 0


def test_filter_that_takes_everything_matches_the_baseline(market):
    report = _run(market, 1.0)
    assert report["filtered"] == report["baseline"]
    assert report["filter"]["skipped"] == 0
    assert report["filter"]["taken"] > 0


def test_filter_that_rejects_everything_never_trades(market):
    report = _run(market, 0.0)
    assert report["filtered"]["n_trades"] == 0
    assert report["filtered"]["total_pnl"] == 0
    assert report["filter"]["taken"] == 0
    assert report["filter"]["skipped"] > 0


def test_filter_is_inactive_before_active_from(market):
    """With active_from after the data ends, rejecting everything changes nothing."""
    report = _run(market, 0.0, active_from=datetime(2030, 1, 1, tzinfo=UTC))
    assert report["filtered"] == report["baseline"]
    assert report["filter"]["skipped"] == 0


def test_report_records_threshold_and_active_from(market):
    report = _run(market, 1.0)
    assert report["filter"]["threshold"] == 0.5
    assert report["filter"]["active_from"] == "2021-01-04"


def test_tearsheet_is_not_written(market, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _run(market, 1.0)
    assert list(tmp_path.iterdir()) == []


class FakeConnection:
    def __init__(self, columns, rows):
        self.description = [type("C", (), {"name": c})() for c in columns]
        self.rows, self.params = rows, None

    def cursor(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.params = params

    def fetchall(self):
        return self.rows


def test_read_bars_converts_nanosecond_timestamps_and_sorts():
    columns = ["symbol", "ts", "open", "high", "low", "close", "volume"]
    rows = [
        ("BBB", 1514937600000000000, 1, 1, 1, 1, 1),
        ("AAA", 1514937600000000000, 2, 2, 2, 2, 2),
        ("AAA", 1514851200000000000, 3, 3, 3, 3, 3),
    ]
    conn = FakeConnection(columns, rows)
    bars = read_bars(conn, ["AAA", "BBB"])
    assert conn.params == (["AAA", "BBB"],)
    assert bars["symbol"].tolist() == ["AAA", "AAA", "BBB"]
    assert bars["ts"].iloc[0] == pd.Timestamp("2018-01-02", tz="UTC")
    assert bars["close"].tolist() == [3, 2, 1]


def test_read_features_returns_named_columns():
    conn = FakeConnection(["symbol_a", "zscore"], [("KO", 1.5)])
    assert read_features(conn).to_dict("records") == [{"symbol_a": "KO", "zscore": 1.5}]


def test_classifier_scorer_uses_features_in_model_order():
    class Model:
        def predict_proba(self, frame):
            assert frame.columns.tolist() == PAIR_FEATURES
            return np.array([[0.3, 0.7]])

    values = dict.fromkeys(reversed(PAIR_FEATURES), 1.0)
    assert score_features(Model(), values) == 0.7
    assert ClassifierScorer(Model()).predict_proba(values) == 0.7
