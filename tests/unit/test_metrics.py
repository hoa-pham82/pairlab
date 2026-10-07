"""Tests for performance metrics: known-answer, edge cases, and oracle vs quantstats."""

from __future__ import annotations

import math
import warnings
from datetime import datetime, timezone

import numpy as np
import pytest

from pairlab.metrics.performance import PerformanceMetrics, compute_metrics, max_drawdown
from tests.fixtures.bars import utc


def _equity_curve(values: list[float], start_year: int = 2020) -> list[tuple[datetime, float]]:
    from datetime import timedelta
    base = utc(start_year, 1, 2)
    return [(base + timedelta(days=i), v) for i, v in enumerate(values)]


class TestMaxDrawdown:
    def test_monotone_up_no_drawdown(self):
        mdd, dur = max_drawdown(np.array([100.0, 105.0, 110.0, 115.0]))
        assert mdd == 0.0
        assert dur == 0

    def test_known_drawdown(self):
        """Peak 110, trough 88 → MDD = (110-88)/110 ≈ 0.2."""
        eq = np.array([100.0, 105.0, 110.0, 99.0, 88.0, 95.0, 110.0])
        mdd, _ = max_drawdown(eq)
        expected = (110 - 88) / 110
        assert abs(mdd - expected) < 1e-9

    def test_duration_counted(self):
        """Peak at index 1 (110), trough at index 4 (95), back above peak at index 5 → underwater for 3 bars."""
        eq = np.array([100.0, 110.0, 105.0, 100.0, 95.0, 112.0])
        _, dur = max_drawdown(eq)
        assert dur == 3


class TestComputeMetrics:
    def test_single_point_returns_empty(self):
        m = compute_metrics([( utc(2020, 1, 2), 1_000_000.0)])
        assert m.total_return == 0.0
        assert math.isnan(m.sharpe)

    def test_zero_volatility_sharpe_nan_with_warning(self):
        """Constant equity → zero vol → Sharpe NaN with a warning."""
        eq = _equity_curve([1_000_000.0] * 50)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            m = compute_metrics(eq)
        assert math.isnan(m.sharpe)
        assert any("Zero volatility" in str(warning.message) for warning in w)

    def test_monotone_up_mdd_zero(self):
        eq = _equity_curve([100_000 + i * 100 for i in range(252)])
        m = compute_metrics(eq)
        assert m.max_drawdown == 0.0

    def test_all_losses_profit_factor_zero_range(self):
        """All negative returns → profit_factor should be 0 or NaN (no wins)."""
        eq = _equity_curve([100_000 - i * 100 for i in range(50)])
        m = compute_metrics(eq)
        # No fills → profit_factor from trade metrics is NaN
        assert math.isnan(m.profit_factor)

    def test_positive_sharpe_for_uptrend(self):
        rng = np.random.default_rng(0)
        # Strong positive drift
        eq_vals = list(np.cumprod(1 + rng.normal(0.001, 0.005, 252)) * 1_000_000)
        eq = _equity_curve(eq_vals)
        m = compute_metrics(eq)
        assert m.sharpe > 0

    def test_total_return_known(self):
        """From 1000 to 1100 = 10% return."""
        eq = _equity_curve([1000.0, 1050.0, 1100.0])
        m = compute_metrics(eq)
        assert abs(m.total_return - 0.10) < 1e-9

    def test_negative_return(self):
        eq = _equity_curve([1000.0, 900.0])
        m = compute_metrics(eq)
        assert m.total_return == pytest.approx(-0.1, abs=1e-9)

    @pytest.mark.parametrize("start,end,expected_return", [
        (100.0, 100.0, 0.0),   # EP: no change
        (100.0, 200.0, 1.0),   # EP: double
        (100.0, 50.0, -0.5),   # EP: halve
    ])
    def test_total_return_partitions(self, start, end, expected_return):
        eq = _equity_curve([start, end])
        m = compute_metrics(eq)
        assert abs(m.total_return - expected_return) < 1e-9
