"""Tests for transaction cost model."""

from __future__ import annotations

import pytest

from pairlab.config import CostConfig
from pairlab.execution.costs import (
    borrow_cost_daily,
    commission,
    half_spread_cost,
    slippage_cost,
    total_cost,
)


def _cfg(**kwargs) -> CostConfig:
    defaults = dict(
        commission_bps=5.0,
        commission_min_usd=1.0,
        half_spread_bps=2.0,
        slippage_factor=0.1,
        borrow_cost_bps_yr=50.0,
    )
    defaults.update(kwargs)
    return CostConfig(**defaults)


class TestCommission:
    def test_bps_formula(self):
        cfg = _cfg(commission_bps=10.0, commission_min_usd=0.0)
        # 100 shares @ $50 = $5000; 10 bps = $5
        assert abs(commission(100, 50.0, cfg) - 5.0) < 0.01

    def test_minimum_enforced(self):
        cfg = _cfg(commission_bps=1.0, commission_min_usd=5.0)
        # 1 share @ $1 = $1 notional; 1 bps = $0.0001 → minimum $5 applies
        assert commission(1, 1.0, cfg) == 5.0

    def test_negative_quantity(self):
        """Short sell: same cost as long."""
        cfg = _cfg(commission_bps=5.0, commission_min_usd=0.0)
        assert abs(commission(-100, 50.0, cfg) - commission(100, 50.0, cfg)) < 1e-9


class TestSlippage:
    def test_zero_volume_zero_slippage(self):
        """Halt — no volume, no slippage (order shouldn't fill anyway)."""
        assert slippage_cost(100, 50.0, 0, 0.01, _cfg()) == 0.0

    def test_zero_quantity(self):
        assert slippage_cost(0, 50.0, 1_000_000, 0.01, _cfg()) == 0.0

    def test_monotonic_in_participation(self):
        """Higher participation (bigger order relative to volume) → higher slippage."""
        cfg = _cfg(slippage_factor=0.1)
        s_small = slippage_cost(100, 50.0, 1_000_000, 0.01, cfg)
        s_large = slippage_cost(10_000, 50.0, 1_000_000, 0.01, cfg)
        assert s_large > s_small

    def test_monotonic_in_volatility(self):
        """Higher daily vol → higher slippage."""
        cfg = _cfg(slippage_factor=0.1)
        s_low = slippage_cost(1000, 50.0, 1_000_000, 0.005, cfg)
        s_high = slippage_cost(1000, 50.0, 1_000_000, 0.02, cfg)
        assert s_high > s_low


class TestBorrowCost:
    def test_annual_rate_correct(self):
        """50 bps/yr on $100k notional = $500/yr / 252 days."""
        cfg = _cfg(borrow_cost_bps_yr=50.0)
        daily = borrow_cost_daily(100_000.0, cfg)
        assert abs(daily - 100_000 * 50 / 10_000 / 252) < 0.01

    def test_zero_notional(self):
        assert borrow_cost_daily(0.0, _cfg()) == 0.0


class TestTotalCost:
    def test_returns_tuple(self):
        result = total_cost(100, 50.0, 1_000_000, 0.01, _cfg())
        assert isinstance(result, tuple) and len(result) == 2

    def test_total_positive(self):
        total, slip = total_cost(100, 50.0, 1_000_000, 0.01, _cfg())
        assert total > 0
        assert slip >= 0
        assert total >= slip
