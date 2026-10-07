"""Tests for Portfolio — cash tracking, position accounting, leverage, delistings."""

from __future__ import annotations

import pytest

from pairlab.config import CostConfig, PortfolioConfig
from pairlab.events import DelistEvent, FillEvent, MarketEvent, SignalEvent
from pairlab.portfolio import Portfolio
from tests.fixtures.bars import utc

T1 = utc(2024, 1, 2)
T2 = utc(2024, 1, 3)


def _port(**kwargs) -> Portfolio:
    pcfg = PortfolioConfig(
        initial_cash=kwargs.pop("initial_cash", 1_000_000.0),
        gross_notional_per_pair=100_000.0,
        max_gross_leverage=kwargs.pop("max_gross_leverage", 4.0),
        delisting_return=kwargs.pop("delisting_return", -0.30),
    )
    ccfg = CostConfig()
    return Portfolio(pcfg, ccfg)


def _market(symbol="KO", close=60.0, ts=T1):
    return MarketEvent(ts=ts, symbol=symbol, open=close, high=close * 1.01,
                       low=close * 0.99, close=close, volume=1e6)


def _signal(symbol="KO", direction=1, target=50_000.0, ts=T1):
    return SignalEvent(ts=ts, symbol=symbol, direction=direction, target_dollar=target)


def _fill(symbol="KO", qty=100.0, price=60.0, commission=10.0, slippage=1.0, ts=T1):
    return FillEvent(ts=ts, symbol=symbol, quantity=qty, fill_price=price,
                     commission=commission, slippage=slippage)


class TestEquityInvariant:
    def test_equity_equals_cash_plus_positions(self):
        """Equity must equal cash + sum(position market values) at every bar."""
        port = _port()
        port.on_market(_market("KO", 60.0))
        port.on_fill(_fill("KO", 100, 60.0, 5.0, 0.5))  # buy 100 @ 60
        port.on_market(_market("KO", 62.0))
        port.record_equity(T1)
        expected = port.cash + 100 * 62.0
        assert abs(port.equity - expected) < 0.01

    def test_initial_equity_equals_cash(self):
        port = _port()
        assert abs(port.equity - 1_000_000.0) < 0.01

    def test_equity_flat_after_close(self):
        """Open + close a position; equity should be initial minus costs."""
        port = _port()
        port.on_market(_market("KO", 60.0))
        port.on_fill(_fill("KO", 100, 60.0, 10.0, 1.0))    # open
        port.on_market(_market("KO", 60.0))                  # same price
        port.on_fill(_fill("KO", -100, 60.0, 10.0, 1.0))   # close
        assert "KO" not in port.get_positions()
        # Costs: 2 × ($10 + $1) = $22
        assert abs(port.equity - (1_000_000.0 - 22.0)) < 0.01


class TestOrderGeneration:
    def test_signal_generates_order(self):
        port = _port()
        port.on_market(_market("KO", 60.0))
        orders = port.on_signal(_signal("KO", direction=1, target=60_000.0))
        assert len(orders) == 1
        assert orders[0].quantity > 0

    def test_close_signal_no_position(self):
        """Close signal with no open position → no order."""
        port = _port()
        orders = port.on_signal(_signal("KO", direction=0))
        assert orders == []

    def test_close_signal_opens_sell_order(self):
        port = _port()
        port.on_market(_market("KO", 60.0))
        port.on_fill(_fill("KO", 100, 60.0, 5.0, 0.5))
        orders = port.on_signal(_signal("KO", direction=0))
        assert len(orders) == 1
        assert orders[0].quantity < 0

    def test_leverage_limit_blocks_order(self):
        """Reject signal when gross notional would exceed leverage cap."""
        port = _port(initial_cash=100_000.0, max_gross_leverage=1.0)
        port.on_market(_market("KO", 60.0))
        # Try to buy $200k notional on $100k cash with 1× leverage cap
        orders = port.on_signal(_signal("KO", direction=1, target=200_000.0))
        assert orders == []

    def test_unknown_price_no_order(self):
        """No last price known → no order."""
        port = _port()
        orders = port.on_signal(_signal("KO", direction=1))
        assert orders == []


class TestDelistingHandling:
    def test_delist_removes_position(self):
        port = _port()
        port.on_market(_market("KO", 60.0))
        port.on_fill(_fill("KO", 100, 60.0, 5.0, 0.5))
        assert "KO" in port.get_positions()
        port.on_delist(DelistEvent(ts=T2, symbol="KO", last_price=60.0))
        assert "KO" not in port.get_positions()

    def test_delist_no_position_is_noop(self):
        port = _port()
        port.on_delist(DelistEvent(ts=T2, symbol="KO", last_price=60.0))
        assert port.equity == 1_000_000.0


class TestBorrowCostAccrual:
    def test_short_position_accrues_daily_cost(self):
        port = _port()
        port.on_market(_market("KO", 100.0))
        port.on_fill(_fill("KO", -100, 100.0, 5.0, 0.5))  # short 100 shares
        initial_cash = port.cash
        port.on_market(_market("KO", 100.0))               # next day
        # Borrow cost = 100 shares × $100 × 50bps/yr / 252 ≈ $0.20/day
        assert port.cash < initial_cash

    def test_long_position_no_borrow_cost(self):
        port = _port()
        port.on_market(_market("KO", 100.0))
        port.on_fill(_fill("KO", 100, 100.0, 5.0, 0.5))   # long 100 shares
        initial_cash = port.cash
        port.on_market(_market("KO", 100.0))
        assert port.cash == initial_cash
