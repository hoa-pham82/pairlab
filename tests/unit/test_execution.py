"""Tests for SimulatedExecutionHandler."""

from __future__ import annotations

import pytest

from pairlab.config import CostConfig
from pairlab.events import FillEvent, MarketEvent, OrderEvent
from pairlab.execution.simulated import SimulatedExecutionHandler
from tests.fixtures.bars import utc

T1 = utc(2024, 1, 2)
T2 = utc(2024, 1, 3)


def _cfg() -> CostConfig:
    return CostConfig(commission_bps=5.0, commission_min_usd=1.0,
                      half_spread_bps=2.0, slippage_factor=0.1, borrow_cost_bps_yr=50.0)


def _bar(ts, symbol="KO", volume=1_000_000.0, open_=60.0):
    return MarketEvent(ts=ts, symbol=symbol, open=open_, high=61.0, low=59.5, close=60.5, volume=volume)


def _order(ts, symbol="KO", qty=100.0):
    return OrderEvent(ts=ts, symbol=symbol, quantity=qty)


class TestFillAtNextOpen:
    def test_fills_at_next_bar_open(self):
        h = SimulatedExecutionHandler(_cfg())
        h.on_order(_order(T1))
        fills = h.on_market(_bar(T2, open_=65.0))
        assert len(fills) == 1
        assert fills[0].fill_price == 65.0

    def test_fill_ts_matches_market_ts(self):
        h = SimulatedExecutionHandler(_cfg())
        h.on_order(_order(T1))
        fills = h.on_market(_bar(T2))
        assert fills[0].ts == T2

    def test_zero_quantity_order_ignored(self):
        h = SimulatedExecutionHandler(_cfg())
        h.on_order(_order(T1, qty=0.0))
        fills = h.on_market(_bar(T2))
        assert fills == []

    def test_halt_rejects_order(self):
        """Volume == 0 = trading halt; order stays pending."""
        h = SimulatedExecutionHandler(_cfg())
        h.on_order(_order(T1))
        fills = h.on_market(_bar(T2, volume=0))
        assert fills == []

    def test_wrong_symbol_not_filled(self):
        h = SimulatedExecutionHandler(_cfg())
        h.on_order(_order(T1, symbol="KO"))
        fills = h.on_market(_bar(T2, symbol="PEP"))
        assert fills == []

    def test_commission_positive(self):
        h = SimulatedExecutionHandler(_cfg())
        h.on_order(_order(T1))
        fill = h.on_market(_bar(T2))[0]
        assert fill.commission > 0

    def test_short_order(self):
        h = SimulatedExecutionHandler(_cfg())
        h.on_order(_order(T1, qty=-100.0))
        fills = h.on_market(_bar(T2))
        assert len(fills) == 1
        assert fills[0].quantity == -100.0

    def test_cancel_pending(self):
        h = SimulatedExecutionHandler(_cfg())
        h.on_order(_order(T1, symbol="KO"))
        h.cancel_pending("KO")
        fills = h.on_market(_bar(T2))
        assert fills == []
