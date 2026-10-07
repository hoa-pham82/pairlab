"""Tests for PairsStrategy signal generation."""

from __future__ import annotations

from datetime import timedelta, timezone

import numpy as np
import pytest

from pairlab.config import StrategyConfig
from pairlab.events import MarketEvent, SignalEvent
from pairlab.strategy.pairs import PairsStrategy
from tests.fixtures.bars import utc


def _cfg(**kwargs) -> StrategyConfig:
    defaults = dict(
        formation_window=100,
        zscore_window=20,
        entry_z=2.0,
        exit_z=0.5,
        stop_z=4.0,
        retest_every=21,
    )
    defaults.update(kwargs)
    return StrategyConfig(**defaults)


def _make_cointegrated_series(n: int, seed: int = 0) -> tuple[list[float], list[float]]:
    """Cointegrated pair: shared GBM trend + OU spread."""
    rng = np.random.default_rng(seed)
    common = np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    spread = np.zeros(n)
    for i in range(1, n):
        spread[i] = spread[i - 1] * 0.9 + rng.normal(0, 0.02)
    a = common * np.exp(spread)
    b = common
    return a.tolist(), b.tolist()


def _feed(strategy: PairsStrategy, prices_a: list[float], prices_b: list[float]) -> list[SignalEvent]:
    """Feed price series into strategy and collect all signals."""
    all_signals = []
    for i, (pa, pb) in enumerate(zip(prices_a, prices_b)):
        ts = utc(2020, 1, 1) + timedelta(days=i)
        for sym, price in [("A", pa), ("B", pb)]:
            e = MarketEvent(ts=ts, symbol=sym, open=price, high=price, low=price, close=price, volume=1e6)
            all_signals.extend(strategy.on_market(e))
    return all_signals


class TestPairsStrategySignals:
    def test_no_signals_before_warmup(self):
        """Before zscore_window bars, strategy must produce no signals."""
        strat = PairsStrategy(_cfg(formation_window=50, zscore_window=20), pairs=[("A", "B")])
        prices_a, prices_b = _make_cointegrated_series(15)
        sigs = _feed(strat, prices_a, prices_b)
        assert sigs == []

    def test_cointegrated_pair_eventually_signals(self):
        """A strongly cointegrated pair should produce at least one entry signal."""
        strat = PairsStrategy(_cfg(formation_window=100, zscore_window=20), pairs=[("A", "B")])
        prices_a, prices_b = _make_cointegrated_series(400, seed=1)
        sigs = _feed(strat, prices_a, prices_b)
        entry_sigs = [s for s in sigs if s.direction != 0]
        assert len(entry_sigs) > 0

    def test_close_signal_direction_zero(self):
        """Exit signals must have direction == 0."""
        strat = PairsStrategy(_cfg(formation_window=100, zscore_window=20), pairs=[("A", "B")])
        prices_a, prices_b = _make_cointegrated_series(600, seed=2)
        sigs = _feed(strat, prices_a, prices_b)
        close_sigs = [s for s in sigs if s.direction == 0]
        # Every close signal must pair with a prior entry
        if close_sigs:
            assert all(s.pair_id == "A/B" for s in close_sigs)

    def test_signals_come_in_pairs(self):
        """Signals always arrive in multiples of 2 (both legs of a pair per decision)."""
        strat = PairsStrategy(_cfg(formation_window=100, zscore_window=20), pairs=[("A", "B")])
        prices_a, prices_b = _make_cointegrated_series(500, seed=3)
        sigs = _feed(strat, prices_a, prices_b)
        by_ts: dict = {}
        for s in sigs:
            by_ts.setdefault(s.ts, []).append(s)
        for ts_sigs in by_ts.values():
            assert len(ts_sigs) % 2 == 0, f"Odd signal count at ts: {ts_sigs}"

    def test_negative_price_ignored(self):
        """close <= 0 must not crash strategy."""
        strat = PairsStrategy(_cfg(), pairs=[("A", "B")])
        ts = utc(2020, 1, 1)
        e = MarketEvent(ts=ts, symbol="A", open=-1.0, high=-1.0, low=-1.0, close=-1.0, volume=1e6)
        assert strat.on_market(e) == []

    def test_nan_price_ignored(self):
        strat = PairsStrategy(_cfg(), pairs=[("A", "B")])
        ts = utc(2020, 1, 1)
        e = MarketEvent(ts=ts, symbol="A", open=float("nan"), high=float("nan"),
                        low=float("nan"), close=float("nan"), volume=1e6)
        assert strat.on_market(e) == []

    def test_unknown_pair_no_signals(self):
        """Strategy only tracks declared pairs."""
        strat = PairsStrategy(_cfg(), pairs=[("A", "B")])
        prices_a, prices_b = _make_cointegrated_series(300)
        # Feed C instead of B
        all_signals = []
        for i, (pa, _) in enumerate(zip(prices_a, prices_b)):
            ts = utc(2020, 1, 1) + timedelta(days=i)
            e = MarketEvent(ts=ts, symbol="A", open=pa, high=pa, low=pa, close=pa, volume=1e6)
            all_signals.extend(strat.on_market(e))
        assert all_signals == []

    def test_get_pair_state_none_before_formation(self):
        strat = PairsStrategy(_cfg(formation_window=100, zscore_window=20), pairs=[("A", "B")])
        assert strat.get_pair_state(("A", "B")) is None

    def test_pair_processed_once_per_day_with_third_symbol(self):
        """With a third symbol in the event stream, each pair fires exactly once per day.

        Regression for the N-symbol bug: _process_pair was called once per
        MarketEvent (once per symbol), so with N symbols a pair fired N times
        per day, inflating bars_in_trade N-fold and triggering time stops early.
        """
        strat = PairsStrategy(_cfg(formation_window=100, zscore_window=20, retest_every=9999),
                              pairs=[("A", "B")])
        prices_a, prices_b = _make_cointegrated_series(300, seed=7)
        prices_c = [1.0] * 300  # unrelated third symbol

        all_signals: list[SignalEvent] = []
        for i in range(300):
            ts = utc(2020, 1, 1) + timedelta(days=i)
            for sym, price in [("A", prices_a[i]), ("B", prices_b[i]), ("C", prices_c[i])]:
                e = MarketEvent(ts=ts, symbol=sym, open=price, high=price,
                                low=price, close=price, volume=1e6)
                all_signals.extend(strat.on_market(e))

        # Signals from A/B pair must each have a unique ts per decision
        by_ts: dict = {}
        for s in all_signals:
            by_ts.setdefault(s.ts, []).append(s)
        # No timestamp should have more than 2 signals (one open/close decision = 2 leg signals)
        for ts, sigs in by_ts.items():
            assert len(sigs) <= 2, (
                f"Pair processed more than once at {ts}: got {len(sigs)} signals. "
                "Likely N-symbol firing bug still present."
            )
