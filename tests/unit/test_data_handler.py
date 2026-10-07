"""Tests for PointInTimeDataHandler — correct ordering and no-lookahead guarantee."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from pairlab.data.handler import PointInTimeDataHandler
from pairlab.data.universe import Universe
from pairlab.events import DelistEvent, MarketEvent
from tests.fixtures.bars import DATES_5, make_bars, utc


def _simple_handler(symbols=("KO", "PEP"), dates=DATES_5):
    bars = make_bars(list(symbols), dates)
    universe = Universe.from_bar_data(bars)
    return PointInTimeDataHandler(bars, universe)


class TestOrdering:
    def test_events_come_out_in_ts_order(self):
        h = _simple_handler()
        events = list(h.iter_events())
        tss = [e.ts for e in events]
        assert tss == sorted(tss)

    def test_all_bars_emitted(self):
        h = _simple_handler(symbols=["KO", "PEP"], dates=DATES_5)
        events = [e for e in h.iter_events() if isinstance(e, MarketEvent)]
        assert len(events) == 2 * len(DATES_5)

    def test_has_more_false_after_drain(self):
        h = _simple_handler()
        list(h.iter_events())
        assert not h.has_more()

    def test_next_events_empty_when_exhausted(self):
        h = _simple_handler(dates=[DATES_5[0]])
        h.next_events()
        assert h.next_events() == []


class TestNoLookahead:
    """Mutating a future bar must not change past decisions."""

    def test_get_history_excludes_future_bars(self):
        """get_history with end_ts = day 3 must not include day 4 or 5."""
        bars = make_bars(["KO"], DATES_5, price_fn=lambda s, i: 100.0 + i)
        universe = Universe.from_bar_data(bars)
        h = PointInTimeDataHandler(bars, universe)
        hist = h.get_history("KO", end_ts=DATES_5[2], n_bars=100)
        assert len(hist) == 3  # days 0, 1, 2
        assert hist["ts"].max() == DATES_5[2]

    def test_get_history_respects_n_bars(self):
        bars = make_bars(["KO"], DATES_5)
        universe = Universe.from_bar_data(bars)
        h = PointInTimeDataHandler(bars, universe)
        hist = h.get_history("KO", end_ts=DATES_5[-1], n_bars=2)
        assert len(hist) == 2


class TestUniverse:
    def test_delisted_symbol_excluded(self):
        """After delisting date, symbol must not appear in MarketEvents."""
        dates = DATES_5
        bars = make_bars(["KO", "GONE"], dates)
        # GONE delists after day 2
        master = pd.DataFrame([
            {"symbol": "KO", "listed_at": dates[0], "delisted_at": pd.NaT},
            {"symbol": "GONE", "listed_at": dates[0], "delisted_at": dates[2]},
        ])
        universe = Universe(master)
        h = PointInTimeDataHandler(bars, universe)
        all_events = list(h.iter_events())
        market_events = [e for e in all_events if isinstance(e, MarketEvent)]
        gone_market = [e for e in market_events if e.symbol == "GONE"]
        # GONE should only appear in bars at ts <= its delisting date (exclusive)
        for e in gone_market:
            assert e.ts < dates[2], f"GONE appeared at {e.ts} after delisting {dates[2]}"

    def test_delist_event_emitted(self):
        """A DelistEvent must be emitted at the delisting timestamp."""
        dates = DATES_5
        bars = make_bars(["KO", "GONE"], dates)
        master = pd.DataFrame([
            {"symbol": "KO", "listed_at": dates[0], "delisted_at": pd.NaT},
            {"symbol": "GONE", "listed_at": dates[0], "delisted_at": dates[2]},
        ])
        universe = Universe(master)
        h = PointInTimeDataHandler(bars, universe)
        delist_events = [e for e in h.iter_events() if isinstance(e, DelistEvent)]
        assert any(e.symbol == "GONE" for e in delist_events)

    def test_delist_event_not_emitted_twice(self):
        dates = DATES_5
        bars = make_bars(["GONE"], dates)
        master = pd.DataFrame([
            {"symbol": "GONE", "listed_at": dates[0], "delisted_at": dates[2]},
        ])
        universe = Universe(master)
        h = PointInTimeDataHandler(bars, universe)
        delist_events = [e for e in h.iter_events() if isinstance(e, DelistEvent) and e.symbol == "GONE"]
        assert len(delist_events) == 1


class TestCurrentTs:
    def test_current_ts_before_drain(self):
        h = _simple_handler()
        assert h.current_ts == DATES_5[0]

    def test_current_ts_none_after_drain(self):
        h = _simple_handler(dates=[DATES_5[0]])
        list(h.iter_events())
        assert h.current_ts is None

    def test_current_ts_advances(self):
        h = _simple_handler(dates=DATES_5[:3])
        h.next_events()
        assert h.current_ts == DATES_5[1]


class TestEdgeCases:
    def test_empty_bars_no_events(self):
        bars = pd.DataFrame(columns=["symbol", "ts", "open", "high", "low", "close", "volume"])
        bars["ts"] = pd.to_datetime(bars["ts"], utc=True)
        universe = Universe.from_bar_data(bars)
        h = PointInTimeDataHandler(bars, universe)
        assert list(h.iter_events()) == []

    def test_single_bar(self):
        bars = make_bars(["KO"], [DATES_5[0]])
        universe = Universe.from_bar_data(bars)
        h = PointInTimeDataHandler(bars, universe)
        events = list(h.iter_events())
        assert len(events) == 1
        assert isinstance(events[0], MarketEvent)

    def test_zero_volume_bar_still_emitted(self):
        """Volume == 0 is a trading halt; handler emits it, execution layer rejects it."""
        bars = make_bars(["KO"], [DATES_5[0]])
        bars.loc[0, "volume"] = 0.0
        universe = Universe.from_bar_data(bars)
        h = PointInTimeDataHandler(bars, universe)
        events = list(h.iter_events())
        assert len(events) == 1
        assert events[0].volume == 0.0

    def test_delist_event_last_price_zero_when_no_history(self):
        """Delisted symbol with no bars → last_price == 0.0 (not a crash)."""
        dates = DATES_5
        bars = make_bars(["KO"], dates)   # only KO has bars
        master = pd.DataFrame([
            {"symbol": "KO",   "listed_at": dates[0], "delisted_at": pd.NaT},
            {"symbol": "GHOST", "listed_at": dates[0], "delisted_at": dates[2]},
        ])
        universe = Universe(master)
        h = PointInTimeDataHandler(bars, universe)
        delist_events = [e for e in h.iter_events() if isinstance(e, DelistEvent) and e.symbol == "GHOST"]
        assert len(delist_events) == 1
        assert delist_events[0].last_price == 0.0
