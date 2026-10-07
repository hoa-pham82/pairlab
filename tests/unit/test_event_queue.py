"""Tests for the deterministic EventQueue heap."""

from datetime import datetime, timezone

import pytest

from pairlab.event_queue import EventQueue
from pairlab.events import FillEvent, MarketEvent, SignalEvent

T1 = datetime(2024, 1, 2, tzinfo=timezone.utc)
T2 = datetime(2024, 1, 3, tzinfo=timezone.utc)


def _market(ts, symbol="KO"):
    return MarketEvent(ts=ts, symbol=symbol, open=60.0, high=61.0, low=59.5, close=60.5, volume=1e6)


def _signal(ts, symbol="KO"):
    return SignalEvent(ts=ts, symbol=symbol, direction=1, target_dollar=50_000.0)


def _fill(ts, symbol="KO"):
    return FillEvent(ts=ts, symbol=symbol, quantity=830.0, fill_price=60.5, commission=5.0, slippage=0.3)


class TestEventQueueOrdering:
    def test_earlier_ts_first(self):
        q = EventQueue()
        q.push(_market(T2))
        q.push(_market(T1))
        assert q.pop().ts == T1

    def test_same_ts_priority_order(self):
        """MarketEvent must come before SignalEvent at the same timestamp."""
        q = EventQueue()
        q.push(_signal(T1))
        q.push(_market(T1))
        assert isinstance(q.pop(), MarketEvent)
        assert isinstance(q.pop(), SignalEvent)

    def test_same_ts_same_priority_fifo(self):
        """Two MarketEvents at same ts must come out in insertion order."""
        q = EventQueue()
        a = _market(T1, symbol="KO")
        b = _market(T1, symbol="PEP")
        q.push(a)
        q.push(b)
        assert q.pop() is a
        assert q.pop() is b

    def test_len_and_bool(self):
        q = EventQueue()
        assert not q
        assert len(q) == 0
        q.push(_market(T1))
        assert q
        assert len(q) == 1
        q.pop()
        assert not q

    def test_pop_empty_raises(self):
        q = EventQueue()
        with pytest.raises(IndexError):
            q.pop()

    def test_peek_ts_empty(self):
        assert EventQueue().peek_ts() is None

    def test_peek_ts_does_not_remove(self):
        q = EventQueue()
        q.push(_market(T1))
        assert q.peek_ts() == T1
        assert len(q) == 1

    def test_iter_drains_in_order(self):
        q = EventQueue()
        q.push(_market(T2))
        q.push(_signal(T1))
        q.push(_market(T1))
        events = list(q)
        assert events[0].ts == T1
        assert isinstance(events[0], MarketEvent)
        assert events[1].ts == T1
        assert isinstance(events[1], SignalEvent)
        assert events[2].ts == T2
        assert not q

    def test_many_events_sorted(self):
        import random
        rng = random.Random(0)
        q = EventQueue()
        tss = [datetime(2024, 1, i + 1, tzinfo=timezone.utc) for i in range(30)]
        shuffled = tss * 2
        rng.shuffle(shuffled)
        for ts in shuffled:
            q.push(_market(ts))
        out = list(q)
        assert out == sorted(out, key=lambda e: (e.ts, e.priority))
