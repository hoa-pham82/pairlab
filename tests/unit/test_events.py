"""Tests for event dataclasses and priority ordering."""

from datetime import datetime, timezone

import pytest

from pairlab.events import (
    DelistEvent,
    FillEvent,
    MarketEvent,
    OrderEvent,
    Priority,
    SignalEvent,
)

TS = datetime(2024, 1, 2, tzinfo=timezone.utc)


def test_market_event_frozen():
    e = MarketEvent(ts=TS, symbol="KO", open=60.0, high=61.0, low=59.5, close=60.5, volume=1e6)
    with pytest.raises((AttributeError, TypeError)):
        e.close = 99.0  # type: ignore[misc]


def test_priority_ordering():
    assert Priority.MARKET < Priority.SIGNAL < Priority.ORDER < Priority.FILL < Priority.DELIST


@pytest.mark.parametrize("event_cls,kwargs", [
    (MarketEvent, dict(ts=TS, symbol="KO", open=60.0, high=61.0, low=59.5, close=60.5, volume=1e6)),
    (SignalEvent, dict(ts=TS, symbol="KO", direction=1, target_dollar=50_000.0)),
    (OrderEvent, dict(ts=TS, symbol="KO", quantity=830.0)),
    (FillEvent, dict(ts=TS, symbol="KO", quantity=830.0, fill_price=60.5, commission=5.0, slippage=0.3)),
    (DelistEvent, dict(ts=TS, symbol="KO", last_price=0.01)),
])
def test_event_priority_field_matches_class(event_cls, kwargs):
    """Priority field must equal the class-level Priority constant."""
    event = event_cls(**kwargs)
    expected = {
        MarketEvent: Priority.MARKET,
        SignalEvent: Priority.SIGNAL,
        OrderEvent: Priority.ORDER,
        FillEvent: Priority.FILL,
        DelistEvent: Priority.DELIST,
    }[event_cls]
    assert event.priority == expected


def test_signal_event_close_direction():
    e = SignalEvent(ts=TS, symbol="PEP", direction=0, target_dollar=0.0, pair_id="KO/PEP")
    assert e.direction == 0
    assert e.pair_id == "KO/PEP"
