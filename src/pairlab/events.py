"""Event types passed between backtester components."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum


class Priority(IntEnum):
    """Lower value = processed first when timestamps are equal."""
    MARKET = 10
    SIGNAL = 20
    ORDER = 30
    FILL = 40
    DELIST = 50


@dataclass(frozen=True, order=False)
class MarketEvent:
    """One OHLCV bar for a symbol, ready for consumption by strategies."""
    ts: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    priority: int = field(default=Priority.MARKET, init=False, compare=False)


@dataclass(frozen=True, order=False)
class SignalEvent:
    """A trading signal produced by a strategy."""
    ts: datetime
    symbol: str
    direction: int        # +1 long, -1 short, 0 close
    target_dollar: float  # desired notional in USD
    pair_id: str = ""     # e.g. "KO/PEP"
    priority: int = field(default=Priority.SIGNAL, init=False, compare=False)


@dataclass(frozen=True, order=False)
class OrderEvent:
    """An order produced by the portfolio, sent to execution."""
    ts: datetime
    symbol: str
    quantity: float       # shares; positive = buy, negative = sell
    order_type: str = "MARKET"
    pair_id: str = ""
    priority: int = field(default=Priority.ORDER, init=False, compare=False)


@dataclass(frozen=True, order=False)
class FillEvent:
    """Confirmation that an order was (simulated) filled."""
    ts: datetime
    symbol: str
    quantity: float       # actual shares filled
    fill_price: float
    commission: float
    slippage: float
    pair_id: str = ""
    priority: int = field(default=Priority.FILL, init=False, compare=False)


@dataclass(frozen=True, order=False)
class DelistEvent:
    """A symbol is removed from the universe (delisting, merger, etc.)."""
    ts: datetime
    symbol: str
    last_price: float
    reason: str = "delisted"
    priority: int = field(default=Priority.DELIST, init=False, compare=False)


# Union type for type hints
Event = MarketEvent | SignalEvent | OrderEvent | FillEvent | DelistEvent
