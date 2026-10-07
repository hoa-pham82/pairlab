"""PointInTimeDataHandler — serves bars strictly up to the current simulation time."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Generator

import pandas as pd

from pairlab.data.universe import Universe
from pairlab.events import DelistEvent, MarketEvent

_ONE_SECOND = timedelta(seconds=1)


class PointInTimeDataHandler:
    """Yields MarketEvents in chronological order; never exposes future bars.

    On each call to `next_events`, returns all events at the next timestamp.
    Emits DelistEvent for symbols removed from the universe.
    """

    def __init__(self, bars: pd.DataFrame, universe: Universe) -> None:
        """
        Args:
            bars: normalised daily-bars DataFrame (columns: symbol, ts, open, high, low, close, volume).
            universe: point-in-time symbol master.
        """
        self._bars = bars.copy()
        self._universe = universe
        # store as tz-aware Python datetimes for arithmetic safety
        self._sorted_ts: list[datetime] = [
            ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
            for ts in sorted(bars["ts"].unique())
        ]
        self._idx = 0
        self._emitted_delistings: set[str] = set()

    @property
    def current_ts(self) -> datetime | None:
        if self._idx >= len(self._sorted_ts):
            return None
        return self._sorted_ts[self._idx]

    def has_more(self) -> bool:
        return self._idx < len(self._sorted_ts)

    def next_events(self) -> list[MarketEvent | DelistEvent]:
        """Return all events at the next timestamp. Advances the internal clock."""
        if not self.has_more():
            return []

        ts = self._sorted_ts[self._idx]
        self._idx += 1

        # Symbols active as-of this ts
        active = self._universe.as_of(ts)

        # Check for delistings: symbols that *were* active at ts-1 but aren't now
        events: list[MarketEvent | DelistEvent] = []
        for sym, delist_ts in [
            (s, self._universe.delist_ts(s))
            for s in self._universe.as_of(ts - _ONE_SECOND) if s not in active
        ]:
            if sym not in self._emitted_delistings and delist_ts is not None and delist_ts <= ts:
                last_price = self._last_price(sym, ts)
                events.append(DelistEvent(ts=ts, symbol=sym, last_price=last_price))
                self._emitted_delistings.add(sym)

        # Market events for bars at this ts, for active symbols only
        snapshot = self._bars[(self._bars["ts"] == ts) & (self._bars["symbol"].isin(active))]
        for _, row in snapshot.iterrows():
            events.append(MarketEvent(
                ts=row["ts"].to_pydatetime() if hasattr(row["ts"], "to_pydatetime") else row["ts"],
                symbol=row["symbol"],
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            ))

        return events

    def get_history(self, symbol: str, end_ts: datetime, n_bars: int) -> pd.DataFrame:
        """Return up to n_bars of history for symbol with ts <= end_ts (no lookahead)."""
        df = self._bars[
            (self._bars["symbol"] == symbol) & (self._bars["ts"] <= end_ts)
        ].tail(n_bars)
        return df.reset_index(drop=True)

    def _last_price(self, symbol: str, as_of: datetime) -> float:
        hist = self._bars[(self._bars["symbol"] == symbol) & (self._bars["ts"] <= as_of)]
        if hist.empty:
            return 0.0
        return float(hist.iloc[-1]["close"])

    def iter_events(self) -> Generator[MarketEvent | DelistEvent, None, None]:
        """Yield all events in order. Convenience wrapper for tests."""
        while self.has_more():
            yield from self.next_events()
