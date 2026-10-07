"""Abstract base class for strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pairlab.events import MarketEvent, SignalEvent


class BaseStrategy(ABC):
    """All strategies implement this interface."""

    @abstractmethod
    def on_market(self, event: MarketEvent) -> list[SignalEvent]:
        """Process a market event and return zero or more signals."""
        ...
