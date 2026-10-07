"""Meta-label filter: wrap a strategy and skip entries the model scores too low."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from ml.dataset import PAIR_FEATURES
from pairlab.events import MarketEvent, SignalEvent
from pairlab.strategy.base import BaseStrategy

FeatureLookup = Callable[[str, datetime], dict[str, float] | None]


@dataclass
class FilterStats:
    """How many pair entries the filter took, skipped, or could not score."""

    taken: int = 0
    skipped: int = 0
    unscored: int = 0


class MetaLabelFilter(BaseStrategy):
    """Passes a strategy's signals through, dropping low-probability pair entries.

    An entry is skipped when the model's probability is below ``threshold``.
    The matching exit signals of a skipped entry are dropped too. Entries with
    no features, or before ``active_from``, pass through unfiltered.
    """

    def __init__(
        self,
        inner: BaseStrategy,
        model,
        features: FeatureLookup,
        threshold: float = 0.5,
        active_from: datetime | None = None,
    ) -> None:
        """
        Args:
            inner: the strategy that proposes trades.
            model: anything with ``predict_proba(features: dict) -> float``.
            features: returns the features known for a pair at a timestamp.
            active_from: the filter only acts on entries at or after this time.
        """
        self._inner = inner
        self._model = model
        self._features = features
        self._threshold = threshold
        self._active_from = active_from
        self._skipped_pairs: set[str] = set()
        self.stats = FilterStats()

    def on_market(self, event: MarketEvent) -> list[SignalEvent]:
        """Return the inner strategy's signals minus filtered entries and their exits."""
        signals = self._inner.on_market(event)
        entry_decisions: dict[str, bool] = {}
        released: set[str] = set()
        kept: list[SignalEvent] = []
        for signal in signals:
            if signal.direction == 0:
                if signal.pair_id in self._skipped_pairs:
                    released.add(signal.pair_id)
                else:
                    kept.append(signal)
                continue
            if signal.pair_id not in entry_decisions:
                entry_decisions[signal.pair_id] = self._take_entry(signal)
            if entry_decisions[signal.pair_id]:
                kept.append(signal)
        self._skipped_pairs -= released
        return kept

    def _take_entry(self, signal: SignalEvent) -> bool:
        if self._active_from is not None and signal.ts < self._active_from:
            return True
        values = self._features(signal.pair_id, signal.ts)
        if values is None:
            self.stats.unscored += 1
            return True
        if self._model.predict_proba(values) >= self._threshold:
            self.stats.taken += 1
            return True
        self.stats.skipped += 1
        self._skipped_pairs.add(signal.pair_id)
        return False


def frame_feature_lookup(features: pd.DataFrame) -> FeatureLookup:
    """Build a lookup over pair-feature rows, keyed by pair and calendar day.

    Only the row dated the same day as the signal is returned, never a later
    one. Pair IDs use the strategy's ``A/B`` form.

    Args:
        features: rows with ``symbol_a``, ``symbol_b``, ``event_timestamp`` and
            the model's feature columns.
    """
    keys = (
        features["symbol_a"]
        + "/"
        + features["symbol_b"]
        + "|"
        + pd.to_datetime(features["event_timestamp"]).dt.strftime("%Y-%m-%d")
    )
    rows = dict(zip(keys, features[PAIR_FEATURES].to_dict("records"), strict=True))

    def lookup(pair_id: str, ts: datetime) -> dict[str, float] | None:
        row = rows.get(f"{pair_id}|{ts:%Y-%m-%d}")
        if row is None or any(pd.isna(value) for value in row.values()):
            return None
        return row

    return lookup
