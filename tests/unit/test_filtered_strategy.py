"""Tests for the meta-label filter that wraps a strategy."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest
from ml.dataset import PAIR_FEATURES
from ml.filtered_strategy import MetaLabelFilter, frame_feature_lookup

from pairlab.events import MarketEvent, SignalEvent
from pairlab.strategy.base import BaseStrategy

DAY = datetime(2024, 3, 1, tzinfo=UTC)
FEATURES = dict.fromkeys(PAIR_FEATURES, 1.0)


def _event(ts: datetime = DAY) -> MarketEvent:
    return MarketEvent(ts=ts, symbol="KO", open=1.0, high=1.0, low=1.0, close=1.0, volume=1.0)


def _entry(pair: str = "KO/PEP", ts: datetime = DAY) -> list[SignalEvent]:
    a, b = pair.split("/")
    return [
        SignalEvent(ts=ts, symbol=a, direction=1, target_dollar=50_000.0, pair_id=pair),
        SignalEvent(ts=ts, symbol=b, direction=-1, target_dollar=50_000.0, pair_id=pair),
    ]


def _exit(pair: str = "KO/PEP", ts: datetime = DAY) -> list[SignalEvent]:
    a, b = pair.split("/")
    return [
        SignalEvent(ts=ts, symbol=a, direction=0, target_dollar=0.0, pair_id=pair),
        SignalEvent(ts=ts, symbol=b, direction=0, target_dollar=0.0, pair_id=pair),
    ]


class Scripted(BaseStrategy):
    """Inner strategy that replays a fixed list of signal batches."""

    def __init__(self, *batches: list[SignalEvent]):
        self._batches = list(batches)

    def on_market(self, event):
        return self._batches.pop(0) if self._batches else []


class FixedModel:
    def __init__(self, prob: float):
        self.prob = prob
        self.seen: list[dict] = []

    def predict_proba(self, features):
        self.seen.append(features)
        return self.prob


def _filter(inner, prob, lookup=lambda pair, ts: FEATURES, **kwargs) -> MetaLabelFilter:
    return MetaLabelFilter(inner, FixedModel(prob), lookup, **kwargs)


# Boundary values for the decision: probability just below, at, and just above
# the threshold. An entry is taken when prob >= threshold.
@pytest.mark.parametrize(("prob", "taken"), [(0.49, False), (0.5, True), (0.51, True)])
def test_threshold_boundary(prob, taken):
    strategy = _filter(Scripted(_entry()), prob, threshold=0.5)
    assert (len(strategy.on_market(_event())) == 2) is taken
    assert (strategy.stats.taken, strategy.stats.skipped) == ((1, 0) if taken else (0, 1))


def test_both_legs_are_dropped_and_the_model_is_asked_once_per_pair():
    model = FixedModel(0.1)
    strategy = MetaLabelFilter(Scripted(_entry()), model, lambda pair, ts: FEATURES)
    assert strategy.on_market(_event()) == []
    assert len(model.seen) == 1


def test_exit_of_a_skipped_entry_is_dropped_then_the_pair_can_trade_again():
    inner = Scripted(_entry(), _exit(), _entry(), _exit())
    model = FixedModel(0.1)
    strategy = MetaLabelFilter(inner, model, lambda pair, ts: FEATURES)
    assert strategy.on_market(_event()) == []
    assert strategy.on_market(_event()) == []
    model.prob = 0.9
    assert len(strategy.on_market(_event())) == 2
    assert len(strategy.on_market(_event())) == 2


def test_exit_of_a_taken_entry_passes_through():
    strategy = _filter(Scripted(_entry(), _exit()), 0.9)
    strategy.on_market(_event())
    assert [s.direction for s in strategy.on_market(_event())] == [0, 0]


def test_pairs_are_decided_independently():
    lookup = {"KO/PEP": FEATURES, "XOM/CVX": {**FEATURES, "zscore": -9.0}}

    class ByZscore:
        def predict_proba(self, features):
            return 0.9 if features["zscore"] > 0 else 0.1

    strategy = MetaLabelFilter(
        Scripted(_entry("KO/PEP") + _entry("XOM/CVX")), ByZscore(), lambda p, ts: lookup[p]
    )
    assert {s.pair_id for s in strategy.on_market(_event())} == {"KO/PEP"}


def test_entry_without_features_passes_and_is_counted_unscored():
    strategy = _filter(Scripted(_entry()), 0.0, lookup=lambda pair, ts: None)
    assert len(strategy.on_market(_event())) == 2
    assert strategy.stats.unscored == 1


# Boundary: the filter acts from active_from on, not before.
@pytest.mark.parametrize(("days_before", "filtered"), [(1, False), (0, True), (-1, True)])
def test_active_from_boundary(days_before, filtered):
    ts = DAY - timedelta(days=days_before)
    strategy = _filter(Scripted(_entry(ts=ts)), 0.0, active_from=DAY)
    assert (strategy.on_market(_event(ts)) == []) is filtered


def test_no_signals_gives_no_signals():
    assert _filter(Scripted(), 0.9).on_market(_event()) == []


class TestFrameFeatureLookup:
    def _frame(self) -> pd.DataFrame:
        days = pd.to_datetime(["2024-02-29", "2024-03-01", "2024-03-04"])
        return pd.DataFrame(
            {
                "symbol_a": "KO",
                "symbol_b": "PEP",
                "event_timestamp": days,
                **{name: [1.0, 2.0, 3.0] for name in PAIR_FEATURES},
            }
        )

    def test_returns_the_row_of_the_same_day(self):
        lookup = frame_feature_lookup(self._frame())
        assert lookup("KO/PEP", DAY) == dict.fromkeys(PAIR_FEATURES, 2.0)

    def test_never_returns_a_later_row(self):
        """No lookahead: a day with no row gets nothing, not the next day's features."""
        lookup = frame_feature_lookup(self._frame())
        assert lookup("KO/PEP", datetime(2024, 3, 2, tzinfo=UTC)) is None

    def test_changing_future_rows_does_not_change_todays_answer(self):
        frame = self._frame()
        before = frame_feature_lookup(frame)("KO/PEP", DAY)
        frame.loc[frame["event_timestamp"] > "2024-03-01", PAIR_FEATURES] = 99.0
        assert frame_feature_lookup(frame)("KO/PEP", DAY) == before

    def test_unknown_pair_and_nan_features_give_none(self):
        frame = self._frame()
        frame.loc[1, "zscore"] = float("nan")
        lookup = frame_feature_lookup(frame)
        assert lookup("XOM/CVX", DAY) is None
        assert lookup("KO/PEP", DAY) is None
