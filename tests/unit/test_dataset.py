"""Tests for training-frame assembly and the purged time split."""

from __future__ import annotations

import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from ml.dataset import (
    PAIR_FEATURES,
    entity_frame,
    features_and_target,
    load_training_frame,
    purged_time_split,
)
from pandas.tseries.offsets import BDay


def _frame(dates) -> pd.DataFrame:
    dates = pd.to_datetime(list(dates))
    return pd.DataFrame(
        {
            "pair_date_id": [f"KO__PEP|{d.date()}" for d in dates],
            "label": [i % 2 for i in range(len(dates))],
            "symbol_pair": "KO__PEP",
            "event_timestamp": dates,
            **{name: 1.0 for name in PAIR_FEATURES},
        }
    )


class FakeStore:
    """Stands in for feast.FeatureStore: returns the entity frame plus features."""

    def __init__(self):
        self.requested: list[str] = []

    def get_historical_features(self, entity_df, features):
        self.requested = features
        result = entity_df.assign(**{name: 0.5 for name in PAIR_FEATURES})
        return type("Job", (), {"to_df": lambda _self: result.iloc[::-1]})()


class TestEntityFrame:
    def test_splits_id_into_pair_and_timestamp(self):
        labels = pd.DataFrame({"pair_date_id": ["KO__PEP|2020-03-02"], "label": [1]})
        out = entity_frame(labels)
        assert out["symbol_pair"].tolist() == ["KO__PEP"]
        assert out["event_timestamp"].tolist() == [pd.Timestamp("2020-03-02")]
        assert out["label"].tolist() == [1]


class TestLoadTrainingFrame:
    def test_requests_pair_features_and_sorts_by_time(self):
        store = FakeStore()
        labels = pd.DataFrame(
            {"pair_date_id": ["KO__PEP|2020-03-02", "KO__PEP|2020-03-03"], "label": [1, 0]}
        )
        frame = load_training_frame(store, labels)
        assert store.requested == [f"pair_daily_fv:{name}" for name in PAIR_FEATURES]
        assert frame["event_timestamp"].is_monotonic_increasing
        assert set(PAIR_FEATURES) <= set(frame.columns)


# ---------------------------------------------------------------------------
# Equivalence partitions for purged_time_split:
#   EP1: train row whose label window ends before validation starts → kept
#   EP2: train row whose label window reaches validation            → purged
#   EP3: row on or after the validation start                       → validation
# Boundary: label window ending one business day before / exactly on the
# validation start.
# ---------------------------------------------------------------------------


class TestPurgedTimeSplit:
    HORIZON = 5
    VALID_START = pd.Timestamp("2024-03-01")  # a Friday

    @pytest.mark.parametrize(
        ("bars_before_valid", "kept"),
        [(HORIZON + 1, True), (HORIZON, False), (1, False)],
    )
    def test_purge_boundary(self, bars_before_valid, kept):
        entry = self.VALID_START - BDay(bars_before_valid)
        frame = _frame([entry, self.VALID_START])
        train, valid = purged_time_split(frame, valid_fraction=0.5, horizon_bars=self.HORIZON)
        assert (len(train) == 1) is kept
        assert valid["event_timestamp"].tolist() == [self.VALID_START]

    def test_validation_is_the_latest_fraction(self):
        frame = _frame(pd.bdate_range("2024-01-01", periods=100))
        train, valid = purged_time_split(frame, valid_fraction=0.25, horizon_bars=5)
        assert len(valid) == 25
        assert len(train) == 75 - 5

    def test_rows_sharing_the_split_date_all_go_to_validation(self):
        frame = _frame(["2024-01-01", "2024-03-01", "2024-03-01", "2024-03-04"])
        _, valid = purged_time_split(frame, valid_fraction=0.5, horizon_bars=5)
        assert len(valid) == 3

    def test_empty_frame(self):
        train, valid = purged_time_split(_frame([]))
        assert train.empty and valid.empty

    @pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
    def test_invalid_fraction_rejected(self, fraction):
        with pytest.raises(ValueError):
            purged_time_split(_frame(["2024-01-01"]), valid_fraction=fraction)

    @settings(
        max_examples=100, deadline=None, suppress_health_check=[HealthCheck.differing_executors]
    )
    @given(
        offsets=st.lists(st.integers(0, 400), min_size=2, max_size=60),
        horizon=st.integers(1, 30),
    )
    def test_no_training_label_window_reaches_validation(self, offsets, horizon):
        dates = [pd.Timestamp("2022-01-03") + BDay(o) for o in offsets]
        train, valid = purged_time_split(_frame(dates), horizon_bars=horizon)
        if not train.empty:
            latest_label_end = (train["event_timestamp"] + BDay(horizon)).max()
            assert latest_label_end < valid["event_timestamp"].min()
        assert len(train) + len(valid) <= len(dates)


def test_features_and_target():
    x, y = features_and_target(_frame(["2024-01-01", "2024-01-02"]))
    assert x.columns.tolist() == PAIR_FEATURES
    assert y.tolist() == [0, 1]
