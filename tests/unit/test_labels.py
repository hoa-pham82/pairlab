"""Tests for meta-label construction from pair z-scores."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from ml.labels import LabelConfig, build_label_table, label_entries, pair_date_id

CFG = LabelConfig(entry_z=2.0, exit_z=0.5, stop_z=4.0, horizon_bars=3)


def _z(*values: float) -> pd.Series:
    return pd.Series(values, dtype=float)


# ---------------------------------------------------------------------------
# Equivalence partitions for label_entries (one entry at bar 1):
#   EP1: no entry            — |z| never above entry_z          → no labels
#   EP2: reverts in horizon  — |z| < exit_z before stop         → 1
#   EP3: stops out           — |z| > stop_z before reverting    → 0
#   EP4: times out           — neither within horizon_bars      → 0
#   EP5: censored            — series ends before it is resolved → dropped
# Boundary values: each threshold at just-below / exactly / just-above, and
# reversion on the last bar of the horizon vs one bar too late.
# ---------------------------------------------------------------------------


class TestLabelEntries:
    @pytest.mark.parametrize(
        ("entry_value", "is_entry"),
        [(1.99, False), (2.0, False), (2.01, True), (-2.01, True)],
    )
    def test_entry_threshold_boundary(self, entry_value, is_entry):
        labels = label_entries(_z(0.0, entry_value, 0.0, 0.0, 0.0), CFG)
        assert (len(labels) == 1) is is_entry

    @pytest.mark.parametrize(
        ("exit_value", "expected"),
        [(0.49, 1), (0.5, 0), (0.51, 0), (-0.49, 1)],
    )
    def test_exit_threshold_boundary(self, exit_value, expected):
        labels = label_entries(_z(0.0, 2.5, exit_value, 1.0, 1.0, 1.0), CFG)
        assert labels.tolist() == [expected]

    @pytest.mark.parametrize(
        ("stop_value", "expected"),
        [(3.99, 1), (4.0, 1), (4.01, 0), (-4.01, 0)],
    )
    def test_stop_threshold_boundary(self, stop_value, expected):
        """A stop on the next bar beats a reversion one bar later."""
        labels = label_entries(_z(0.0, 2.5, stop_value, 0.0, 0.0, 0.0), CFG)
        assert labels.tolist() == [expected]

    @pytest.mark.parametrize(
        ("bars_until_revert", "expected"),
        [(1, 1), (3, 1), (4, 0)],
    )
    def test_horizon_boundary(self, bars_until_revert, expected):
        path = [1.0] * 8
        path[bars_until_revert - 1] = 0.0
        labels = label_entries(_z(0.0, 2.5, *path), CFG)
        assert labels.tolist() == [expected]

    def test_no_entry_gives_no_labels(self):
        assert label_entries(_z(0.0, 1.0, -1.5, 0.3), CFG).empty

    def test_empty_series(self):
        assert label_entries(_z(), CFG).empty

    def test_unresolved_entry_near_end_is_dropped(self):
        """EP5: two bars left of a three-bar horizon and no outcome yet."""
        assert label_entries(_z(0.0, 2.5, 1.0, 1.0), CFG).empty

    def test_resolved_entry_near_end_is_kept(self):
        assert label_entries(_z(0.0, 2.5, 0.1), CFG).tolist() == [1]

    def test_consecutive_bars_in_zone_are_one_entry(self):
        labels = label_entries(_z(0.0, 2.5, 2.6, 2.7, 0.0), CFG)
        assert labels.index.tolist() == [1]

    def test_re_entry_after_leaving_zone_is_a_new_entry(self):
        labels = label_entries(_z(0.0, 2.5, 0.0, 2.5, 0.0), CFG)
        assert labels.index.tolist() == [1, 3]

    def test_nan_never_counts_as_entry_or_outcome(self):
        labels = label_entries(_z(np.nan, 2.5, np.nan, 0.1, 0.0), CFG)
        assert labels.tolist() == [1]

    def test_labels_keep_the_series_index(self):
        idx = pd.date_range("2024-01-01", periods=4, freq="D")
        labels = label_entries(pd.Series([0.0, 2.5, 0.1, 0.0], index=idx), CFG)
        assert labels.index.tolist() == [idx[1]]

    @settings(
        max_examples=100, deadline=None, suppress_health_check=[HealthCheck.differing_executors]
    )
    @given(
        z=st.lists(st.floats(-6, 6, allow_nan=False), min_size=1, max_size=40),
        tail=st.lists(st.floats(-6, 6, allow_nan=False), min_size=1, max_size=10),
    )
    def test_label_depends_only_on_its_own_horizon(self, z, tail):
        """Appending later bars never changes a label already resolved."""
        before = label_entries(_z(*z), CFG)
        after = label_entries(_z(*z, *tail), CFG)
        assert after.loc[before.index].tolist() == before.tolist()


class TestLabelConfig:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"exit_z": 2.0, "entry_z": 2.0},
            {"entry_z": 4.0, "stop_z": 4.0},
            {"exit_z": -0.1},
            {"horizon_bars": 0},
        ],
    )
    def test_invalid_thresholds_rejected(self, kwargs):
        with pytest.raises(ValueError):
            LabelConfig(**kwargs)


class TestBuildLabelTable:
    def _features(self) -> pd.DataFrame:
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        ko = pd.DataFrame(
            {
                "symbol_a": "KO",
                "symbol_b": "PEP",
                "event_timestamp": dates,
                "zscore": [0.0, 2.5, 0.1, 0.0, 0.0],
            }
        )
        xom = pd.DataFrame(
            {
                "symbol_a": "XOM",
                "symbol_b": "CVX",
                "event_timestamp": dates,
                "zscore": [0.0, -2.5, -4.5, 0.0, 0.0],
            }
        )
        return pd.concat([xom, ko], ignore_index=True)

    def test_two_columns_one_row_per_entry(self):
        table = build_label_table(self._features(), CFG)
        assert table.columns.tolist() == ["pair_date_id", "label"]
        assert table.to_dict("records") == [
            {"pair_date_id": "KO__PEP|2024-01-02", "label": 1},
            {"pair_date_id": "XOM__CVX|2024-01-02", "label": 0},
        ]

    def test_rows_out_of_order_give_the_same_table(self):
        shuffled = self._features().sample(frac=1.0, random_state=0)
        pd.testing.assert_frame_equal(
            build_label_table(shuffled, CFG), build_label_table(self._features(), CFG)
        )

    def test_no_entries_gives_empty_table_with_columns(self):
        flat = self._features().assign(zscore=0.0)
        table = build_label_table(flat, CFG)
        assert table.empty
        assert table.columns.tolist() == ["pair_date_id", "label"]


def test_pair_date_id_format():
    assert pair_date_id("KO", "PEP", pd.Timestamp("2020-03-02 00:00")) == "KO__PEP|2020-03-02"
