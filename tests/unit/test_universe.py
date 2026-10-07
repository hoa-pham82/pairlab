"""Tests for Universe (point-in-time symbol master)."""

from __future__ import annotations

from datetime import timezone

import pandas as pd
import pytest

from pairlab.data.universe import Universe
from tests.fixtures.bars import DATES_5, make_bars, utc


def _make_master(rows):
    df = pd.DataFrame(rows)
    if "delisted_at" not in df.columns:
        df["delisted_at"] = pd.NaT
    return df


class TestUniverseAsOf:
    def test_symbol_not_yet_listed(self):
        master = _make_master([{"symbol": "KO", "listed_at": DATES_5[2], "delisted_at": pd.NaT}])
        u = Universe(master)
        assert "KO" not in u.as_of(DATES_5[1])
        assert "KO" in u.as_of(DATES_5[2])

    def test_delisted_symbol_excluded(self):
        master = _make_master([{"symbol": "KO", "listed_at": DATES_5[0], "delisted_at": DATES_5[3]}])
        u = Universe(master)
        assert "KO" in u.as_of(DATES_5[2])
        assert "KO" not in u.as_of(DATES_5[3])

    def test_still_active(self):
        master = _make_master([{"symbol": "KO", "listed_at": DATES_5[0], "delisted_at": pd.NaT}])
        u = Universe(master)
        assert "KO" in u.as_of(DATES_5[-1])

    def test_from_bar_data(self):
        bars = make_bars(["KO", "PEP"], DATES_5)
        u = Universe.from_bar_data(bars)
        assert "KO" in u.as_of(DATES_5[-1])
        assert "PEP" in u.as_of(DATES_5[-1])

    def test_missing_column_raises(self):
        with pytest.raises(ValueError, match="(?i)symbol master must have columns"):
            Universe(pd.DataFrame([{"symbol": "KO"}]))

    @pytest.mark.parametrize("ts_idx,expected_count", [
        (0, 2),  # both listed on day 0
        (2, 1),  # GONE delisted on day 2
        (4, 1),  # still 1 after delisting
    ])
    def test_count_as_of(self, ts_idx, expected_count):
        master = _make_master([
            {"symbol": "KO",   "listed_at": DATES_5[0], "delisted_at": pd.NaT},
            {"symbol": "GONE", "listed_at": DATES_5[0], "delisted_at": DATES_5[2]},
        ])
        u = Universe(master)
        assert len(u.as_of(DATES_5[ts_idx])) == expected_count
