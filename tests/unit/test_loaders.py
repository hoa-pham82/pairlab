"""Tests for data loaders — file loading and normalisation."""

from __future__ import annotations

import io
from pathlib import Path
import pandas as pd
import pytest

from pairlab.data.loaders import _normalise, load_csv, load_parquet
from tests.fixtures.bars import DATES_5, make_bars


def _sample_csv(tmp_path: Path) -> Path:
    bars = make_bars(["KO", "PEP"], DATES_5[:3])
    p = tmp_path / "bars.csv"
    bars.to_csv(p, index=False)
    return p


def _sample_parquet(tmp_path: Path) -> Path:
    bars = make_bars(["KO", "PEP"], DATES_5[:3])
    p = tmp_path / "bars.parquet"
    bars.to_parquet(p, index=False)
    return p


class TestLoadCSV:
    def test_loads_all_rows(self, tmp_path):
        p = _sample_csv(tmp_path)
        df = load_csv(p)
        assert len(df) == 6  # 2 symbols × 3 dates

    def test_ts_is_utc(self, tmp_path):
        df = load_csv(_sample_csv(tmp_path))
        assert str(df["ts"].dt.tz) == "UTC"

    def test_required_columns_present(self, tmp_path):
        df = load_csv(_sample_csv(tmp_path))
        for col in ("symbol", "ts", "open", "high", "low", "close", "volume"):
            assert col in df.columns

    def test_sorted_by_symbol_then_ts(self, tmp_path):
        df = load_csv(_sample_csv(tmp_path))
        expected = df.sort_values(["symbol", "ts"])
        pd.testing.assert_frame_equal(df.reset_index(drop=True), expected.reset_index(drop=True))


class TestLoadParquet:
    def test_loads_all_rows(self, tmp_path):
        df = load_parquet(_sample_parquet(tmp_path))
        assert len(df) == 6

    def test_ts_is_utc(self, tmp_path):
        df = load_parquet(_sample_parquet(tmp_path))
        assert str(df["ts"].dt.tz) == "UTC"


class TestNormalise:
    def _base(self):
        return pd.DataFrame([{
            "symbol": "KO", "ts": "2024-01-02",
            "open": 60.0, "high": 61.0, "low": 59.5, "close": 60.5, "volume": 1e6,
        }])

    def test_date_column_renamed_to_ts(self):
        df = self._base().rename(columns={"ts": "date"})
        out = _normalise(df)
        assert "ts" in out.columns

    def test_datetime_column_renamed_to_ts(self):
        df = self._base().rename(columns={"ts": "datetime"})
        out = _normalise(df)
        assert "ts" in out.columns

    def test_timestamp_column_renamed_to_ts(self):
        df = self._base().rename(columns={"ts": "timestamp"})
        out = _normalise(df)
        assert "ts" in out.columns

    def test_missing_required_column_raises(self):
        df = self._base().drop(columns=["close"])
        with pytest.raises(ValueError, match="Missing columns"):
            _normalise(df)

    def test_naive_ts_localised_to_utc(self):
        df = self._base()
        df["ts"] = pd.to_datetime(df["ts"])  # no tz
        out = _normalise(df)
        assert str(out["ts"].dt.tz) == "UTC"

    def test_aware_ts_converted_to_utc(self):
        df = self._base()
        df["ts"] = pd.to_datetime(df["ts"]).dt.tz_localize("US/Eastern")
        out = _normalise(df)
        assert str(out["ts"].dt.tz) == "UTC"

    def test_string_ts_converted(self):
        df = self._base()
        out = _normalise(df)
        assert pd.api.types.is_datetime64_any_dtype(out["ts"])

    def test_columns_lowercased(self):
        df = self._base().rename(columns=str.upper)
        df = df.rename(columns={"TS": "ts"})  # keep ts for the check
        out = _normalise(df)
        assert all(c == c.lower() for c in out.columns)


