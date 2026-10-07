"""Integration test: full event loop from bars to metrics."""

from __future__ import annotations

from datetime import date, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pairlab.backtest import run_backtest
from pairlab.config import BacktestConfig
from pairlab.data.loaders import _normalise
from tests.fixtures.bars import DATES_5, make_bars, utc


def _cointegrated_bars(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """Synthetic bars for A and B with a cointegrated relationship."""
    rng = np.random.default_rng(seed)
    common = np.exp(np.cumsum(rng.normal(0, 0.005, n)))
    spread = np.zeros(n)
    for i in range(1, n):
        spread[i] = spread[i - 1] * 0.9 + rng.normal(0, 0.01)
    prices_a = common * np.exp(spread) * 50
    prices_b = common * 40

    base = utc(2020, 1, 2)
    rows = []
    for i in range(n):
        ts = base + timedelta(days=i)
        for sym, price in [("A", prices_a[i]), ("B", prices_b[i])]:
            rows.append({
                "symbol": sym, "ts": ts,
                "open": price * 0.999, "high": price * 1.005,
                "low": price * 0.995, "close": price,
                "volume": 1_000_000.0,
            })
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.sort_values(["symbol", "ts"]).reset_index(drop=True)


def _cfg(start=date(2020, 1, 2), end=date(2020, 10, 28), output_dir=None) -> BacktestConfig:
    return BacktestConfig(
        name="integration_test",
        start_date=start,
        end_date=end,
        output_dir=Path(output_dir) if output_dir else Path("/tmp/pairlab_test"),
        symbols=["A", "B"],
    )


class TestBacktestLoop:
    def test_runs_without_error(self):
        bars = _cointegrated_bars()
        result = run_backtest(_cfg(), bars)
        assert result.metrics is not None
        assert len(result.equity_curve) > 0

    def test_equity_positive_at_all_times(self):
        """Equity should never go negative (no unlimited leverage)."""
        bars = _cointegrated_bars()
        result = run_backtest(_cfg(), bars)
        for _, eq in result.equity_curve:
            assert eq > 0, f"Equity went negative: {eq}"

    def test_fills_have_valid_prices(self):
        bars = _cointegrated_bars()
        result = run_backtest(_cfg(), bars)
        for fill in result.fills:
            assert fill.fill_price > 0
            assert fill.quantity != 0

    def test_no_fills_when_symbols_empty(self):
        """Empty symbol list with no cointegrated pairs → no fills."""
        bars = _cointegrated_bars(n=50)
        cfg = BacktestConfig(
            name="no_pairs",
            start_date=date(2020, 1, 2),
            end_date=date(2020, 2, 19),
            output_dir=Path("/tmp/pairlab_test"),
        )
        result = run_backtest(cfg, bars, pairs=[])
        assert result.fills == []


class TestTearsheetOutput:
    def test_tearsheet_files_created(self, tmp_path):
        bars = _cointegrated_bars()
        cfg = _cfg(output_dir=str(tmp_path))
        run_backtest(cfg, bars)
        assert (tmp_path / "tearsheet.html").exists()
        assert (tmp_path / "results.json").exists()

    def test_results_json_has_metrics(self, tmp_path):
        import json
        bars = _cointegrated_bars()
        cfg = _cfg(output_dir=str(tmp_path))
        run_backtest(cfg, bars)
        data = json.loads((tmp_path / "results.json").read_text())
        assert "metrics" in data
        assert "sharpe" in data["metrics"]
