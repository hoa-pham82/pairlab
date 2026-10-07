"""Shared test fixtures for bar DataFrames."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd


def make_bars(
    symbols: list[str],
    dates: list[datetime],
    price_fn=None,
) -> pd.DataFrame:
    """Build a minimal bars DataFrame from symbols × dates."""
    rows = []
    for sym in symbols:
        for i, ts in enumerate(dates):
            price = price_fn(sym, i) if price_fn else float(100 + i)
            rows.append({
                "symbol": sym,
                "ts": ts,
                "open": price,
                "high": price * 1.01,
                "low": price * 0.99,
                "close": price,
                "volume": 1_000_000.0,
            })
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.sort_values(["symbol", "ts"]).reset_index(drop=True)


def utc(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=timezone.utc)


DATES_5 = [utc(2024, 1, d) for d in range(2, 7)]
