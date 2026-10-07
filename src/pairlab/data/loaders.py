"""Load bar data from Parquet/CSV files, or Yahoo Finance."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd


_REQUIRED_COLS = {"symbol", "ts", "open", "high", "low", "close", "volume"}


def load_parquet(path: Path) -> pd.DataFrame:
    """Load and normalise a daily-bars Parquet file."""
    df = pd.read_parquet(path)
    return _normalise(df)


def load_csv(path: Path) -> pd.DataFrame:
    """Load and normalise a daily-bars CSV file."""
    df = pd.read_csv(path, parse_dates=["ts"])
    return _normalise(df)


def load_yahoo(symbols: list[str], start: date, end: date) -> pd.DataFrame:
    """Download adjusted daily bars from Yahoo Finance via yfinance."""
    import yfinance as yf

    raw = yf.download(
        symbols,
        start=start.isoformat(),
        end=end.isoformat(),
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        frames = []
        for sym in symbols:
            if sym not in raw.columns.get_level_values(1):
                continue
            sub = raw.xs(sym, axis=1, level=1).copy()
            sub.columns = sub.columns.str.lower()
            sub["symbol"] = sym
            sub = sub.rename(columns={"date": "ts"})
            sub.index.name = "ts"
            sub = sub.reset_index()
            frames.append(sub)
        df = pd.concat(frames, ignore_index=True)
    else:
        raw.columns = raw.columns.str.lower()
        raw["symbol"] = symbols[0]
        raw.index.name = "ts"
        df = raw.reset_index()

    return _normalise(df)



def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure consistent column names, UTC timestamps, and non-negative prices."""
    df = df.rename(columns=str.lower)
    # accept 'date', 'datetime', 'timestamp' as the time column
    for alt in ("date", "datetime", "timestamp"):
        if alt in df.columns and "ts" not in df.columns:
            df = df.rename(columns={alt: "ts"})

    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # normalise timestamps to UTC datetime
    if not pd.api.types.is_datetime64_any_dtype(df["ts"]):
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
    elif df["ts"].dt.tz is None:
        df["ts"] = df["ts"].dt.tz_localize("UTC")
    else:
        df["ts"] = df["ts"].dt.tz_convert("UTC")

    df = df.sort_values(["symbol", "ts"]).reset_index(drop=True)
    return df
