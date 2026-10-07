"""Point-in-time universe derived from a symbol master (SCD2-style)."""

from __future__ import annotations

from datetime import datetime

import pandas as pd


class Universe:
    """Returns the set of tradeable symbols as of a given date, without lookahead.

    The symbol master has columns: symbol, listed_at, delisted_at (NaT = still active).
    """

    def __init__(self, master: pd.DataFrame) -> None:
        """
        Args:
            master: DataFrame with columns [symbol, listed_at, delisted_at].
        """
        required = {"symbol", "listed_at", "delisted_at"}
        if not required.issubset(master.columns):
            raise ValueError(f"Symbol master must have columns {required}")
        self._master = master.copy()
        # normalise to UTC
        for col in ("listed_at", "delisted_at"):
            if self._master[col].dtype == object:
                self._master[col] = pd.to_datetime(self._master[col], utc=True)
            elif self._master[col].dt.tz is None:
                self._master[col] = self._master[col].dt.tz_localize("UTC")

    def as_of(self, ts: datetime) -> set[str]:
        """Symbols listed on or before ts and not yet delisted (or delisted after ts)."""
        m = self._master
        active = m[(m["listed_at"] <= ts) & (m["delisted_at"].isna() | (m["delisted_at"] > ts))]
        return set(active["symbol"])

    def delist_ts(self, symbol: str) -> datetime | None:
        """Return the delisting timestamp for symbol, or None if still active."""
        rows = self._master[self._master["symbol"] == symbol]
        if rows.empty:
            return None
        ts = rows.iloc[0]["delisted_at"]
        return None if pd.isna(ts) else ts.to_pydatetime()

    @classmethod
    def from_bar_data(cls, df: pd.DataFrame) -> "Universe":
        """Build a trivial universe from bar data: listed = first bar, delisted = never."""
        summary = (
            df.groupby("symbol")["ts"]
            .agg(listed_at="min")
            .reset_index()
        )
        summary["delisted_at"] = pd.NaT
        return cls(summary)
