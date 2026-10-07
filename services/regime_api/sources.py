"""Where the regime API gets price history."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import psycopg2

_HISTORY_SQL = """
    SELECT close_a, close_b FROM (
        SELECT close_a, close_b, event_timestamp
        FROM gold.feat_pair_daily
        WHERE symbol_a = %s AND symbol_b = %s
        ORDER BY event_timestamp DESC
        LIMIT %s
    ) latest
    ORDER BY event_timestamp
"""


class PriceSource(Protocol):
    """Looks up recent closing prices for a pair."""

    def get_closes(self, pair_id: str, n_bars: int) -> tuple[np.ndarray, np.ndarray] | None:
        """Return (close_a, close_b), oldest first, or None for an unknown pair."""

    def ping(self) -> bool:
        """Return True if the source can be reached."""


class PostgresPriceSource:
    """Reads pair closes from gold.feat_pair_daily."""

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    def get_closes(self, pair_id: str, n_bars: int) -> tuple[np.ndarray, np.ndarray] | None:
        symbol_a, symbol_b = pair_id.split("__")
        with psycopg2.connect(self._dsn) as conn, conn.cursor() as cur:
            cur.execute(_HISTORY_SQL, (symbol_a, symbol_b, n_bars))
            rows = cur.fetchall()
        if not rows:
            return None
        closes = np.array(rows, dtype=float)
        return closes[:, 0], closes[:, 1]

    def ping(self) -> bool:
        try:
            with psycopg2.connect(self._dsn, connect_timeout=2) as conn, conn.cursor() as cur:
                cur.execute("SELECT 1")
            return True
        except psycopg2.Error:
            return False
