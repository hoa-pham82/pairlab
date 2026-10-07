"""Write generated bars and symbol master to Postgres bronze + vendor schemas."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

_DEFAULT_DSN = "postgresql://pairlab:pairlab@localhost:5432/pairlab"


def write_symbol_master(master_df: pd.DataFrame, dsn: str = _DEFAULT_DSN) -> int:
    """Upsert symbol master rows into vendor.symbols. Returns row count inserted."""
    rows = [
        (
            str(r["symbol"]),
            str(r.get("name", r["symbol"])),
            str(r.get("sector", "Unknown")),
            pd.Timestamp(r["listed_at"]).date(),
            pd.Timestamp(r["delisted_at"]).date() if pd.notna(r.get("delisted_at")) else None,
        )
        for _, r in master_df.iterrows()
    ]
    with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO vendor.symbols (symbol, name, sector, listed_at, delisted_at)
            VALUES %s
            ON CONFLICT (symbol, listed_at) DO UPDATE
                SET delisted_at = EXCLUDED.delisted_at
            """,
            rows,
        )
        conn.commit()
    log.info("Upserted %d symbol master rows", len(rows))
    return len(rows)


def write_raw_bars(bars_df: pd.DataFrame, dsn: str = _DEFAULT_DSN, batch_size: int = 5_000) -> int:
    """Append raw bars to bronze.raw_daily_bars. Preserves dupes intentionally (bronze = raw)."""
    cols = ["symbol", "ts", "open", "high", "low", "close", "volume"]
    optional = {"adj_close": None, "schema_version": 1}
    df = bars_df.copy()
    for col, default in optional.items():
        if col not in df.columns:
            df[col] = default

    rows = [
        (
            str(r["symbol"]),
            pd.Timestamp(r["ts"]).to_pydatetime(),
            float(r.get("schema_version", 1)),
            float(r["open"]),
            float(r["high"]),
            float(r["low"]),
            float(r["close"]),
            float(r["volume"]),
            float(r["adj_close"]) if pd.notna(r.get("adj_close")) else None,
            "generator",
        )
        for _, r in df.iterrows()
    ]

    total = 0
    with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            execute_values(
                cur,
                """
                INSERT INTO bronze.raw_daily_bars
                    (symbol, ts, schema_version, open, high, low, close, volume, adj_close, source_file)
                VALUES %s
                """,
                batch,
            )
            total += len(batch)
            log.info("  Inserted batch %d/%d (%d rows)", i // batch_size + 1, -(-len(rows) // batch_size), len(batch))
        conn.commit()

    log.info("Inserted %d raw bar rows into bronze.raw_daily_bars", total)
    return total
