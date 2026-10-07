"""Export gold.obt_pair_backtest_input to Parquet and run a backtest.

Postgres access lives here (platform/script code), not in the engine (src/pairlab).
The engine only ever reads a normalised Parquet file.

Usage:
    uv run python scripts/backtest_from_gold.py [--dsn "host=... dbname=..."]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pairlab.data.loaders import _normalise, load_parquet


_DEFAULT_DSN = "host=localhost port=5432 dbname=pairlab user=pairlab password=pairlab"
_PARQUET_PATH = Path("data/gold_obt.parquet")


def _load_obt_from_postgres(dsn: str) -> pd.DataFrame:
    """Read gold.obt_pair_backtest_input from Postgres and return a normalised DataFrame.

    ts is stored as Unix nanoseconds; converted here before handing to the engine.
    """
    import psycopg2  # intentional: infra import stays outside src/pairlab

    with psycopg2.connect(dsn) as conn:
        df = pd.read_sql_query(
            "SELECT symbol_a AS symbol, "
            "  to_timestamp(ts / 1e9) AT TIME ZONE 'UTC' AS ts, "
            "  open_a AS open, high_a AS high, low_a AS low, "
            "  close_a AS close, volume_a AS volume "
            "FROM gold.obt_pair_backtest_input",
            conn,
        )
    return _normalise(df)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsn", default=_DEFAULT_DSN)
    args = parser.parse_args()

    print("Loading bars from gold.obt_pair_backtest_input …")
    df = _load_obt_from_postgres(args.dsn)
    _PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_PARQUET_PATH, index=False)
    print(f"Exported {len(df):,} rows → {_PARQUET_PATH}")

    from pairlab.backtest import run_backtest
    from pairlab.config import BacktestConfig
    import yaml

    cfg = BacktestConfig(
        **yaml.safe_load(Path("configs/backtest/gold_obt.yaml").read_text())
    )
    cfg.data_path = _PARQUET_PATH
    bars = load_parquet(_PARQUET_PATH)
    result = run_backtest(cfg, bars)
    m = result.metrics
    print(f"Sharpe={m.sharpe:.3f}  PnL={m.total_pnl:+,.0f}  Trades={m.n_trades}")


if __name__ == "__main__":
    main()
