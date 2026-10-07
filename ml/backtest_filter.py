"""Backtest the pairs strategy with and without the meta-label filter.

Usage:
    uv run python -m ml.backtest_filter [--active-from 2024-05-20]

Reads bars and pair features from the warehouse and the model from
``models/meta_label.joblib``. The filter only acts from ``--active-from`` on,
which should be the start of the model's validation period, so the comparison
is out of sample.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from ml.filtered_strategy import MetaLabelFilter, frame_feature_lookup
from ml.scoring import ClassifierScorer
from ml.train import load_model
from pairlab.backtest import run_backtest
from pairlab.config import BacktestConfig
from pairlab.strategy.pairs import PairsStrategy

DEFAULT_DSN = "postgresql://pairlab:pairlab@localhost:5432/pairlab"


def compare_with_filter(
    bars: pd.DataFrame,
    features: pd.DataFrame,
    model,
    cfg: BacktestConfig,
    pairs: list[tuple[str, str]],
    active_from: datetime,
    threshold: float = 0.5,
) -> dict:
    """Run the same backtest twice: plain strategy, and strategy behind the filter.

    Args:
        bars: normalised daily bars for every leg of ``pairs``.
        features: pair-feature rows the filter looks up by pair and day.
        model: anything with ``predict_proba(features: dict) -> float``.
        active_from: the filter passes every entry before this time.

    Returns:
        ``baseline`` and ``filtered`` metrics, and the ``filter`` entry counts.
    """
    cfg = cfg.model_copy(update={"output_dir": None})
    baseline = run_backtest(cfg, bars, pairs=pairs)
    filtered_strategy = MetaLabelFilter(
        PairsStrategy(cfg.strategy, pairs),
        model,
        frame_feature_lookup(features),
        threshold=threshold,
        active_from=active_from,
    )
    filtered = run_backtest(cfg, bars, pairs=pairs, strategy=filtered_strategy)
    return {
        "baseline": asdict(baseline.metrics),
        "filtered": asdict(filtered.metrics),
        "filter": {
            **asdict(filtered_strategy.stats),
            "threshold": threshold,
            "active_from": active_from.date().isoformat(),
        },
    }


def read_bars(conn, symbols: list[str]) -> pd.DataFrame:
    """Load daily bars for ``symbols`` from gold.fact_daily_bar (ts is epoch nanoseconds)."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT symbol, ts, open, high, low, close, volume FROM gold.fact_daily_bar "
            "WHERE symbol = ANY(%s)",
            (symbols,),
        )
        frame = pd.DataFrame(cur.fetchall(), columns=[c.name for c in cur.description])
    frame["ts"] = pd.to_datetime(frame["ts"], unit="ns", utc=True)
    return frame.sort_values(["symbol", "ts"]).reset_index(drop=True)


def read_features(conn) -> pd.DataFrame:
    """Load pair features from gold.feat_pair_daily."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT symbol_a, symbol_b, event_timestamp, zscore, hedge_ratio, spread_vol, "
            "correlation_60d FROM gold.feat_pair_daily"
        )
        return pd.DataFrame(cur.fetchall(), columns=[c.name for c in cur.description])


if __name__ == "__main__":
    import psycopg2

    parser = argparse.ArgumentParser()
    parser.add_argument("--dsn", default=os.environ.get("POSTGRES_URL", DEFAULT_DSN))
    parser.add_argument("--model-path", default="models/meta_label.joblib")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2018, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 12, 31))
    parser.add_argument("--active-from", type=date.fromisoformat, default=date(2024, 5, 20))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out")
    args = parser.parse_args()

    with psycopg2.connect(args.dsn) as connection:
        feature_rows = read_features(connection)
        pair_list = sorted(
            set(zip(feature_rows["symbol_a"], feature_rows["symbol_b"], strict=True))
        )
        bar_rows = read_bars(connection, sorted({s for pair in pair_list for s in pair}))

    report = compare_with_filter(
        bar_rows,
        feature_rows,
        ClassifierScorer(load_model(args.model_path)),
        BacktestConfig(start_date=args.start, end_date=args.end),
        pair_list,
        datetime.combine(args.active_from, datetime.min.time(), tzinfo=UTC),
        args.threshold,
    )
    print(json.dumps(report, indent=2, default=str))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, default=str))
