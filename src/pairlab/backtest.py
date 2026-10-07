"""Backtest engine — event-loop orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from pairlab.config import BacktestConfig
from pairlab.data.handler import PointInTimeDataHandler
from pairlab.data.universe import Universe
from pairlab.event_queue import EventQueue
from pairlab.events import DelistEvent, FillEvent, MarketEvent
from pairlab.execution.simulated import SimulatedExecutionHandler
from pairlab.metrics.performance import PerformanceMetrics, compute_metrics
from pairlab.metrics.tearsheet import save_tearsheet
from pairlab.portfolio import Portfolio
from pairlab.strategy.base import BaseStrategy
from pairlab.strategy.pairs import PairsStrategy


@dataclass
class BacktestResult:
    """Outputs of a completed backtest run."""
    metrics: PerformanceMetrics
    equity_curve: list[tuple[datetime, float]]
    fills: list[FillEvent]
    config: BacktestConfig


def run_backtest(
    cfg: BacktestConfig,
    bars,           # pd.DataFrame
    universe: Universe | None = None,
    pairs: list[tuple[str, str]] | None = None,
    strategy: BaseStrategy | None = None,
) -> BacktestResult:
    """Run the full event-driven backtest loop.

    Args:
        cfg: validated BacktestConfig.
        bars: normalised daily-bars DataFrame.
        universe: optional point-in-time universe; built from bars if None.
        pairs: candidate pairs; defaults to all pairs from symbols in bars.
        strategy: strategy instance; defaults to PairsStrategy(cfg.strategy, pairs).

    Returns:
        BacktestResult with metrics, equity curve, fills.
    """
    import pandas as pd

    if universe is None:
        universe = Universe.from_bar_data(bars)

    symbols = list(cfg.symbols) or list(bars["symbol"].unique())

    if pairs is None:
        pairs = [(a, b) for i, a in enumerate(symbols) for b in symbols[i + 1:]]

    # Filter to backtest date range
    start_ts = pd.Timestamp(cfg.start_date, tz="UTC")
    end_ts = pd.Timestamp(cfg.end_date, tz="UTC")
    bars_filtered = bars[(bars["ts"] >= start_ts) & (bars["ts"] <= end_ts)]

    handler = PointInTimeDataHandler(bars_filtered, universe)
    if strategy is None:
        strategy = PairsStrategy(cfg.strategy, pairs)
    portfolio = Portfolio(cfg.portfolio, cfg.costs)
    execution = SimulatedExecutionHandler(cfg.costs)
    queue = EventQueue()

    all_fills: list[FillEvent] = []

    while handler.has_more():
        # Load next batch into queue
        for event in handler.next_events():
            queue.push(event)

        # Process all events at this timestamp
        while queue:
            event = queue.pop()

            if isinstance(event, MarketEvent):
                # Fill previous bar's orders at today's open — no lookahead
                fills = execution.on_market(event)
                for fill in fills:
                    portfolio.on_fill(fill)
                    all_fills.append(fill)

                # Update MTM prices, generate signals from today's close
                portfolio.on_market(event)
                signals = strategy.on_market(event)
                for sig in signals:
                    orders = portfolio.on_signal(sig)
                    for order in orders:
                        execution.on_order(order)  # queued for next bar's open

            elif isinstance(event, DelistEvent):
                execution.cancel_pending(event.symbol)
                portfolio.on_delist(event)

        portfolio.record_equity(handler.current_ts or end_ts.to_pydatetime())

    equity_curve = portfolio.get_equity_series()
    metrics = compute_metrics(equity_curve, all_fills)

    if cfg.output_dir:
        save_tearsheet(cfg.name, metrics, equity_curve, cfg.output_dir)

    return BacktestResult(
        metrics=metrics,
        equity_curve=equity_curve,
        fills=all_fills,
        config=cfg,
    )
