"""Performance metrics: Sharpe, Sortino, max drawdown, profit factor, and more."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import datetime

import numpy as np


@dataclass
class PerformanceMetrics:
    """All computed performance metrics for a backtest run."""
    total_return: float
    annualised_return: float
    annualised_vol: float
    sharpe: float           # NaN when vol == 0
    sortino: float          # NaN when downside vol == 0
    max_drawdown: float     # positive value; e.g. 0.15 = 15%
    max_drawdown_duration_days: int
    calmar: float
    profit_factor: float    # inf when no losing trades
    hit_rate: float
    avg_holding_bars: float
    turnover_annual: float
    exposure: float         # fraction of time with any open position
    total_pnl: float
    n_trades: int


def compute_metrics(
    equity_curve: list[tuple[datetime, float]],
    fills: list = None,
    trading_days_per_year: int = 252,
) -> PerformanceMetrics:
    """Compute performance metrics from an equity curve.

    Args:
        equity_curve: list of (timestamp, equity) pairs in chronological order.
        fills: list of FillEvent (optional; used for trade-level stats).
        trading_days_per_year: annualisation factor.

    Returns:
        PerformanceMetrics dataclass.
    """
    if len(equity_curve) < 2:
        return _empty_metrics()

    tss, equities = zip(*equity_curve)
    eq = np.array(equities, dtype=float)
    initial = eq[0]
    final = eq[-1]

    # Daily returns
    returns = np.diff(eq) / eq[:-1]

    # Annualisation
    n_days = len(returns)
    years = n_days / trading_days_per_year

    total_return = (final - initial) / initial
    annualised_return = (1 + total_return) ** (1 / max(years, 1e-9)) - 1

    vol = float(np.std(returns, ddof=1))
    if vol == 0 or np.isnan(vol):
        warnings.warn("Zero volatility — Sharpe is undefined (NaN)", stacklevel=2)
        annualised_vol = 0.0
        sharpe = float("nan")
        sortino = float("nan")
    else:
        annualised_vol = vol * np.sqrt(trading_days_per_year)
        sharpe = annualised_return / annualised_vol

        downside = returns[returns < 0]
        downside_vol = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0
        if downside_vol == 0:
            sortino = float("nan")
        else:
            sortino = annualised_return / (downside_vol * np.sqrt(trading_days_per_year))

    mdd, mdd_dur = max_drawdown(eq)
    calmar = annualised_return / mdd if mdd > 0 else float("inf")

    # Trade-level metrics (from fills)
    n_trades, pf, hit, avg_hold, turnover, exposure = _trade_metrics(fills or [], initial, n_days)
    total_pnl = final - initial

    return PerformanceMetrics(
        total_return=total_return,
        annualised_return=annualised_return,
        annualised_vol=annualised_vol,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=mdd,
        max_drawdown_duration_days=mdd_dur,
        calmar=calmar,
        profit_factor=pf,
        hit_rate=hit,
        avg_holding_bars=avg_hold,
        turnover_annual=turnover,
        exposure=exposure,
        total_pnl=total_pnl,
        n_trades=n_trades,
    )


def max_drawdown(equity: np.ndarray) -> tuple[float, int]:
    """Maximum drawdown and its duration in bars.

    Returns:
        (max_drawdown_fraction, max_drawdown_duration_bars)
    """
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / np.where(peak == 0, 1, peak)
    mdd = float(drawdown.max())

    # Duration: longest consecutive period below peak
    underwater = drawdown > 0
    max_dur = 0
    cur = 0
    for u in underwater:
        if u:
            cur += 1
            max_dur = max(max_dur, cur)
        else:
            cur = 0

    return mdd, max_dur


def _trade_metrics(
    fills: list,
    initial_equity: float,
    n_bars: int,
) -> tuple[int, float, float, float, float, float]:
    """Extract trade-level metrics from fills."""
    if not fills:
        return 0, float("nan"), float("nan"), 0.0, 0.0, 0.0

    by_symbol: dict[str, list] = {}
    for f in fills:
        by_symbol.setdefault(f.symbol, []).append(f)

    pnls: list[float] = []
    holding_bars: list[int] = []
    total_volume = 0.0
    bars_with_position = 0  # approximation: bars between first and last fill

    for sym_fills in by_symbol.values():
        queue = []
        for f in sorted(sym_fills, key=lambda x: x.ts):
            if f.quantity > 0:
                queue.append(f)
            elif queue and f.quantity < 0:
                entry = queue.pop(0)
                pnl = -f.quantity * (f.fill_price - entry.fill_price) - f.commission - entry.commission
                pnls.append(pnl)
                dur = (f.ts - entry.ts).days
                holding_bars.append(dur)
            total_volume += abs(f.quantity) * f.fill_price

    n_trades = len(pnls)
    if n_trades == 0:
        return 0, float("nan"), float("nan"), 0.0, 0.0, 0.0

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    hit_rate = len(wins) / n_trades

    if not losses:
        pf = float("inf")
    elif sum(abs(p) for p in losses) == 0:
        pf = float("inf")
    else:
        pf = sum(wins) / sum(abs(p) for p in losses)

    avg_hold = float(np.mean(holding_bars)) if holding_bars else 0.0
    turnover = total_volume / initial_equity / (n_bars / 252) if n_bars > 0 else 0.0
    exposure = float(n_trades) / n_bars if n_bars > 0 else 0.0

    return n_trades, pf, hit_rate, avg_hold, turnover, exposure


def _empty_metrics() -> PerformanceMetrics:
    nan = float("nan")
    return PerformanceMetrics(
        total_return=0.0, annualised_return=0.0, annualised_vol=0.0,
        sharpe=nan, sortino=nan, max_drawdown=0.0, max_drawdown_duration_days=0,
        calmar=nan, profit_factor=nan, hit_rate=nan,
        avg_holding_bars=0.0, turnover_annual=0.0, exposure=0.0,
        total_pnl=0.0, n_trades=0,
    )
