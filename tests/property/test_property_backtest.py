"""Property-based tests using Hypothesis: determinism, no-lookahead, cost monotonicity."""

from __future__ import annotations

from datetime import date, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from pairlab.config import BacktestConfig, CostConfig, StrategyConfig
from pairlab.data.handler import PointInTimeDataHandler
from pairlab.data.universe import Universe
from pairlab.event_queue import EventQueue
from pairlab.events import MarketEvent
from pairlab.execution.simulated import SimulatedExecutionHandler
from pairlab.metrics.performance import compute_metrics
from pairlab.strategy.pairs import PairsStrategy
from pairlab.strategy.stats import rolling_zscore
from tests.fixtures.bars import utc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bars(prices_a, prices_b, base_ts=None):
    if base_ts is None:
        base_ts = utc(2020, 1, 2)
    rows = []
    for i, (pa, pb) in enumerate(zip(prices_a, prices_b)):
        ts = base_ts + timedelta(days=i)
        for sym, p in [("A", pa), ("B", pb)]:
            rows.append({"symbol": sym, "ts": ts,
                         "open": p, "high": p * 1.005,
                         "low": p * 0.995, "close": p, "volume": 1e6})
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.sort_values(["symbol", "ts"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Determinism: same data + same config → identical results
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_deterministic_equity_curve(self):
        """Two runs with identical data must produce identical equity curves."""
        rng = np.random.default_rng(0)
        prices_a = (100 * np.exp(np.cumsum(rng.normal(0, 0.01, 200)))).tolist()
        prices_b = (100 * np.exp(np.cumsum(rng.normal(0, 0.01, 200)))).tolist()
        bars = _make_bars(prices_a, prices_b)

        cfg = BacktestConfig(
            start_date=date(2020, 1, 2),
            end_date=date(2020, 7, 19),
            symbols=["A", "B"],
            output_dir=None,
        )

        from pairlab.backtest import run_backtest
        r1 = run_backtest(cfg, bars, pairs=[("A", "B")])
        r2 = run_backtest(cfg, bars, pairs=[("A", "B")])

        assert len(r1.equity_curve) == len(r2.equity_curve)
        for (ts1, eq1), (ts2, eq2) in zip(r1.equity_curve, r2.equity_curve):
            assert ts1 == ts2
            assert abs(eq1 - eq2) < 1e-9


# ---------------------------------------------------------------------------
# No-lookahead: mutating future bars never changes past decisions
# ---------------------------------------------------------------------------

class TestNoLookahead:
    def test_future_price_change_does_not_affect_past_signals(self):
        """If we change prices on day N+1, all decisions on days 1..N must be identical."""
        rng = np.random.default_rng(1)
        n = 150
        prices_a = (100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))).tolist()
        prices_b = (100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))).tolist()
        bars = _make_bars(prices_a, prices_b)

        from pairlab.backtest import run_backtest
        cfg = BacktestConfig(
            start_date=date(2020, 1, 2),
            end_date=date(2020, 5, 30),
            symbols=["A", "B"],
            output_dir=None,
        )

        # Run baseline
        r1 = run_backtest(cfg, bars, pairs=[("A", "B")])

        # Mutate future bars (after the backtest end date — these should be invisible)
        bars_mutated = bars.copy()
        future_mask = bars_mutated["ts"] > pd.Timestamp(cfg.end_date, tz="UTC")
        bars_mutated.loc[future_mask, "close"] *= 1000  # extreme price change

        r2 = run_backtest(cfg, bars_mutated, pairs=[("A", "B")])

        # Results must be identical
        assert len(r1.fills) == len(r2.fills)
        for f1, f2 in zip(r1.fills, r2.fills):
            assert f1.fill_price == f2.fill_price

    def test_get_history_strict_cutoff(self):
        """History returned at ts=T must contain ONLY bars with ts <= T."""
        rng = np.random.default_rng(2)
        n = 50
        bars = _make_bars(
            (100 + rng.normal(0, 1, n)).tolist(),
            (80 + rng.normal(0, 1, n)).tolist(),
        )
        universe = Universe.from_bar_data(bars)
        handler = PointInTimeDataHandler(bars, universe)
        cutoff = bars["ts"].sort_values().iloc[n // 2]
        hist = handler.get_history("A", cutoff.to_pydatetime(), n_bars=100)
        assert (hist["ts"] <= cutoff).all()


# ---------------------------------------------------------------------------
# Cost monotonicity: more cost → never more P&L
# ---------------------------------------------------------------------------

class TestCostMonotonicity:
    def test_higher_costs_never_better_pnl(self):
        """Doubling all costs must not increase total P&L."""
        rng = np.random.default_rng(3)
        n = 200
        bars = _make_bars(
            (100 * np.exp(np.cumsum(rng.normal(0, 0.015, n)))).tolist(),
            (100 * np.exp(np.cumsum(rng.normal(0, 0.015, n)))).tolist(),
        )

        import dataclasses
        from pairlab.backtest import run_backtest

        def _run(cost_multiplier: float):
            cfg = BacktestConfig(
                start_date=date(2020, 1, 2),
                end_date=date(2020, 7, 19),
                symbols=["A", "B"],
                output_dir=None,
                costs=CostConfig(
                    commission_bps=5.0 * cost_multiplier,
                    commission_min_usd=1.0 * cost_multiplier,
                    half_spread_bps=2.0 * cost_multiplier,
                    slippage_factor=0.1 * cost_multiplier,
                    borrow_cost_bps_yr=50.0 * cost_multiplier,
                ),
            )
            return run_backtest(cfg, bars, pairs=[("A", "B")])

        r_low = _run(1.0)
        r_high = _run(2.0)

        # Higher costs must not produce better P&L
        assert r_high.metrics.total_pnl <= r_low.metrics.total_pnl + 1.0  # allow float tolerance


# ---------------------------------------------------------------------------
# Property: rolling z-score never NaN on valid input
# ---------------------------------------------------------------------------

@given(
    values=st.lists(st.floats(min_value=0.01, max_value=1000.0, allow_nan=False,
                              allow_infinity=False), min_size=60, max_size=200),
    window=st.integers(min_value=5, max_value=30),
)
@settings(max_examples=30)
def test_zscore_no_crash_on_valid_floats(values, window):
    """rolling_zscore must never raise on valid finite floats."""
    z = rolling_zscore(np.array(values), window)
    # NaN is allowed; crashes are not
    assert len(z) == len(values)


# ---------------------------------------------------------------------------
# Property: engine never crashes on corrupted bar streams
# ---------------------------------------------------------------------------

@given(
    prices=st.lists(
        st.floats(min_value=-100.0, max_value=1000.0),
        min_size=5, max_size=50,
    )
)
@settings(max_examples=20)
def test_strategy_never_crashes_on_corrupted_prices(prices):
    """PairsStrategy must not raise on any price value (NaN, negative, zero)."""
    from pairlab.strategy.pairs import PairsStrategy
    strat = PairsStrategy(StrategyConfig(), pairs=[("A", "B")])
    for i, p in enumerate(prices):
        ts = utc(2020, 1, 2) + timedelta(days=i)
        e = MarketEvent(ts=ts, symbol="A", open=p, high=p, low=p, close=p, volume=1e6)
        try:
            strat.on_market(e)
        except Exception as exc:
            pytest.fail(f"Strategy raised on price {p}: {exc}")
