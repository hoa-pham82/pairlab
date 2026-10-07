"""PairsStrategy — cointegration-based pairs trading signal generation."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from pairlab.config import StrategyConfig
from pairlab.events import MarketEvent, SignalEvent
from pairlab.strategy.base import BaseStrategy
from pairlab.strategy.stats import (
    check_cointegration,
    compute_spread,
    rolling_zscore,
)


@dataclass
class PairState:
    """Runtime state for one active pair."""
    symbol_a: str
    symbol_b: str
    hedge_ratio: float
    half_life: float
    spread_history: deque[float] = field(default_factory=lambda: deque(maxlen=500))
    position: int = 0          # +1 long spread, -1 short spread, 0 flat
    bars_in_trade: int = 0
    bars_since_retest: int = 0
    active: bool = True        # False = regime broken; stop trading


class PairsStrategy(BaseStrategy):
    """Cointegration pairs strategy.

    Formation window: test cointegration, estimate hedge ratio and half-life.
    Trading window: enter on |z| > entry_z, exit on |z| < exit_z, stop on |z| > stop_z.
    """

    def __init__(self, config: StrategyConfig, pairs: list[tuple[str, str]]) -> None:
        """
        Args:
            config: strategy hyperparameters.
            pairs: list of (symbol_a, symbol_b) pairs to consider.
        """
        self._cfg = config
        self._candidate_pairs = pairs
        self._price_history: defaultdict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=config.formation_window + config.zscore_window + 10)
        )
        self._pair_state: dict[tuple[str, str], PairState] = {}
        self._bar_count = 0

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    def on_market(self, event: MarketEvent) -> list[SignalEvent]:
        """Update price history, run formation/retest if due, emit signals."""
        if event.close <= 0 or np.isnan(event.close):
            return []

        self._price_history[event.symbol].append(event.close)
        self._bar_count += 1

        signals: list[SignalEvent] = []

        for pair in self._candidate_pairs:
            sig = self._process_pair(pair, event.ts)
            signals.extend(sig)

        return signals

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_pair(self, pair: tuple[str, str], ts: datetime) -> list[SignalEvent]:
        sym_a, sym_b = pair
        hist_a = list(self._price_history[sym_a])
        hist_b = list(self._price_history[sym_b])

        n = min(len(hist_a), len(hist_b), self._cfg.formation_window)
        if n < self._cfg.zscore_window + 1:
            return []   # not enough history yet

        log_a = np.log(np.array(hist_a[-n:], dtype=float))
        log_b = np.log(np.array(hist_b[-n:], dtype=float))

        state = self._pair_state.get(pair)

        # Formation or periodic re-test
        if state is None or state.bars_since_retest >= self._cfg.retest_every:
            result = check_cointegration(log_a, log_b)

            if state is None:
                if result.pvalue > self._cfg.coint_pvalue_entry:
                    return []  # not cointegrated; don't add to active pairs
                state = PairState(
                    symbol_a=sym_a,
                    symbol_b=sym_b,
                    hedge_ratio=result.hedge_ratio,
                    half_life=result.half_life,
                )
                self._pair_state[pair] = state
            else:
                if result.pvalue > self._cfg.coint_pvalue_drop and state.position == 0:
                    state.active = False
                    return []
                if result.pvalue <= self._cfg.coint_pvalue_drop:
                    state.hedge_ratio = result.hedge_ratio
                    state.half_life = result.half_life
                    state.active = True
            state.bars_since_retest = 0
        else:
            state.bars_since_retest += 1

        if not state.active:
            return []

        # Compute current z-score
        spread_series = compute_spread(log_a, log_b, state.hedge_ratio)
        zscores = rolling_zscore(spread_series, self._cfg.zscore_window)
        z = zscores[-1]

        if np.isnan(z):
            return []

        state.spread_history.append(float(spread_series[-1]))

        pair_id = f"{sym_a}/{sym_b}"
        signals: list[SignalEvent] = []

        # Time stop
        max_hold = (
            int(state.half_life * self._cfg.time_stop_halflife_multiple)
            if np.isfinite(state.half_life) else 9999
        )

        if state.position != 0:
            state.bars_in_trade += 1

        # Exit conditions (check before entry)
        if state.position != 0:
            should_exit = (
                abs(z) < self._cfg.exit_z           # mean reversion
                or abs(z) > self._cfg.stop_z        # stop-loss
                or state.bars_in_trade >= max_hold  # time stop
            )
            if should_exit:
                signals.extend(self._close_pair(sym_a, sym_b, pair_id, ts))
                state.position = 0
                state.bars_in_trade = 0
                return signals

        # Entry conditions
        if state.position == 0:
            if z > self._cfg.entry_z:
                # spread too high → short spread: short A, long B
                signals.extend(self._open_pair(sym_a, sym_b, pair_id, ts, direction=-1))
                state.position = -1
                state.bars_in_trade = 0
            elif z < -self._cfg.entry_z:
                # spread too low → long spread: long A, short B
                signals.extend(self._open_pair(sym_a, sym_b, pair_id, ts, direction=+1))
                state.position = +1
                state.bars_in_trade = 0

        return signals

    def _open_pair(
        self, sym_a: str, sym_b: str, pair_id: str, ts: datetime, direction: int
    ) -> list[SignalEvent]:
        cfg = self._cfg
        notional = 50_000.0   # half gross notional per leg; portfolio sizes it exactly
        return [
            SignalEvent(ts=ts, symbol=sym_a, direction=direction, target_dollar=notional, pair_id=pair_id),
            SignalEvent(ts=ts, symbol=sym_b, direction=-direction, target_dollar=notional, pair_id=pair_id),
        ]

    def _close_pair(
        self, sym_a: str, sym_b: str, pair_id: str, ts: datetime
    ) -> list[SignalEvent]:
        return [
            SignalEvent(ts=ts, symbol=sym_a, direction=0, target_dollar=0.0, pair_id=pair_id),
            SignalEvent(ts=ts, symbol=sym_b, direction=0, target_dollar=0.0, pair_id=pair_id),
        ]

    def get_pair_state(self, pair: tuple[str, str]) -> PairState | None:
        """Return the current state for a pair (useful for diagnostics/tests)."""
        return self._pair_state.get(pair)
