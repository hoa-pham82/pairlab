"""Portfolio — tracks positions, cash, and mark-to-market value."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from pairlab.config import CostConfig, PortfolioConfig
from pairlab.events import DelistEvent, FillEvent, MarketEvent, OrderEvent, SignalEvent
from pairlab.execution.costs import borrow_cost_daily


@dataclass
class Position:
    """A single open position."""
    symbol: str
    quantity: float        # positive = long, negative = short
    avg_price: float
    pair_id: str = ""

    @property
    def is_short(self) -> bool:
        return self.quantity < 0

    def market_value(self, price: float) -> float:
        return self.quantity * price

    def unrealised_pnl(self, price: float) -> float:
        return self.quantity * (price - self.avg_price)


class Portfolio:
    """Manages cash, positions, and generates orders from signals.

    Equity = cash + sum(position.market_value(last_price)).
    """

    def __init__(self, cfg: PortfolioConfig, cost_cfg: CostConfig) -> None:
        self._cfg = cfg
        self._cost_cfg = cost_cfg
        self._cash: float = cfg.initial_cash
        self._positions: dict[str, Position] = {}
        self._last_price: dict[str, float] = {}
        self._equity_history: list[tuple[datetime, float]] = []
        self._realised_pnl: float = 0.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def gross_notional(self) -> float:
        return sum(abs(p.market_value(self._last_price.get(p.symbol, p.avg_price)))
                   for p in self._positions.values())

    @property
    def equity(self) -> float:
        return self._cash + sum(
            p.market_value(self._last_price.get(s, p.avg_price))
            for s, p in self._positions.items()
        )

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def on_market(self, event: MarketEvent) -> None:
        """Update last-known price and accrue short-borrow cost."""
        self._last_price[event.symbol] = event.close
        pos = self._positions.get(event.symbol)
        if pos and pos.is_short:
            short_notional = abs(pos.market_value(event.close))
            cost = borrow_cost_daily(short_notional, self._cost_cfg)
            self._cash -= cost

    def on_signal(self, signal: SignalEvent) -> list[OrderEvent]:
        """Convert a signal into an order, enforcing leverage limits."""
        if signal.direction == 0:
            return self._close_position(signal)
        return self._open_or_adjust(signal)

    def on_fill(self, fill: FillEvent) -> None:
        """Update cash and positions when a fill arrives."""
        total_txn_cost = fill.commission + fill.slippage
        cost = abs(fill.quantity) * fill.fill_price
        if fill.quantity > 0:
            self._cash -= cost + total_txn_cost
        else:
            self._cash += cost - total_txn_cost

        sym = fill.symbol
        if sym in self._positions:
            pos = self._positions[sym]
            new_qty = pos.quantity + fill.quantity
            if abs(new_qty) < 1e-9:
                # Position closed; book realised P&L
                self._realised_pnl += pos.quantity * (fill.fill_price - pos.avg_price)
                del self._positions[sym]
            else:
                # Partial or reversal — recalculate average price
                if (pos.quantity > 0) == (fill.quantity > 0):
                    avg = (pos.quantity * pos.avg_price + fill.quantity * fill.fill_price) / new_qty
                else:
                    avg = pos.avg_price  # closing side; keep cost basis
                self._positions[sym] = Position(sym, new_qty, avg, fill.pair_id)
        else:
            self._positions[sym] = Position(sym, fill.quantity, fill.fill_price, fill.pair_id)

    def on_delist(self, event: DelistEvent) -> None:
        """Force-close a delisted position at the configured delisting return."""
        pos = self._positions.pop(event.symbol, None)
        if pos is None:
            return
        last = self._last_price.get(event.symbol, event.last_price)
        exit_price = last * (1 + self._cfg.delisting_return) if pos.quantity < 0 else last
        self._realised_pnl += pos.quantity * (exit_price - pos.avg_price)
        self._cash += pos.quantity * exit_price

    def record_equity(self, ts: datetime) -> None:
        """Append current equity to history."""
        self._equity_history.append((ts, self.equity))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open_or_adjust(self, signal: SignalEvent) -> list[OrderEvent]:
        current_pos = self._positions.get(signal.symbol)
        price = self._last_price.get(signal.symbol)
        if price is None or price <= 0:
            return []

        target_qty = signal.direction * signal.target_dollar / price

        # Enforce leverage
        projected_notional = self.gross_notional + abs(target_qty) * price
        if projected_notional > self._cfg.max_gross_leverage * self._cfg.initial_cash:
            return []

        current_qty = current_pos.quantity if current_pos else 0.0
        delta = target_qty - current_qty
        if abs(delta) < 1e-9:
            return []

        return [OrderEvent(
            ts=signal.ts,
            symbol=signal.symbol,
            quantity=round(delta, 4),
            pair_id=signal.pair_id,
        )]

    def _close_position(self, signal: SignalEvent) -> list[OrderEvent]:
        pos = self._positions.get(signal.symbol)
        if pos is None:
            return []
        return [OrderEvent(
            ts=signal.ts,
            symbol=signal.symbol,
            quantity=-pos.quantity,
            pair_id=signal.pair_id,
        )]

    def get_equity_series(self) -> list[tuple[datetime, float]]:
        """Return equity history as (ts, equity) pairs."""
        return list(self._equity_history)

    def get_positions(self) -> dict[str, Position]:
        return dict(self._positions)
