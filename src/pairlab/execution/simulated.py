"""SimulatedExecutionHandler — fills orders at the next bar's open price."""

from __future__ import annotations

from pairlab.config import CostConfig
from pairlab.events import FillEvent, MarketEvent, OrderEvent
from pairlab.execution.costs import total_cost


class SimulatedExecutionHandler:
    """Fills orders at next-bar open.  Rejects orders when volume == 0 (halt)."""

    def __init__(self, cfg: CostConfig) -> None:
        self._cfg = cfg
        self._pending: list[OrderEvent] = []
        self._last_bar: dict[str, MarketEvent] = {}

    def on_order(self, order: OrderEvent) -> None:
        """Queue an order for execution at the next market open."""
        if order.quantity == 0:
            return
        self._pending.append(order)

    def on_market(self, event: MarketEvent) -> list[FillEvent]:
        """Try to fill all pending orders for this symbol at today's open."""
        self._last_bar[event.symbol] = event

        fills: list[FillEvent] = []
        remaining: list[OrderEvent] = []

        for order in self._pending:
            if order.symbol != event.symbol:
                remaining.append(order)
                continue

            if event.volume == 0:
                # Trading halt — reject order, leave it pending for next open
                remaining.append(order)
                continue

            fill_price = event.open
            daily_vol_pct = (event.high - event.low) / event.open if event.open > 0 else 0.01
            cost, slip = total_cost(
                order.quantity, fill_price, event.volume, daily_vol_pct, self._cfg
            )

            fills.append(FillEvent(
                ts=event.ts,
                symbol=order.symbol,
                quantity=order.quantity,
                fill_price=fill_price,
                commission=cost - slip,
                slippage=slip,
                pair_id=order.pair_id,
            ))

        self._pending = remaining
        return fills

    def cancel_pending(self, symbol: str) -> None:
        """Cancel all pending orders for a symbol (e.g., on delisting)."""
        self._pending = [o for o in self._pending if o.symbol != symbol]
