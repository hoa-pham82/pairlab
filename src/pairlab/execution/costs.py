"""Transaction cost model: commission, spread, slippage, short-borrow."""

from __future__ import annotations

import math

from pairlab.config import CostConfig


def commission(quantity: float, price: float, cfg: CostConfig) -> float:
    """One-way commission: max(bps * notional, minimum)."""
    notional = abs(quantity) * price
    return max(cfg.commission_bps / 10_000 * notional, cfg.commission_min_usd)


def half_spread_cost(quantity: float, price: float, cfg: CostConfig) -> float:
    """Half-spread paid on entry and exit."""
    notional = abs(quantity) * price
    return cfg.half_spread_bps / 10_000 * notional


def slippage_cost(
    quantity: float,
    price: float,
    daily_volume: float,
    daily_vol_pct: float,
    cfg: CostConfig,
) -> float:
    """Market-impact slippage: k * daily_vol_pct * sqrt(participation).

    participation = |quantity| / daily_volume.
    Returns zero when daily_volume == 0 (halt) or quantity == 0.
    """
    if daily_volume <= 0 or quantity == 0:
        return 0.0
    participation = abs(quantity) / daily_volume
    impact_pct = cfg.slippage_factor * daily_vol_pct * math.sqrt(participation)
    return abs(quantity) * price * impact_pct


def borrow_cost_daily(
    short_notional: float,
    cfg: CostConfig,
) -> float:
    """Daily short-borrow cost accrued on gross short notional."""
    return short_notional * cfg.borrow_cost_bps_yr / 10_000 / 252


def total_cost(
    quantity: float,
    fill_price: float,
    daily_volume: float,
    daily_vol_pct: float,
    cfg: CostConfig,
) -> tuple[float, float]:
    """Total one-way transaction cost (commission + spread + slippage).

    Returns:
        (total_cost_usd, slippage_usd) for record-keeping.
    """
    c = commission(quantity, fill_price, cfg)
    s = half_spread_cost(quantity, fill_price, cfg)
    slip = slippage_cost(quantity, fill_price, daily_volume, daily_vol_pct, cfg)
    return c + s + slip, slip
