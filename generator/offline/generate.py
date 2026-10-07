"""Generate synthetic daily bars: common-factor GBM + cointegrated OU spreads."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class GeneratorConfig:
    seed: int = 42
    n_symbols: int = 20
    n_cointegrated_pairs: int = 4
    start: str = "2018-01-02"
    days: int = 1260
    ou_theta: float = 0.07
    ou_sigma: float = 0.015
    gbm_drift: float = 0.00005
    gbm_vol: float = 0.012
    delist_rate: float = 0.03
    skew_zipf_exponent: float = 1.2
    duplicate_rate: float = 0.02
    schema_evolution: bool = True
    schema_change_date: str = "2020-06-01"
    regime_shift_date: str | None = None
    regime_shift_multiplier: float = 5.0
    # Sector change: one or two symbols reclassified on a known date (exercises SCD2 MERGE)
    sector_change_date: str | None = None
    sector_change_symbols: list[str] = field(default_factory=list)
    sector_change_new_value: str = "Energy-Alt"
    # Streaming
    burst_probability: float = 0.05
    late_arrival_probability: float = 0.03
    late_arrival_max_seconds: int = 30
    streaming_duplicate_rate: float = 0.015


def _business_dates(start: str, n: int) -> list[datetime]:
    """Generate n business dates starting from start."""
    base = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    dates = []
    current = base
    while len(dates) < n:
        if current.weekday() < 5:
            dates.append(current)
        current += timedelta(days=1)
    return dates


def _symbol_names(n: int) -> list[str]:
    base = ["KO", "PEP", "XOM", "CVX", "JPM", "BAC", "GS", "MS",
            "AAPL", "MSFT", "GOOG", "META", "AMZN", "NFLX", "TSLA",
            "WMT", "TGT", "HD", "LOW", "NKE", "AXP", "V", "MA", "PYPL",
            "JNJ", "PFE", "MRK", "ABT", "UNH", "CVS"]
    while len(base) < n:
        base.append(f"SYM{len(base):03d}")
    return base[:n]


def _sector(sym: str) -> str:
    sector_map = {
        "KO": "Cons-Staples", "PEP": "Cons-Staples",
        "XOM": "Energy", "CVX": "Energy",
        "JPM": "Financials", "BAC": "Financials", "GS": "Financials", "MS": "Financials",
        "AAPL": "Technology", "MSFT": "Technology", "GOOG": "Technology",
        "META": "Technology", "AMZN": "Technology", "NFLX": "Technology", "TSLA": "Technology",
        "WMT": "Cons-Disc", "TGT": "Cons-Disc", "HD": "Cons-Disc", "LOW": "Cons-Disc",
        "NKE": "Cons-Disc",
    }
    return sector_map.get(sym, "Other")


def generate_bars(
    cfg: GeneratorConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, list[tuple[str, str]], list[dict]]:
    """Generate bars, symbol master, cointegrated pair list, and sector change events.

    Returns:
        (bars_df, symbol_master_df, cointegrated_pairs, sector_changes)
        sector_changes: list of {symbol, effective_date, old_sector, new_sector}
    """
    rng = np.random.default_rng(cfg.seed)
    dates = _business_dates(cfg.start, cfg.days)
    symbols = _symbol_names(cfg.n_symbols)
    n = len(dates)

    # Common market factor (GBM)
    common = np.exp(np.cumsum(rng.normal(cfg.gbm_drift, cfg.gbm_vol, n)))

    base_prices = rng.uniform(20, 200, len(symbols))
    betas = rng.uniform(0.5, 1.5, len(symbols))

    n_pairs = min(cfg.n_cointegrated_pairs, len(symbols) // 2)
    pair_indices = [(i * 2, i * 2 + 1) for i in range(n_pairs)]
    cointegrated_pairs = [(symbols[a], symbols[b]) for a, b in pair_indices]

    # Build OU spreads
    ou_spreads: dict[tuple[int, int], np.ndarray] = {}
    for ia, ib in pair_indices:
        spread = np.zeros(n)
        sigma = cfg.ou_sigma
        for t in range(1, n):
            if cfg.regime_shift_date and dates[t].date().isoformat() >= cfg.regime_shift_date:
                sigma = cfg.ou_sigma * cfg.regime_shift_multiplier
            spread[t] = spread[t - 1] - cfg.ou_theta * spread[t - 1] + rng.normal(0, sigma)
        ou_spreads[(ia, ib)] = spread

    # Generate price paths
    price_paths: dict[str, np.ndarray] = {}
    for i, sym in enumerate(symbols):
        idio = np.exp(np.cumsum(rng.normal(0, cfg.gbm_vol * 0.3, n)))
        prices = base_prices[i] * (common ** betas[i]) * idio
        for (ia, ib), spread in ou_spreads.items():
            if i == ib:
                prices = price_paths[symbols[ia]] * np.exp(-spread) / np.exp(
                    np.log(base_prices[ib]) - np.log(base_prices[ia])
                )
        price_paths[sym] = np.maximum(prices, 0.01)

    # Zipf-distributed volumes AND tick symbol weights (not just volume scaling)
    zipf_weights = np.array([1.0 / (i + 1) ** cfg.skew_zipf_exponent for i in range(len(symbols))])
    zipf_weights /= zipf_weights.sum()
    base_volumes = (zipf_weights * 10_000_000).astype(int) + 100_000

    # Delist schedule
    delistings: dict[str, int] = {}
    if cfg.delist_rate > 0:
        daily_rate = cfg.delist_rate / 252
        non_pair_symbols = [s for s in symbols if not any(s in p for p in cointegrated_pairs)]
        for sym in non_pair_symbols:
            if rng.random() < daily_rate * n:
                delist_day = int(rng.integers(n // 2, n))
                delistings[sym] = delist_day

    # Assemble rows
    rows = []
    for i, sym in enumerate(symbols):
        prices = price_paths[sym]
        delist_day = delistings.get(sym, n)
        for t, ts in enumerate(dates[:delist_day]):
            vol_noise = rng.lognormal(0, 0.3)
            volume = float(base_volumes[i] * vol_noise)
            close = float(prices[t])
            daily_range = close * cfg.gbm_vol * rng.uniform(0.5, 2.0)
            rows.append({
                "symbol": sym,
                "ts": ts,
                "open": max(close - daily_range / 2, 0.01),
                "high": close + daily_range / 2,
                "low": max(close - daily_range / 2, 0.01),
                "close": close,
                "volume": volume,
            })

    bars = pd.DataFrame(rows)
    bars["ts"] = pd.to_datetime(bars["ts"], utc=True)
    bars = bars.sort_values(["symbol", "ts"]).reset_index(drop=True)

    # Symbol master — build SCD2-ready rows with sector change events
    sector_changes: list[dict] = []
    master_rows = []
    for sym in symbols:
        listed_at = dates[0]
        delist_day = delistings.get(sym)
        delisted_at = dates[delist_day] if delist_day and delist_day < len(dates) else None
        original_sector = _sector(sym)

        if (
            cfg.sector_change_date
            and sym in cfg.sector_change_symbols
            and listed_at.date().isoformat() < cfg.sector_change_date
        ):
            # Row 1: original sector, valid until change date
            change_ts = datetime.fromisoformat(cfg.sector_change_date).replace(tzinfo=timezone.utc)
            master_rows.append({
                "symbol": sym,
                "listed_at": listed_at,
                "delisted_at": delisted_at,
                "sector": original_sector,
                "valid_from": listed_at,
                "valid_to": change_ts,
                "is_current": False,
            })
            # Row 2: new sector, currently active
            master_rows.append({
                "symbol": sym,
                "listed_at": listed_at,
                "delisted_at": delisted_at,
                "sector": cfg.sector_change_new_value,
                "valid_from": change_ts,
                "valid_to": None,
                "is_current": True,
            })
            sector_changes.append({
                "symbol": sym,
                "effective_date": cfg.sector_change_date,
                "old_sector": original_sector,
                "new_sector": cfg.sector_change_new_value,
            })
        else:
            master_rows.append({
                "symbol": sym,
                "listed_at": listed_at,
                "delisted_at": delisted_at,
                "sector": original_sector,
                "valid_from": listed_at,
                "valid_to": None,
                "is_current": True,
            })

    master = pd.DataFrame(master_rows)
    master["listed_at"] = pd.to_datetime(master["listed_at"], utc=True)
    master["delisted_at"] = pd.to_datetime(master["delisted_at"], utc=True)
    master["valid_from"] = pd.to_datetime(master["valid_from"], utc=True)
    master["valid_to"] = pd.to_datetime(master["valid_to"], utc=True)

    # Expose zipf_weights so ingest_to_platform can build a weighted tick sampler
    bars.attrs["zipf_weights"] = dict(zip(symbols, zipf_weights.tolist()))

    return bars, master, cointegrated_pairs, sector_changes
