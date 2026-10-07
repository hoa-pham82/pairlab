"""Statistical tools for pairs trading: hedge ratio, cointegration, half-life, z-score."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint


@dataclass
class CointResult:
    """Output of a cointegration test."""
    pvalue: float
    hedge_ratio: float   # β from OLS: log(A) = α + β·log(B)
    half_life: float     # mean-reversion half-life in bars


def estimate_hedge_ratio(log_a: np.ndarray, log_b: np.ndarray) -> float:
    """OLS estimate of β in log(A) = α + β·log(B).

    Returns:
        hedge_ratio β (slope of regression).
    """
    if len(log_a) < 2 or np.std(log_b) == 0:
        return 1.0
    X = np.column_stack([np.ones(len(log_b)), log_b])
    coef, *_ = np.linalg.lstsq(X, log_a, rcond=None)
    return float(coef[1])


def check_cointegration(log_a: np.ndarray, log_b: np.ndarray) -> CointResult:
    """Engle-Granger cointegration test with OLS hedge ratio and OU half-life.

    Args:
        log_a: log-prices of asset A.
        log_b: log-prices of asset B.

    Returns:
        CointResult with pvalue, hedge_ratio, and half_life.
    """
    if len(log_a) < 20:
        return CointResult(pvalue=1.0, hedge_ratio=1.0, half_life=np.inf)

    beta = estimate_hedge_ratio(log_a, log_b)
    spread = log_a - beta * log_b

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, pvalue, _ = coint(log_a, log_b)

    hl = estimate_half_life(spread)
    return CointResult(pvalue=float(pvalue), hedge_ratio=beta, half_life=hl)


def estimate_half_life(spread: np.ndarray) -> float:
    """Ornstein–Uhlenbeck half-life: −log(2) / log(1 + θ̂).

    θ̂ is the AR(1) mean-reversion coefficient from OLS on lagged spread.
    Returns inf if the series is not mean-reverting (θ̂ >= 0).
    """
    if len(spread) < 3:
        return np.inf
    y = np.diff(spread)
    x = spread[:-1] - spread[:-1].mean()
    if np.std(x) == 0:
        return np.inf
    X = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    theta = float(coef[1])
    if theta >= 0:
        return np.inf  # random walk or trend; not mean-reverting
    return float(-np.log(2) / np.log(1 + theta))


def rolling_zscore(series: np.ndarray, window: int) -> np.ndarray:
    """Rolling z-score: (x − mean) / std over the last `window` bars.

    Returns NaN where std == 0 or fewer than `window` observations exist.
    """
    out = np.full(len(series), np.nan)
    for i in range(window - 1, len(series)):
        window_data = series[i - window + 1 : i + 1]
        mu = window_data.mean()
        sigma = window_data.std(ddof=1)
        if sigma == 0 or np.isnan(sigma):
            continue
        out[i] = (series[i] - mu) / sigma
    return out


def compute_spread(log_a: np.ndarray, log_b: np.ndarray, beta: float) -> np.ndarray:
    """Spread series: log(A) − β·log(B)."""
    return log_a - beta * log_b
