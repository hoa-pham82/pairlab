"""Regime assessment for a pair: cointegration p-value and spread PSI."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from ml.drift import population_stability_index, spread_change_psi

from pairlab.strategy.stats import check_cointegration


class Regime(StrEnum):
    """How healthy the pair relationship looks."""

    STABLE = "stable"
    SHIFTING = "shifting"
    BROKEN = "broken"


@dataclass(frozen=True)
class RegimeThresholds:
    """Cut-offs above which a pair counts as shifting or broken."""

    pvalue_shifting: float = 0.05
    pvalue_broken: float = 0.10
    psi_shifting: float = 0.20
    psi_broken: float = 0.40


@dataclass(frozen=True)
class RegimeAssessment:
    """Result of assessing one pair."""

    regime: Regime
    coint_pvalue: float
    psi: float
    bars_used: int


def classify(
    coint_pvalue: float, psi: float, thresholds: RegimeThresholds | None = None
) -> Regime:
    """Map the two statistics to a regime; the worse of the two wins."""
    t = thresholds or RegimeThresholds()
    if coint_pvalue > t.pvalue_broken or psi > t.psi_broken:
        return Regime.BROKEN
    if coint_pvalue > t.pvalue_shifting or psi > t.psi_shifting:
        return Regime.SHIFTING
    return Regime.STABLE


def assess_pair(
    close_a: np.ndarray,
    close_b: np.ndarray,
    reference_bars: int = 252,
    current_bars: int = 63,
    thresholds: RegimeThresholds | None = None,
) -> RegimeAssessment:
    """Assess a pair from its closing prices, oldest first.

    The cointegration test runs on the latest ``reference_bars``. PSI compares
    daily spread changes over the latest ``current_bars`` with those of the
    ``reference_bars`` before them, using the hedge ratio of that reference
    window. Changes are used because spread levels are strongly autocorrelated.

    Raises:
        ValueError: on mismatched lengths, too little history, or bad prices.
    """
    close_a = np.asarray(close_a, dtype=float)
    close_b = np.asarray(close_b, dtype=float)
    needed = reference_bars + current_bars
    if len(close_a) != len(close_b):
        raise ValueError("price series must have the same length")
    if len(close_a) < needed:
        raise ValueError(f"need {needed} bars, got {len(close_a)}")
    if not (np.all(np.isfinite(close_a)) and np.all(np.isfinite(close_b))):
        raise ValueError("prices must be finite")
    if np.any(close_a <= 0) or np.any(close_b <= 0):
        raise ValueError("prices must be positive")

    log_a, log_b = np.log(close_a[-needed:]), np.log(close_b[-needed:])
    pvalue = check_cointegration(log_a[-reference_bars:], log_b[-reference_bars:]).pvalue

    psi = spread_change_psi(log_a, log_b, reference_bars)

    return RegimeAssessment(classify(pvalue, psi, thresholds), pvalue, psi, needed)


__all__ = [
    "Regime",
    "RegimeAssessment",
    "RegimeThresholds",
    "assess_pair",
    "classify",
    "population_stability_index",
]
