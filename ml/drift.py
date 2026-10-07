"""Data-drift check on pair features: PSI of spread changes and of z-scores.

Usage:
    uv run python -m ml.drift [--pushgateway http://localhost:9091]

Prints ``drift`` or ``stable`` as its last line, so a scheduler can branch on it.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

PSI_BINS = 5
PSI_DRIFT_THRESHOLD = 0.40
_EPSILON = 1e-4


@dataclass(frozen=True)
class DriftReport:
    """PSI per pair (spread changes) and overall (z-score), and the verdict."""

    spread_psi: dict[str, float]
    zscore_psi: float
    threshold: float

    @property
    def drifted_pairs(self) -> list[str]:
        return sorted(pair for pair, psi in self.spread_psi.items() if psi > self.threshold)

    @property
    def drifted(self) -> bool:
        return bool(self.drifted_pairs)


def population_stability_index(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """PSI between two samples, using quantile bins of ``reference``.

    0 means identical distributions; larger means more drift.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    if len(reference) == 0 or len(current) == 0:
        raise ValueError("PSI needs non-empty samples")
    if bins < 2:
        raise ValueError("bins must be >= 2")

    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)[1:-1]))
    ref_share = _bin_shares(reference, edges)
    cur_share = _bin_shares(current, edges)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def _bin_shares(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Share of values per bin, floored so empty bins do not give log(0)."""
    counts = np.bincount(np.searchsorted(edges, values, side="left"), minlength=len(edges) + 1)
    return np.maximum(counts / len(values), _EPSILON)


def spread_change_psi(log_a: np.ndarray, log_b: np.ndarray, reference_bars: int) -> float:
    """PSI of daily spread changes: bars after ``reference_bars`` vs the reference bars.

    The spread uses the OLS hedge ratio of the reference window. Changes are
    compared because spread levels are strongly autocorrelated.
    """
    beta = np.polyfit(log_b[:reference_bars], log_a[:reference_bars], 1)[0]
    changes = np.diff(log_a - beta * log_b)
    split = reference_bars - 1
    return population_stability_index(changes[:split], changes[split:], PSI_BINS)


def drift_report(
    features: pd.DataFrame,
    reference_bars: int = 252,
    current_bars: int = 63,
    threshold: float = PSI_DRIFT_THRESHOLD,
) -> DriftReport:
    """Compare each pair's latest ``current_bars`` with the ``reference_bars`` before them.

    Args:
        features: rows with ``symbol_a``, ``symbol_b``, ``event_timestamp``,
            ``close_a``, ``close_b``, ``zscore``.

    Pairs with less history than both windows are skipped.
    """
    needed = reference_bars + current_bars
    spread_psi: dict[str, float] = {}
    reference_z: list[np.ndarray] = []
    current_z: list[np.ndarray] = []

    for (sym_a, sym_b), pair in features.groupby(["symbol_a", "symbol_b"], sort=True):
        recent = pair.sort_values("event_timestamp").tail(needed)
        if len(recent) < needed:
            continue
        log_a = np.log(recent["close_a"].to_numpy(dtype=float))
        log_b = np.log(recent["close_b"].to_numpy(dtype=float))
        spread_psi[f"{sym_a}__{sym_b}"] = spread_change_psi(log_a, log_b, reference_bars)
        z = recent["zscore"].to_numpy(dtype=float)
        reference_z.append(z[:reference_bars][np.isfinite(z[:reference_bars])])
        current_z.append(z[reference_bars:][np.isfinite(z[reference_bars:])])

    zscore_psi = float("nan")
    if reference_z and sum(map(len, reference_z)) and sum(map(len, current_z)):
        zscore_psi = population_stability_index(
            np.concatenate(reference_z), np.concatenate(current_z), PSI_BINS
        )
    return DriftReport(spread_psi, zscore_psi, threshold)


def push_report(report: DriftReport, gateway: str, job: str = "pairlab_drift") -> None:
    """Send the report to a Prometheus Pushgateway."""
    from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

    registry = CollectorRegistry()
    spread = Gauge(
        "pairlab_spread_change_psi", "PSI of daily spread changes", ["pair_id"], registry=registry
    )
    for pair, psi in report.spread_psi.items():
        spread.labels(pair).set(psi)
    if not np.isnan(report.zscore_psi):
        Gauge("pairlab_zscore_psi", "PSI of z-scores, all pairs", registry=registry).set(
            report.zscore_psi
        )
    Gauge("pairlab_drift_detected", "1 if any pair drifted", registry=registry).set(
        int(report.drifted)
    )
    Gauge(
        "pairlab_drift_last_run_seconds", "Unix time of the check", registry=registry
    ).set_to_current_time()
    push_to_gateway(gateway, job=job, registry=registry)


def read_pair_features(dsn: str) -> pd.DataFrame:
    """Load the drift inputs from the offline feature table gold.feat_pair_daily."""
    import psycopg2

    sql = (
        "SELECT symbol_a, symbol_b, event_timestamp, close_a, close_b, zscore "
        "FROM gold.feat_pair_daily"
    )
    with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(sql)
        return pd.DataFrame(cur.fetchall(), columns=[c.name for c in cur.description])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dsn",
        default=os.environ.get(
            "POSTGRES_URL", "postgresql://pairlab:pairlab@localhost:5432/pairlab"
        ),
    )
    parser.add_argument("--pushgateway", default=os.environ.get("PUSHGATEWAY_URL"))
    args = parser.parse_args()

    result = drift_report(read_pair_features(args.dsn))
    for pair_id, value in result.spread_psi.items():
        print(f"{pair_id}: spread-change PSI {value:.3f}")
    print(f"z-score PSI {result.zscore_psi:.3f}; drifted pairs: {result.drifted_pairs}")
    if args.pushgateway:
        push_report(result, args.pushgateway)
    print("drift" if result.drifted else "stable")
