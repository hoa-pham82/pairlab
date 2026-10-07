"""Meta-labels: did a pairs entry signal revert before stopping out?"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ID_COLUMN = "pair_date_id"
LABEL_COLUMN = "label"


@dataclass(frozen=True)
class LabelConfig:
    """Thresholds that define an entry signal and its outcome."""

    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 4.0
    horizon_bars: int = 20

    def __post_init__(self) -> None:
        if not 0 <= self.exit_z < self.entry_z < self.stop_z:
            raise ValueError("thresholds must satisfy 0 <= exit_z < entry_z < stop_z")
        if self.horizon_bars < 1:
            raise ValueError("horizon_bars must be >= 1")


def pair_date_id(symbol_a: str, symbol_b: str, ts: pd.Timestamp) -> str:
    """Build the label key, e.g. ``KO__PEP|2020-03-02``."""
    return f"{symbol_a}__{symbol_b}|{pd.Timestamp(ts).date().isoformat()}"


def label_entries(zscore: pd.Series, cfg: LabelConfig | None = None) -> pd.Series:
    """Label every entry bar of one pair's z-score series.

    An entry bar is one where |z| moves above ``entry_z``. Its label is 1 if
    |z| falls below ``exit_z`` within ``horizon_bars`` before exceeding
    ``stop_z``, else 0. Entries too close to the end to be resolved are dropped.

    Returns:
        Series of 0/1 indexed by the entry bars' index values.
    """
    cfg = cfg or LabelConfig()
    abs_z = np.abs(zscore.to_numpy(dtype=float))
    n = len(abs_z)
    labels: dict[object, int] = {}

    for i in range(n):
        in_zone = abs_z[i] > cfg.entry_z
        was_in_zone = i > 0 and abs_z[i - 1] > cfg.entry_z
        if not in_zone or was_in_zone:
            continue
        outcome = _outcome(abs_z, i, cfg)
        if outcome is not None:
            labels[zscore.index[i]] = outcome

    return pd.Series(labels, name=LABEL_COLUMN, dtype="int64")


def _outcome(abs_z: np.ndarray, entry: int, cfg: LabelConfig) -> int | None:
    """Return 1 (reverted), 0 (stopped or timed out), or None (not yet known)."""
    last = min(entry + cfg.horizon_bars, len(abs_z) - 1)
    for j in range(entry + 1, last + 1):
        if abs_z[j] > cfg.stop_z:
            return 0
        if abs_z[j] < cfg.exit_z:
            return 1
    horizon_complete = entry + cfg.horizon_bars <= len(abs_z) - 1
    return 0 if horizon_complete else None


def build_label_table(features: pd.DataFrame, cfg: LabelConfig | None = None) -> pd.DataFrame:
    """Build the two-column label table from pair features.

    Args:
        features: rows with ``symbol_a``, ``symbol_b``, ``event_timestamp``, ``zscore``.

    Returns:
        DataFrame with columns ``pair_date_id`` and ``label``, one row per entry signal.
    """
    rows: list[tuple[str, int]] = []
    for (sym_a, sym_b), pair in features.groupby(["symbol_a", "symbol_b"], sort=True):
        z = pair.sort_values("event_timestamp").set_index("event_timestamp")["zscore"]
        for ts, label in label_entries(z, cfg).items():
            rows.append((pair_date_id(sym_a, sym_b, ts), int(label)))
    return pd.DataFrame(rows, columns=[ID_COLUMN, LABEL_COLUMN])
