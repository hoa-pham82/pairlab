"""Inject data problems: duplicates, schema evolution, NaN prices."""

from __future__ import annotations

import numpy as np
import pandas as pd


def inject_duplicates(df: pd.DataFrame, rate: float, seed: int = 0) -> pd.DataFrame:
    """Duplicate approximately `rate` fraction of rows."""
    if rate <= 0:
        return df
    rng = np.random.default_rng(seed)
    n_dups = max(1, int(len(df) * rate))
    dup_idx = rng.choice(len(df), size=n_dups, replace=False)
    dups = df.iloc[dup_idx].copy()
    return pd.concat([df, dups], ignore_index=True)


def apply_schema_evolution(
    df: pd.DataFrame,
    change_date: str,
    seed: int = 0,
) -> pd.DataFrame:
    """Simulate schema evolution:
    - Before change_date: no adj_close column (v1)
    - After change_date: add adj_close = close * adjustment_factor (v2)
    - 0.5% of rows after change_date have 'vol' instead of 'volume' (v3 breaking)
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    change_ts = pd.Timestamp(change_date, tz="UTC")

    after_mask = df["ts"] >= change_ts
    df.loc[after_mask, "adj_close"] = df.loc[after_mask, "close"] * rng.uniform(0.995, 1.005, after_mask.sum())

    # v3-breaking: rename 'volume' to 'vol' in a small fraction of post-change rows
    breaking_mask = after_mask & (rng.random(len(df)) < 0.005)
    if breaking_mask.any():
        df.loc[breaking_mask, "vol"] = df.loc[breaking_mask, "volume"]
        df.loc[breaking_mask, "volume"] = np.nan

    return df


def inject_skew(df: pd.DataFrame) -> pd.DataFrame:
    """Volume skew is already embedded by the generator; this is a no-op marker."""
    return df
