"""Training data: labels joined to Feast features, and a leakage-safe time split."""

from __future__ import annotations

import pandas as pd
from pandas.tseries.offsets import BDay

from ml.labels import ID_COLUMN, LABEL_COLUMN

ENTITY_COLUMN = "symbol_pair"
TIME_COLUMN = "event_timestamp"
PAIR_FEATURES = ["zscore", "hedge_ratio", "spread_vol", "correlation_60d"]
PAIR_FEATURE_VIEW = "pair_daily_fv"


def entity_frame(labels: pd.DataFrame) -> pd.DataFrame:
    """Split ``pair_date_id`` into the Feast entity key and event timestamp."""
    parts = labels[ID_COLUMN].str.split("|", n=1, expand=True)
    return labels.assign(
        **{ENTITY_COLUMN: parts[0], TIME_COLUMN: pd.to_datetime(parts[1])}
    )


def load_training_frame(store, labels: pd.DataFrame) -> pd.DataFrame:
    """Join labels with point-in-time pair features from the Feast offline store.

    Args:
        store: a ``feast.FeatureStore`` (or anything with ``get_historical_features``).
        labels: the two-column label table.
    """
    refs = [f"{PAIR_FEATURE_VIEW}:{name}" for name in PAIR_FEATURES]
    frame = store.get_historical_features(entity_df=entity_frame(labels), features=refs).to_df()
    return frame.sort_values([TIME_COLUMN, ENTITY_COLUMN]).reset_index(drop=True)


def purged_time_split(
    frame: pd.DataFrame, valid_fraction: float = 0.25, horizon_bars: int = 20
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by time, dropping training rows whose label window reaches validation.

    Validation is the latest ``valid_fraction`` of rows. A training row is kept
    only if its entry date plus ``horizon_bars`` business days is before the
    first validation date.

    Returns:
        (train, valid)
    """
    if not 0 < valid_fraction < 1:
        raise ValueError("valid_fraction must be between 0 and 1")
    if frame.empty:
        return frame.copy(), frame.copy()

    ordered = frame.sort_values(TIME_COLUMN).reset_index(drop=True)
    first_valid = int(len(ordered) * (1 - valid_fraction))
    valid_start = ordered[TIME_COLUMN].iloc[first_valid]

    valid = ordered[ordered[TIME_COLUMN] >= valid_start]
    label_end = ordered[TIME_COLUMN] + BDay(horizon_bars)
    train = ordered[label_end < valid_start]
    return train.reset_index(drop=True), valid.reset_index(drop=True)


def features_and_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return the model inputs and the 0/1 target."""
    return frame[PAIR_FEATURES], frame[LABEL_COLUMN].astype(int)
