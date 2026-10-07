"""Read pair features from the warehouse and write gold.label_pair_reversion.

Usage:
    uv run python -m ml.build_labels [--horizon-bars 20]
"""

from __future__ import annotations

import argparse
import os

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

from ml.labels import ID_COLUMN, LABEL_COLUMN, LabelConfig, build_label_table

DEFAULT_DSN = "postgresql://pairlab:pairlab@localhost:5432/pairlab"
LABEL_TABLE = "gold.label_pair_reversion"

_FEATURES_SQL = "SELECT symbol_a, symbol_b, event_timestamp, zscore FROM gold.feat_pair_daily"
_DDL = f"""
    CREATE TABLE IF NOT EXISTS {LABEL_TABLE} (
        {ID_COLUMN} TEXT PRIMARY KEY,
        {LABEL_COLUMN} SMALLINT NOT NULL
    )
"""


def read_pair_features(conn) -> pd.DataFrame:
    """Load the columns needed for labeling from gold.feat_pair_daily."""
    with conn.cursor() as cur:
        cur.execute(_FEATURES_SQL)
        columns = [c.name for c in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=columns)


def write_label_table(conn, labels: pd.DataFrame) -> int:
    """Replace the label table contents with ``labels``. Returns rows written."""
    rows = list(labels[[ID_COLUMN, LABEL_COLUMN]].itertuples(index=False, name=None))
    with conn.cursor() as cur:
        cur.execute(_DDL)
        cur.execute(f"TRUNCATE {LABEL_TABLE}")
        execute_values(
            cur, f"INSERT INTO {LABEL_TABLE} ({ID_COLUMN}, {LABEL_COLUMN}) VALUES %s", rows
        )
    conn.commit()
    return len(rows)


def run(dsn: str, cfg: LabelConfig) -> pd.DataFrame:
    """Build labels from warehouse features and store them."""
    with psycopg2.connect(dsn) as conn:
        labels = build_label_table(read_pair_features(conn), cfg)
        written = write_label_table(conn, labels)
    positive = labels[LABEL_COLUMN].mean() if written else float("nan")
    print(f"Wrote {written} labels to {LABEL_TABLE} (positive rate {positive:.3f})")
    return labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsn", default=os.environ.get("POSTGRES_URL", DEFAULT_DSN))
    parser.add_argument("--horizon-bars", type=int, default=LabelConfig.horizon_bars)
    args = parser.parse_args()
    run(args.dsn, LabelConfig(horizon_bars=args.horizon_bars))
