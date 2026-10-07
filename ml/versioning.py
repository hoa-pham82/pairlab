"""Incremental data versioning: snapshot the training frame as a Delta table."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from deltalake import DeltaTable, write_deltalake

from ml.dataset import TIME_COLUMN
from ml.labels import ID_COLUMN


@dataclass(frozen=True)
class DataVersion:
    """Delta table version holding a training frame, and what changed to reach it."""

    version: int
    rows_inserted: int
    rows_updated: int
    rows_deleted: int

    @property
    def changed(self) -> bool:
        return bool(self.rows_inserted or self.rows_updated or self.rows_deleted)


def snapshot_training_frame(
    frame: pd.DataFrame, table_uri: str, storage_options: dict[str, str] | None = None
) -> DataVersion:
    """Store ``frame`` as the next version of a Delta table, writing only the changes.

    Rows are keyed by ``pair_date_id``. An unchanged frame creates no new version.

    Args:
        table_uri: local path or ``s3://`` URI of the Delta table.
        storage_options: object-store settings passed to delta-rs.
    """
    frame = _normalise(frame)
    if not DeltaTable.is_deltatable(table_uri, storage_options):
        write_deltalake(table_uri, frame, storage_options=storage_options)
        return DataVersion(0, rows_inserted=len(frame), rows_updated=0, rows_deleted=0)

    table = DeltaTable(table_uri, storage_options=storage_options)
    stored = _normalise(table.to_pandas())
    upserts = _rows_not_in(frame, stored)
    updated = int(upserts[ID_COLUMN].isin(stored[ID_COLUMN]).sum())
    removed_ids = sorted(set(stored[ID_COLUMN]) - set(frame[ID_COLUMN]))

    if not upserts.empty:
        (
            table.merge(
                upserts,
                predicate=f"target.{ID_COLUMN} = source.{ID_COLUMN}",
                source_alias="source",
                target_alias="target",
            )
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute()
        )
    if removed_ids:
        quoted = ", ".join("'" + i.replace("'", "''") + "'" for i in removed_ids)
        table.delete(f"{ID_COLUMN} IN ({quoted})")

    return DataVersion(
        version=DeltaTable(table_uri, storage_options=storage_options).version(),
        rows_inserted=len(upserts) - updated,
        rows_updated=updated,
        rows_deleted=len(removed_ids),
    )


def load_training_frame_version(
    table_uri: str, version: int, storage_options: dict[str, str] | None = None
) -> pd.DataFrame:
    """Read the training frame exactly as it was at ``version`` (time travel)."""
    table = DeltaTable(table_uri, version=version, storage_options=storage_options)
    return _normalise(table.to_pandas())


def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
    """Give frames from pandas and from Delta the same dtypes, order, and index."""
    out = frame.copy()
    out[TIME_COLUMN] = pd.to_datetime(out[TIME_COLUMN], utc=True).astype("datetime64[us, UTC]")
    return out.sort_values(ID_COLUMN).reset_index(drop=True)


def _rows_not_in(new: pd.DataFrame, old: pd.DataFrame) -> pd.DataFrame:
    """Rows of ``new`` with no identical row in ``old``."""
    columns = list(new.columns)
    flagged = new.merge(old[columns].drop_duplicates(), how="left", on=columns, indicator=True)
    return new[(flagged["_merge"] == "left_only").to_numpy()]
