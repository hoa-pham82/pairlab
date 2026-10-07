"""Tests for incremental Delta snapshots of the training frame."""

from __future__ import annotations

import pandas as pd
import pytest
from deltalake import DeltaTable
from ml.versioning import load_training_frame_version, snapshot_training_frame


def _frame(n: int, zscore: float = 2.5) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame(
        {
            "pair_date_id": [f"KO__PEP|{d.date()}" for d in dates],
            "label": [i % 2 for i in range(n)],
            "symbol_pair": "KO__PEP",
            "event_timestamp": dates,
            "zscore": zscore,
        }
    )


@pytest.fixture
def uri(tmp_path) -> str:
    return str(tmp_path / "training_frame")


# ---------------------------------------------------------------------------
# Equivalence partitions for snapshot_training_frame:
#   EP1: table does not exist      → version 0, all rows inserted
#   EP2: identical frame           → no new version
#   EP3: new rows only             → new version, only those rows inserted
#   EP4: existing row changed      → new version, row updated
#   EP5: row no longer in frame    → new version, row deleted
# ---------------------------------------------------------------------------


def test_first_snapshot_is_version_zero(uri):
    result = snapshot_training_frame(_frame(5), uri)
    assert (result.version, result.rows_inserted, result.changed) == (0, 5, True)


def test_identical_frame_creates_no_new_version(uri):
    snapshot_training_frame(_frame(5), uri)
    result = snapshot_training_frame(_frame(5), uri)
    assert result.version == 0
    assert not result.changed


def test_row_order_does_not_count_as_a_change(uri):
    snapshot_training_frame(_frame(5), uri)
    result = snapshot_training_frame(_frame(5).iloc[::-1], uri)
    assert not result.changed


@pytest.mark.parametrize("extra_rows", [1, 3])
def test_new_rows_are_the_only_rows_written(uri, extra_rows):
    snapshot_training_frame(_frame(5), uri)
    result = snapshot_training_frame(_frame(5 + extra_rows), uri)
    assert (result.rows_inserted, result.rows_updated, result.rows_deleted) == (extra_rows, 0, 0)
    assert result.version == 1
    added = DeltaTable(uri).history(1)[0]["operationMetrics"]
    assert int(added["num_target_rows_inserted"]) == extra_rows
    assert int(added["num_target_files_removed"]) == 0


def test_changed_row_is_updated(uri):
    snapshot_training_frame(_frame(5), uri)
    changed = _frame(5)
    changed.loc[2, "zscore"] = -3.0
    result = snapshot_training_frame(changed, uri)
    assert (result.rows_inserted, result.rows_updated, result.rows_deleted) == (0, 1, 0)
    assert load_training_frame_version(uri, result.version)["zscore"].tolist()[2] == -3.0


def test_removed_row_is_deleted(uri):
    snapshot_training_frame(_frame(5), uri)
    result = snapshot_training_frame(_frame(4), uri)
    assert (result.rows_inserted, result.rows_updated, result.rows_deleted) == (0, 0, 1)
    assert len(load_training_frame_version(uri, result.version)) == 4


def test_old_versions_stay_readable(uri):
    """Time travel: version 0 still returns the first frame after later changes."""
    snapshot_training_frame(_frame(5), uri)
    latest = snapshot_training_frame(_frame(8), uri)
    assert len(load_training_frame_version(uri, 0)) == 5
    assert len(load_training_frame_version(uri, latest.version)) == 8


def test_round_trip_preserves_values(uri):
    frame = _frame(5)
    snapshot_training_frame(frame, uri)
    stored = load_training_frame_version(uri, 0)
    assert stored["pair_date_id"].tolist() == frame["pair_date_id"].tolist()
    assert stored["label"].tolist() == frame["label"].tolist()
    assert stored["event_timestamp"].dt.tz_localize(None).tolist() == frame[
        "event_timestamp"
    ].tolist()
