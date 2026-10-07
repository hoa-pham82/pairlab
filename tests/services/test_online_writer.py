"""Tests for stream job 2: Kafka bars into the Feast online store."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st
from services.online_writer import writer
from services.online_writer.writer import (
    FEATURE_VIEW,
    latest_per_symbol,
    parse_bar,
    process_batch,
    run,
)


def bar(symbol="KO", minute=0, close=1.0, **overrides) -> dict:
    """A Flink-formatted bar (TIMESTAMP(3) serialized in SQL format)."""
    return {
        "symbol": symbol,
        "window_start": f"2026-10-07 12:{minute:02d}:00",
        "window_end": f"2026-10-07 12:{minute + 1:02d}:00",
        "open": 1.0, "high": 2.0, "low": 0.5, "close": close, "volume": 10.0, "tick_count": 3,
        **overrides,
    }


def raw(**kwargs) -> bytes:
    return json.dumps(bar(**kwargs)).encode()


class FakeStore:
    def __init__(self, fail: bool = False):
        self.writes: list[tuple[str, pd.DataFrame]] = []
        self.fail = fail

    def write_to_online_store(self, feature_view_name, df):
        if self.fail:
            raise ConnectionError("redis down")
        self.writes.append((feature_view_name, df))


class FakeConsumer:
    def __init__(self, batches: list[list[bytes]]):
        self.batches = list(batches)
        self.commits = 0
        self.polls = 0

    def poll(self, timeout_ms, max_records):
        self.polls += 1
        if not self.batches:
            return {}
        return {"p0": [SimpleNamespace(value=v) for v in self.batches.pop(0)]}

    def commit(self):
        self.commits += 1


class TestParseBar:
    def test_valid_bar_is_typed_and_utc(self):
        row = parse_bar(raw())
        assert row["event_timestamp"] == pd.Timestamp("2026-10-07 12:00:00", tz="UTC")
        assert row["tick_count"] == 3 and isinstance(row["tick_count"], int)
        assert row["close"] == 1.0

    def test_iso_timestamp_with_zone_is_accepted(self):
        row = parse_bar(raw(window_start="2026-10-07T14:00:00+02:00"))
        assert row["event_timestamp"] == pd.Timestamp("2026-10-07 12:00:00", tz="UTC")

    # Partitions of malformed input: not JSON, not an object, missing field, wrong type,
    # empty/absent key, null timestamp.
    @pytest.mark.parametrize(
        "payload",
        [
            b"not json",
            b"[1, 2]",
            json.dumps({k: v for k, v in bar().items() if k != "close"}).encode(),
            raw(volume="lots"),
            raw(volume=None),
            raw(symbol=""),
            raw(symbol=None),
            raw(window_start=None),
            raw(window_start="yesterday-ish"),
        ],
        ids=["not-json", "array", "missing-close", "text-volume", "null-volume",
             "empty-symbol", "null-symbol", "null-ts", "bad-ts"],
    )
    def test_malformed_is_none(self, payload):
        assert parse_bar(payload) is None


class TestLatestPerSymbol:
    def test_empty(self):
        assert latest_per_symbol([]).empty

    def test_late_bar_does_not_replace_newer_one(self):
        rows = [parse_bar(raw(minute=5, close=5.0)), parse_bar(raw(minute=3, close=3.0))]
        frame = latest_per_symbol(rows)
        assert frame["close"].tolist() == [5.0]

    @given(st.lists(st.tuples(st.sampled_from(["KO", "PEP", "XOM"]), st.integers(0, 58)),
                    min_size=1, max_size=30))
    def test_one_row_per_symbol_holding_its_newest_minute(self, keys):
        rows = [parse_bar(raw(symbol=s, minute=m, close=float(m))) for s, m in keys]
        frame = latest_per_symbol(rows)
        expected = {}
        for s, m in keys:
            expected[s] = max(expected.get(s, -1), m)
        assert dict(zip(frame["symbol"], frame["close"], strict=True)) == {
            s: float(m) for s, m in expected.items()
        }


class TestProcessBatch:
    def test_writes_the_feature_view_and_skips_bad_messages(self):
        store = FakeStore()
        before = writer.BAD_MESSAGES._value.get()
        written = process_batch([raw(symbol="KO"), b"junk", raw(symbol="PEP")], store)
        assert written == 2
        name, frame = store.writes[0]
        assert name == FEATURE_VIEW
        assert set(frame["symbol"]) == {"KO", "PEP"}
        assert writer.BAD_MESSAGES._value.get() == before + 1

    def test_all_bad_writes_nothing(self):
        store = FakeStore()
        assert process_batch([b"junk"], store) == 0
        assert store.writes == []


class TestRun:
    def test_commits_once_per_non_empty_batch(self):
        consumer = FakeConsumer([[raw()], [], [raw(symbol="PEP")]])
        store = FakeStore()
        run(consumer, store, should_stop=lambda: consumer.polls >= 3)
        assert consumer.commits == 2
        assert len(store.writes) == 2

    def test_failed_write_is_not_committed(self):
        consumer = FakeConsumer([[raw()]])
        with pytest.raises(ConnectionError):
            run(consumer, FakeStore(fail=True), should_stop=lambda: consumer.polls >= 1)
        assert consumer.commits == 0


def test_main_wires_kafka_and_feast_from_the_environment(monkeypatch):
    import sys

    created = {}

    class Kafka:
        def __init__(self, topic, **kwargs):
            created["topic"], created["kafka"] = topic, kwargs

    class Feast:
        def __init__(self, repo_path):
            created["repo"] = repo_path

    monkeypatch.setitem(sys.modules, "kafka", SimpleNamespace(KafkaConsumer=Kafka))
    monkeypatch.setitem(sys.modules, "feast", SimpleNamespace(FeatureStore=Feast))
    monkeypatch.setattr(writer, "start_http_server", lambda port: created.setdefault("port", port))
    monkeypatch.setattr(writer, "run", lambda consumer, store: created.setdefault("ran", True))
    monkeypatch.setenv("FEAST_REPO", "/repo")
    writer.main()
    assert created["topic"] == "bars.1m"
    assert created["kafka"]["enable_auto_commit"] is False
    assert created["repo"] == "/repo"
    assert created["ran"] and created["port"] == 9108
