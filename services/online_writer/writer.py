"""Consume `bars.1m` from Kafka and write the latest bar per symbol to the Feast online store.

Job 1 (Flink SQL) writes the same bars to the offline store (`silver.stg_bars_1m`); this
job keeps `bars_1m_fv` fresh in Redis within seconds instead of waiting for materialize.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Iterable
from typing import Protocol

import pandas as pd
from prometheus_client import Counter, Gauge, start_http_server

FEATURE_VIEW = "bars_1m_fv"
FEATURES = ("open", "high", "low", "close", "volume", "tick_count")

log = logging.getLogger(__name__)
ROWS_WRITTEN = Counter("online_writer_rows_total", "Bars written to the online store")
BAD_MESSAGES = Counter("online_writer_bad_messages_total", "Messages skipped as malformed")
LAST_BAR_TS = Gauge("online_writer_last_bar_timestamp_seconds", "window_start of the newest bar")


class OnlineStore(Protocol):
    """The part of feast.FeatureStore this job uses."""

    def write_to_online_store(self, feature_view_name: str, df: pd.DataFrame) -> None:
        """Persist rows of ``feature_view_name`` to the online store."""


class Consumer(Protocol):
    """The part of kafka.KafkaConsumer this job uses."""

    def poll(self, timeout_ms: int, max_records: int) -> dict:
        """Return {partition: [records]} fetched within ``timeout_ms``."""

    def commit(self) -> None:
        """Commit the offsets of the records returned so far."""


def parse_bar(raw: bytes) -> dict | None:
    """Decode one Flink JSON bar; return None when it is malformed."""
    try:
        bar = json.loads(raw)
        row = {"symbol": bar["symbol"],
               "event_timestamp": pd.to_datetime(bar["window_start"], utc=True)}
        row.update({name: float(bar[name]) for name in FEATURES})
    except (ValueError, KeyError, TypeError):
        return None
    if not isinstance(row["symbol"], str) or not row["symbol"] or pd.isna(row["event_timestamp"]):
        return None
    row["tick_count"] = int(row["tick_count"])
    return row


def latest_per_symbol(rows: Iterable[dict]) -> pd.DataFrame:
    """Keep only the newest bar of each symbol, so a late bar never overwrites a newer one."""
    frame = pd.DataFrame(list(rows), columns=["symbol", "event_timestamp", *FEATURES])
    if frame.empty:
        return frame
    frame = frame.sort_values("event_timestamp").drop_duplicates("symbol", keep="last")
    return frame.reset_index(drop=True)


def process_batch(raw_messages: Iterable[bytes], store: OnlineStore) -> int:
    """Write one batch to the online store and return how many rows were written."""
    rows = []
    for raw in raw_messages:
        bar = parse_bar(raw)
        if bar is None:
            BAD_MESSAGES.inc()
        else:
            rows.append(bar)
    frame = latest_per_symbol(rows)
    if frame.empty:
        return 0
    store.write_to_online_store(FEATURE_VIEW, frame)
    ROWS_WRITTEN.inc(len(frame))
    LAST_BAR_TS.set(frame["event_timestamp"].max().timestamp())
    return len(frame)


def run(consumer: Consumer, store: OnlineStore, should_stop: Callable[[], bool] = lambda: False,
        max_records: int = 500) -> None:
    """Poll, write, then commit: a crash before commit replays the batch (at-least-once)."""
    while not should_stop():
        batches = consumer.poll(timeout_ms=1000, max_records=max_records)
        raw = [record.value for records in batches.values() for record in records]
        if raw:
            written = process_batch(raw, store)
            consumer.commit()
            log.info("wrote %d bars from %d messages", written, len(raw))


def main() -> None:
    """Connect to Kafka and Feast from the environment and run forever."""
    from feast import FeatureStore
    from kafka import KafkaConsumer

    logging.basicConfig(level=logging.INFO)
    start_http_server(int(os.environ.get("METRICS_PORT", "9108")))
    consumer = KafkaConsumer(
        os.environ.get("BARS_TOPIC", "bars.1m"),
        bootstrap_servers=os.environ.get("KAFKA_BOOTSTRAP", "redpanda:9092"),
        group_id=os.environ.get("CONSUMER_GROUP", "online-writer"),
        enable_auto_commit=False,
        auto_offset_reset="earliest",
    )
    run(consumer, FeatureStore(repo_path=os.environ.get("FEAST_REPO", "platform/feature_repo")))


if __name__ == "__main__":
    main()
