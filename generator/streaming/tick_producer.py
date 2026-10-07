"""Produce synthetic tick stream to JSONL (Phase 1) and Redpanda/Kafka (Phase 2)."""

from __future__ import annotations

import json
import random
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


def generate_ticks(
    symbols: list[str],
    n_ticks: int,
    output_path: Path,
    seed: int = 42,
    burst_probability: float = 0.05,
    late_arrival_probability: float = 0.03,
    late_arrival_max_seconds: int = 30,
    duplicate_rate: float = 0.015,
    zipf_weights: dict[str, float] | None = None,
) -> dict:
    """Write a JSONL tick stream with bursts, late arrivals, and duplicate IDs.

    Each line: {"trade_id", "symbol", "price", "quantity", "event_time", "ingest_ts"}
    event_time is epoch milliseconds (int), matching Flink SQL `BIGINT` schema.

    Args:
        zipf_weights: per-symbol probability weights (Zipf-distributed).
            If None, symbols are chosen uniformly.

    Returns:
        Quality summary dict.
    """
    rng = random.Random(seed)
    sym_list = list(symbols)
    weights = [zipf_weights[s] for s in sym_list] if zipf_weights else None

    base_prices = {sym: rng.uniform(30, 200) for sym in sym_list}
    base_ts = datetime(2024, 1, 2, 9, 30, tzinfo=timezone.utc)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    n_late = 0
    n_burst = 0
    n_dup = 0
    recent_ids: list[str] = []

    with output_path.open("w") as f:
        ts = base_ts
        while total < n_ticks:
            sym = rng.choices(sym_list, weights=weights, k=1)[0]
            price = base_prices[sym] * (1 + rng.gauss(0, 0.0005))
            base_prices[sym] = price

            is_burst = rng.random() < burst_probability
            batch_size = rng.randint(5, 20) if is_burst else 1
            if is_burst:
                n_burst += batch_size

            for _ in range(batch_size):
                trade_id = str(uuid.UUID(int=rng.getrandbits(128)))
                event_ts = ts

                if rng.random() < late_arrival_probability:
                    delay = rng.randint(1, late_arrival_max_seconds)
                    event_ts = ts - timedelta(seconds=delay)
                    n_late += 1

                if recent_ids and rng.random() < duplicate_rate:
                    trade_id = rng.choice(recent_ids)
                    n_dup += 1

                tick = {
                    "trade_id": trade_id,
                    "symbol": sym,
                    "price": round(price, 4),
                    "quantity": rng.randint(1, 500),
                    "event_time": int(event_ts.timestamp() * 1000),  # epoch ms int
                    "ingest_ts": int(ts.timestamp() * 1000),
                }
                f.write(json.dumps(tick) + "\n")
                recent_ids.append(trade_id)
                if len(recent_ids) > 100:
                    recent_ids.pop(0)

                total += 1
                if total >= n_ticks:
                    break

            ts += timedelta(milliseconds=rng.randint(10, 500))

    return {
        "total_ticks": total,
        "n_late_arrivals": n_late,
        "n_bursts": n_burst,
        "n_duplicates": n_dup,
        "late_pct": round(n_late / total, 4) if total else 0,
        "dup_pct": round(n_dup / total, 4) if total else 0,
    }


def produce_to_kafka(
    symbols: list[str],
    n_ticks: int,
    bootstrap_servers: str = "localhost:19092",
    topic: str = "ticks.raw",
    seed: int = 42,
    burst_probability: float = 0.05,
    late_arrival_probability: float = 0.03,
    late_arrival_max_seconds: int = 30,
    duplicate_rate: float = 0.015,
    zipf_weights: dict[str, float] | None = None,
) -> dict:
    """Produce tick events directly to Redpanda/Kafka.

    Each message value: JSON with fields matching ticks_to_bars_1m.sql source schema:
    {symbol, price, quantity, trade_id, event_time (epoch ms int)}

    Returns:
        Quality summary dict (same shape as generate_ticks).
    """
    from kafka import KafkaProducer  # kafka-python-ng, already a dependency

    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode(),
    )

    rng = random.Random(seed)
    sym_list = list(symbols)
    weights = [zipf_weights[s] for s in sym_list] if zipf_weights else None

    base_prices = {sym: rng.uniform(30, 200) for sym in sym_list}
    base_ts = datetime.now(tz=timezone.utc)

    total = 0
    n_late = 0
    n_burst = 0
    n_dup = 0
    recent_ids: list[str] = []

    ts = base_ts
    while total < n_ticks:
        sym = rng.choices(sym_list, weights=weights, k=1)[0]
        price = base_prices[sym] * (1 + rng.gauss(0, 0.0005))
        base_prices[sym] = price

        is_burst = rng.random() < burst_probability
        batch_size = rng.randint(5, 20) if is_burst else 1
        if is_burst:
            n_burst += batch_size

        for _ in range(batch_size):
            trade_id = str(uuid.UUID(int=rng.getrandbits(128)))
            event_ts = ts

            if rng.random() < late_arrival_probability:
                delay = rng.randint(1, late_arrival_max_seconds)
                event_ts = ts - timedelta(seconds=delay)
                n_late += 1

            if recent_ids and rng.random() < duplicate_rate:
                trade_id = rng.choice(recent_ids)
                n_dup += 1

            tick = {
                "symbol": sym,
                "price": round(price, 4),
                "quantity": float(rng.randint(1, 500)),
                "trade_id": trade_id,
                "event_time": int(event_ts.timestamp() * 1000),
            }
            producer.send(topic, tick)
            recent_ids.append(trade_id)
            if len(recent_ids) > 100:
                recent_ids.pop(0)

            total += 1
            if total >= n_ticks:
                break

        ts += timedelta(milliseconds=rng.randint(10, 500))

    # Send flush ticks (event_time = now + 3 min) to advance the Flink watermark
    # and close any open tumbling windows before the producer exits.
    flush_ts = int((datetime.now(tz=timezone.utc) + timedelta(minutes=3)).timestamp() * 1000)
    for sym in sym_list:
        producer.send(topic, {
            "symbol": sym,
            "price": round(base_prices[sym], 4),
            "quantity": 1.0,
            "trade_id": f"flush-{sym}-{uuid.uuid4().hex[:8]}",
            "event_time": flush_ts,
        })

    producer.flush()
    producer.close()

    return {
        "total_ticks": total,
        "n_late_arrivals": n_late,
        "n_bursts": n_burst,
        "n_duplicates": n_dup,
        "late_pct": round(n_late / total, 4) if total else 0,
        "dup_pct": round(n_dup / total, 4) if total else 0,
    }
