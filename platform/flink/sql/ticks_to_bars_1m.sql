-- Flink SQL: ticks → 1-minute OHLCV bars with watermark-based late-arrival handling.
--
-- Source  : Kafka topic ticks.raw  (JSON: symbol, event_time, price, quantity, trade_id)
-- Sink    : Kafka topic bars.1m    (JSON: symbol, window_start, open, high, low, close, volume)
-- Sink    : Postgres silver.stg_bars_1m (via JDBC connector)
--
-- Late arrival handling: WATERMARK DELAY 30 SECONDS
--   Ticks arriving up to 30 s after their window closes are included.
--   Ticks beyond 30 s are silently dropped by Flink's watermark mechanism.
--   (Flink SQL has no side-output API; late-data capture requires the DataStream API.)
--
-- Dedup: ROW_NUMBER() on (symbol, trade_id) by proctime keeps first seen copy.

-- ────────────────────────────────────────────────────────────────────────────
-- 1. Source table: read JSON ticks from Kafka / Redpanda
-- ────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ticks_raw (
    symbol          STRING,
    price           DOUBLE,
    quantity        DOUBLE,
    trade_id        STRING,
    event_time      BIGINT,          -- Unix milliseconds from producer (JSON key: event_time)
    proctime        AS PROCTIME(),   -- processing time for dedup ordering
    event_ts        AS TO_TIMESTAMP_LTZ(event_time, 3),
    WATERMARK FOR event_ts AS event_ts - INTERVAL '30' SECOND
) WITH (
    'connector'                     = 'kafka',
    'topic'                         = 'ticks.raw',
    'properties.bootstrap.servers'  = 'redpanda:9092',
    'properties.group.id'           = 'flink-ticks-consumer',
    'scan.startup.mode'             = 'earliest-offset',
    'format'                        = 'json'
);

-- ────────────────────────────────────────────────────────────────────────────
-- 2. Dedup view: drop duplicate trade_ids within a processing-time window
-- ────────────────────────────────────────────────────────────────────────────
CREATE TEMPORARY VIEW ticks_deduped AS
SELECT symbol, event_ts, price, quantity, trade_id
FROM (
    SELECT *,
        ROW_NUMBER() OVER (
            PARTITION BY symbol, trade_id
            ORDER BY proctime ASC
        ) AS rn
    FROM ticks_raw
) t
WHERE rn = 1;

-- ────────────────────────────────────────────────────────────────────────────
-- 3. Sink: 1-minute bars to Kafka bars.1m
-- ────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS bars_1m_kafka (
    symbol       STRING,
    window_start TIMESTAMP(3),
    window_end   TIMESTAMP(3),
    `open`       DOUBLE,
    high         DOUBLE,
    low          DOUBLE,
    `close`      DOUBLE,
    volume       DOUBLE,
    tick_count   BIGINT
) WITH (
    'connector'                     = 'kafka',
    'topic'                         = 'bars.1m',
    'properties.bootstrap.servers'  = 'redpanda:9092',
    'format'                        = 'json'
);

-- ────────────────────────────────────────────────────────────────────────────
-- 4. Sink: 1-minute bars to Postgres (JDBC)
--    Column names must match silver.stg_bars_1m exactly.
-- ────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS bars_1m_pg (
    symbol       STRING,
    window_start TIMESTAMP(3),
    `open`       DOUBLE,
    high         DOUBLE,
    low          DOUBLE,
    `close`      DOUBLE,
    volume       DOUBLE,
    tick_count   BIGINT,
    PRIMARY KEY (symbol, window_start) NOT ENFORCED
) WITH (
    'connector'  = 'jdbc',
    'url'        = 'jdbc:postgresql://postgres:5432/pairlab',
    'table-name' = 'silver.stg_bars_1m',
    'username'   = 'pairlab',
    'password'   = 'pairlab',
    'driver'     = 'org.postgresql.Driver'
);

-- ────────────────────────────────────────────────────────────────────────────
-- 5. Main aggregation: tumbling 1-minute window OHLCV.
--    One statement set = one Flink job, so CD can cancel and resubmit it by name.
-- ────────────────────────────────────────────────────────────────────────────
SET 'pipeline.name' = 'ticks_to_bars_1m';

EXECUTE STATEMENT SET
BEGIN

INSERT INTO bars_1m_kafka
SELECT
    symbol,
    window_start,
    window_end,
    FIRST_VALUE(price) AS `open`,
    MAX(price)         AS high,
    MIN(price)         AS low,
    LAST_VALUE(price)  AS `close`,
    SUM(quantity)      AS volume,
    COUNT(*)           AS tick_count
FROM TABLE(
    TUMBLE(TABLE ticks_deduped, DESCRIPTOR(event_ts), INTERVAL '1' MINUTE)
)
GROUP BY symbol, window_start, window_end;

-- Mirror same result to Postgres (window_end omitted — not in stg_bars_1m schema)
INSERT INTO bars_1m_pg
SELECT
    symbol,
    window_start,
    FIRST_VALUE(price),
    MAX(price),
    MIN(price),
    LAST_VALUE(price),
    SUM(quantity),
    COUNT(*)
FROM TABLE(
    TUMBLE(TABLE ticks_deduped, DESCRIPTOR(event_ts), INTERVAL '1' MINUTE)
)
GROUP BY symbol, window_start, window_end;

END;
