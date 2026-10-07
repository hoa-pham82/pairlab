-- pairlab warehouse schema
-- Run automatically by Postgres on first container start.
-- Multiple databases are created here; Airflow and MLflow
-- use their own databases to avoid schema collisions.

-- ── Extra databases ─────────────────────────────────────────────────────────

SELECT 'CREATE DATABASE airflow' WHERE NOT EXISTS (
  SELECT FROM pg_database WHERE datname = 'airflow'
)\gexec

SELECT 'CREATE DATABASE mlflow' WHERE NOT EXISTS (
  SELECT FROM pg_database WHERE datname = 'mlflow'
)\gexec

-- ── Source / vendor (Bronze raw) ─────────────────────────────────────────────

CREATE SCHEMA IF NOT EXISTS vendor;
CREATE SCHEMA IF NOT EXISTS bronze;
CREATE SCHEMA IF NOT EXISTS silver;
CREATE SCHEMA IF NOT EXISTS gold;
CREATE SCHEMA IF NOT EXISTS feast;
CREATE SCHEMA IF NOT EXISTS platform;

-- Symbol master supplied by "another department"
CREATE TABLE IF NOT EXISTS vendor.symbols (
    symbol          TEXT        NOT NULL,
    name            TEXT,
    sector          TEXT,
    listed_at       DATE        NOT NULL,
    delisted_at     DATE,
    PRIMARY KEY (symbol, listed_at)
);

-- Raw daily bars — exactly as received (may have dupes, nulls, schema drift)
CREATE TABLE IF NOT EXISTS bronze.raw_daily_bars (
    ingest_ts       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    schema_version  SMALLINT    NOT NULL DEFAULT 1,
    symbol          TEXT        NOT NULL,
    ts              TIMESTAMPTZ NOT NULL,
    open            DOUBLE PRECISION,
    high            DOUBLE PRECISION,
    low             DOUBLE PRECISION,
    close           DOUBLE PRECISION,
    volume          DOUBLE PRECISION,
    adj_close       DOUBLE PRECISION,   -- added in schema v2
    source_file     TEXT
);
CREATE INDEX IF NOT EXISTS idx_raw_daily_bars_symbol_ts
    ON bronze.raw_daily_bars (symbol, ts);

-- Raw ticks from Redpanda consumer
CREATE TABLE IF NOT EXISTS bronze.raw_ticks (
    ingest_ts       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    trade_id        TEXT        NOT NULL,
    symbol          TEXT        NOT NULL,
    event_ts        TIMESTAMPTZ NOT NULL,
    price           DOUBLE PRECISION,
    quantity        DOUBLE PRECISION,
    side            CHAR(1)
);
CREATE INDEX IF NOT EXISTS idx_raw_ticks_symbol_event_ts
    ON bronze.raw_ticks (symbol, event_ts);

-- Quarantine for rejected rows (schema v3 breaking changes, out-of-range values)
CREATE TABLE IF NOT EXISTS bronze.stg_rejected_daily_bars (
    ingest_ts       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    symbol          TEXT,
    ts              TIMESTAMPTZ,
    reject_reason   TEXT,
    raw_json        JSONB
);

-- ── Silver ───────────────────────────────────────────────────────────────────

-- Deduped, schema-normalised, split-adjusted daily bars
CREATE TABLE IF NOT EXISTS silver.stg_daily_bars (
    symbol          TEXT            NOT NULL,
    ts              TIMESTAMPTZ     NOT NULL,
    open            DOUBLE PRECISION NOT NULL,
    high            DOUBLE PRECISION NOT NULL,
    low             DOUBLE PRECISION NOT NULL,
    close           DOUBLE PRECISION NOT NULL,
    adj_close       DOUBLE PRECISION NOT NULL,
    volume          DOUBLE PRECISION NOT NULL,
    split_factor    DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    schema_version  SMALLINT        NOT NULL,
    PRIMARY KEY (symbol, ts)
);

-- Deduped 1-min bars from Flink (trade_id dedupe already done in Flink)
CREATE TABLE IF NOT EXISTS silver.stg_bars_1m (
    symbol          TEXT            NOT NULL,
    window_start    TIMESTAMPTZ     NOT NULL,
    open            DOUBLE PRECISION NOT NULL,
    high            DOUBLE PRECISION NOT NULL,
    low             DOUBLE PRECISION NOT NULL,
    close           DOUBLE PRECISION NOT NULL,
    volume          DOUBLE PRECISION NOT NULL,
    tick_count      INT             NOT NULL,
    PRIMARY KEY (symbol, window_start)
);

-- ── Gold: dimensions ─────────────────────────────────────────────────────────

-- SCD2 symbol dimension: one row per (symbol, valid_from_ts) period.
-- Survivorship-bias fix: Universe.as_of(ts) queries valid_from_ts <= ts < valid_to_ts.
CREATE TABLE IF NOT EXISTS gold.dim_symbol (
    dim_symbol_key  SERIAL          PRIMARY KEY,
    symbol          TEXT            NOT NULL,
    name            TEXT,
    sector          TEXT,
    listed_at       DATE            NOT NULL,
    delisted_at     DATE,
    valid_from_ts   TIMESTAMPTZ     NOT NULL,
    valid_to_ts     TIMESTAMPTZ,            -- NULL = currently active
    is_current      BOOLEAN         NOT NULL DEFAULT TRUE,
    change_reason   TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_dim_symbol_current
    ON gold.dim_symbol (symbol) WHERE is_current = TRUE;
CREATE INDEX IF NOT EXISTS idx_dim_symbol_symbol_period
    ON gold.dim_symbol (symbol, valid_from_ts, valid_to_ts);

CREATE TABLE IF NOT EXISTS gold.dim_date (
    date_key        INT             PRIMARY KEY,  -- YYYYMMDD
    date_val        DATE            NOT NULL UNIQUE,
    year            SMALLINT,
    quarter         SMALLINT,
    month           SMALLINT,
    week            SMALLINT,
    day_of_week     SMALLINT,
    is_trading_day  BOOLEAN         DEFAULT TRUE
);

-- ── Gold: facts ──────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS gold.fact_daily_bar (
    symbol          TEXT            NOT NULL,
    ts              TIMESTAMPTZ     NOT NULL,
    open            DOUBLE PRECISION NOT NULL,
    high            DOUBLE PRECISION NOT NULL,
    low             DOUBLE PRECISION NOT NULL,
    close           DOUBLE PRECISION NOT NULL,
    adj_close       DOUBLE PRECISION NOT NULL,
    volume          DOUBLE PRECISION NOT NULL,
    daily_return    DOUBLE PRECISION,
    log_return      DOUBLE PRECISION,
    PRIMARY KEY (symbol, ts)
);
-- Covering index for backtester point-in-time queries
CREATE INDEX IF NOT EXISTS idx_fact_daily_bar_ts_symbol
    ON gold.fact_daily_bar (ts, symbol);

CREATE TABLE IF NOT EXISTS gold.fact_bar_1m (
    symbol          TEXT            NOT NULL,
    window_start    TIMESTAMPTZ     NOT NULL,
    open            DOUBLE PRECISION NOT NULL,
    high            DOUBLE PRECISION NOT NULL,
    low             DOUBLE PRECISION NOT NULL,
    close           DOUBLE PRECISION NOT NULL,
    volume          DOUBLE PRECISION NOT NULL,
    tick_count      INT             NOT NULL,
    PRIMARY KEY (symbol, window_start)
);

-- ── Gold: features ───────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS gold.feat_symbol_daily (
    symbol          TEXT            NOT NULL,
    ts              TIMESTAMPTZ     NOT NULL,   -- feature observation time
    event_timestamp TIMESTAMPTZ     NOT NULL,   -- Feast requires this
    created         TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    vol_21d         DOUBLE PRECISION,
    vol_63d         DOUBLE PRECISION,
    ret_21d         DOUBLE PRECISION,
    ret_63d         DOUBLE PRECISION,
    atr_14d         DOUBLE PRECISION,
    PRIMARY KEY (symbol, ts)
);

CREATE TABLE IF NOT EXISTS gold.feat_pair_daily (
    symbol_a        TEXT            NOT NULL,
    symbol_b        TEXT            NOT NULL,
    ts              TIMESTAMPTZ     NOT NULL,
    event_timestamp TIMESTAMPTZ     NOT NULL,
    created         TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    hedge_ratio     DOUBLE PRECISION,
    coint_pvalue    DOUBLE PRECISION,
    half_life       DOUBLE PRECISION,
    zscore          DOUBLE PRECISION,
    spread_vol      DOUBLE PRECISION,
    correlation_60d DOUBLE PRECISION,
    PRIMARY KEY (symbol_a, symbol_b, ts)
);

-- ── Gold: labels ─────────────────────────────────────────────────────────────

-- Meta-labeling target: did the spread revert within H bars before stop-loss?
CREATE TABLE IF NOT EXISTS gold.label_pair_reversion (
    pair_date_id    TEXT            PRIMARY KEY,   -- e.g. KO__PEP|2020-03-02
    label           SMALLINT        NOT NULL       -- 1=reverted, 0=stopped out or timed out
);

-- ── Gold: OBT (One Big Table for backtester) ─────────────────────────────────

-- Point-in-time filtered at query time by backtester.
-- Joins fact_daily_bar + feat_pair_daily + label_pair_reversion.
CREATE TABLE IF NOT EXISTS gold.obt_pair_backtest_input (
    symbol_a        TEXT            NOT NULL,
    symbol_b        TEXT            NOT NULL,
    ts              TIMESTAMPTZ     NOT NULL,
    open_a          DOUBLE PRECISION,
    close_a         DOUBLE PRECISION,
    volume_a        DOUBLE PRECISION,
    open_b          DOUBLE PRECISION,
    close_b         DOUBLE PRECISION,
    volume_b        DOUBLE PRECISION,
    hedge_ratio     DOUBLE PRECISION,
    coint_pvalue    DOUBLE PRECISION,
    half_life       DOUBLE PRECISION,
    zscore          DOUBLE PRECISION,
    spread_vol      DOUBLE PRECISION,
    label           SMALLINT,
    PRIMARY KEY (symbol_a, symbol_b, ts)
);
CREATE INDEX IF NOT EXISTS idx_obt_backtest_ts
    ON gold.obt_pair_backtest_input (ts);

-- ── Feast offline store tables ───────────────────────────────────────────────

-- Feast reads from gold.feat_symbol_daily and gold.feat_pair_daily directly.
-- These views alias them for the Feast OfflineStore configuration.
CREATE OR REPLACE VIEW feast.feat_symbol_daily AS
    SELECT * FROM gold.feat_symbol_daily;

CREATE OR REPLACE VIEW feast.feat_pair_daily AS
    SELECT * FROM gold.feat_pair_daily;

-- ── Utility: populate dim_date 2020-2030 ─────────────────────────────────────
INSERT INTO gold.dim_date (date_key, date_val, year, quarter, month, week, day_of_week, is_trading_day)
SELECT
    TO_CHAR(d, 'YYYYMMDD')::INT,
    d,
    EXTRACT(YEAR  FROM d)::SMALLINT,
    EXTRACT(QUARTER FROM d)::SMALLINT,
    EXTRACT(MONTH FROM d)::SMALLINT,
    EXTRACT(WEEK  FROM d)::SMALLINT,
    EXTRACT(DOW   FROM d)::SMALLINT,
    EXTRACT(DOW   FROM d) NOT IN (0, 6)   -- Mon-Fri = trading day
FROM generate_series('2020-01-01'::DATE, '2030-12-31'::DATE, '1 day'::INTERVAL) d
ON CONFLICT (date_key) DO NOTHING;

-- ── Platform job status ───────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS platform.job_status (
    job         TEXT        NOT NULL,
    ran_at      TIMESTAMP   NOT NULL DEFAULT now(),
    rows_out    BIGINT,
    status      TEXT        NOT NULL DEFAULT 'ok',
    PRIMARY KEY (job, ran_at)
);
