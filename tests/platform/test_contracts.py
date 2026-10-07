"""Contract tests: the Flink job and Feast views must agree with the warehouse DDL."""

from __future__ import annotations

import re
from datetime import timedelta
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
FLINK_SQL = (ROOT / "platform/flink/sql/ticks_to_bars_1m.sql").read_text()
INIT_SQL = (ROOT / "platform/warehouse/init.sql").read_text()


def _ddl_body(sql: str, table: str) -> str:
    match = re.search(rf"CREATE TABLE IF NOT EXISTS {re.escape(table)} \((.*?)\n\)", sql, re.S)
    assert match, f"{table} not found"
    return match.group(1)


def _columns(body: str) -> list[str]:
    names = []
    for line in body.splitlines():
        line = line.split("--")[0].strip().strip(",")
        if not line or line.upper().startswith("PRIMARY KEY"):
            continue
        names.append(line.split()[0].strip("`").lower())
    return names


def _primary_key(body: str) -> list[str]:
    match = re.search(r"PRIMARY KEY \(([^)]*)\)", body)
    return [c.strip() for c in match.group(1).split(",")] if match else []


class TestFlinkOfflineSink:
    sink = _ddl_body(FLINK_SQL, "bars_1m_pg")
    table = _ddl_body(INIT_SQL, "silver.stg_bars_1m")

    def test_sink_columns_match_the_table(self):
        assert _columns(self.sink) == _columns(self.table)

    def test_sink_upserts_on_the_table_key(self):
        assert _primary_key(self.sink) == _primary_key(self.table) == ["symbol", "window_start"]

    def test_job_runs_as_one_named_statement_set(self):
        assert "SET 'pipeline.name' = 'ticks_to_bars_1m';" in FLINK_SQL
        assert FLINK_SQL.count("EXECUTE STATEMENT SET") == 1

    def test_watermark_covers_the_generated_lateness(self):
        delay = int(re.search(r"event_ts - INTERVAL '(\d+)' SECOND", FLINK_SQL).group(1))
        config = yaml.safe_load((ROOT / "generator/configs/default.yaml").read_text())
        assert delay >= config["late_arrival_max_seconds"]


@pytest.fixture(scope="module")
def features():
    pytest.importorskip("feast")
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "pairlab_features", ROOT / "platform/feature_repo/features.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Boundary: a daily view must survive a weekend (Fri -> Mon = 3 days); bars need minutes.
@pytest.mark.parametrize(
    ("view", "ttl"),
    [
        ("symbol_daily_fv", timedelta(days=3)),
        ("pair_daily_fv", timedelta(days=3)),
        ("bars_1m_fv", timedelta(minutes=10)),
    ],
)
def test_feature_view_ttl(features, view, ttl):
    assert getattr(features, view).ttl == ttl


@pytest.mark.parametrize(
    ("view", "table"),
    [
        ("symbol_daily_fv", "gold.feat_symbol_daily"),
        ("pair_daily_fv", "gold.feat_pair_daily"),
        ("bars_1m_fv", "silver.stg_bars_1m"),
    ],
)
def test_feature_view_fields_exist_in_the_table(features, view, table):
    fields = {f.name for f in getattr(features, view).schema}
    assert fields <= set(_columns(_ddl_body(INIT_SQL, table)))
