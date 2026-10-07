"""Tests for synthetic data generator and problem injection."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from generator.offline.generate import GeneratorConfig, generate_bars
from generator.problems.inject import apply_schema_evolution, inject_duplicates
from generator.streaming.tick_producer import generate_ticks


def _small_cfg(**kwargs) -> GeneratorConfig:
    defaults = dict(n_symbols=4, n_cointegrated_pairs=2, days=60, seed=7)
    defaults.update(kwargs)
    return GeneratorConfig(**defaults)


class TestGenerateBars:
    def test_correct_symbol_count(self):
        cfg = _small_cfg(n_symbols=4)
        bars, master, pairs, _ = generate_bars(cfg)
        assert bars["symbol"].nunique() == 4

    def test_correct_row_count_approx(self):
        cfg = _small_cfg(n_symbols=4, days=60, delist_rate=0.0)
        bars, *_ = generate_bars(cfg)
        # 4 symbols × ~60 business days = ~240 rows (weekends excluded)
        assert 200 <= len(bars) <= 300

    def test_required_columns_present(self):
        bars, *_ = generate_bars(_small_cfg())
        for col in ("symbol", "ts", "open", "high", "low", "close", "volume"):
            assert col in bars.columns

    def test_prices_positive(self):
        bars, *_ = generate_bars(_small_cfg())
        assert (bars["close"] > 0).all()

    def test_symbol_master_has_listed_at(self):
        _, master, *_ = generate_bars(_small_cfg())
        assert "listed_at" in master.columns
        assert "delisted_at" in master.columns

    def test_determinism(self):
        """Same seed → identical output."""
        cfg = _small_cfg(seed=42)
        bars1, *_ = generate_bars(cfg)
        bars2, *_ = generate_bars(cfg)
        pd.testing.assert_frame_equal(bars1, bars2)

    def test_cointegrated_pairs_in_output(self):
        cfg = _small_cfg(n_symbols=4, n_cointegrated_pairs=2)
        _, _, pairs, _ = generate_bars(cfg)
        assert len(pairs) == 2

    def test_ts_is_utc(self):
        bars, *_ = generate_bars(_small_cfg())
        assert str(bars["ts"].dt.tz) == "UTC"

    def test_regime_shift_config(self):
        cfg = _small_cfg(
            n_symbols=4, days=120,
            regime_shift_date="2022-03-01",
            regime_shift_multiplier=10.0,
        )
        bars, *_ = generate_bars(cfg)
        assert len(bars) > 0

    def test_sector_change_produces_two_master_rows(self):
        """Sector change creates two SCD2 rows: one closed, one current."""
        cfg = _small_cfg(
            n_symbols=4, days=120, start="2021-01-04",
            sector_change_date="2021-06-01",
            sector_change_symbols=["KO"],
            sector_change_new_value="Beverages",
        )
        _, master, _, changes = generate_bars(cfg)
        ko_rows = master[master["symbol"] == "KO"]
        assert len(ko_rows) == 2, "sector change should produce two SCD2 rows for KO"
        assert len(changes) == 1
        assert changes[0]["symbol"] == "KO"
        assert changes[0]["new_sector"] == "Beverages"
        closed = ko_rows[~ko_rows["is_current"]]
        current = ko_rows[ko_rows["is_current"]]
        assert len(closed) == 1 and len(current) == 1
        assert current.iloc[0]["sector"] == "Beverages"


class TestInjectDuplicates:
    def test_adds_rows(self):
        bars, *_ = generate_bars(_small_cfg())
        original_len = len(bars)
        duped = inject_duplicates(bars, rate=0.05, seed=0)
        assert len(duped) > original_len

    def test_zero_rate_no_change(self):
        bars, *_ = generate_bars(_small_cfg())
        duped = inject_duplicates(bars, rate=0.0)
        assert len(duped) == len(bars)

    def test_approximate_rate(self):
        bars, *_ = generate_bars(_small_cfg(days=200))
        original_len = len(bars)
        duped = inject_duplicates(bars, rate=0.02, seed=0)
        added = len(duped) - original_len
        expected = int(original_len * 0.02)
        assert abs(added - expected) <= 2


class TestSchemaEvolution:
    def test_adj_close_added_after_change_date(self):
        bars, *_ = generate_bars(_small_cfg(days=120))
        evolved = apply_schema_evolution(bars, "2022-03-15")
        after = evolved[evolved["ts"] >= pd.Timestamp("2022-03-15", tz="UTC")]
        assert "adj_close" in evolved.columns
        if len(after) > 0:
            assert after["adj_close"].notna().any()

    def test_no_adj_close_before_change_date(self):
        bars, *_ = generate_bars(_small_cfg(days=60, start="2022-01-03"))
        evolved = apply_schema_evolution(bars, "2022-06-01")
        before = evolved[evolved["ts"] < pd.Timestamp("2022-06-01", tz="UTC")]
        assert before["adj_close"].isna().all() if "adj_close" in evolved.columns else True


class TestTickProducer:
    def test_produces_correct_count(self, tmp_path):
        out = tmp_path / "ticks.jsonl"
        stats = generate_ticks(["A", "B", "C"], n_ticks=100, output_path=out, seed=0)
        assert stats["total_ticks"] == 100

    def test_file_created(self, tmp_path):
        out = tmp_path / "ticks.jsonl"
        generate_ticks(["A"], n_ticks=50, output_path=out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_late_arrival_percentage(self, tmp_path):
        out = tmp_path / "ticks.jsonl"
        stats = generate_ticks(
            ["A", "B"],
            n_ticks=1000,
            output_path=out,
            late_arrival_probability=0.1,
            seed=1,
        )
        assert stats["late_pct"] > 0

    def test_duplicate_rate_non_zero(self, tmp_path):
        out = tmp_path / "ticks.jsonl"
        stats = generate_ticks(
            ["A", "B"],
            n_ticks=500,
            output_path=out,
            duplicate_rate=0.05,
            seed=2,
        )
        assert stats["dup_pct"] > 0

    def test_valid_json_lines(self, tmp_path):
        import json
        out = tmp_path / "ticks.jsonl"
        generate_ticks(["A"], n_ticks=20, output_path=out)
        lines = out.read_text().strip().split("\n")
        for line in lines:
            tick = json.loads(line)
            assert "trade_id" in tick
            assert "price" in tick
            assert tick["price"] > 0
            assert isinstance(tick["event_time"], int), "event_time must be epoch ms int"

    def test_zipf_weights_skew_symbol_counts(self, tmp_path):
        """With Zipf weights, high-ranked symbol should have more ticks than low-ranked."""
        import json
        out = tmp_path / "ticks_zipf.jsonl"
        symbols = ["A", "B", "C", "D"]
        # A gets weight 1.0, D gets ~0.08 (Zipf exponent 1.2)
        weights = {s: 1.0 / (i + 1) ** 1.2 for i, s in enumerate(symbols)}
        total = sum(weights.values())
        weights = {s: v / total for s, v in weights.items()}
        generate_ticks(symbols, n_ticks=2000, output_path=out, seed=5, zipf_weights=weights)
        counts: dict[str, int] = {}
        for line in out.read_text().strip().split("\n"):
            sym = json.loads(line)["symbol"]
            counts[sym] = counts.get(sym, 0) + 1
        assert counts.get("A", 0) > counts.get("D", 0), "A should have more ticks than D"
