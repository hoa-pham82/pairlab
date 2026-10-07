"""Tests for BacktestConfig and sub-configs."""

from datetime import date

import pytest

from pairlab.config import BacktestConfig, StrategyConfig


def _base():
    return dict(start_date=date(2020, 1, 1), end_date=date(2023, 12, 31))


class TestStrategyConfigValidation:
    def test_entry_below_stop(self):
        """entry_z must be less than stop_z."""
        with pytest.raises(ValueError):
            StrategyConfig(entry_z=5.0, exit_z=0.5, stop_z=3.0)

    def test_entry_above_exit(self):
        """entry_z must be greater than exit_z."""
        with pytest.raises(ValueError):
            StrategyConfig(entry_z=0.3, exit_z=0.5, stop_z=4.0)

    @pytest.mark.parametrize("entry_z,exit_z,stop_z", [
        (2.0, 0.5, 4.0),  # nominal
        (1.5, 0.0, 3.0),  # exit at zero
        (2.0, 1.9, 4.0),  # tight exit
    ])
    def test_valid_thresholds(self, entry_z, exit_z, stop_z):
        cfg = StrategyConfig(entry_z=entry_z, exit_z=exit_z, stop_z=stop_z)
        assert cfg.entry_z == entry_z


class TestBacktestConfigValidation:
    def test_end_before_start_raises(self):
        with pytest.raises(ValueError):
            BacktestConfig(start_date=date(2023, 1, 1), end_date=date(2022, 12, 31))

    def test_defaults_sensible(self):
        cfg = BacktestConfig(**_base())
        assert cfg.portfolio.initial_cash == 1_000_000.0
        assert cfg.costs.commission_bps == 5.0
        assert cfg.seed == 42

    def test_cost_bps_non_negative(self):
        from pairlab.config import CostConfig
        with pytest.raises(ValueError):
            CostConfig(commission_bps=-1.0)

    def test_symbols_default_empty(self):
        cfg = BacktestConfig(**_base())
        assert cfg.symbols == []
