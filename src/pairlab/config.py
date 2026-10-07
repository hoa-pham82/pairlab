"""Pydantic-validated backtest configuration."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator


class CostConfig(BaseModel):
    commission_bps: float = Field(5.0, ge=0, description="One-way commission in basis points")
    commission_min_usd: float = Field(1.0, ge=0)
    half_spread_bps: float = Field(2.0, ge=0)
    slippage_factor: float = Field(0.1, ge=0, description="k in slippage = k * vol * sqrt(participation)")
    borrow_cost_bps_yr: float = Field(50.0, ge=0, description="Annual short-borrow cost in bps")


class StrategyConfig(BaseModel):
    formation_window: int = Field(252, ge=20, description="Bars used to estimate hedge ratio and test cointegration")
    zscore_window: int = Field(60, ge=5, description="Rolling window for spread z-score")
    entry_z: float = Field(2.0, gt=0)
    exit_z: float = Field(0.5, ge=0)
    stop_z: float = Field(4.0, gt=0)
    coint_pvalue_entry: float = Field(0.05, gt=0, lt=1, description="Max p-value to open a pair")
    coint_pvalue_drop: float = Field(0.10, gt=0, lt=1, description="p-value above which a live pair is closed")
    retest_every: int = Field(21, ge=1, description="Bars between cointegration re-tests")
    time_stop_halflife_multiple: float = Field(3.0, gt=0, description="Close if held > N * half-life bars")

    @model_validator(mode="after")
    def entry_above_exit(self) -> "StrategyConfig":
        if self.entry_z <= self.exit_z:
            raise ValueError("entry_z must be > exit_z")
        if self.stop_z <= self.entry_z:
            raise ValueError("stop_z must be > entry_z")
        return self


class PortfolioConfig(BaseModel):
    initial_cash: float = Field(1_000_000.0, gt=0)
    gross_notional_per_pair: float = Field(100_000.0, gt=0, description="Total two-leg notional per pair trade")
    max_gross_leverage: float = Field(4.0, gt=0)
    delisting_return: float = Field(-0.30, description="Forced return on a bad delisting (e.g. fraud)")


class BacktestConfig(BaseModel):
    name: str = "backtest"
    data_path: Path = Path("data/daily_bars.parquet")
    symbol_master_path: Path | None = None
    start_date: date
    end_date: date
    symbols: list[str] = Field(default_factory=list, description="Empty = use all in data_path")

    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    costs: CostConfig = Field(default_factory=CostConfig)

    output_dir: Path | None = Path("results")
    seed: int = 42

    @field_validator("end_date")
    @classmethod
    def end_after_start(cls, v: date, info: object) -> date:
        if hasattr(info, "data") and "start_date" in info.data and v <= info.data["start_date"]:
            raise ValueError("end_date must be after start_date")
        return v
