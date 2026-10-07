"""CLI entry point: pairlab backtest --config path/to/config.yaml"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(help="pairlab — pairs trading research platform", no_args_is_help=True)


@app.callback()
def _root(
    version: Optional[bool] = typer.Option(None, "--version", is_eager=True, help="Show version"),
) -> None:
    if version:
        typer.echo("pairlab 0.1.0")
        raise typer.Exit()


@app.command()
def backtest(
    config: Path = typer.Option(..., "--config", "-c", help="Path to BacktestConfig YAML"),
) -> None:
    """Run a backtest from a YAML config file."""
    import yaml
    from pairlab.backtest import run_backtest
    from pairlab.config import BacktestConfig
    from pairlab.data.loaders import load_parquet, load_csv

    raw = yaml.safe_load(config.read_text())
    cfg = BacktestConfig(**raw)

    typer.echo(f"Loading data from {cfg.data_path}…")
    if cfg.data_path.suffix == ".parquet":
        bars = load_parquet(cfg.data_path)
    else:
        bars = load_csv(cfg.data_path)

    typer.echo(f"Running backtest: {cfg.name}")
    result = run_backtest(cfg, bars)

    m = result.metrics
    typer.echo(f"\n{'='*40}")
    typer.echo(f"  Sharpe:     {m.sharpe:.3f}" if m.sharpe == m.sharpe else "  Sharpe:     N/A")
    typer.echo(f"  Total P&L:  ${m.total_pnl:,.2f}")
    typer.echo(f"  Max DD:     {m.max_drawdown * 100:.1f}%")
    typer.echo(f"  # Trades:   {m.n_trades}")
    typer.echo(f"{'='*40}")
    if cfg.output_dir:
        typer.echo(f"\nTear sheet saved to {cfg.output_dir}/tearsheet.html")
