"""Generate HTML + PNG tear sheet from backtest results."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from pairlab.metrics.performance import PerformanceMetrics


_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>{name} Tear Sheet</title>
<style>
  body {{ font-family: monospace; max-width: 900px; margin: 2em auto; background: #111; color: #eee; }}
  h1 {{ color: #7ec8e3; }} h2 {{ color: #aaa; border-bottom: 1px solid #333; }}
  table {{ border-collapse: collapse; width: 100%; }}
  td, th {{ padding: 6px 12px; text-align: left; }}
  tr:nth-child(even) {{ background: #1a1a1a; }}
  .good {{ color: #7ec8e3; }} .bad {{ color: #e37e7e; }} .na {{ color: #666; }}
  img {{ max-width: 100%; margin: 1em 0; border: 1px solid #333; }}
</style></head>
<body>
<h1>{name}</h1>
<p>Generated {ts}</p>
<h2>Performance Summary</h2>
<table>
{rows}
</table>
<h2>Equity Curve</h2>
<img src="equity_curve.png" alt="equity curve">
<h2>Drawdown</h2>
<img src="drawdown.png" alt="drawdown">
<h2>Monthly Returns</h2>
<img src="monthly_returns.png" alt="monthly returns">
</body></html>"""


def _fmt(val: float, pct: bool = False, decimals: int = 2) -> str:
    if val != val:  # NaN
        return '<span class="na">N/A</span>'
    if pct:
        css = "good" if val >= 0 else "bad"
        return f'<span class="{css}">{val * 100:.{decimals}f}%</span>'
    css = "good" if val >= 0 else "bad"
    return f'<span class="{css}">{val:.{decimals}f}</span>'


def _metric_rows(m: PerformanceMetrics) -> str:
    rows = [
        ("Total Return", _fmt(m.total_return, pct=True)),
        ("Annualised Return", _fmt(m.annualised_return, pct=True)),
        ("Annualised Vol", _fmt(m.annualised_vol, pct=True)),
        ("Sharpe Ratio", _fmt(m.sharpe)),
        ("Sortino Ratio", _fmt(m.sortino)),
        ("Max Drawdown", _fmt(-m.max_drawdown, pct=True)),
        ("Max DD Duration (bars)", str(m.max_drawdown_duration_days)),
        ("Calmar Ratio", _fmt(m.calmar)),
        ("Profit Factor", "∞" if m.profit_factor == float("inf") else _fmt(m.profit_factor)),
        ("Hit Rate", _fmt(m.hit_rate, pct=True)),
        ("Avg Holding (bars)", f"{m.avg_holding_bars:.1f}"),
        ("Annual Turnover", _fmt(m.turnover_annual)),
        ("Exposure", _fmt(m.exposure, pct=True)),
        ("Total P&L", f"${m.total_pnl:,.2f}"),
        ("# Trades", str(m.n_trades)),
    ]
    return "\n".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows)


def save_tearsheet(
    name: str,
    metrics: PerformanceMetrics,
    equity_curve: list[tuple[datetime, float]],
    output_dir: Path,
) -> Path:
    """Write HTML tear sheet, PNG charts, and results.json to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    _save_charts(equity_curve, output_dir)
    _save_json(name, metrics, output_dir)

    html = _HTML_TEMPLATE.format(
        name=name,
        ts=datetime.now().strftime("%Y-%m-%d %H:%M"),
        rows=_metric_rows(metrics),
    )
    html_path = output_dir / "tearsheet.html"
    html_path.write_text(html)
    return html_path


def _save_charts(equity_curve: list[tuple[datetime, float]], output_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    if len(equity_curve) < 2:
        return

    tss, equities = zip(*equity_curve)
    eq = np.array(equities)

    # Equity curve
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(list(tss), eq, color="#7ec8e3", linewidth=1.5)
    ax.set_title("Equity Curve")
    ax.set_facecolor("#111")
    fig.patch.set_facecolor("#111")
    ax.tick_params(colors="#aaa")
    fig.savefig(output_dir / "equity_curve.png", dpi=100, bbox_inches="tight")
    plt.close(fig)

    # Drawdown
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / np.where(peak == 0, 1, peak)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.fill_between(list(tss), -dd, color="#e37e7e", alpha=0.7)
    ax.set_title("Drawdown")
    ax.set_facecolor("#111")
    fig.patch.set_facecolor("#111")
    ax.tick_params(colors="#aaa")
    fig.savefig(output_dir / "drawdown.png", dpi=100, bbox_inches="tight")
    plt.close(fig)

    # Monthly returns heatmap (simplified bar chart)
    returns = np.diff(eq) / eq[:-1]
    fig, ax = plt.subplots(figsize=(10, 3))
    colors = ["#7ec8e3" if r >= 0 else "#e37e7e" for r in returns[-60:]]
    ax.bar(range(len(returns[-60:])), returns[-60:] * 100, color=colors, width=1.0)
    ax.set_title("Last 60 Daily Returns (%)")
    ax.set_facecolor("#111")
    fig.patch.set_facecolor("#111")
    ax.tick_params(colors="#aaa")
    fig.savefig(output_dir / "monthly_returns.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def _save_json(name: str, metrics: PerformanceMetrics, output_dir: Path) -> None:
    def _clean(v):
        if v != v:
            return None
        if v == float("inf"):
            return "inf"
        return v

    data = {
        "name": name,
        "generated": datetime.now().isoformat(),
        "metrics": {k: _clean(v) for k, v in metrics.__dict__.items()},
    }
    (output_dir / "results.json").write_text(json.dumps(data, indent=2))
