"""Generate all synthetic datasets: offline Parquet + streaming JSONL + quality summary."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parents[2]))

from generator.offline.generate import GeneratorConfig, generate_bars
from generator.problems.inject import apply_schema_evolution, inject_duplicates
from generator.streaming.tick_producer import generate_ticks


def _top_k_share(bars: pd.DataFrame, k: int = 5) -> dict:
    """Return the Zipf top-k symbols by total volume and their share."""
    vol_by_sym = bars.groupby("symbol")["volume"].sum().sort_values(ascending=False)
    top_k = vol_by_sym.head(k)
    total = vol_by_sym.sum()
    return {sym: round(float(v / total), 4) for sym, v in top_k.items()}


def run(config_path: str, output_dir: str) -> None:
    cfg_raw = yaml.safe_load(Path(config_path).read_text())
    cfg = GeneratorConfig(**{k: v for k, v in cfg_raw.items() if hasattr(GeneratorConfig, k)})
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Generating {cfg.n_symbols} symbols × {cfg.days} bars …")
    bars, master, pairs, sector_changes = generate_bars(cfg)

    # Inject problems
    bars_raw = bars.copy()
    if cfg.duplicate_rate > 0:
        bars = inject_duplicates(bars, cfg.duplicate_rate, cfg.seed)
    if cfg.schema_evolution:
        bars = apply_schema_evolution(bars, cfg.schema_change_date, cfg.seed)

    # Write Parquet
    bars_path = out / "daily_bars.parquet"
    bars.to_parquet(bars_path, index=False)
    print(f"Wrote {bars_path} ({len(bars):,} rows)")

    master_path = out / "symbol_master.parquet"
    master.to_parquet(master_path, index=False)
    print(f"Wrote {master_path}")

    # Streaming JSONL (with Zipf weights for symbol selection)
    tick_path = out / "ticks.jsonl"
    symbols = list(bars["symbol"].unique())[:10]
    zipf_weights = bars.attrs.get("zipf_weights")
    tick_weights = {s: zipf_weights[s] for s in symbols} if zipf_weights else None
    stream_stats = generate_ticks(
        symbols=symbols,
        n_ticks=10_000,
        output_path=tick_path,
        seed=cfg.seed,
        zipf_weights=tick_weights,
    )
    print(f"Wrote {tick_path} ({stream_stats['total_ticks']:,} ticks)")

    # Quality summary
    raw_rows = len(bars_raw)
    dup_rows = len(bars) - raw_rows
    change_ts = pd.Timestamp(cfg.schema_change_date, tz="UTC")
    post_mask = bars_raw["ts"] >= change_ts
    nulls_per_version = {
        "v1_volume_nulls": int(bars_raw[~post_mask]["volume"].isna().sum()),
        "v2_adj_close_nulls": int(
            bars.get("adj_close", pd.Series(dtype=float)).isna().sum()
            if "adj_close" in bars.columns else raw_rows
        ),
    }

    summary = {
        "config": cfg_raw,
        "offline": {
            "symbols": int(bars["symbol"].nunique()),
            "total_rows": len(bars),
            "raw_rows": raw_rows,
            "duplicate_rows_injected": dup_rows,
            "duplicate_pct": round(dup_rows / raw_rows, 4),
            "distinct_trade_ids": "N/A (daily bars — no trade IDs)",
            "cointegrated_pairs": [f"{a}/{b}" for a, b in pairs],
            "schema_evolution": cfg.schema_evolution,
            "schema_change_date": cfg.schema_change_date,
            "nulls_per_version": nulls_per_version,
            "top_5_volume_share": _top_k_share(bars_raw, k=5),
            "sector_changes": sector_changes,
        },
        "streaming": stream_stats,
    }
    summary_path = out / "quality_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Quality summary → {summary_path}")
    print(f"Top-5 symbol volume share: {summary['offline']['top_5_volume_share']}")
    if sector_changes:
        print(f"Sector changes: {sector_changes}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="generator/configs/default.yaml")
    p.add_argument("--output", default="data/generated")
    args = p.parse_args()
    run(args.config, args.output)
