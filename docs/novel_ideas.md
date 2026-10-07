# Novel ideas

Two techniques in this project that go beyond the course material. Both are
already in the code, and each has a test or a result file that shows it works.

| # | Idea | Where | Proof |
|---|---|---|---|
| 1 | Meta-labeling with a purged time split, judged by a backtest | `ml/` | [`pngs/phase3_backtest_filter.json`](pngs/phase3_backtest_filter.json) |
| 2 | Metamorphic tests that show the backtester cannot see the future | `tests/property/`, `tests/unit/` | `tests/property/test_property_backtest.py` |

---

## Idea 1 — Meta-labeling with a purged time split, judged by a backtest

### What it is

Three techniques from quantitative finance (described in Marcos López de
Prado, *Advances in Financial Machine Learning*), used together:

1. **Meta-labeling.** The model does not choose trades. The pairs strategy
   proposes an entry; the model answers one question: *is this entry likely to
   work?* A "no" skips the trade.
2. **Purged time split.** A label looks 20 trading days ahead. A training row
   dated shortly before the validation period therefore has a label that was
   decided *inside* the validation period. `purged_time_split` removes those
   rows.
3. **Judging the model by a backtest.** The model is scored by running the
   backtest engine twice on the same bars, with and without the model as a
   filter, and comparing profit and risk.

### Why we chose it

- A random train/validation split, the usual default, leaks the future here:
  neighbouring signals share most of their 20-day outcome window. The model
  would look better in validation than it is.
- AUC measures ranking, not money. A filter can have a good AUC and still skip
  the profitable trades. The backtest measures what the business cares about.
- It keeps the roles clean. Statistics decide when to enter; the model only
  vetoes. If the model is poor, the strategy still runs.

### How it works here

| Step | Code |
|---|---|
| Label each entry: 1 if the spread reverts within 20 bars before the stop-loss, else 0 | `ml/labels.py` |
| Time split with a purge gap of 20 business days | `ml/dataset.py` → `purged_time_split` |
| Wrap the strategy; drop an entry when the model's probability is below 0.5 | `ml/filtered_strategy.py` → `MetaLabelFilter` |
| Run both backtests and write the comparison | `ml/backtest_filter.py` |

The filter only acts from 2024-05-20, the first day of the model's validation
period. Before that date the two runs are identical, so any difference comes
from data the model never trained on.

### Proof it worked

**The purge removes the overlapping rows.** On the default dataset the 301
labelled entries become 224 training rows and 76 validation rows; one row is
purged because its label window reaches the validation period
(see [ml.md](ml.md) §2).

**The comparison ran end to end.** From
[`pngs/phase3_backtest_filter.json`](pngs/phase3_backtest_filter.json):

| Metric | Without filter | With filter |
|---|---|---|
| Sharpe ratio | 1.58 | 1.62 |
| Total P&L | $47,673 | $48,100 |
| Max drawdown | 0.22% | 0.22% |
| Leg trades | 516 | 496 |
| Entries scored by the model | – | 61 (51 taken, 10 skipped) |

**What this does and does not show.** The technique works: the filter skipped
10 of 61 out-of-sample entries and the engine produced a comparable result for
both runs. It does *not* show that the model adds value. The difference of
$427 is inside the noise for this few trades, and the model's validation AUC
is 0.38, which is no better than chance. The generated spreads are random by
construction, so that is the expected outcome.

**Tests.**

| Test file | What it checks |
|---|---|
| `tests/unit/test_dataset.py` | Purge: a training row whose label window ends before validation is kept; one whose window reaches validation is removed. Boundary: one business day before, and exactly on, the validation start |
| `tests/unit/test_labels.py` | A Hypothesis test: adding later data never changes a label that was already decided |
| `tests/unit/test_filtered_strategy.py` | Threshold boundary (0.49 / 0.50 / 0.51); a skipped entry also drops its exit; the feature lookup never returns a later day's row |
| `tests/unit/test_backtest_filter.py` | A filter that accepts everything reproduces the baseline exactly; one that rejects everything never trades |

**Reproduce** (needs the `core` profile running and DP1–DP3 loaded):

```
uv run python -m ml.backtest_filter --out docs/pngs/phase3_backtest_filter.json
uv run pytest tests/unit/test_dataset.py tests/unit/test_filtered_strategy.py \
    tests/unit/test_backtest_filter.py -o addopts=""
```

> **TODO (screenshot):** terminal output of the two commands above, saved to
> `docs/pngs/` and linked here.

---

## Idea 2 — Metamorphic tests that show the backtester cannot see the future

### What it is

A backtest has no answer key: nobody knows the "correct" profit for a
strategy, so a test cannot simply compare the output with an expected number.
**Metamorphic testing** checks a *relation between two runs* instead: change
the input in a known way and assert how the output must, or must not, change.

The most damaging backtest bug is **lookahead**: a decision on day *N* that
used a price from day *N+1*. It makes every result look better than it could
be in real trading, and nothing crashes.

### Why we chose it

- Lookahead cannot be found by reading results. A relation such as "changing
  the future must not change the past" catches it directly.
- The same approach covers two more properties that a trustworthy backtester
  needs: the same input always gives the same result, and higher trading costs
  never give a higher profit.
- The engine is written for this. Strategies read prices only through
  `PointInTimeDataHandler`, and orders fill at the next bar's open.

### How it works here

| Relation | Change to the input | What must hold | Test |
|---|---|---|---|
| No lookahead | Multiply every close after the backtest end date by 1000 | Same number of fills, same fill prices | `TestNoLookahead::test_future_price_change_does_not_affect_past_signals` |
| History cut-off | Ask for history at a timestamp in the middle of the data | Every returned bar has `ts <= cutoff` | `TestNoLookahead::test_get_history_strict_cutoff` |
| Determinism | Run twice on identical data | Identical equity curve (to 1e-9) | `TestDeterminism::test_deterministic_equity_curve` |
| Cost monotonicity | Double every cost | Total P&L does not go up | `TestCostMonotonicity::test_higher_costs_never_better_pnl` |
| Robustness (Hypothesis) | Random prices, including negative, zero and NaN | The strategy never raises | `test_strategy_never_crashes_on_corrupted_prices` |
| Robustness (Hypothesis) | Random finite floats and window sizes | `rolling_zscore` never raises | `test_zscore_no_crash_on_valid_floats` |

All six are in `tests/property/test_property_backtest.py`. Two more tests of
the cut-off are in `tests/unit/test_data_handler.py` (`TestNoLookahead`).

The same rule is tested where the ML model joins the engine:
`tests/unit/test_filtered_strategy.py::TestFrameFeatureLookup::test_never_returns_a_later_row` checks
that the filter's feature lookup returns the row dated the same day as the
signal and never a later one.

### Proof it worked

- The tests above are in the repository and are part of `uv run pytest`.
- The engine they protect produced the demo result in
  [`pngs/phase1_demo_results.json`](pngs/phase1_demo_results.json) and the
  equity curve below.

![Equity curve of the Phase 1 demo backtest](pngs/phase1_demo_equity_curve.png)

*Equity curve from the Phase 1 demo run on generated data. It shows the engine
runs end to end; it is not evidence of a profitable strategy.*

**Reproduce:**

```
uv run pytest tests/property tests/unit/test_data_handler.py -o addopts="" -v
```

> **TODO (screenshot):** the `-v` output of the command above showing the test
> names passing, saved to `docs/pngs/` and linked here.

### Known limit

The no-lookahead relation changes bars *after the backtest end date*. That
proves the engine does not read past the end of the run. A stricter version
would change a bar in the middle of the run and assert that every fill before
that bar is unchanged. The cut-off test on `get_history` covers the middle of
the data at the data-handler level, but not through a full backtest.
