# Web APIs

Two FastAPI services. Both are async, validate input with Pydantic, expose
`/healthz`, `/readyz` and Prometheus `/metrics`, and take their data sources as
constructor arguments so tests can replace them.

| Service | Port | Main endpoint | Purpose |
|---|---|---|---|
| `signal_api` | 8000 | `POST /signal` | Online features for a pair → model → take the trade or not |
| `regime_api` | 8001 | `POST /regime` | Drift detection: is the pair relationship still intact? |

Run them locally (needs the `core` Compose profile):

```
uv run uvicorn services.signal_api.app:build_default_app --factory --port 8000
uv run uvicorn services.regime_api.app:build_default_app --factory --port 8001
```

## 1. signal_api

**Flow:** request with a pair ID → features from the Feast online store (Redis)
→ meta-label model → response.

```
POST /signal   {"pair_id": "KO__PEP"}
200            {"pair_id": "KO__PEP", "prob": 0.705, "take_trade": true,
                "threshold": 0.5, "model_version": "file-741300eb30cb",
                "features": {"zscore": -1.60, "hedge_ratio": 0.39, ...}}
```

| Situation | Status |
|---|---|
| Pair ID not in the form `AAA__BBB` | 422, before any lookup |
| No online features for the pair | 404 |
| A feature is NaN or infinite | 422, no prediction is made |
| Model or feature store not loaded | 503 |

- `GET /healthz` — the process is alive (Kubernetes liveness probe).
- `GET /readyz` — model and feature store are loaded (readiness probe); 503 otherwise.
- **Async:** the endpoint is `async`; the blocking Feast and model calls run in
  a thread pool, so one slow lookup does not stall other requests.
- **Settings:** `FEAST_REPO` and `TAKE_THRESHOLD` environment variables.
- **Model source:** with `MLFLOW_TRACKING_URI` set, the registry model
  `meta_label` at alias `production` (`MODEL_NAME`, `MODEL_ALIAS` to change).
  Without it, the joblib file at `MODEL_PATH`.

### Model A/B test

When the MLflow registry has a model at the `challenger` alias, `signal_api`
splits requests between it and the `production` model (the champion).

- The split is by a stable hash of the pair ID, so a pair always gets the same
  model. `CHALLENGER_SHARE` (default 0.5) sets the share sent to the challenger.
- Raising the share only moves pairs from champion to challenger, never back.
- The response carries `variant` (`champion` or `challenger`) and the
  `model_version` that scored it.
- `signal_api_decisions_total{take_trade, model_version, variant}` lets a
  dashboard compare take rates per variant.
- With no challenger in the registry, everything is `champion`.

This is a paper-signal comparison: both variants only produce decisions; no
live money follows either. In trading, a live A/B test would mean trading real
money on the worse model.

## 2. regime_api

**Flow:** request with a pair ID → last 315 daily closes from the warehouse →
two statistics → regime.

```
POST /regime   {"pair_id": "KO__PEP"}
200            {"pair_id": "KO__PEP", "regime": "stable",
                "coint_pvalue": 0.0095, "psi": 0.123, "bars_used": 315}
```

| Statistic | What it measures | Shifting above | Broken above |
|---|---|---|---|
| Cointegration p-value, latest 252 bars | Do the two prices still move together? | 0.05 | 0.10 |
| PSI of daily spread changes, latest 63 bars vs the 252 before | Has the spread started behaving differently? | 0.20 | 0.40 |

The worse of the two decides. The p-value cut-offs are the strategy's own
(PLAN.md §4.1). The PSI cut-offs were calibrated on generated data: without
drift the 95th percentile is about 0.15; after a regime shift the median is 0.64.

**Current result on the default dataset:** all four pairs come back `stable`
(p-values 0.001–0.022, PSI 0.11–0.15). With the generator's first spread
settings three of four came back `broken` although no drift was injected: the
spread was too noisy for the cointegration test to see the link in one year
(only 28% of one-year windows passed). The generator now uses `ou_sigma` 0.005
and `ou_theta` 0.10.

| Situation | Status |
|---|---|
| Unknown pair | 404 |
| Fewer than 315 bars, or a zero, negative, NaN or infinite price | 422 |
| Warehouse unreachable | `/readyz` returns 503 |

## 3. Metrics

| Metric | Labels |
|---|---|
| `<service>_requests_total` | `route`, `status` |
| `<service>_request_seconds` (histogram) | `route` |
| `signal_api_decisions_total` | `take_trade`, `model_version` |
| `regime_api_assessments_total` | `regime` |
| `regime_api_spread_psi` (gauge) | `pair_id` |

## 4. Tests

```
uv run pytest tests/services -o addopts="" --cov=services --cov-report=term-missing --cov-fail-under=90
```

- **Coverage:** 92 tests, 95% of `services/`. Every application module is at
  100%; the uncovered lines are the Locust scenario file, which is a load-test
  script. Output saved in
  [`pngs/phase3_services_coverage.txt`](pngs/phase3_services_coverage.txt).
- **Fixtures and fakes:** `tests/services/conftest.py` replaces Feast, the
  model and Postgres with in-memory fakes, so the tests need no running stack.
- **Equivalence partitions and boundary values:** documented in comments above
  the parametrized tests — pair-ID format, decision threshold (0.49 / 0.5 /
  0.51), regime cut-offs (exactly at / just above), history length (315 / 314).
- **Property-based idempotency (Hypothesis):** the same request always returns
  the same response, for random feature values and for a real LightGBM model.
- **Load test (Locust):** [`pngs/locust_signal_api.html`](pngs/locust_signal_api.html).
  50 users for 30 s against the live stack on a laptop: 468 requests/s, median
  59 ms, 95th percentile 140 ms, 0 failures.

Dockerfiles exist at `services/signal_api/Dockerfile` and `services/regime_api/Dockerfile` but have not been built.

- **Mutation testing (mutmut):** 614 mutants of `services/`, `ml/labels.py` and
  `ml/dataset.py`: 513 killed, 71 survived, 30 crashed the test process.
  Score 83.6% counting crashes as not killed (87.8% of the mutants that ran to
  a verdict). Result saved in
  [`pngs/phase3_mutation_score.txt`](pngs/phase3_mutation_score.txt).
  Most survivors are in the PSI and regime-assessment arithmetic and in
  `LabelConfig` validation messages; the crashes are mutants of Prometheus
  metric and app setup code.

Not done yet: Compose wiring, Kubernetes deployment.
