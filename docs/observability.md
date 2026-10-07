# Observability

## Stack

| Component | Port | Purpose |
|-----------|------|---------|
| Prometheus | 9090 | Time-series metric collection |
| Grafana | 3000 | Dashboards (`admin` / `admin`) |
| cAdvisor | 8084 | Container CPU/memory/network metrics |
| Pushgateway | 9091 | Batch-job and DAG metrics (drift DAG) |

Start with `docker compose -f platform/compose.yml --profile obs up -d`.

## Prometheus scrape targets

| Job | Target | What it exposes |
|-----|--------|----------------|
| `signal_api` | `signal_api:8000/metrics` | FastAPI request rate, latency histograms, prediction counters |
| `regime_api` | `regime_api:8001/metrics` | FastAPI request rate, latency histograms, regime label counters |
| `llama_cpp` | `llama-cpp:8080/metrics` | Prompt/generation token rates, queue depth |
| `cadvisor` | `cadvisor:8080` | Per-container CPU, memory, network I/O |
| `pushgateway` | `pushgateway:9091` | Drift DAG gauges (see below) |

Config: `platform/docker/prometheus.yml`.  Pushgateway uses `honor_labels: true` so the `pair_id` label from the drift DAG is preserved.

## Grafana dashboard — "Pairlab Platform"

Auto-provisioned from `platform/docker/grafana-dashboard-pairlab.json`.  Four sections:

### API — Requests
- **signal_api request rate** — `rate(http_requests_total{job="signal_api"}[1m])` by method/handler/status
- **regime_api request rate** — same for regime_api
- **signal_api latency p50/p95/p99** — `histogram_quantile` over `http_request_duration_seconds_bucket`
- **regime_api latency p50/p95/p99** — same

### Drift Monitoring
Metrics pushed by `ml/drift.py` via `ml_drift_dag.py`.

| Metric | Type | Meaning |
|--------|------|---------|
| `pairlab_drift_detected{pair_id}` | Gauge | 1 = drift detected, 0 = OK |
| `pairlab_spread_change_psi{pair_id}` | Gauge | PSI of spread-change distribution vs training baseline |
| `pairlab_zscore_psi{pair_id}` | Gauge | PSI of z-score distribution vs training baseline |
| `pairlab_drift_last_run_seconds` | Gauge | Unix timestamp of last drift check |

PSI thresholds: < 0.1 green, 0.1–0.2 yellow (slight drift), > 0.2 red (retrain triggered).

### Container Resources (cAdvisor)
- CPU usage % by container
- Memory usage MB by container
- Network receive/transmit bytes/s

### LLM (llama.cpp)
- Prompt token throughput (`llamacpp_prompt_tokens_total`)
- Generation token throughput (`llamacpp_tokens_predicted_total`)

## Adding new metrics

From a FastAPI service: the `prometheus-client` library is in the services extra; use `Counter`, `Histogram` and `Gauge` from it and expose `/metrics` with `make_asgi_app()` mounted at that path.

From a DAG or script: push to the Pushgateway at `http://pushgateway:9091` (or `http://localhost:9091` from the host):

```python
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
registry = CollectorRegistry()
g = Gauge("my_metric", "description", ["label"], registry=registry)
g.labels(label="value").set(42)
push_to_gateway("pushgateway:9091", job="my_job", registry=registry)
```
