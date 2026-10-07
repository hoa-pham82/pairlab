# pairlab — Pairs-Trading Research Platform

A pairs-trading research system built as coursework and as a quant /
data-engineering portfolio piece. It covers a backtesting engine, the data
platform that feeds it, an ML model that filters trades, and an LLM analyst
that explains the result.

---

## Table of Contents

1. [Business domain](#1-business-domain)
2. [System deployment diagram](#2-system-deployment-diagram)
3. [Deployable units](#3-deployable-units)
4. [Repository layout](#4-repository-layout)
5. [Quick start](#5-quick-start)
6. [Status](#6-status)
7. [Tests](#7-tests)
8. [Docs](#8-docs)
9. [Scope and known omissions](#9-scope-and-known-omissions)

---

## 1. Business domain

**Pairs trading** is a market-neutral strategy. Find two stocks whose prices
move together (they are _cointegrated_), wait for the gap between them to
widen, then bet that it closes again: buy the cheap one, sell the expensive
one.

A research team needs four things to do this safely:

| Need                                                    | What this project builds                                                 |
| ------------------------------------------------------- | ------------------------------------------------------------------------ |
| Clean, point-in-time price data                         | Generator → Spark → Delta Lake and Postgres (bronze, silver, gold)       |
| A backtester that cannot cheat                          | `src/pairlab`: event-driven engine, no lookahead, realistic costs        |
| A way to skip weak signals                              | Meta-label model served by `signal_api`; drift detection by `regime_api` |
| A plain-language answer to "is KO/PEP still tradeable?" | An LLM analyst agent that calls the two APIs as tools                    |

New to the domain? Start with
[docs/learning/business-overview.md](docs/learning/business-overview.md).

---

## 2. System deployment diagram

![pairlab system deployment diagram](docs/pngs/architecture.png)

_Each icon is a deployable unit or a person using the system. Arrows follow
the data and are numbered in flow order; colours separate the flows. Source: [docs/architecture_diagram.py](docs/architecture_diagram.py),
editable copy in [docs/pngs/architecture.drawio](docs/pngs/architecture.drawio)._

The picture was drawn before the last changes. Where it differs from the code,
the code is right:

| In the diagram                                     | In the code today                                                  |
| -------------------------------------------------- | ------------------------------------------------------------------ |
| Ollama, Qwen 7B                                    | llama.cpp server, Qwen2.5-3B-Instruct                              |
| "Binance WS tick producer"                         | Generated ticks from `generator/streaming/tick_producer.py`        |
| GitHub Actions pushes an image to GHCR and deploys | CI runs lint, tests and a Docker build; it does not push or deploy |
| "champion alias"                                   | MLflow alias `production`                                          |
| Not shown                                          | Loki, Promtail, Tempo, cAdvisor, Pushgateway, the agent API        |

> **TODO:** regenerate `docs/pngs/architecture.png` so it matches the table in
> section 3.

---

## 3. Deployable units

Everything runs in Docker Compose (`platform/compose.yml`), grouped into
profiles so only the needed part is started.

| Profile    | Unit                          | Role                                                                                 | Host port                  |
| ---------- | ----------------------------- | ------------------------------------------------------------------------------------ | -------------------------- |
| `core`     | PostgreSQL                    | Warehouse (bronze / silver / gold), Feast offline store, Airflow and MLflow metadata | 5432                       |
| `core`     | MinIO                         | S3-compatible object store: `vendor-raw`, `delta-lake`, `mlflow-artifacts`           | 4566 (API), 9001 (console) |
| `core`     | Redis                         | Feast online store                                                                   | 6379                       |
| `stream`   | Redpanda (+ console)          | Kafka-compatible broker: topics `ticks.raw`, `bars.1m`                               | 19092, 8080 (console)      |
| `stream`   | Flink JobManager, TaskManager | Ticks → 1-minute bars                                                                | 8081                       |
| `platform` | Spark master, worker          | Batch jobs DP1, DP2, DP3                                                             | 8082                       |
| `platform` | Airflow                       | Runs the pipelines in order                                                          | 8083                       |
| `ml`       | MLflow                        | Experiment tracking, model registry                                                  | 5001                       |
| `ml`       | `signal_api`                  | FastAPI: online features → model → take the trade or not                             | 8000                       |
| `ml`       | `regime_api`                  | FastAPI: drift detection (cointegration p-value and PSI)                             | 8001                       |
| `obs`      | Prometheus, Pushgateway       | Metrics                                                                              | 9090, 9091                 |
| `obs`      | Grafana                       | Dashboards                                                                           | 3000                       |
| `obs`      | cAdvisor                      | Container CPU, memory, network                                                       | 8084                       |
| `obs`      | Loki, Promtail, Tempo         | Logs and traces                                                                      | 3100, 3200                 |
| `llm`      | llama.cpp server              | Local LLM, OpenAI-compatible API                                                     | 8085                       |
| `llm`      | `mcp_server`                  | FastMCP: tools `get_pair_features`, `check_regime`                                   | 8002                       |
| `llm`      | `agent_api`                   | Analyst agents (features, regime, coordinator)                                       | 8003                       |
| `gateway`  | NGINX                         | HTTPS, Basic Auth and rate limit in front of the APIs, MLflow and Grafana            | 8443                       |

The backtesting engine (`src/pairlab`) is a Python package and CLI. It imports
no infrastructure, so it also runs with none of the above.

---

## 4. Repository layout

```
src/pairlab/          Backtesting engine: events, data handler, strategy, portfolio,
                      execution, costs, metrics, tear sheet, CLI. No infrastructure imports.
tests/
  unit/               Engine, generator and ML unit tests
  property/           Property-based and metamorphic tests (determinism, no lookahead)
  integration/        Full backtest loop
  services/           APIs, MCP server, agents (fakes instead of running services)
generator/            Synthetic market data
  configs/            default.yaml, drift.yaml, e2e-small.yaml
  offline/            Daily bars and symbol master
  streaming/          Tick stream (JSONL file or Redpanda)
  problems/           Injected data problems (duplicates, schema evolution)
  sinks/              MinIO and Postgres writers
  scripts/            generate_all.py, ingest_to_platform.py
platform/
  compose.yml         All deployable units, by profile
  spark/jobs/         dp1_ingest_bronze (baseline and optimized), dp2_silver_gold,
                      dp3_features, optimize_delta
  flink/sql/          ticks_to_bars_1m.sql
  airflow/dags/       dp1, dp2, dp3, feast_materialize, ml_train, ml_drift
  feature_repo/       Feast entities and feature views
  warehouse/init.sql  Postgres schemas, tables, indexes
  docker/             Dockerfiles; Prometheus, Grafana, Loki, Tempo, NGINX config
ml/                   Labels, training table, training, pipeline, data versioning,
                      drift check, meta-label filter
services/             signal_api, regime_api, mcp_server, agent
notebooks/            ml.ipynb, agent_demo.ipynb
scripts/              backtest_from_gold.py, download_model.sh
configs/backtest/     Backtest configurations
docs/                 One file per rubric area; docs/learning/ for concepts;
                      docs/pngs/ for proof
.github/workflows/    CI: lint → tests → Docker build
PLAN.md               Scope, phases and rubric mapping
```

---

## 5. Quick start

```bash
# macOS only: LightGBM needs the OpenMP library
brew install libomp

# Install
uv sync --all-extras

# Run the tests
uv run pytest

# Run a backtest on generated data (no Docker needed)
uv run python generator/scripts/generate_all.py
uv run pairlab backtest --config configs/backtest/demo.yaml
```

To start the platform, copy `.env.example` to `.env` first and set
`MINIO_ROOT_USER`, `MINIO_ROOT_PASSWORD` and `AIRFLOW__CORE__FERNET_KEY`.

```bash
# Storage: Postgres, MinIO, Redis
docker compose -f platform/compose.yml --env-file .env --profile core up -d

# Add streaming, Spark, Airflow, MLflow and the two APIs
docker compose -f platform/compose.yml --env-file .env \
  --profile core --profile stream --profile platform --profile ml up -d

# Add observability, the LLM layer, the gateway
docker compose -f platform/compose.yml --env-file .env \
  --profile obs --profile llm --profile gateway up -d
```

Spark and Flink need connector JARs, and the LLM needs a model file. Download
them once:

```bash
bash platform/spark/download_jars.sh
bash platform/flink/download_jars.sh
bash scripts/download_model.sh
```

| UI                | Address                    | Login                                           |
| ----------------- | -------------------------- | ----------------------------------------------- |
| Airflow           | http://localhost:8083      | `admin` / `admin`                               |
| Spark master      | http://localhost:8082      | –                                               |
| Flink             | http://localhost:8081      | –                                               |
| Redpanda console  | http://localhost:8080      | –                                               |
| MinIO console     | http://localhost:9001      | values from `.env`                              |
| MLflow            | http://localhost:5001      | –                                               |
| Grafana           | http://localhost:3000      | `admin` / `admin`                               |
| `signal_api` docs | http://localhost:8000/docs | –                                               |
| Gateway           | https://localhost:8443     | `GATEWAY_USER` / `GATEWAY_PASSWORD` from `.env` |

---

## 6. Status

| Phase | Focus                                                   | State                                   |
| ----- | ------------------------------------------------------- | --------------------------------------- |
| 1     | Engine, generator, tests, CI                            | Built                                   |
| 2     | Data platform: Spark, Delta Lake, Flink, Airflow, Feast | Built, with the gaps listed below       |
| 3     | ML model, APIs, observability, LLM agents               | Built on Docker Compose                 |
| 4     | Gateway, logs and traces, IaC, Kubernetes               | Gateway added; the rest is in section 9 |

Known gaps in Phase 2, from [docs/review-2026-10-07.md](docs/review-2026-10-07.md)
and [docs/processing_jobs.md](docs/processing_jobs.md):

- No DataHub; no lineage or data-contract screenshots.
- The Spark skew handling is not yet shown against a baseline in the Spark UI.
- No baseline Flink job to compare with the optimized one.
- `dim_symbol` does not yet keep the history of a sector change.
- Most UI screenshots (Spark, Flink, Airflow DP2 and DP3, DBeaver) are still to
  be captured; each doc marks them with **TODO**.

---

## 7. Tests

```bash
uv run pytest                                         # everything, with the coverage gate
uv run pytest tests/unit -o addopts=""                # one folder, without the gate
uv run pytest tests/services -o addopts="" --cov=services --cov-fail-under=90
```

The coverage gate is 90% across `pairlab`, `ml` and `services` (set in
`pyproject.toml`).

| Folder               | Test functions | What they cover                                                                                 |
| -------------------- | -------------- | ----------------------------------------------------------------------------------------------- |
| `tests/unit/`        | 229            | Engine parts, generator, labels, training table, pipeline, versioning, drift, meta-label filter |
| `tests/property/`    | 6              | Determinism, no lookahead, cost monotonicity, random-input robustness                           |
| `tests/integration/` | 6              | A full backtest from bars to tear sheet                                                         |
| `tests/services/`    | 94             | `signal_api`, `regime_api`, MCP server, agents, LLM benchmark                                   |

Counts are test functions; a parametrized function counts once.

| Technique                                  | Where to look                                                           | Proof                                                     |
| ------------------------------------------ | ----------------------------------------------------------------------- | --------------------------------------------------------- |
| Fixtures and fakes for the APIs            | `tests/services/conftest.py`                                            | [coverage output](docs/pngs/phase3_services_coverage.txt) |
| Equivalence partitions and boundary values | Comments above the parametrized tests, e.g. `tests/unit/test_labels.py` | [docs/web_apis.md](docs/web_apis.md) §4                   |
| Property-based idempotency (Hypothesis)    | `tests/services/test_signal_api.py`, `tests/unit/test_labels.py`        | [docs/web_apis.md](docs/web_apis.md) §4                   |
| Metamorphic no-lookahead tests             | `tests/property/test_property_backtest.py`                              | [docs/novel_ideas.md](docs/novel_ideas.md)                |
| Mutation testing (mutmut)                  | `[tool.mutmut]` in `pyproject.toml`                                     | [mutation score](docs/pngs/phase3_mutation_score.txt)     |
| Load test (Locust)                         | `services/signal_api/locustfile.py`                                     | [HTML report](docs/pngs/locust_signal_api.html)           |

---

## 8. Docs

`README.md` is the summary. The detail for each rubric area is in `docs/`.

**Data platform**

| Doc                                                | Content                                                                 |
| -------------------------------------------------- | ----------------------------------------------------------------------- |
| [docs/generator.md](docs/generator.md)             | Generator configuration, injected data problems, data volume and format |
| [docs/processing_jobs.md](docs/processing_jobs.md) | Spark DP1–DP3 (baseline and optimized) and the Flink job                |
| [docs/data_storage.md](docs/data_storage.md)       | Object store, Delta Lake, Postgres, Redis                               |
| [docs/orchestration.md](docs/orchestration.md)     | Airflow DAGs, stages, connections and variables                         |
| [docs/schema_design.md](docs/schema_design.md)     | Tables per zone, SCD2 dimension, feature tables, naming, indexes        |
| [docs/feature_store.md](docs/feature_store.md)     | Feast feature views and the reason for each TTL                         |
| [docs/docker.md](docs/docker.md)                   | Images, multi-stage builds, Compose profiles                            |

**ML, serving and agents**

| Doc                                                  | Content                                                                                       |
| ---------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| [docs/ml.md](docs/ml.md)                             | Label table, training table, notebook, training pipeline, backtest with and without the model |
| [docs/versioning.md](docs/versioning.md)             | Model versions in MLflow; incremental data versions in Delta                                  |
| [docs/web_apis.md](docs/web_apis.md)                 | `signal_api`, `regime_api`, metrics, tests, load test                                         |
| [docs/observability.md](docs/observability.md)       | Prometheus, Grafana dashboard, drift metrics                                                  |
| [docs/llm_agents.md](docs/llm_agents.md)             | llama.cpp, benchmark, MCP server, agents, telemetry                                           |
| [docs/low_level_design.md](docs/low_level_design.md) | Five key ML components and the design patterns used                                           |

**Across the project**

| Doc                                        | Content                                      |
| ------------------------------------------ | -------------------------------------------- |
| [docs/novel_ideas.md](docs/novel_ideas.md) | Two techniques beyond the course, with proof |
| [docs/pngs/](docs/pngs/)                   | Screenshots and proof files                  |

---

## 9. Scope and known omissions

The system runs on one laptop with Docker Compose. These rubric items are not
built:

| Area                 | Not built                                                                   | Used instead                                                                                                                                                                   |
| -------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Governance           | DataHub lineage, data contracts                                             | SQL checks in each DAG's `validate` task                                                                                                                                       |
| Kubernetes           | Helm charts, rolling update, KEDA autoscaling, KServe, service mesh         | Compose services with health checks                                                                                                                                            |
| ML pipelines         | Kubeflow, distributed training                                              | Airflow DAG `ml_train`                                                                                                                                                         |
| Agents on Kubernetes | Agent registry, sandbox, multi-replica agents, warm-up mode                 | `mcp_server` and `agent_api` as Compose services                                                                                                                               |
| RAG                  | Chunking and embedding pipeline                                             | –                                                                                                                                                                              |
| CI/CD                | One pipeline per data pipeline, API and job; automatic deploy               | One CI workflow: lint, tests, Docker build                                                                                                                                     |
| IaC                  | Terraform, Ansible                                                          | `platform/compose.yml`                                                                                                                                                         |
| Security             | Vault                                                                       | `.env` (git-ignored) and `.env.example`                                                                                                                                        |
| A/B testing          | Live-money A/B (deliberately: it would trade real money on the worse model) | Paper-signal split in `signal_api` (MLflow `production` vs `challenger`) and a session split in `agent_api` (two prompt sets), compared offline by `services/agent/ab_eval.py` |
