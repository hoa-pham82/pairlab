# Docker and Docker Compose

The whole system runs on one laptop with Docker Compose. This page lists the
images, explains how the ones built here are kept small, and records where the
size measurements go.

---

## 1. Compose file and profiles

File: `platform/compose.yml`. Services are grouped into **profiles** so that
only the part being worked on is started and the laptop's memory is not
spent on idle services.

| Profile | Services |
|---|---|
| `core` | `postgres`, `minio`, `minio-init`, `redis` |
| `stream` | `redpanda`, `redpanda-console`, `redpanda-init`, `flink-jobmanager`, `flink-taskmanager` |
| `platform` | `spark-master`, `spark-worker`, `airflow` |
| `ml` | `mlflow`, `signal_api`, `regime_api` |
| `obs` | `prometheus`, `grafana`, `cadvisor`, `pushgateway`, `loki`, `promtail`, `tempo` |
| `llm` | `llama-cpp`, `mcp_server`, `agent_api` |
| `gateway` | `nginx` |

```bash
docker compose -f platform/compose.yml --env-file .env --profile core up -d
docker compose -f platform/compose.yml --env-file .env \
    --profile core --profile stream --profile platform up -d
```

`--env-file .env` is required: the MinIO credentials and the Airflow Fernet
key are read from it. Copy `.env.example` to `.env` first.

What Compose is used for beyond starting containers:

| Feature | Example |
|---|---|
| Health checks and start order | `airflow` waits for `postgres` to be healthy; `agent_api` waits for `mcp_server` |
| One-shot setup containers | `minio-init` creates the buckets; `redpanda-init` creates the topics |
| Named volumes | `postgres_data`, `minio_data`, `redpanda_data` keep data across restarts |
| Log rotation | Every service: JSON logs, 10 MB per file, 3 files |
| Configuration as mounted files | Spark jobs, Airflow DAGs, Prometheus and Grafana config are mounted, so a change needs no rebuild |

---

## 2. Images

**Built in this repository**

| Image | Dockerfile | Used by | Multi-stage |
|---|---|---|---|
| Engine | `platform/docker/engine.Dockerfile` | CI build; `pairlab` CLI | Yes |
| Services | `services/signal_api/Dockerfile` | `signal_api`, `mcp_server`, `agent_api` (same image, different command) | Yes |
| Services | `services/regime_api/Dockerfile` | `regime_api` (differs only in port and command) | Yes |
| Airflow | `platform/docker/airflow.Dockerfile` | `airflow` | No |
| MLflow | `platform/docker/mlflow.Dockerfile` | `mlflow` | No |
| Gateway | `platform/docker/nginx.Dockerfile` | `nginx` | No |

**Pulled as published**

`postgres:16-alpine`, `redis:7-alpine`, `cgr.dev/chainguard/minio`,
`redpandadata/redpanda:v24.1.1`, `flink:1.19-scala_2.12`, `apache/spark:3.5.3`,
`prom/prometheus`, `grafana/grafana`, `grafana/loki`, `grafana/tempo`,
`ghcr.io/ggml-org/llama.cpp:server`. Postgres, Redis and the NGINX base are
Alpine variants.

---

## 3. How the built images are kept small

### Multi-stage build

A Python image needs a package installer and build tools to *install*
dependencies, but not to *run* the program. A multi-stage build uses two
images: the first installs, the second copies only the result.

```dockerfile
# Stage 1: build — has uv, resolves and installs dependencies into a virtualenv
FROM python:3.11-slim AS build
RUN pip install --no-cache-dir uv==0.12.23
COPY pyproject.toml .
COPY src/ src/
RUN uv venv /app/.venv && uv sync --no-dev --no-extra platform --no-extra llm --no-extra ml …

# Stage 2: runtime — only the virtualenv and the source
FROM python:3.11-slim AS runtime
COPY --from=build /app/.venv /app/.venv
COPY --from=build /app/src /app/src
```

(`platform/docker/engine.Dockerfile`, shortened.)

### Each optimization and what it removes

| Optimization | Where | What it keeps out of the final image |
|---|---|---|
| Multi-stage build | Engine, both service Dockerfiles | `uv`, `pip`'s caches, anything created while installing |
| `python:3.11-slim` base, not the full `python:3.11` | Engine, services, MLflow | Compilers and development headers |
| Install only the extras the image needs | Engine: no `platform`, `ml`, `llm`, `dev`. Services: only `services` | Spark, Airflow, test and LLM packages |
| `pip install --no-cache-dir` | All Dockerfiles | pip's download cache |
| Delete `__pycache__` and `.pyc` after install | Engine, services | Compiled bytecode (Python recreates it when needed) |
| `apt-get install --no-install-recommends`, then remove `/var/lib/apt/lists` | Services, Airflow | Suggested packages and the package index |
| `Dockerfile.dockerignore` allow-list | Services | Everything except `pyproject.toml`, `uv.lock`, `src`, `ml`, `services`, `platform/feature_repo` is kept out of the build context |
| Copy `pyproject.toml` and `uv.lock` before the source | Services | Nothing from the image; it lets Docker reuse the dependency layer when only code changes, so rebuilds are fast |

Not a size matter, but part of the same files: the service images run as a
non-root user (`app`) and declare a `HEALTHCHECK` on `/healthz`.

---

## 4. Image sizes before and after

> **TODO (measure and fill in).** No size in this table has been measured yet.
> `PLAN.md` records "engine image: 47 MB"; that is smaller than the
> uncompressed `python:3.11-slim` base, so treat it as unverified and measure
> again.

| Image | Before: single stage | After: multi-stage | Reduction |
|---|---|---|---|
| Engine | TODO | TODO | TODO |
| `signal_api` | TODO | TODO | TODO |
| `regime_api` | TODO | TODO | TODO |

**How to measure.** The `build` stage contains everything a single-stage
image would, so its size is the "before" number.

```bash
# After: the image as shipped
docker build -f platform/docker/engine.Dockerfile -t pairlab-engine:multi .

# Before: stop at the build stage
docker build -f platform/docker/engine.Dockerfile --target build -t pairlab-engine:single .

docker images pairlab-engine --format "table {{.Tag}}\t{{.Size}}"

# Which layers the size comes from
docker history pairlab-engine:multi
```

Repeat with `services/signal_api/Dockerfile` (tag `pairlab-signal_api`) and
`services/regime_api/Dockerfile`.

> **TODO (screenshot):** the output of `docker images` for both tags of each
> image, and of `docker history` for the engine image.

---

## 5. Known gaps

- **No `.dockerignore` at the repository root.** The engine image is built
  with the repository root as its context, so Docker first sends everything
  there, including `.venv/`, the connector JARs and the model file in
  `models/` (about 2 GB). The final image is unaffected, but the build is
  slow. An allow-list like the services' `Dockerfile.dockerignore` fixes it.
- **A failed install can pass unnoticed.** The install step in the engine and
  service Dockerfiles ends with `… || true`, which also hides a failure of
  `uv sync` earlier in the same command. Putting the cache clean-up in its own
  `RUN` line removes the risk.
- **The engine image is not built with the lock file.** It copies
  `pyproject.toml` only, so dependency versions can differ between builds.
  The service images use `uv sync --frozen` with `uv.lock`.
- **Airflow, MLflow and the gateway are single-stage.** They add packages to
  a published base image and have no build tools to leave behind, so a second
  stage would save little.
