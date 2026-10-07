# syntax=docker/dockerfile:1
# Build from the repo root: docker build -f services/signal_api/Dockerfile -t pairlab-signal_api .

# Stage 1: build — install the engine plus the "services" extra into a venv
FROM python:3.11-slim AS build

RUN pip install --no-cache-dir uv==0.12.23

WORKDIR /app
COPY pyproject.toml uv.lock ./

RUN uv venv /app/.venv && \
    uv sync --frozen --no-dev --no-install-project --extra services \
    && find /app/.venv -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Stage 2: runtime — venv, engine source, and service code only
FROM python:3.11-slim AS runtime

# libgomp1: OpenMP runtime LightGBM needs to load a model
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home app

WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY src/ src/
COPY ml/ ml/
COPY services/ services/
COPY platform/feature_repo/ platform/feature_repo/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src:/app" \
    PYTHONUNBUFFERED=1

USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --retries=5 CMD curl -fs http://localhost:8000/healthz || exit 1

CMD ["uvicorn", "services.signal_api.app:build_default_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
