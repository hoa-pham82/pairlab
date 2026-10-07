# syntax=docker/dockerfile:1
# Build from the repo root: docker build -f services/online_writer/Dockerfile -t pairlab-online_writer .

# Stage 1: build — install the "services" extra into a venv
FROM python:3.11-slim AS build

RUN pip install --no-cache-dir uv==0.12.23

WORKDIR /app
COPY pyproject.toml uv.lock ./

RUN uv venv /app/.venv && \
    uv sync --frozen --no-dev --no-install-project --extra services \
    && find /app/.venv -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Stage 2: runtime — venv, job code and the Feast repo definition only
FROM python:3.11-slim AS runtime

RUN useradd --create-home app

WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY services/__init__.py services/__init__.py
COPY services/online_writer/ services/online_writer/
COPY platform/feature_repo/ platform/feature_repo/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app" \
    PYTHONUNBUFFERED=1

USER app
EXPOSE 9108
HEALTHCHECK --interval=15s --timeout=3s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:9108/metrics')"

CMD ["python", "-m", "services.online_writer.writer"]
