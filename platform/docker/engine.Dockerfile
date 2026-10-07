# syntax=docker/dockerfile:1

# Stage 1: build — install all deps into a venv
FROM python:3.11-slim AS build

RUN pip install --no-cache-dir uv==0.12.23

WORKDIR /app
COPY pyproject.toml .
COPY src/ src/

# Install only engine deps (no platform, no dev)
RUN uv venv /app/.venv && \
    uv sync --no-dev --no-extra platform --no-extra llm --no-extra ml \
    && find /app/.venv -name "*.pyc" -delete \
    && find /app/.venv -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Stage 2: runtime — only the venv + app source
FROM python:3.11-slim AS runtime

WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY --from=build /app/src /app/src
COPY configs/ configs/

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src"

ENTRYPOINT ["pairlab"]
CMD ["--help"]
