# syntax=docker/dockerfile:1
# Bakes the exported production model into a built signal_api image (used by CD).
# docker build -f services/signal_api/package.Dockerfile \
#   --build-arg BASE_IMAGE=pairlab-signal_api:base --build-arg MODEL_VERSION=meta_label-v3 .
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG MODEL_VERSION
COPY --chown=app build/model/meta_label.joblib /app/model/meta_label.joblib

ENV MODEL_SOURCE=file \
    MODEL_PATH=/app/model/meta_label.joblib \
    MODEL_VERSION=${MODEL_VERSION}

LABEL org.opencontainers.image.description="signal_api with ${MODEL_VERSION} baked in"
