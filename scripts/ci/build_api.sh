#!/usr/bin/env bash
# Build an API image from this checkout (never the deploy dir); for signal_api, bake in the registry's production model.
# Usage: scripts/ci/build_api.sh signal_api|regime_api
# Writes IMAGE=... (and MODEL_VERSION=...) to $GITHUB_OUTPUT when it is set.
source "$(dirname "$0")/lib.sh"

service=${1:?usage: $0 signal_api|regime_api}
tag=$(release_tag)
image="pairlab-$service:$tag"
network="${PAIRLAB_DOCKER_NETWORK:-pairlab-network}"

log "building $service base image"
docker build -q -f "$SOURCE_DIR/services/$service/Dockerfile" -t "pairlab-$service:base-$tag" "$SOURCE_DIR"

if [[ "$service" == signal_api ]]; then
  out="$SOURCE_DIR/build/model"
  rm -rf "$out" && mkdir -p "$out"
  log "exporting meta_label@production from MLflow"
  version=$(docker run --rm --network "$network" -v "$out:/out" "pairlab-$service:base-$tag" \
      python -m services.signal_api.package_model --tracking-uri http://mlflow:5000 --out /out \
    | sed -n 's/^MODEL_VERSION=//p')
  [[ -n "$version" ]] || die "no production model in the registry"
  log "packaging $version into $image"
  docker build -q -f "$SOURCE_DIR/services/signal_api/package.Dockerfile" \
    --build-arg BASE_IMAGE="pairlab-$service:base-$tag" --build-arg MODEL_VERSION="$version" \
    -t "$image" "$SOURCE_DIR"
  [[ -n "${GITHUB_OUTPUT:-}" ]] && echo "MODEL_VERSION=$version" >> "$GITHUB_OUTPUT"
else
  docker tag "pairlab-$service:base-$tag" "$image"
fi

[[ -n "${GITHUB_OUTPUT:-}" ]] && echo "IMAGE=$image" >> "$GITHUB_OUTPUT"
log "built $image ($(docker image inspect -f '{{.Size}}' "$image" | awk '{printf "%.0f MB", $1/1e6}'))"
