#!/usr/bin/env bash
# Roll an API image out to kind with Helm; --atomic rolls back by itself if pods never get ready.
# Usage: scripts/ci/deploy_api.sh signal_api|regime_api [EXPECTED_MODEL_VERSION]
source "$(dirname "$0")/lib.sh"

service=${1:?usage: $0 signal_api|regime_api [EXPECTED_MODEL_VERSION]}
expected_version=${2:-}
tag=$(release_tag)
release=${service//_/-}
case "$service" in
  signal_api) port=30080 ;;
  regime_api) port=30081 ;;
  *) die "unknown service $service" ;;
esac

log "loading pairlab-$service:$tag into kind"
kind load docker-image "pairlab-$service:$tag" --name "$KIND_CLUSTER" >/dev/null

log "helm upgrade $release -> $tag"
helm upgrade --install "$release" "$DEPLOY_DIR/deploy/helm/pairlab-service" \
  -f "$DEPLOY_DIR/deploy/helm/values/$release.yaml" \
  --set image.tag="$tag" --atomic --wait --timeout 6m
helm history "$release" --max 3

ready() { curl -fsS "http://localhost:$port/readyz" >/dev/null; }
wait_for 120 5 ready || die "$release is not ready on :$port"
log "$release ready on :$port"

if [[ "$service" == signal_api && -n "$expected_version" ]]; then
  for pair in KO__PEP XOM__CVX JPM__BAC GS__MS; do
    body=$(curl -fsS -X POST "http://localhost:$port/signal" \
      -H 'content-type: application/json' -d "{\"pair_id\": \"$pair\"}" || true)
    if [[ $(jq -r '.variant // empty' <<<"$body") == champion ]]; then
      served=$(jq -r .model_version <<<"$body")
      [[ "$served" == "$expected_version" ]] || die "champion serves $served, expected $expected_version"
      log "champion serves $served for $pair"
      exit 0
    fi
  done
  log "WARNING: no smoke pair was routed to the champion; version not checked"
fi
