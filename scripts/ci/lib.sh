# Shared settings and helpers for the CD scripts; source it, don't run it.
set -euo pipefail

CI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$(cd "$CI_DIR/../.." && pwd)"
DEPLOY_DIR="${PAIRLAB_DEPLOY_DIR:-$HOME/pairlab-deploy}"
COMPOSE_PROJECT="${PAIRLAB_COMPOSE_PROJECT:-platform}"
KIND_CLUSTER="${PAIRLAB_KIND_CLUSTER:-pairlab}"
export KUBECONFIG="${PAIRLAB_KUBECONFIG:-$DEPLOY_DIR/deploy/kind/kubeconfig}"
# All profiles, so named services and their dependencies resolve; commands still act only
# on the services they name.
export COMPOSE_PROFILES="${PAIRLAB_COMPOSE_PROFILES:-core,stream,platform,ml,obs,secrets,llm,gateway}"

log() {
  local now
  now=$(date +%H:%M:%S)
  printf '[cd %s] %s\n' "$now" "$*"
}

die() {
  log "ERROR: $*"
  exit 1
}

# docker compose against the deployed release, never the developer's working tree.
compose() {
  docker compose -p "$COMPOSE_PROJECT" -f "$DEPLOY_DIR/platform/compose.yml" \
    --env-file "$DEPLOY_DIR/.env" "$@"
}

release_tag() {
  if [[ -n "${PAIRLAB_RELEASE:-}" ]]; then
    echo "${PAIRLAB_RELEASE:0:12}"
  elif [[ -f "$DEPLOY_DIR/RELEASE" ]]; then
    cut -c1-12 "$DEPLOY_DIR/RELEASE"
  else
    echo dev
  fi
}

# Retry "$@" every $2 seconds until it succeeds or $1 seconds pass.
wait_for() {
  local timeout=$1 interval=$2; shift 2
  local deadline=$((SECONDS + timeout))
  until "$@"; do
    (( SECONDS < deadline )) || return 1
    sleep "$interval"
  done
}
