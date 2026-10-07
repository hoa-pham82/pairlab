#!/usr/bin/env bash
# Recreate compose services whose definition or mounts changed, and wait until healthy.
# Usage: scripts/ci/refresh_services.sh airflow spark-master spark-worker
source "$(dirname "$0")/lib.sh"

(( $# > 0 )) || die "usage: $0 SERVICE..."
log "refreshing: $*"
compose up -d --no-deps --wait --wait-timeout 600 "$@"
