#!/usr/bin/env bash
# Build a compose service's image from this checkout, tagged with the release.
# Usage: scripts/ci/build_image.sh airflow|online_writer
source "$(dirname "$0")/lib.sh"

tag=$(release_tag)
case "${1:?usage: $0 airflow|online_writer}" in
  airflow)
    # compose names the image <project>-<service>; :latest is what `up` runs.
    docker build -f "$SOURCE_DIR/platform/docker/airflow.Dockerfile" \
      -t "$COMPOSE_PROJECT-airflow:$tag" -t "$COMPOSE_PROJECT-airflow:latest" "$SOURCE_DIR/platform/docker"
    image="$COMPOSE_PROJECT-airflow:$tag"
    ;;
  online_writer)
    docker build -f "$SOURCE_DIR/services/online_writer/Dockerfile" \
      -t "pairlab-online_writer:$tag" "$SOURCE_DIR"
    image="pairlab-online_writer:$tag"
    ;;
  *) die "usage: $0 airflow|online_writer" ;;
esac
log "built $image"
