#!/usr/bin/env bash
# Stream job 2: start the release's bars.1m -> Feast online store consumer and check it consumes.
# The image must already exist: scripts/ci/build_image.sh online_writer
source "$(dirname "$0")/lib.sh"

export ONLINE_WRITER_TAG
ONLINE_WRITER_TAG=$(release_tag)
docker image inspect "pairlab-online_writer:$ONLINE_WRITER_TAG" >/dev/null \
  || die "pairlab-online_writer:$ONLINE_WRITER_TAG is not built"

compose up -d --no-deps --wait --wait-timeout 180 online_writer

group_stable() {
  compose exec -T redpanda rpk group describe online-writer 2>/dev/null | grep -qE '^STATE +Stable'
}
wait_for 120 5 group_stable || die "consumer group online-writer has no active member"
compose exec -T redpanda rpk group describe online-writer
curl -fsS localhost:9108/metrics | grep -E '^online_writer_(rows|bad_messages)_total' || true
log "online_writer $ONLINE_WRITER_TAG is consuming bars.1m"
