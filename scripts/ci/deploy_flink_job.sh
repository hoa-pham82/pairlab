#!/usr/bin/env bash
# Stream job 1: replace the running Flink SQL job (ticks -> 1-min bars -> Kafka + Postgres).
# Cancel + resubmit replays Kafka from the earliest offset; the Postgres sink upserts on
# (symbol, window_start), so the offline store ends up with the same rows.
source "$(dirname "$0")/lib.sh"

job_name=ticks_to_bars_1m
flink=${PAIRLAB_FLINK_URL:-http://localhost:8081}

jobs_in_state() {
  curl -fsS "$flink/jobs/overview" \
    | jq -r --arg n "$job_name" --arg s "$1" '.jobs[] | select(.name == $n and .state == $s) | .jid'
}

flink_up() { curl -fsS "$flink/overview" >/dev/null; }
wait_for 180 5 flink_up || die "Flink REST API is not reachable at $flink"

for jid in $(jobs_in_state RUNNING); do
  log "cancelling $job_name ($jid)"
  curl -fsS -X PATCH "$flink/jobs/$jid?mode=cancel" >/dev/null
done
no_running() { [[ -z "$(jobs_in_state RUNNING)" ]]; }
wait_for 120 5 no_running || die "old $job_name did not stop"

log "submitting $job_name"
compose cp "$DEPLOY_DIR/platform/flink/sql/$job_name.sql" "flink-jobmanager:/tmp/$job_name.sql"
compose exec -T flink-jobmanager /opt/flink/bin/sql-client.sh -f "/tmp/$job_name.sql" \
  | grep -E "Job ID|ERROR|Exception" || true

all_tasks_running() {
  local jid
  jid=$(jobs_in_state RUNNING | head -1)
  [[ -n "$jid" ]] && curl -fsS "$flink/jobs/$jid" \
    | jq -e '[.vertices[].status] | all(. == "RUNNING")' >/dev/null
}
wait_for 180 5 all_tasks_running || die "$job_name did not reach RUNNING"
log "$job_name is RUNNING ($(jobs_in_state RUNNING | head -1))"
