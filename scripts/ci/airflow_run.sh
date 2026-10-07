#!/usr/bin/env bash
# Deploy gate for a DAG: no import errors, unpause, trigger one run, wait for success.
# Usage: scripts/ci/airflow_run.sh DAG_ID [TIMEOUT_SECONDS]
source "$(dirname "$0")/lib.sh"

dag_id=${1:?usage: $0 DAG_ID [TIMEOUT_SECONDS]}
timeout=${2:-2700}
run_id="cd__$(release_tag)__$(date -u +%Y%m%dT%H%M%S)"

af() { compose exec -T airflow airflow "$@" 2>/dev/null; }

dag_is_loaded() { af dags list -o json | jq -e --arg d "$dag_id" 'any(.[]; .dag_id == $d)' >/dev/null; }

log "re-parsing DAG files"
af dags reserialize >/dev/null || true
errors=$(af dags list-import-errors -o json || echo '[]')
if [[ "$errors" != "[]" && "$errors" != *"No data found"* ]]; then
  echo "$errors"
  die "Airflow reports DAG import errors"
fi
wait_for 120 5 dag_is_loaded || die "$dag_id is not loaded in Airflow"

af dags unpause "$dag_id" >/dev/null
af dags trigger "$dag_id" -r "$run_id" >/dev/null
log "triggered $dag_id run $run_id (timeout ${timeout}s)"

run_state() {
  af dags list-runs -d "$dag_id" -o json \
    | jq -r --arg r "$run_id" '.[] | select(.run_id == $r) | .state'
}

deadline=$((SECONDS + timeout))
state=""
while (( SECONDS < deadline )); do
  state=$(run_state || true)
  case "$state" in
    success) log "$dag_id $run_id: success"; exit 0 ;;
    failed)  break ;;
  esac
  sleep 20
done

af tasks states-for-dag-run "$dag_id" "$run_id" -o table || true
die "$dag_id $run_id ended as '${state:-timeout}'"
