#!/usr/bin/env bash
# Read the model registry over MLflow's REST API.
# Usage: scripts/ci/mlflow_model.sh latest | alias ALIAS     (prints a version number)
source "$(dirname "$0")/lib.sh"

mlflow=${PAIRLAB_MLFLOW_URL:-http://localhost:5001}
name=${PAIRLAB_MODEL_NAME:-meta_label}

case "${1:?usage: $0 latest|alias ALIAS}" in
  latest)
    curl -fsS -G "$mlflow/api/2.0/mlflow/model-versions/search" \
      --data-urlencode "filter=name='$name'" --data-urlencode "max_results=1000" \
      | jq -r '[.model_versions[]?.version | tonumber] | max // 0'
    ;;
  alias)
    curl -fsS -G "$mlflow/api/2.0/mlflow/registered-models/alias" \
      --data-urlencode "name=$name" --data-urlencode "alias=${2:?alias name}" \
      | jq -r '.model_version.version'
    ;;
  *) die "usage: $0 latest|alias ALIAS" ;;
esac
