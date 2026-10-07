#!/usr/bin/env bash
# Feast steps for the materialize pipeline, run inside the Airflow container (same Feast as the DAG).
# Usage: scripts/ci/feast.sh apply | check-online [PAIR_ID]
source "$(dirname "$0")/lib.sh"

feast_py() { compose exec -T airflow /opt/ml-venv/bin/python "$@"; }

case "${1:?usage: $0 apply|check-online [PAIR_ID]}" in
  apply)
    compose exec -T airflow /opt/ml-venv/bin/feast -c /opt/airflow/feature_repo apply
    ;;
  check-online)
    pair=${2:-KO__PEP}
    feast_py - "$pair" <<'EOF'
import sys
from feast import FeatureStore

pair = sys.argv[1]
names = ["zscore", "hedge_ratio", "spread_vol", "correlation_60d"]
row = FeatureStore(repo_path="/opt/airflow/feature_repo").get_online_features(
    features=[f"pair_daily_fv:{n}" for n in names], entity_rows=[{"symbol_pair": pair}]
).to_dict()
missing = [n for n in names if row[n][0] is None]
print({n: row[n][0] for n in names})
sys.exit(f"online store has no {missing} for {pair}" if missing else 0)
EOF
    log "online features present"
    ;;
  *) die "usage: $0 apply|check-online [PAIR_ID]" ;;
esac
