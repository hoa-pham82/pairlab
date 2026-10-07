#!/usr/bin/env bash
# Submit the Flink SQL ticks→bars job to the running Flink cluster.
# Run from repo root:  bash platform/flink/jobs/submit_ticks_to_bars.sh

set -euo pipefail

FLINK_SQL_FILE="platform/flink/sql/ticks_to_bars_1m.sql"
FLINK_CONTAINER="platform-flink-jobmanager-1"

# Copy SQL file into the container and run it via Flink SQL client
docker cp "$FLINK_SQL_FILE" "$FLINK_CONTAINER:/tmp/ticks_to_bars_1m.sql"

docker exec "$FLINK_CONTAINER" \
    /opt/flink/bin/sql-client.sh \
    -f /tmp/ticks_to_bars_1m.sql

echo "Flink SQL job submitted. Check Flink UI: http://localhost:8081"
