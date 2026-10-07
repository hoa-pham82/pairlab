#!/usr/bin/env bash
# Download Flink connector JARs for the ticks → 1-min bars pipeline.
# Run once from repo root: bash platform/flink/download_jars.sh
# JARs are git-ignored; re-run if containers are recreated.

set -euo pipefail

DEST="platform/flink/jars"
MAVEN="https://repo1.maven.org/maven2"

mkdir -p "$DEST"

download() {
    local url="$1"
    local file="$DEST/$(basename "$url")"
    [ -f "$file" ] && echo "  skip $(basename "$url")" && return 0
    echo "  downloading $(basename "$url") …"
    curl -fsSL -o "$file" "$url"
}

echo "Downloading Flink connector JARs → $DEST/"

download "$MAVEN/org/apache/flink/flink-connector-kafka/3.2.0-1.19/flink-connector-kafka-3.2.0-1.19.jar"
download "$MAVEN/org/apache/kafka/kafka-clients/3.7.0/kafka-clients-3.7.0.jar"
download "$MAVEN/org/apache/flink/flink-connector-jdbc/3.2.0-1.19/flink-connector-jdbc-3.2.0-1.19.jar"
download "$MAVEN/org/postgresql/postgresql/42.7.3/postgresql-42.7.3.jar"

echo "Done. $(ls "$DEST"/*.jar | wc -l | tr -d ' ') JARs in $DEST/"
echo "Recreate Flink containers to pick them up:"
echo "  docker compose -f platform/compose.yml --profile core --profile stream up -d --force-recreate flink-jobmanager flink-taskmanager"
