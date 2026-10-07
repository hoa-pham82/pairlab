#!/usr/bin/env bash
# Download Spark extra JARs into platform/spark/jars/ (one-time setup).
# Total download: ~277 MB (mostly aws-java-sdk-bundle).
# Safe to re-run — skips already-downloaded files.
#
# Usage:  bash platform/spark/download_jars.sh

set -euo pipefail

JARS_DIR="$(cd "$(dirname "$0")/jars" && pwd)"
MAVEN="https://repo1.maven.org/maven2"

download() {
    local url="$1"
    local file="$JARS_DIR/$(basename "$url")"
    if [[ -f "$file" ]]; then
        echo "  already have $(basename "$file")"
    else
        echo "  downloading $(basename "$url") ..."
        curl -fL --progress-bar -o "$file" "$url"
    fi
}

echo "=== Downloading Spark extra JARs to $JARS_DIR ==="

# S3A filesystem connector (needed for s3a:// URIs → LocalStack)
download "$MAVEN/org/apache/hadoop/hadoop-aws/3.3.4/hadoop-aws-3.3.4.jar"

# AWS SDK v1 bundle — required by hadoop-aws 3.3.4  (~268 MB, one-time)
download "$MAVEN/com/amazonaws/aws-java-sdk-bundle/1.12.262/aws-java-sdk-bundle-1.12.262.jar"

# Delta Lake (Spark 3.5 / Scala 2.12)
download "$MAVEN/io/delta/delta-spark_2.12/3.2.0/delta-spark_2.12-3.2.0.jar"
download "$MAVEN/io/delta/delta-storage/3.2.0/delta-storage-3.2.0.jar"

# PostgreSQL JDBC (for DP2/DP3 JDBC sink)
download "$MAVEN/org/postgresql/postgresql/42.7.3/postgresql-42.7.3.jar"

echo ""
echo "Done. $(ls "$JARS_DIR"/*.jar | wc -l | tr -d ' ') JARs in $JARS_DIR"
echo ""
echo "Restart the Spark cluster to pick them up:"
echo "  docker compose -f platform/compose.yml --profile platform restart spark-master spark-worker"
