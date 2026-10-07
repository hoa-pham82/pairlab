#!/usr/bin/env bash
# Copy a release (this checkout or an unpacked build artifact) into the deploy directory.
# Usage: scripts/ci/release.sh            (env: PAIRLAB_RELEASE=<git sha>, PAIRLAB_ENV_FILE=<.env text>)
source "$(dirname "$0")/lib.sh"

mkdir -p "$DEPLOY_DIR"
log "releasing $SOURCE_DIR -> $DEPLOY_DIR"

# Runtime state lives only in the deploy dir: excluded paths are never copied or deleted.
rsync -a --delete \
  --exclude '.git/' --exclude '.venv/' --exclude '__pycache__/' --exclude '.pytest_cache/' \
  --exclude '/.env' --exclude '/RELEASE' --exclude '/data/' --exclude '/models/' \
  --exclude '/build/' --exclude '/mutants/' \
  --exclude '/platform/spark/jars/*.jar' --exclude '/platform/flink/jars/*.jar' \
  --exclude '/platform/feature_repo/data/' \
  --exclude '/deploy/kind/kubeconfig' --exclude '/deploy/terraform/' \
  "$SOURCE_DIR/" "$DEPLOY_DIR/"

# The Feast registry is state: seed it once, then feast apply/materialize own it.
registry="$DEPLOY_DIR/platform/feature_repo/data/registry.db"
if [[ ! -f "$registry" && -f "$SOURCE_DIR/platform/feature_repo/data/registry.db" ]]; then
  mkdir -p "$(dirname "$registry")"
  cp "$SOURCE_DIR/platform/feature_repo/data/registry.db" "$registry"
fi

if [[ -n "${PAIRLAB_ENV_FILE:-}" ]]; then
  umask 077
  printf '%s\n' "$PAIRLAB_ENV_FILE" > "$DEPLOY_DIR/.env"
fi
[[ -f "$DEPLOY_DIR/.env" ]] || die "no $DEPLOY_DIR/.env: set the PAIRLAB_ENV_FILE secret"

for job in spark flink; do
  if ! ls "$DEPLOY_DIR/platform/$job/jars/"*.jar >/dev/null 2>&1; then
    log "downloading $job connector jars (one-time)"
    mkdir -p "$DEPLOY_DIR/platform/$job/jars"
    (cd "$DEPLOY_DIR" && bash "platform/$job/download_jars.sh")
  fi
done

if [[ -n "${PAIRLAB_RELEASE:-}" ]]; then
  echo "$PAIRLAB_RELEASE" > "$DEPLOY_DIR/RELEASE"
elif git -C "$SOURCE_DIR" rev-parse HEAD >/dev/null 2>&1; then
  git -C "$SOURCE_DIR" rev-parse HEAD > "$DEPLOY_DIR/RELEASE"
fi

mkdir -p "$(dirname "$KUBECONFIG")"
if kind get clusters 2>/dev/null | grep -qx "$KIND_CLUSTER"; then
  kind get kubeconfig --name "$KIND_CLUSTER" > "$KUBECONFIG"
fi
log "release $(release_tag) is in place"
