#!/bin/sh
# Seed KV v2 secrets into Vault dev server.
# Runs as an init container; idempotent (vault kv put overwrites).
set -e

export VAULT_ADDR="${VAULT_ADDR:-http://vault:8200}"
export VAULT_TOKEN="${VAULT_TOKEN:-pairlab-root}"

echo "Waiting for Vault..."
until vault status 2>/dev/null | grep -q "Initialized.*true"; do sleep 1; done

# Enable KV v2 (idempotent — ignore "already enabled" error)
vault secrets enable -path=secret kv-v2 2>/dev/null || true

vault kv put secret/minio \
  access_key="${MINIO_ROOT_USER}" \
  secret_key="${MINIO_ROOT_PASSWORD}"

vault kv put secret/postgres \
  host="postgres" \
  port="5432" \
  dbname="pairlab" \
  user="pairlab" \
  password="pairlab"

vault kv put secret/gateway \
  user="${GATEWAY_USER:-pairlab}" \
  password="${GATEWAY_PASSWORD:-pairlab}"

echo "Vault seeded: secret/minio, secret/postgres, secret/gateway"
