#!/usr/bin/env bash
# Download the Qwen2.5-3B-Instruct GGUF model for the llama.cpp inference server.
# Run once from repo root: bash scripts/download_model.sh
# The models/ directory is git-ignored; re-run after a clean checkout.
#
# Fallback: Qwen2.5-1.5B if RAM is under 8 GB.

set -euo pipefail

MODELS_DIR="models"
HF_BASE="https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main"
MODEL_FILE="qwen2.5-3b-instruct-q4_k_m.gguf"

mkdir -p "$MODELS_DIR"

TARGET="$MODELS_DIR/$MODEL_FILE"
if [ -f "$TARGET" ]; then
    echo "Model already present: $TARGET"
    exit 0
fi

echo "Downloading $MODEL_FILE (~2.0 GB) …"
curl -L --progress-bar -o "$TARGET" "$HF_BASE/$MODEL_FILE"
echo "Saved to $TARGET"
echo "Start the llm profile: docker compose -f platform/compose.yml --profile llm up -d llama-cpp"
