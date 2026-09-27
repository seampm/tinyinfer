#!/usr/bin/env bash
# Download models for tinyinfer. Weights are never committed (see .gitignore).
set -euo pipefail

MODELS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/models"
mkdir -p "$MODELS_DIR"

echo "==> Downloading TinyStories 15M (Karpathy llama2.c) for development..."
# TinyStories via llama2.c repo
if [ ! -f "$MODELS_DIR/stories15M.bin" ]; then
  curl -L -o "$MODELS_DIR/stories15M.bin" \
    https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin || \
    echo "WARN: stories15M.bin download failed, see scripts/reference.py for manual steps"
fi

echo "==> TinyLlama-1.1B (Hugging Face) for final benchmarks..."
echo "    Manual step (large): use scripts/download_tinyllama.py or:"
echo "    huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0 --local-dir $MODELS_DIR/tinyllama-1.1b"
echo ""
echo "Done. Models live in $MODELS_DIR/ (gitignored)."
