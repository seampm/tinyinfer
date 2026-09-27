#!/usr/bin/env bash
# Download models for tinyinfer. Weights are never committed (see .gitignore).
set -euo pipefail

MODELS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/models"
mkdir -p "$MODELS_DIR"

echo "==> TinyStories 15M (nickypro/tinyllama-15M: Llama-2 arch, converted from"
echo "    karpathy/tinyllamas) for development. Weights are converted F16->F32"
echo "    on download because tinyinfer is fp32-first (Phase 2 design decision)."
M15="$MODELS_DIR/tinyllama-15m"
mkdir -p "$M15"
export NO_PROXY="localhost,127.0.0.1,::1" no_proxy="localhost,127.0.0.1,::1"
for f in config.json tokenizer.json tokenizer_config.json; do
  if [ ! -f "$M15/$f" ]; then
    curl -L -o "$M15/$f" "https://huggingface.co/nickypro/tinyllama-15M/resolve/main/$f"
  fi
done
if [ ! -f "$M15/model.safetensors" ]; then
  curl -L -o "$M15/model-f16.safetensors" \
    "https://huggingface.co/nickypro/tinyllama-15M/resolve/main/model.safetensors"
  python3 - "$M15" << 'PYEOF'
import os, sys
from safetensors.torch import load_file, save_file
d = sys.argv[1]
t = load_file(os.path.join(d, "model-f16.safetensors"))
save_file({k: v.float() for k, v in t.items()}, os.path.join(d, "model.safetensors"))
os.remove(os.path.join(d, "model-f16.safetensors"))
print("converted model.safetensors to F32")
PYEOF
fi

echo "==> TinyLlama-1.1B (Hugging Face) for final benchmarks..."
echo "    Manual step (large):"
echo "    huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0 --local-dir $MODELS_DIR/tinyllama-1.1b"
echo ""
echo "Done. Models live in $MODELS_DIR/ (gitignored)."
