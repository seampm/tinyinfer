#!/usr/bin/env bash
# Phase 8: final benchmark suite — tinyinfer vs llama.cpp, same prompt/tokens.
# Writes results JSON to bench/results.json and prints a markdown table.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PROMPT="Once upon a time, there was a little girl named Lily."
N=128
LLAMA=~/workspace/llama.cpp/build/bin
OUT=bench/results.json

ti_bench() { # model quant -> tok/s
  local qflag=""; [ "$2" = "i8" ] && qflag="--quant i8"
  ./build/tinyinfer bench "$PROMPT" --max-tokens $N --model "models/$1" $qflag 2>/dev/null \
    | awk 'NR==3{print $6}'
}

lc_bench() { # gguf name -> tok/s (llama-bench tg column)
  "$LLAMA/llama-bench" -m "models/gguf/$1.gguf" -p 8 -n $N 2>/dev/null \
    | awk -F'|' 'NR==3{gsub(/ /,"",$4); print $4}'
}

ti15_f32=$(ti_bench tinyllama-15m f32); echo "tinyinfer 15M f32: $ti15_f32"
ti15_i8=$(ti_bench tinyllama-15m i8);   echo "tinyinfer 15M i8:  $ti15_i8"
ti11_f32=$(ti_bench tinyllama-1.1b f32); echo "tinyinfer 1.1B f32: $ti11_f32"
lc15_f32=$(lc_bench tinyllama-15m-f32); echo "llama.cpp 15M f32: $lc15_f32"
lc11_f32=$(lc_bench tinyllama-1.1b-f32); echo "llama.cpp 1.1B f32: $lc11_f32"

python3 - "$ti15_f32" "$ti15_i8" "$ti11_f32" "$lc15_f32" "$lc11_f32" << 'PYEOF'
import json, sys
r = {
  "prompt_tokens": 8, "gen_tokens": 128,
  "tinyinfer_15m_f32": float(sys.argv[1]),
  "tinyinfer_15m_i8": float(sys.argv[2]),
  "tinyinfer_1.1b_f32": float(sys.argv[3]),
  "llamacpp_15m_f32": float(sys.argv[4]),
  "llamacpp_1.1b_f32": float(sys.argv[5]),
}
json.dump(r, open("bench/results.json", "w"), indent=2)
print(json.dumps(r, indent=2))
PYEOF
