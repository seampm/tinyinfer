# tinyinfer — LLM Inference Engine in C++

A from-scratch inference engine for Llama-architecture models (TinyStories 15M/110M for dev, TinyLlama-1.1B for final benchmarks) with no ML frameworks.

> Status: Phase 6 — KV-cache benchmarked: 67 tok/s cached vs 1.9 tok/s naive recompute (35.9x speedup, token-identical). SIMD/OpenMP kernels and int8/int4 quantization coming per the project plan.

## Demo

_TODO: Add terminal GIF once CLI generates text end-to-end._

## Architecture

```
Prompt → Tokenizer (BPE) → Embedding
  → N × [ RMSNorm → Attention (RoPE + GQA) + residual
        → RMSNorm → SwiGLU FFN + residual ]
  → Final RMSNorm → LM head → Logits → Sampler → Tokens
```

- Each layer: RMSNorm → self-attention with rotary position embeddings (RoPE) and grouped-query attention → residual add → RMSNorm → SwiGLU feed-forward → residual add
- After all layers: final RMSNorm + output projection to logits
- KV-cache stores per-layer K/V so each decode step only computes the newest token

## Build and Run

```bash
# Fetch models (never committed)
./scripts/download_models.sh

# Build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Run (model dir holds model.safetensors + config.json + tokenizer.json)
./build/tinyinfer prompt "Once upon a time" --max-tokens 50
./build/tinyinfer prompt "Once upon a time" --max-tokens 50 --temperature 0.8 --top-k 40 --top-p 0.9 --seed 42
./build/tinyinfer tokenize "Once upon a time"   # debug: print token ids
```

Requirements: CMake 3.20+, C++20 compiler, Python 3.10+ (for scripts).

## Benchmarks

| Model | Precision | Tokens/sec | Time to first token | Peak memory |
|-------|-----------|------------|---------------------|-------------|
| TinyStories-15M (dev) | fp32, KV cache | 67.0 (greedy decode) | 0.08 s | TODO |
| TinyStories-15M (dev) | fp32, no cache (baseline) | 1.87 (greedy decode) | 0.07 s | TODO |
| TinyLlama-1.1B | fp32 | TODO | TODO | TODO |
| TinyLlama-1.1B | int8 | TODO | TODO | TODO |
| TinyLlama-1.1B | int4 (stretch) | TODO | TODO | TODO |
| llama.cpp (reference) | — | TODO | TODO | TODO |

Charts: `bench/results/` (TODO)

Context: llama.cpp is the reference. This engine aims to document where the gap comes from — e.g. "reaches ~40% of llama.cpp's speed because X".

## Design Decisions and Tradeoffs

- **safetensors parsed by hand + mmap**: 8-byte header length, JSON header, raw data; mmap for near-instant loads. Good interview talking point.
- **fp32 first, optimize later**: simplest correct kernels first, verified layer-by-layer against Hugging Face (1e-4 tolerance), then OpenMP/SIMD/quantization.
- **No ML frameworks**: all tensor ops hand-written with GoogleTest unit tests vs NumPy reference values.
- **Prior work**: llama.cpp and llama2.c exist — used only as reference when stuck, never copied. Benchmarked against for context.

## What's Next

- [x] Phase 1: Python reference (`scripts/reference.py`) saving per-layer outputs
- [x] Phase 2: safetensors loader with mmap
- [x] Phase 3: math ops (matvec, RMSNorm, softmax, RoPE, SiLU/SwiGLU) + tests
- [x] Phase 4: forward pass, layer-by-layer verification
- [x] Phase 5: BPE tokenizer, sampling, CLI (greedy output token-identical to PyTorch)
- [x] Phase 6: KV cache (+ before/after numbers: 35.9x vs naive recompute)
- [ ] Phase 7: OpenMP, AVX2/NEON, int8/int4 quantization + perplexity on WikiText-2
- [ ] Phase 8: final benchmarks, charts, demo GIF

## Design Log

See `DESIGN_LOG.md` — every >1hr bug, every optimization with before/after numbers, every tradeoff considered. Interview prep gold.

## License

MIT
