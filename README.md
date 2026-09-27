# tinyinfer — LLM Inference Engine in C++

A from-scratch inference engine for Llama-architecture models (TinyStories 15M/110M for dev, TinyLlama-1.1B for final benchmarks) with no ML frameworks.

> Status: Phase 0 — repo scaffolded. Forward pass, tokenizer, KV-cache, SIMD/OpenMP kernels, and int8/int4 quantization coming per the project plan.

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

# Run
./build/tinyinfer --model tinyllama --prompt "Once upon a time" --tokens 200
```

Requirements: CMake 3.20+, C++20 compiler, Python 3.10+ (for scripts).

## Benchmarks

| Model | Precision | Tokens/sec | Time to first token | Peak memory |
|-------|-----------|------------|---------------------|-------------|
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

- [ ] Phase 1: Python reference (`scripts/reference.py`) saving per-layer outputs
- [ ] Phase 2: safetensors loader with mmap
- [ ] Phase 3: math ops (matvec, RMSNorm, softmax, RoPE, SiLU/SwiGLU) + tests
- [ ] Phase 4: forward pass, layer-by-layer verification
- [ ] Phase 5: BPE tokenizer, sampling, CLI
- [ ] Phase 6: KV cache (+ before/after numbers)
- [ ] Phase 7: OpenMP, AVX2/NEON, int8/int4 quantization + perplexity on WikiText-2
- [ ] Phase 8: final benchmarks, charts, demo GIF

## Design Log

See `DESIGN_LOG.md` — every >1hr bug, every optimization with before/after numbers, every tradeoff considered. Interview prep gold.

## License

MIT
