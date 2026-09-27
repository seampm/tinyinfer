# tinyinfer benchmarks

Quick numbers live behind `tinyinfer bench` (no separate bench binary):

```bash
./build/tinyinfer bench "Once upon a time" --max-tokens 64
```

It runs greedy generation with the KV cache vs naive per-token recompute
(`forward_full`), asserts the token streams are identical, and prints
ttft / decode tok/s for both.

`bench/results/` will hold the Phase 7-8 result tables:
- tokens/sec, time to first token, peak memory
- fp32 vs int8 vs int4 vs llama.cpp baseline

Measured so far (TinyStories-15M, fp32, 64 greedy tokens):
- cached: 67.0 tok/s decode — naive: 1.87 tok/s — speedup 35.9x
