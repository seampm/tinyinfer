# tinyinfer

An LLM inference engine in C++, written from scratch. No ML frameworks, no copied code. The safetensors parser, BPE tokenizer, math kernels, and quantization are all hand-written.

**[Live demo](https://seampm.github.io/tinyinfer/)** — terminal replay, benchmark charts, architecture walkthrough.

![demo](bench/demo.gif)

## What it does

- **Load**: Llama checkpoints (`model.safetensors` + `config.json`) via a hand-written safetensors parser over mmap
- **Tokenize**: BPE from `tokenizer.json`, byte-exact vs the Python `tokenizers` library on 21/21 test strings
- **Run**: full Llama forward pass — RMSNorm, GQA attention with NeoX RoPE, SwiGLU MLP, KV cache — verified layer-by-layer against Hugging Face within 1e-4
- **Sample**: greedy, temperature, top-k, top-p with seeded RNG (greedy output token-identical to PyTorch)
- **Quantize**: per-row symmetric int8 weights (`--quant i8`), validated by WikiText-2 perplexity
- **Measure**: `bench` (cached vs naive tok/s), `perplexity` (WikiText-2)

## Benchmarks

2-core VM, AVX2+FMA, `g++ -O3 -march=native`, OpenMP 2 threads. Median of 3 runs, 15-token prompt, 128 generated tokens, greedy. llama.cpp rev `2b129cc`, F32 GGUF.

| model | engine | precision | decode tok/s |
|---|---|---|---|
| TinyStories-15M | tinyinfer | fp32 | **276** |
| TinyStories-15M | tinyinfer | int8 | **606** |
| TinyStories-15M | llama.cpp | fp32 | 274 |
| TinyLlama-1.1B | tinyinfer | fp32 | **5.1** |
| TinyLlama-1.1B | llama.cpp | fp32 | 6.9 |

101% of llama.cpp on the 15M model, 74% on 1.1B. int8 is 2.19x fp32 on 15M. The 1.1B gap is kernel micro-optimization (cache blocking, blocked accumulation) that isn't done yet.

![throughput](bench/charts/throughput.png)

### How it got there (15M decode tok/s)

| configuration | tok/s |
|---|---|
| scalar fp32, KV cache | 67 |
| + AVX2/FMA, 4-way unrolled accumulators | 276 |
| + per-row int8 quantization | 606 |

![journey](bench/charts/journey.png)

The biggest single win came from benchmarking against llama.cpp: it showed the AVX2 kernel was latency-bound (one accumulator chain against 4–5 cycle FMA latency), not bandwidth-bound as assumed. Unrolling 4x gave 1.6x on the 1.1B immediately. Without a reference to compare against, 3.4 tok/s would have looked like the hardware limit.

### Perplexity (WikiText-2, 19,999 scored tokens)

| precision | perplexity |
|---|---|
| fp32 | 6260.2 |
| int8 | 6199.8 |

int8 changes perplexity by −1.0% (noise). The 2.19x speedup costs nothing measurable in quality.

## Tradeoffs

- Every kernel has a scalar reference and a differential test vs NumPy/HF before optimization. The fast path never diverges from the tested path.
- int8, not int4: 4-bit needs blocked quantization (Q4_0-style) to stay accurate. Honest int8 now, int4 done properly later.
- OpenMP only where it wins: threading the 32k-row lm_head helps; threading the small attention/MLP mats measured slower (overhead > gain).
- Weights are memory-mapped: instant startup, the OS pages in only what's touched. First-token latency includes page faults.
- The from-scratch BPE is byte-exact but ~35x slower than the Python library on megabyte inputs. The perplexity harness accepts pre-tokenized `--ids` for large corpora.

## Architecture

```
prompt text
    │  BPE tokenizer (tokenizer.json: vocab, merges, added tokens)
    ▼
token ids ──► embedding lookup ─────────────────────────────────────────┐
    │                                                                   │
    ▼                                                                   │
┌─ transformer block × N ───────────────────────────────────────────────┤
│   x ──► RMSNorm ──► Q/K/V proj ──► RoPE ──► GQA attention ──► O proj ─┼─► + x (residual)
│   x ──► RMSNorm ──► gate proj ──► SiLU ──╳──► down proj ──────────────┼─► + x (residual)
│                              up proj ──╯    (SwiGLU: silu(gate) * up) │
└───────────────────────────────────────────────────────────────────────┘
    │
    ▼
 RMSNorm ──► lm_head (32k vocab) ──► softmax ──► sampler ──► next token
    ▲
    └── KV cache: keys/values appended per position, never recomputed
```

The hot path per decode step is one matvec per weight matrix (all weights streamed once per token; decode is memory-bandwidth bound). Two matvec kernels: `dot_avx2` (fp32, 4-way unrolled FMA) and `matvec_i8` (per-row int8 + fp32 scales, widened on the fly). OpenMP parallelizes only the 32k-row lm_head; smaller mats run single-threaded (threading them measured slower).

## Quickstart

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
# 15M dev model (download once; gitignored, never committed):
#   nickypro/tinyllama-15M -> models/tinyllama-15m/ (fp32 safetensors)
./build/tinyinfer prompt "Once upon a time" --max-tokens 60
./build/tinyinfer prompt "Once upon a time" --max-tokens 60 --quant i8
./build/tinyinfer bench "Once upon a time" --max-tokens 128 --skip-naive
```

## Future work

- Q4_0-style blocked int4/int8 quantization (int32 accumulation in blocks of 32)
- Cache-blocked matvec for better L2 reuse on large models
- Paged/blocked KV cache for long contexts
- Speculative decoding

## Layout

```
src/            engine, model, ops (SIMD), tokenizer, sampling, CLI
include/        public headers
tests/          GoogleTest suite (42 tests)
bench/          benchmark scripts, charts, demo GIF
scripts/        reference.py (answer key), check_perplexity.py (HF parity)
docs/           architecture.md
DESIGN_LOG.md   every measurement, failed optimization, and tradeoff
```
