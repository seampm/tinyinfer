# Design Log

Running notes: every bug that took >1hr, every optimization with before/after numbers, every tradeoff considered.
Interview prep gold — "walk me through a hard bug" lives here.

## Template

### YYYY-MM-DD — Title
- **Problem:**
- **Root cause:**
- **Fix:**
- **Lesson:**

### Optimization — Name
- **Before:** X tok/s, Y MB
- **After:** X tok/s, Y MB
- **Why it helped:**

### 2026-09-27 — Phase 1 reference validation
- **What:** Validated scripts/reference.py against hf-internal-testing/tiny-random-LlamaForCausalLM (2 layers).
- **Result:** Saved layer_0.pt, layer_1.pt, logits.pt ([1,5,32000]), meta.json with input_ids and 1e-4 tolerance. Script works end-to-end.
- **Gotcha:** This sandbox's NO_PROXY env var is malformed (truncated IPv6 entry) which breaks Python urllib proxy parsing ("Invalid port: ':1]'"). Workaround: override NO_PROXY/no_proxy to "localhost,127.0.0.1,::1" when running. Not an issue on a normal machine.
- **Gotcha 2:** transformers 5.x renamed torch_dtype -> dtype. Script now tries dtype first, falls back to torch_dtype.
- **Lesson:** push_files.py now handles updates (must send existing sha), and GitHub workflow files need the token's `workflow` scope.

### 2026-09-27 — Phase 2 safetensors loader
- **What:** Hand-rolled loader: 8-byte LE header length, nlohmann/json for the
  header dict, whole-file mmap (PROT_READ/MAP_PRIVATE), per-tensor offset
  validation (byte range == numel * dtype_size), zero-copy views into mapping.
- **Result:** 8/8 new tests pass; C++ values match Python-generated fixture exactly.
- **Decisions:** nlohmann/json via FetchContent (same pattern as GoogleTest) —
  the format framing/offsets/mmap are still hand-written; JSON lib is just the
  header dict. F32-first; F16/BF16 parse but kernels come in Phase 3/7.
- **Gotcha:** push_files.py needed the existing blob sha for updates (422
  without it) — fixed.

### 2026-09-27 — Phase 3 math ops
- **What:** matvec, rmsnorm, stable softmax, RoPE (NeoX style), silu/vec_silu,
  vec_mul — all fp32, simplest correct versions in include/tinyinfer/ops.h.
- **Result:** 9/9 new tests pass vs NumPy values from
  scripts/make_ops_reference.py (tolerance 1e-5).
- **Decisions:** RoPE uses GPT-NeoX layout (rotate (x[i], x[i+half])), matching
  HF Llama — this is what Phase 4 verification compares against, so the choice
  is load-bearing. Softmax subtracts max first (test uses inputs ~1000 to prove
  stability). Optimization (OpenMP/SIMD) deferred to Phase 7; signatures stable.

### 2026-09-27 — Phase 4 forward pass
- **What:** include/tinyinfer/model.h + src/model.cpp: LlamaModel with HF
  config.json parsing, mmap'd weights, internal KV cache, incremental
  forward(token, pos) -> logits. forward_debug captures per-layer outputs.
- **Result:** Matches the HF reference layer-by-layer within 1e-4 on the first
  run (2-layer tiny fixture, 4-token prompt) — the per-op verification in
  Phase 3 paid off.
- **Bug (real debugging):** rope() indexed k by q-head (k + h*head_dim) but with
  GQA k only has n_kv_heads heads -> heap corruption, caught by the
  determinism test. Fixed signature to rope(q, k, n_q_heads, n_kv_heads, ...);
  the op test only covered n_q == n_kv, so added Ops.RopeGqa regression test.
  Also hoisted inv_freq/cos/sin out of the head loop.
- **Decisions:** KV cache lives in the model from the start (correctness for
  incremental forward); Phase 6 will benchmark cached vs naive recompute for
  the before/after numbers. Tied embeddings fall back to embed_tokens if
  lm_head.weight is absent. forward() with increasing pos requires reset()
  before each new sequence.

### 2026-09-27 — Phase 5 tokenizer, sampling, CLI (first real generation)
- **What:** include/tinyinfer/tokenizer.h + src/tokenizer.cpp: BPE parsed from
  HF tokenizer.json (vocab, merges, added_tokens); include/tinyinfer/sampling.h
  + src/sampling.cpp (temperature, top-k, top-p, greedy); src/main.cpp CLI with
  `prompt` and `tokenize` subcommands.
- **Model choice:** nickypro/tinyllama-15M (Llama-2 arch, converted from
  karpathy/tinyllamas, safetensors F16). scripts/download_models.sh downloads
  config/tokenizer/weights and casts F16->F32 on the way in (fp32-first design).
  Config: 6 layers, dim 288, 6 heads (MHA), ffn 768, vocab 32000,
  tied embeddings (no lm_head.weight — the Phase 4 fallback handles it).
- **Result:** `./build/tinyinfer prompt "Once upon a time" --max-tokens 60`
  generates a coherent TinyStories-style story at ~45 tok/s (unoptimized fp32).
  C++ greedy decoding is TOKEN-IDENTICAL to PyTorch over 8 tokens
  (scripts/check_generation.py) — the whole pipeline (tokenizer -> model ->
  sampler) verified end-to-end. 36/36 unit tests pass; 21/21 real-tokenizer
  strings match the Python `tokenizers` library exactly
  (scripts/check_tokenizer.py).
- **Bug (real debugging, ~1hr):** first encode() split added special tokens on
  the RAW text, then normalized each piece. Correct for added_tokens with
  "normalized": false, but this tokenizer uses "normalized": true — HF
  normalizes the WHOLE input first, then splits on the NORMALIZED spellings
  ('<s>' -> '▁<s>'), and BPEs the gaps WITHOUT re-normalizing. So 'a<s>bc'
  must NOT treat '<s>' as special (it's glued to 'a'), while '<s>bc' must.
  Found by probing the Python library; both paths now implemented and
  unit-tested (mini_tokenizer.json + mini_tokenizer_raw.json fixtures).
- **Gotcha:** HF returns [] for empty input (short-circuits before the
  normalizer); matched. Also, Python's decode() on THIS tokenizer.json does not
  skip <s>/</s> even with skip_special_tokens=True (quirk of the file); the C++
  decode skips specials, which is the sane behavior and what the CLI wants.
- **Lesson:** "just parse tokenizer.json" is 90% BPE-merges and 10% weird
  normalizer/added-token interaction semantics that you can only discover by
  differential-testing against the reference implementation. The
  check_tokenizer.py differential harness earned its keep twice (the
  normalize-first bug and the empty-string edge).

### 2026-09-27 — Phase 6 KV-cache benchmark (before/after)
- **What:** Added `LlamaModel::forward_full(tokens, n)` — reset + full forward,
  the naive "recompute all K/V every step" baseline (and the primitive Phase 7
  perplexity needs). New `tinyinfer bench` subcommand runs greedy generation
  both ways and reports ttft/decode tok/s; also asserts identical token streams.
- **Correctness:** `tests/test_kvcache.cpp`: incremental (cached) logits match
  `forward_full` recompute within 1e-6 at every prefix length on the tiny
  fixture, plus a reset-isolation test. 38/38 tests pass.
- **Numbers** (TinyStories-15M, fp32, 64 greedy tokens, "Once upon a time"):
  - naive (recompute):  1.87 tok/s decode, ttft 0.069s
  - cached (KV reuse): 67.0  tok/s decode, ttft 0.077s
  - **speedup: 35.9x**; token streams identical.
- **Why it helped:** decode step at position p costs 1 forward cached vs p+1
  forwards naive, so total work is O(n) vs O(n^2/2) — predicted ratio ~35x at
  n=69, measured 35.9x. Theory and measurement agree, which is the real result.
  TTFT is unchanged (prompt processing is a full forward either way).
- **Lesson:** the bench doubles as a 64-token differential test of the cache.
  A cache bug wouldn't just be slow — it would show up as a stream mismatch.

### 2026-09-27 — Phase 7 AVX2 matvec (before/after)
- **What:** hand-wrote an AVX2/FMA dot product (`dot_avx2`: 8 floats/iter,
  `_mm256_fmadd_ps`, horizontal sum, scalar tail) for `matvec`, the documented
  hot spot. `matvec_scalar` kept as the test reference; new
  `Ops.MatvecAvx2MatchesScalar` test (incl. 32000x288 vocab-head shape) agrees
  within 1e-3. Compile-time `#ifdef __AVX2__` dispatch (binary already
  `-march=native`), scalar fallback preserved.
- **Numbers** (15M fp32, 64 greedy tokens): 67.0 -> **322 tok/s (4.8x)**.
- **Why it helped:** the old `sum += row[j]*x[j]` loop has a loop-carried
  dependency; without `-ffast-math` the compiler won't reassociate the
  reduction, so it ran ~scalar/latency-bound. The FMA version does 16
  flops/cycle. This moved decode from compute-bound to memory-bound:
  44MB/token of weights at 322 tok/s ~= 14GB/s, i.e. the machine's bandwidth
  wall — which is exactly what the next measurement confirmed.
- **Lesson:** measure first, then the bottleneck tells you what to do next.

### 2026-09-27 — Phase 7 OpenMP: threads don't help a bandwidth-bound loop
- **What:** `#pragma omp parallel for schedule(static)` over matvec rows,
  gated on `rows >= 4096` so only the 32000-row lm_head/vocab matvec threads
  (small attention/MLP mats would drown in fork overhead).
- **Numbers:** 1 thread 322 tok/s vs 2 threads **319 tok/s** — no gain
  (slightly negative). OpenMP confirmed active (`-fopenmp` in flags).
- **Why:** one thread already saturates ~14GB/s of memory bandwidth on the
  36MB lm_head; a second thread just fights for the same bus. Threads help
  compute-bound work; decode here is a streaming-weight problem.
- **Lesson (the honest one for the README):** the fix for a bandwidth wall
  isn't more threads, it's less traffic — i.e. quantization. Kept the OpenMP
  path anyway: it's correct, tested, and would matter on wider machines or
  for prompt processing (many tokens per forward). The threshold keeps it
  from ever hurting.

### 2026-09-27 — Phase 7 int8 quantization (per-row symmetric)
- **What:** `quantize_i8` (per-row symmetric, scale=max|row|/127) +
  `matvec_i8` (AVX2: int8->int16->int32->fp32 widening, FMA with fp32
  activations, scale applied per row). Model holds a quantized copy of every
  matvec weight (incl. a quantized copy of the embedding table for the tied
  lm head); embeddings/norms stay fp32. `--quant i8` on prompt/bench/
  perplexity.
- **Correctness:** `Ops.QuantizeI8Roundtrip` (half-LSB bound, zero-row scale
  guard), `Ops.MatvecI8MatchesF32` (rel err < 2%), `Forward.Int8CloseToF32`
  (full-model int8 vs fp32 forward, rel err < 5%). All pass.
- **Numbers** (15M, 64 greedy tokens): fp32 322 -> int8 **548 tok/s (1.70x)**.
- **Why only 1.7x, not 4x:** traffic drops 44MB -> ~15MB/token, but the AVX2
  kernel spends ~3 widening instructions per 8 weights on int8->fp32
  conversion — the kernel is no longer pure streaming. A Q8_0-style blocked
  kernel with int32 accumulation (like llama.cpp's) would close this gap;
  noted as future work rather than claimed.
- **Accuracy:** WikiText-2 perplexity, 20k tokens, non-overlapping 256-token
  chunks: fp32 PPL vs int8 PPL measured with `tinyinfer perplexity`
  (fp32 result cross-checked against HF transformers — see check script).
- **Perplexity** (WikiText-2 test, 19999 tokens, non-overlapping 256-token
  chunks, `tinyinfer perplexity --ids`): fp32 PPL **6260.2** vs int8 PPL
  **6199.8** — delta **-1.0%** (int8 trivially *better*, i.e. pure noise).
  Absolute PPL is awful because it's a children's-stories model scored on
  Wikipedia — expected and irrelevant. What matters: per-row symmetric int8
  causes no meaningful degradation. (The eval harness reuses the Phase-5
  PyTorch-verified forward pass + textbook NLL math; fp32 cross-check vs HF
  transformers lives in scripts/check_perplexity.py.)

### 2026-09-27 — Phase 7 perplexity: our BPE is correct but slow on long inputs
- **What:** `tinyinfer perplexity` on WikiText-2 initially tokenized the 1.3MB
  eval text in-process — and never finished: a 50KB prefix alone exceeded a
  120s timeout. The BPE merge loop is effectively O(n^2)-ish on long inputs
  (correctness was the Phase 5 goal; throughput wasn't measured).
- **Workaround (honest):** tokenization is already differentially verified
  (21/21 strings byte-exact vs Python), so the eval harness pre-tokenizes with
  the Python `tokenizers` library (337k tokens in 3.4s) and `perplexity`
  accepts `--ids FILE`. The model measurement is unaffected.
- **Lesson / future work:** a production BPE needs the standard tricks —
  regex pre-tokenization into words, per-word merge caching, priority-queue
  merge selection. Our CLI `prompt` path is fine (short prompts), but long-
  document ingestion would need this. Good interview talking point: "I know
  exactly where my tokenizer is slow and why."
