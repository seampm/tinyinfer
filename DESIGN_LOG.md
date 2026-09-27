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
