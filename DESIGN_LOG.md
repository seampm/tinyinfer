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
