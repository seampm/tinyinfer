# Architecture — Llama Forward Pass (Phase 1)

This is the mental model the C++ engine implements. Verified against
Hugging Face `modeling_llama.py` and Karpathy's "Let's build GPT".

## Forward pass (single token, position `pos`)

```
input_ids -> Embedding -> h

for each layer:
    # 1. Self-attention block
    h_norm  = RMSNorm(h)
    q, k, v = Wq(h_norm), Wk(h_norm), Wv(h_norm)   # GQA: fewer k/v heads than q heads
    q, k    = RoPE(q, k, pos)                      # rotary position embeddings
    attn    = softmax(q @ k^T / sqrt(d)) @ v       # causal mask
    h       = h + Wo(attn)                         # residual

    # 2. Feed-forward block (SwiGLU)
    h_norm  = RMSNorm(h)
    f       = W2( SiLU(W1(h_norm)) * W3(h_norm) )   # elementwise multiply
    h       = h + f                                # residual

logits = LM_head( RMSNorm(h) )                     # vocab-size vector
```

## Key ops (Phase 3)

- **RMSNorm**: `x / sqrt(mean(x^2) + eps) * weight`. No mean subtraction (vs LayerNorm).
- **RoPE**: rotates q/k pairs by angle proportional to position. Implemented as
  interleaved cos/sin applied to head_dim pairs.
- **GQA (Grouped-Query Attention)**: k/v heads < q heads; k/v heads are repeated
  to match q heads. TinyLlama-1.1B uses GQA.
- **SwiGLU**: `SiLU(W1(x)) * W3(x)` then `W2`. SiLU = `x * sigmoid(x)`.
- **Softmax**: numerically stable — subtract max first.
- **Matvec**: matrix-vector multiply is where most runtime goes (memory-bound).

## Reference (answer key)

`scripts/reference.py` runs a fixed prompt through HF transformers and saves
per-layer outputs to `reference_outputs/`:
- `meta.json`: model, prompt, input_ids, tolerance (1e-4)
- `logits.pt`: final logits
- `layer_{i}.pt`: output of each transformer layer

C++ check: each layer's output must match within ~1e-4. When it doesn't,
diff layer-by-layer to find the bug — this catches most bugs in Phase 4.

## Models

- **Dev**: TinyStories 15M/110M (Karpathy llama2.c) — same Llama arch, loads in ~1s.
- **Final**: TinyLlama-1.1B (HF) — real, recognizable, still runs on laptop.

## Resources

- "Attention Is All You Need" (Vaswani et al.)
- Llama papers (Touvron et al.)
- Karpathy's "Let's build GPT" video
- HF `modeling_llama.py` — readable Python reference
