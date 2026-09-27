#pragma once

// Phase 3: standalone math ops, 32-bit floats, simplest correct versions.
// Phase 7: matvec gets an AVX2/FMA dot product + OpenMP over rows.
// Signatures stable; matvec_scalar kept as the test reference.

#include <cstdint>

namespace tinyinfer {

// out = W @ x ; W is rows x cols, row-major. The hot spot: most runtime lives here.
void matvec(const float* W, const float* x, float* out, int rows, int cols);

// Scalar reference (what matvec was in Phase 3); the AVX2 path is tested against it.
void matvec_scalar(const float* W, const float* x, float* out, int rows, int cols);

// Phase 7: int8 weights, fp32 activations.
// out[i] = scale[i] * dot((float)Wq[i,:], x); Wq is rows x cols int8 row-major.
void matvec_i8(const int8_t* Wq, const float* scale, const float* x, float* out,
               int rows, int cols);

// Per-row symmetric int8 quantization of a rows x cols fp32 matrix:
//   scale[r] = max|W[r,:]| / 127 (1.0f if the row is all zeros)
//   Q[r,c] = round(clamp(W[r,c] / scale[r], -127, 127))
void quantize_i8(const float* W, int rows, int cols, int8_t* Q, float* scales);

// RMSNorm (Llama style, no mean subtraction):
//   out[i] = x[i] / sqrt(mean(x^2) + eps) * weight[i]
void rmsnorm(const float* x, const float* weight, float* out, int n, float eps);

// In-place softmax, numerically stable (subtracts max before exp).
void softmax(float* x, int n);

// Rotary position embeddings, GPT-NeoX style (matches HF Llama).
// Rotates q ([n_q_heads * head_dim]) and k ([n_kv_heads * head_dim]) in place
// at position pos. Pair i (0 <= i < head_dim/2): angle = pos / theta^(2i/head_dim),
//   (a, b) = (v[i], v[i + head_dim/2]) ->
//   (a*cos(angle) - b*sin(angle), a*sin(angle) + b*cos(angle))
void rope(float* q, float* k, int n_q_heads, int n_kv_heads, int head_dim, int pos,
          float theta = 10000.0f);

float silu(float x);              // x * sigmoid(x)
void vec_silu(float* x, int n);   // in-place SiLU
void vec_mul(const float* a, const float* b, float* out, int n); // out[i] = a[i]*b[i]

} // namespace tinyinfer
