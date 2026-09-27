// Phase 3: math op implementations (fp32, simple and correct).

#include "tinyinfer/ops.h"

#include <cmath>

namespace tinyinfer {

void matvec(const float* W, const float* x, float* out, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        float sum = 0.0f;
        const float* row = W + static_cast<size_t>(i) * cols;
        for (int j = 0; j < cols; ++j) sum += row[j] * x[j];
        out[i] = sum;
    }
}

void rmsnorm(const float* x, const float* weight, float* out, int n, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; ++i) ss += x[i] * x[i];
    ss = 1.0f / std::sqrt(ss / n + eps);
    for (int i = 0; i < n; ++i) out[i] = x[i] * ss * weight[i];
}

void softmax(float* x, int n) {
    float m = x[0];
    for (int i = 1; i < n; ++i)
        if (x[i] > m) m = x[i];
    float s = 0.0f;
    for (int i = 0; i < n; ++i) {
        x[i] = std::exp(x[i] - m);
        s += x[i];
    }
    for (int i = 0; i < n; ++i) x[i] /= s;
}

void rope(float* q, float* k, int num_heads, int head_dim, int pos, float theta) {
    const int half = head_dim / 2;
    for (int h = 0; h < num_heads; ++h) {
        float* qh = q + static_cast<size_t>(h) * head_dim;
        float* kh = k + static_cast<size_t>(h) * head_dim;
        for (int i = 0; i < half; ++i) {
            const float inv_freq = 1.0f / std::pow(theta, (2.0f * i) / head_dim);
            const float angle = pos * inv_freq;
            const float c = std::cos(angle);
            const float s = std::sin(angle);
            // rotate (v[i], v[i+half]) — NeoX style, matches HF Llama
            const float q0 = qh[i], q1 = qh[i + half];
            qh[i] = q0 * c - q1 * s;
            qh[i + half] = q0 * s + q1 * c;
            const float k0 = kh[i], k1 = kh[i + half];
            kh[i] = k0 * c - k1 * s;
            kh[i + half] = k0 * s + k1 * c;
        }
    }
}

float silu(float x) {
    return x / (1.0f + std::exp(-x));
}

void vec_silu(float* x, int n) {
    for (int i = 0; i < n; ++i) x[i] = silu(x[i]);
}

void vec_mul(const float* a, const float* b, float* out, int n) {
    for (int i = 0; i < n; ++i) out[i] = a[i] * b[i];
}

} // namespace tinyinfer
