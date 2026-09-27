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

void rope(float* q, float* k, int n_q_heads, int n_kv_heads, int head_dim, int pos,
          float theta) {
    const int half = head_dim / 2;
    // inv_freq depends only on i, not on the head — hoist it out of the head loop.
    for (int i = 0; i < half; ++i) {
        const float inv_freq = 1.0f / std::pow(theta, (2.0f * i) / head_dim);
        const float angle = pos * inv_freq;
        const float c = std::cos(angle);
        const float s = std::sin(angle);
        for (int h = 0; h < n_q_heads; ++h) {
            float* qh = q + static_cast<size_t>(h) * head_dim;
            const float a = qh[i], b = qh[i + half];
            qh[i] = a * c - b * s;
            qh[i + half] = a * s + b * c;
        }
        for (int h = 0; h < n_kv_heads; ++h) {
            float* kh = k + static_cast<size_t>(h) * head_dim;
            const float a = kh[i], b = kh[i + half];
            kh[i] = a * c - b * s;
            kh[i + half] = a * s + b * c;
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
