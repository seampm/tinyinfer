// Phase 3: math op implementations (fp32, simple and correct).
// Phase 7: AVX2/FMA matvec + OpenMP over rows; int8 matvec for quantization.

#include "tinyinfer/ops.h"

#include <cmath>

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace tinyinfer {

void matvec_scalar(const float* W, const float* x, float* out, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        float sum = 0.0f;
        const float* row = W + static_cast<size_t>(i) * cols;
        for (int j = 0; j < cols; ++j) sum += row[j] * x[j];
        out[i] = sum;
    }
}

#ifdef __AVX2__
// FMA dot product, 8 floats per iteration; tail handled scalar.
static float dot_avx2(const float* a, const float* b, int n) {
    __m256 acc = _mm256_setzero_ps();
    int j = 0;
    for (; j + 8 <= n; j += 8)
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + j), _mm256_loadu_ps(b + j), acc);
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    float sum = _mm_cvtss_f32(s);
    for (; j < n; ++j) sum += a[j] * b[j];
    return sum;
}
#endif

void quantize_i8(const float* W, int rows, int cols, int8_t* Q, float* scales) {
    for (int i = 0; i < rows; ++i) {
        const float* row = W + static_cast<size_t>(i) * cols;
        float amax = 0.0f;
        for (int j = 0; j < cols; ++j) {
            float a = std::fabs(row[j]);
            if (a > amax) amax = a;
        }
        const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
        scales[i] = scale;
        const float inv = 1.0f / scale;
        int8_t* qrow = Q + static_cast<size_t>(i) * cols;
        for (int j = 0; j < cols; ++j) {
            float v = std::round(row[j] * inv);
            if (v > 127.0f) v = 127.0f;
            if (v < -127.0f) v = -127.0f;
            qrow[j] = static_cast<int8_t>(v);
        }
    }
}

void matvec_i8(const int8_t* Wq, const float* scale, const float* x, float* out,
               int rows, int cols) {
    const bool big =
#ifdef _OPENMP
        rows >= 4096;
#else
        false;
#endif
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (big)
#endif
    for (int i = 0; i < rows; ++i) {
        const int8_t* row = Wq + static_cast<size_t>(i) * cols;
        float dot = 0.0f;
#ifdef __AVX2__
        __m256 acc = _mm256_setzero_ps();
        int j = 0;
        for (; j + 8 <= cols; j += 8) {
            // 8x int8 -> int16 -> int32 -> fp32, FMA with x
            __m128i b8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(row + j));
            __m256 f =
                _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_cvtepi8_epi16(b8)));
            acc = _mm256_fmadd_ps(f, _mm256_loadu_ps(x + j), acc);
        }
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 s = _mm_add_ps(lo, hi);
        s = _mm_hadd_ps(s, s);
        s = _mm_hadd_ps(s, s);
        dot = _mm_cvtss_f32(s);
        for (; j < cols; ++j) dot += static_cast<float>(row[j]) * x[j];
#else
        for (int j = 0; j < cols; ++j) dot += static_cast<float>(row[j]) * x[j];
#endif
        out[i] = dot * scale[i];
    }
}

void matvec(const float* W, const float* x, float* out, int rows, int cols) {
    // Thread only large row counts (the lm_head/vocab matvec); for the small
    // attention/MLP mats OpenMP overhead exceeds the gain (measured Phase 7).
    const bool big =
#ifdef _OPENMP
        rows >= 4096;
#else
        false;
#endif
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (big)
#endif
    for (int i = 0; i < rows; ++i) {
        const float* row = W + static_cast<size_t>(i) * cols;
#ifdef __AVX2__
        out[i] = dot_avx2(row, x, cols);
#else
        float sum = 0.0f;
        for (int j = 0; j < cols; ++j) sum += row[j] * x[j];
        out[i] = sum;
#endif
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
