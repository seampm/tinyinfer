#pragma once

// Phase 5: sampling — temperature, top-k, top-p (nucleus), greedy fallback.

#include <cstdint>
#include <random>

namespace tinyinfer {

struct SampleConfig {
    float temperature = 0.8f;
    int top_k = 40;     // <= 0 disables
    float top_p = 0.9f; // >= 1.0 disables, <= 0 keeps top-1 only
    uint64_t seed = 42;
};

// Returns the next token id. temperature <= 0 (or top_k == 1) -> greedy argmax.
int sample_token(const float* logits, int vocab_size, const SampleConfig& cfg,
                 std::mt19937_64& rng);

} // namespace tinyinfer
