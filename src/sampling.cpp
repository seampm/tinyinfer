// Phase 5: sampling implementation.

#include "tinyinfer/sampling.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace tinyinfer {

int sample_token(const float* logits, int vocab_size, const SampleConfig& cfg,
                 std::mt19937_64& rng) {
    // Greedy fast path.
    if (cfg.temperature <= 0.0f || cfg.top_k == 1 || cfg.top_p <= 0.0f) {
        int best = 0;
        for (int i = 1; i < vocab_size; ++i)
            if (logits[i] > logits[best]) best = i;
        return best;
    }

    // Candidate set: top-k by logit.
    std::vector<int> cand(vocab_size);
    std::iota(cand.begin(), cand.end(), 0);
    int k = vocab_size;
    if (cfg.top_k > 0 && cfg.top_k < vocab_size) {
        k = cfg.top_k;
        std::nth_element(cand.begin(), cand.begin() + k, cand.end(),
                         [&](int a, int b) { return logits[a] > logits[b]; });
        cand.resize(k);
    }

    // Softmax with temperature (stable: subtract max first).
    float m = logits[cand[0]];
    for (int id : cand) m = std::max(m, logits[id]);
    std::vector<float> probs;
    probs.reserve(cand.size());
    for (int id : cand) probs.push_back(std::exp((logits[id] - m) / cfg.temperature));

    // Top-p (nucleus) filtering on the candidate set.
    if (cfg.top_p < 1.0f) {
        std::vector<size_t> order(cand.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&](size_t a, size_t b) { return probs[a] > probs[b]; });
        float total = 0;
        for (float p : probs) total += p;
        float cum = 0;
        size_t keep = 0;
        for (; keep < order.size(); ++keep) {
            cum += probs[order[keep]] / total;
            if (cum >= cfg.top_p) {
                ++keep;
                break;
            }
        }
        if (keep == 0) keep = 1;
        std::vector<int> cand2;
        std::vector<float> probs2;
        cand2.reserve(keep);
        probs2.reserve(keep);
        for (size_t i = 0; i < keep; ++i) {
            cand2.push_back(cand[order[i]]);
            probs2.push_back(probs[order[i]]);
        }
        cand.swap(cand2);
        probs.swap(probs2);
    }

    // Multinomial draw.
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);
    float cum = 0, total = 0;
    for (float p : probs) total += p;
    for (size_t i = 0; i < probs.size(); ++i) {
        cum += probs[i] / total;
        if (r < cum) return cand[i];
    }
    return cand.back();
}

} // namespace tinyinfer
