// Phase 6: KV-cache correctness — incremental (cached) forward must match
// full recompute (forward_full) exactly. Same code path, only difference is
// whether K/V rows are reused or recomputed from scratch.

#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "tinyinfer/model.h"

namespace {

const std::string kDir = std::string(TEST_DATA_DIR) + "/llama_tiny";

TEST(KvCache, FullMatchesIncremental) {
    nlohmann::json manifest;
    {
        std::ifstream f(kDir + "/manifest.json");
        ASSERT_TRUE(f) << "run scripts/make_llama_tiny.py first";
        f >> manifest;
    }
    std::vector<int> input_ids = manifest.at("input_ids").get<std::vector<int>>();
    const int seq = static_cast<int>(input_ids.size());
    ASSERT_GT(seq, 1);

    tinyinfer::LlamaModel model;
    model.load(kDir + "/model.safetensors", kDir + "/config.json");
    const auto& cfg = model.config();
    std::vector<float> logits_cached(cfg.vocab_size), logits_full(cfg.vocab_size);

    // Cached: one incremental pass, snapshot logits at each position.
    std::vector<std::vector<float>> snap(seq);
    model.reset();
    for (int pos = 0; pos < seq; ++pos) {
        model.forward(input_ids[pos], pos, logits_cached.data());
        snap[pos] = logits_cached;
    }

    // Naive: recompute from a clean cache for every prefix length.
    for (int p = 0; p < seq; ++p) {
        model.forward_full(input_ids.data(), p + 1, logits_full.data());
        float max_err = 0.0f;
        for (int i = 0; i < cfg.vocab_size; ++i)
            max_err = std::max(max_err, std::fabs(logits_full[i] - snap[p][i]));
        EXPECT_LT(max_err, 1e-6f)
            << "prefix len " << p + 1 << " max_err=" << max_err;
    }
}

TEST(KvCache, ResetGivesFreshStart) {
    // After reset(), a new sequence must not see the previous one's K/V.
    tinyinfer::LlamaModel model;
    model.load(kDir + "/model.safetensors", kDir + "/config.json");
    const auto& cfg = model.config();
    std::vector<float> logits_a(cfg.vocab_size), logits_b(cfg.vocab_size);

    model.reset();
    model.forward(10, 0, logits_a.data());
    model.forward(20, 1, logits_a.data());

    model.reset();
    model.forward(10, 0, logits_b.data());
    model.forward(20, 1, logits_b.data());

    float max_err = 0.0f;
    for (int i = 0; i < cfg.vocab_size; ++i)
        max_err = std::max(max_err, std::fabs(logits_a[i] - logits_b[i]));
    EXPECT_LT(max_err, 1e-6f) << "max_err=" << max_err;
}

} // namespace
