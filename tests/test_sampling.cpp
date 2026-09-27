// Phase 5: sampling tests.

#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "tinyinfer/sampling.h"

namespace {

const int V = 64;

std::vector<float> LogitsSpike() {
    std::vector<float> l(V, -10.0f);
    l[7] = 5.0f;
    l[20] = 4.0f;
    return l;
}

TEST(Sampling, GreedyPicksArgmax) {
    auto l = LogitsSpike();
    tinyinfer::SampleConfig cfg;
    cfg.temperature = 0.0f;
    std::mt19937_64 rng(1);
    EXPECT_EQ(tinyinfer::sample_token(l.data(), V, cfg, rng), 7);
}

TEST(Sampling, TopK1IsGreedy) {
    auto l = LogitsSpike();
    tinyinfer::SampleConfig cfg;
    cfg.temperature = 1.0f;
    cfg.top_k = 1;
    std::mt19937_64 rng(123);
    for (int i = 0; i < 20; ++i)
        EXPECT_EQ(tinyinfer::sample_token(l.data(), V, cfg, rng), 7);
}

TEST(Sampling, TopPZeroKeepsTop1) {
    auto l = LogitsSpike();
    tinyinfer::SampleConfig cfg;
    cfg.temperature = 1.0f;
    cfg.top_p = 0.0f;
    std::mt19937_64 rng(123);
    EXPECT_EQ(tinyinfer::sample_token(l.data(), V, cfg, rng), 7);
}

TEST(Sampling, FixedSeedReproducible) {
    auto l = LogitsSpike();
    tinyinfer::SampleConfig cfg;
    cfg.temperature = 0.9f;
    cfg.top_k = 10;
    cfg.top_p = 0.95f;
    auto run = [&]() {
        std::mt19937_64 rng(cfg.seed);
        std::vector<int> out;
        for (int i = 0; i < 50; ++i) out.push_back(tinyinfer::sample_token(l.data(), V, cfg, rng));
        return out;
    };
    EXPECT_EQ(run(), run());
}

TEST(Sampling, DifferentSeedsDiverge) {
    auto l = LogitsSpike();
    auto run = [&](uint64_t seed) {
        tinyinfer::SampleConfig cfg;
        cfg.temperature = 1.0f;
        cfg.top_k = 0;
        cfg.top_p = 1.0f;
        cfg.seed = seed;
        std::mt19937_64 rng(seed);
        std::vector<int> out;
        for (int i = 0; i < 50; ++i) out.push_back(tinyinfer::sample_token(l.data(), V, cfg, rng));
        return out;
    };
    EXPECT_NE(run(1), run(2));
}

TEST(Sampling, TopKRestrictsCandidates) {
    auto l = LogitsSpike();
    tinyinfer::SampleConfig cfg;
    cfg.temperature = 100.0f; // near-uniform over candidates
    cfg.top_k = 2;
    cfg.top_p = 1.0f;
    std::mt19937_64 rng(7);
    for (int i = 0; i < 50; ++i) {
        int id = tinyinfer::sample_token(l.data(), V, cfg, rng);
        EXPECT_TRUE(id == 7 || id == 20) << "id=" << id;
    }
}

TEST(Sampling, StaysInRange) {
    std::vector<float> l(V, 0.0f); // uniform logits
    tinyinfer::SampleConfig cfg;
    std::mt19937_64 rng(99);
    for (int i = 0; i < 100; ++i) {
        int id = tinyinfer::sample_token(l.data(), V, cfg, rng);
        EXPECT_GE(id, 0);
        EXPECT_LT(id, V);
    }
}

} // namespace
