// Phase 4: forward-pass test vs the Python reference fixture.
// Fixture: tests/test_data/llama_tiny/ (scripts/make_llama_tiny.py).
// The C++ forward runs incrementally (token-by-token with KV cache) and must
// match the full-sequence HF forward within 1e-4 at every layer and position.

#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "tinyinfer/model.h"

namespace {

const std::string kDir = std::string(TEST_DATA_DIR) + "/llama_tiny";

std::vector<float> ReadBin(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("test: cannot open " + path);
    const size_t n = static_cast<size_t>(f.tellg()) / sizeof(float);
    std::vector<float> v(n);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(v.data()), n * sizeof(float));
    if (!f) throw std::runtime_error("test: short read " + path);
    return v;
}

TEST(Forward, MatchesPythonReference) {
    nlohmann::json manifest;
    {
        std::ifstream f(kDir + "/manifest.json");
        ASSERT_TRUE(f) << "run scripts/make_llama_tiny.py first";
        f >> manifest;
    }
    const float tol = manifest.at("tolerance").get<float>();
    std::vector<int> input_ids = manifest.at("input_ids").get<std::vector<int>>();
    const int seq = static_cast<int>(input_ids.size());

    tinyinfer::LlamaModel model;
    model.load(kDir + "/model.safetensors", kDir + "/config.json");
    const auto& cfg = model.config();
    ASSERT_GT(cfg.n_layers, 0);

    std::vector<float> logits(cfg.vocab_size);
    std::vector<std::vector<float>> layer_outs;

    // Reference tensors: layer_i -> [seq, dim], logits -> [seq, vocab].
    std::vector<std::vector<float>> ref_layers;
    for (int l = 0; l < cfg.n_layers; ++l)
        ref_layers.push_back(ReadBin(kDir + "/layer_" + std::to_string(l) + ".bin"));
    const std::vector<float> ref_logits = ReadBin(kDir + "/logits.bin");

    model.reset();
    for (int pos = 0; pos < seq; ++pos) {
        model.forward_debug(input_ids[pos], pos, logits.data(), layer_outs);
        ASSERT_EQ(static_cast<int>(layer_outs.size()), cfg.n_layers);

        for (int l = 0; l < cfg.n_layers; ++l) {
            const float* ref = ref_layers[l].data() + static_cast<size_t>(pos) * cfg.dim;
            float max_err = 0;
            for (int i = 0; i < cfg.dim; ++i)
                max_err = std::max(max_err, std::fabs(layer_outs[l][i] - ref[i]));
            EXPECT_LT(max_err, tol) << "layer " << l << " pos " << pos
                                    << " max_err=" << max_err;
        }
        const float* ref_l = ref_logits.data() + static_cast<size_t>(pos) * cfg.vocab_size;
        float max_err = 0;
        for (int i = 0; i < cfg.vocab_size; ++i)
            max_err = std::max(max_err, std::fabs(logits[i] - ref_l[i]));
        EXPECT_LT(max_err, tol) << "logits pos " << pos << " max_err=" << max_err;
    }
}

TEST(Forward, IncrementalMatchesFresh) {
    // Same sequence run twice (with reset between) must be bit-identical.
    tinyinfer::LlamaModel model;
    model.load(kDir + "/model.safetensors", kDir + "/config.json");
    const auto& cfg = model.config();
    const std::vector<int> ids = {1, 7, 13, 42};

    auto run = [&]() {
        std::vector<float> all(cfg.vocab_size * ids.size());
        std::vector<float> lg(cfg.vocab_size);
        model.reset();
        for (size_t p = 0; p < ids.size(); ++p) {
            model.forward(ids[p], static_cast<int>(p), lg.data());
            std::copy(lg.begin(), lg.end(), all.begin() + p * cfg.vocab_size);
        }
        return all;
    };
    EXPECT_EQ(run(), run());
}

TEST(Forward, BadInputsThrow) {
    tinyinfer::LlamaModel model;
    model.load(kDir + "/model.safetensors", kDir + "/config.json");
    std::vector<float> lg(model.config().vocab_size);
    EXPECT_THROW(model.forward(-1, 0, lg.data()), std::out_of_range);
    EXPECT_THROW(model.forward(999999, 0, lg.data()), std::out_of_range);
    EXPECT_THROW(model.forward(1, model.config().max_seq_len, lg.data()), std::out_of_range);

    tinyinfer::LlamaModel empty;
    EXPECT_THROW(empty.forward(1, 0, lg.data()), std::runtime_error);
}

} // namespace

TEST(Forward, Int8CloseToF32) {
    // Phase 7: per-row symmetric int8 must not wreck the forward pass.
    nlohmann::json manifest;
    {
        std::ifstream f(kDir + "/manifest.json");
        ASSERT_TRUE(f) << "run scripts/make_llama_tiny.py first";
        f >> manifest;
    }
    std::vector<int> input_ids = manifest.at("input_ids").get<std::vector<int>>();
    const int seq = static_cast<int>(input_ids.size());

    tinyinfer::LlamaModel f32, i8;
    f32.load(kDir + "/model.safetensors", kDir + "/config.json");
    i8.set_quant_mode(tinyinfer::QuantMode::I8);
    i8.load(kDir + "/model.safetensors", kDir + "/config.json");
    const auto& cfg = f32.config();
    std::vector<float> lf(cfg.vocab_size), li(cfg.vocab_size);

    f32.reset();
    double num = 0, den = 0;
    for (int pos = 0; pos < seq; ++pos) {
        f32.forward(input_ids[pos], pos, lf.data());
        i8.forward_full(input_ids.data(), pos + 1, li.data());
        for (int v = 0; v < cfg.vocab_size; ++v) {
            num += (li[v] - lf[v]) * (li[v] - lf[v]);
            den += lf[v] * lf[v];
        }
    }
    EXPECT_LT(std::sqrt(num / den), 0.05) << "int8 forward drifted too far";
}
