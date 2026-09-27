#pragma once

// Phase 4: Llama forward pass.
//
//   model.load(weights.safetensors, config.json)
//   model.reset()                       // before each new sequence
//   model.forward(token, pos, logits)   // incremental, pos = 0,1,2,...
//
// The KV cache lives inside the model (Phase 6 measures its payoff).
// Weight names follow Hugging Face Llama; config is read from HF config.json.

#include <string>
#include <vector>

#include "tinyinfer/loader.h"

namespace tinyinfer {

struct ModelConfig {
    int dim = 0;         // hidden_size
    int n_layers = 0;    // num_hidden_layers
    int n_heads = 0;     // num_attention_heads
    int n_kv_heads = 0;  // num_key_value_heads (GQA; == n_heads for MHA)
    int head_dim = 0;    // dim / n_heads
    int vocab_size = 0;
    int ffn_dim = 0;     // intermediate_size
    int max_seq_len = 0; // max_position_embeddings
    float norm_eps = 1e-6f;
    float rope_theta = 10000.0f;

    static ModelConfig from_hf_json(const std::string& path);
};

class LlamaModel {
public:
    LlamaModel() = default;

    // Memory-maps weights and parses config. Throws std::runtime_error on failure.
    void load(const std::string& weights_path, const std::string& config_path);
    bool is_loaded() const { return loaded_; }
    const ModelConfig& config() const { return cfg_; }

    // Clear the KV cache before starting a new sequence.
    void reset();

    // Incremental forward of one token at position pos.
    // logits_out must hold vocab_size floats. pos must be < max_seq_len.
    void forward(int token, int pos, float* logits_out);

    // Same, but also captures each layer's output hidden state (for tests/debug).
    void forward_debug(int token, int pos, float* logits_out,
                       std::vector<std::vector<float>>& layer_outs);

    // Full forward over tokens[0..n-1] from a clean cache; logits of last token.
    // Naive baseline: recomputes every K/V each call (no cache reuse). Also the
    // primitive Phase 7 perplexity scoring needs.
    void forward_full(const int* tokens, int n, float* logits_out);

private:
    const float* w(const std::string& name) const; // F32 weight row-major
    std::string layer(int l, const std::string& suffix) const;

    ModelConfig cfg_;
    SafetensorsLoader loader_;
    bool loaded_ = false;

    // KV cache: k_cache_[l] / v_cache_[l] hold [max_seq_len][n_kv_heads][head_dim].
    std::vector<std::vector<float>> k_cache_, v_cache_;

    // Scratch buffers (allocated once at load).
    std::vector<float> x_, xb_, xb2_, hb_, hb2_;
    std::vector<float> q_, k_, v_, att_, logits_;
};

} // namespace tinyinfer
