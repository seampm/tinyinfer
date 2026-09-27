// Phase 4: Llama forward pass implementation.

#include "tinyinfer/model.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "tinyinfer/ops.h"

namespace tinyinfer {

ModelConfig ModelConfig::from_hf_json(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("model: cannot open config: " + path);
    nlohmann::json j;
    f >> j;

    ModelConfig c;
    c.dim = j.at("hidden_size").get<int>();
    c.n_layers = j.at("num_hidden_layers").get<int>();
    c.n_heads = j.at("num_attention_heads").get<int>();
    c.n_kv_heads = j.value("num_key_value_heads", c.n_heads);
    c.head_dim = c.dim / c.n_heads;
    c.vocab_size = j.at("vocab_size").get<int>();
    c.ffn_dim = j.at("intermediate_size").get<int>();
    c.max_seq_len = j.at("max_position_embeddings").get<int>();
    c.norm_eps = j.value("rms_norm_eps", 1e-6f);
    if (j.contains("rope_parameters") && j["rope_parameters"].contains("rope_theta"))
        c.rope_theta = j["rope_parameters"]["rope_theta"].get<float>();
    else
        c.rope_theta = j.value("rope_theta", 10000.0f);

    if (c.dim <= 0 || c.n_layers <= 0 || c.n_heads <= 0 || c.dim % c.n_heads != 0)
        throw std::runtime_error("model: invalid config in " + path);
    return c;
}

std::string LlamaModel::layer(int l, const std::string& suffix) const {
    return "model.layers." + std::to_string(l) + "." + suffix;
}

const float* LlamaModel::w(const std::string& name) const {
    const TensorInfo& t = loader_.get(name);
    if (t.dtype != DType::F32)
        throw std::runtime_error("model: expected F32 weights, '" + name + "' is " +
                                 dtype_name(t.dtype));
    return t.as<float>();
}

void LlamaModel::quantize_into_as(const std::string& qname, const std::string& wname) {
    const TensorInfo& t = loader_.get(wname);
    if (t.dtype != DType::F32 || t.shape.size() != 2)
        throw std::runtime_error("model: cannot quantize '" + wname + "'");
    QT qt;
    qt.rows = static_cast<int>(t.shape[0]);
    qt.cols = static_cast<int>(t.shape[1]);
    qt.q.resize(static_cast<size_t>(qt.rows) * qt.cols);
    qt.s.resize(qt.rows);
    quantize_i8(t.as<float>(), qt.rows, qt.cols, qt.q.data(), qt.s.data());
    qmap_[qname] = std::move(qt);
}

void LlamaModel::quantize_into(const std::string& name) {
    quantize_into_as(name, name);
}

void LlamaModel::matvec_w(const std::string& name, const float* x, float* y, int rows,
                           int cols) const {
    if (quant_ == QuantMode::F32) {
        matvec(w(name), x, y, rows, cols);
        return;
    }
    auto it = qmap_.find(name);
    if (it == qmap_.end())
        throw std::runtime_error("model: no quantized weights for '" + name + "'");
    const QT& qt = it->second;
    matvec_i8(qt.q.data(), qt.s.data(), x, y, qt.rows, qt.cols);
}

void LlamaModel::load(const std::string& weights_path, const std::string& config_path) {
    cfg_ = ModelConfig::from_hf_json(config_path);
    loader_.open(weights_path);

    // Sanity-check a few expected tensors exist (fail fast on wrong checkpoint).
    loader_.get("model.embed_tokens.weight");
    loader_.get(layer(0, "self_attn.q_proj.weight"));
    loader_.get("model.norm.weight");

    if (quant_ == QuantMode::I8) {
        // Quantize every matvec weight matrix. The embedding table stays fp32
        // (lookup path), but the lm head needs a quantized copy of it when
        // embeddings are tied.
        for (int l = 0; l < cfg_.n_layers; ++l) {
            quantize_into(layer(l, "self_attn.q_proj.weight"));
            quantize_into(layer(l, "self_attn.k_proj.weight"));
            quantize_into(layer(l, "self_attn.v_proj.weight"));
            quantize_into(layer(l, "self_attn.o_proj.weight"));
            quantize_into(layer(l, "mlp.gate_proj.weight"));
            quantize_into(layer(l, "mlp.up_proj.weight"));
            quantize_into(layer(l, "mlp.down_proj.weight"));
        }
        if (loader_.has("lm_head.weight"))
            quantize_into("lm_head.weight");
        else
            quantize_into_as("lm_head.weight", "model.embed_tokens.weight");
    }

    const size_t kv_elems =
        static_cast<size_t>(cfg_.max_seq_len) * cfg_.n_kv_heads * cfg_.head_dim;
    k_cache_.assign(cfg_.n_layers, std::vector<float>(kv_elems, 0.0f));
    v_cache_.assign(cfg_.n_layers, std::vector<float>(kv_elems, 0.0f));

    x_.resize(cfg_.dim);
    xb_.resize(cfg_.dim);
    xb2_.resize(cfg_.dim);
    hb_.resize(cfg_.ffn_dim);
    hb2_.resize(cfg_.ffn_dim);
    q_.resize(static_cast<size_t>(cfg_.n_heads) * cfg_.head_dim);
    k_.resize(static_cast<size_t>(cfg_.n_kv_heads) * cfg_.head_dim);
    v_.resize(static_cast<size_t>(cfg_.n_kv_heads) * cfg_.head_dim);
    att_.resize(static_cast<size_t>(cfg_.n_heads) * cfg_.max_seq_len);
    logits_.resize(cfg_.vocab_size);

    loaded_ = true;
}

void LlamaModel::reset() {
    for (auto& v : k_cache_) std::fill(v.begin(), v.end(), 0.0f);
    for (auto& v : v_cache_) std::fill(v.begin(), v.end(), 0.0f);
}

void LlamaModel::forward(int token, int pos, float* logits_out) {
    std::vector<std::vector<float>> unused;
    forward_debug(token, pos, logits_out, unused);
}

void LlamaModel::forward_full(const int* tokens, int n, float* logits_out) {
    if (n <= 0) throw std::invalid_argument("model: forward_full needs n > 0");
    reset();
    std::vector<std::vector<float>> unused;
    for (int p = 0; p < n; ++p) forward_debug(tokens[p], p, logits_out, unused);
}

void LlamaModel::forward_debug(int token, int pos, float* logits_out,
                               std::vector<std::vector<float>>& layer_outs) {
    if (!loaded_) throw std::runtime_error("model: not loaded");
    if (token < 0 || token >= cfg_.vocab_size)
        throw std::out_of_range("model: token id out of range");
    if (pos < 0 || pos >= cfg_.max_seq_len)
        throw std::out_of_range("model: position out of range");

    const int dim = cfg_.dim, n_h = cfg_.n_heads, n_kv = cfg_.n_kv_heads;
    const int hd = cfg_.head_dim, ffn = cfg_.ffn_dim;
    const int group = n_h / n_kv; // q heads per kv head (GQA)
    const float attn_scale = 1.0f / std::sqrt(static_cast<float>(hd));

    // 1. token embedding
    const float* embed = w("model.embed_tokens.weight");
    std::memcpy(x_.data(), embed + static_cast<size_t>(token) * dim, dim * sizeof(float));

    layer_outs.clear();
    layer_outs.reserve(cfg_.n_layers);

    // 2. transformer layers
    for (int l = 0; l < cfg_.n_layers; ++l) {
        // attention block
        rmsnorm(x_.data(), w(layer(l, "input_layernorm.weight")), xb_.data(), dim, cfg_.norm_eps);
        matvec_w(layer(l, "self_attn.q_proj.weight"), xb_.data(), q_.data(), n_h * hd, dim);
        matvec_w(layer(l, "self_attn.k_proj.weight"), xb_.data(), k_.data(), n_kv * hd, dim);
        matvec_w(layer(l, "self_attn.v_proj.weight"), xb_.data(), v_.data(), n_kv * hd, dim);
        rope(q_.data(), k_.data(), n_h, n_kv, hd, pos, cfg_.rope_theta);

        // append k/v to cache, then attend over positions 0..pos (causal by construction)
        float* Kc = k_cache_[l].data();
        float* Vc = v_cache_[l].data();
        std::memcpy(Kc + static_cast<size_t>(pos) * n_kv * hd, k_.data(),
                    static_cast<size_t>(n_kv) * hd * sizeof(float));
        std::memcpy(Vc + static_cast<size_t>(pos) * n_kv * hd, v_.data(),
                    static_cast<size_t>(n_kv) * hd * sizeof(float));

        for (int h = 0; h < n_h; ++h) {
            const int kh = h / group;
            const float* qh = q_.data() + static_cast<size_t>(h) * hd;
            float* scores = att_.data() + static_cast<size_t>(h) * cfg_.max_seq_len;
            for (int j = 0; j <= pos; ++j) {
                const float* kj = Kc + (static_cast<size_t>(j) * n_kv + kh) * hd;
                float s = 0.0f;
                for (int d = 0; d < hd; ++d) s += qh[d] * kj[d];
                scores[j] = s * attn_scale;
            }
            softmax(scores, pos + 1);
            float* out = xb_.data() + static_cast<size_t>(h) * hd;
            for (int d = 0; d < hd; ++d) {
                float s = 0.0f;
                for (int j = 0; j <= pos; ++j) {
                    const float* vj = Vc + (static_cast<size_t>(j) * n_kv + kh) * hd;
                    s += scores[j] * vj[d];
                }
                out[d] = s;
            }
        }
        matvec_w(layer(l, "self_attn.o_proj.weight"), xb_.data(), xb2_.data(), dim, n_h * hd);
        for (int i = 0; i < dim; ++i) x_[i] += xb2_[i];

        // SwiGLU MLP block
        rmsnorm(x_.data(), w(layer(l, "post_attention_layernorm.weight")), xb_.data(), dim,
                cfg_.norm_eps);
        matvec_w(layer(l, "mlp.gate_proj.weight"), xb_.data(), hb_.data(), ffn, dim);
        matvec_w(layer(l, "mlp.up_proj.weight"), xb_.data(), hb2_.data(), ffn, dim);
        vec_silu(hb_.data(), ffn);
        vec_mul(hb_.data(), hb2_.data(), hb_.data(), ffn);
        matvec_w(layer(l, "mlp.down_proj.weight"), hb_.data(), xb_.data(), dim, ffn);
        for (int i = 0; i < dim; ++i) x_[i] += xb_[i];

        layer_outs.push_back(x_); // copy: this layer's output hidden state
    }

    // 3. final norm + lm head
    rmsnorm(x_.data(), w("model.norm.weight"), xb_.data(), dim, cfg_.norm_eps);
    if (quant_ == QuantMode::F32) {
        const TensorInfo* head = nullptr;
        if (loader_.has("lm_head.weight"))
            head = &loader_.get("lm_head.weight");
        else
            head = &loader_.get("model.embed_tokens.weight"); // tied embeddings
        matvec(head->as<float>(), xb_.data(), logits_out, cfg_.vocab_size, dim);
    } else {
        matvec_w("lm_head.weight", xb_.data(), logits_out, cfg_.vocab_size, dim);
    }
}

} // namespace tinyinfer
