// Phase 5: tinyinfer CLI.
//
//   tinyinfer prompt "Once upon a time" [--model DIR] [--max-tokens N]
//             [--temperature T] [--top-k K] [--top-p P] [--seed S]
//   tinyinfer tokenize "some text" [--model DIR]   # debug: print token ids

#include <chrono>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "tinyinfer/model.h"
#include "tinyinfer/sampling.h"
#include "tinyinfer/tokenizer.h"

namespace {

void print_usage(const char* prog) {
    std::cout << "Usage:\n"
              << "  " << prog << " prompt \"text\" [--model DIR] [--max-tokens N]\n"
              << "           [--temperature T] [--top-k K] [--top-p P] [--seed S]\n"
              << "  " << prog << " tokenize \"text\" [--model DIR]\n"
              << "\nDefaults: --model models/tinyllama-15m --max-tokens 50\n"
              << "          --temperature 0.8 --top-k 40 --top-p 0.9 --seed 42\n"
              << "temperature <= 0 means greedy.\n";
}

struct Args {
    std::string cmd;
    std::string text;
    std::string model_dir = "models/tinyllama-15m";
    int max_tokens = 50;
    tinyinfer::SampleConfig sample;
};

bool parse_args(int argc, char** argv, Args& a) {
    if (argc < 3) return false;
    a.cmd = argv[1];
    a.text = argv[2];
    for (int i = 3; i < argc; ++i) {
        std::string o = argv[i];
        auto need = [&](const char* name, std::string& out) {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << name << "\n";
                std::exit(1);
            }
            out = argv[++i];
        };
        std::string v;
        if (o == "--model") need(o.c_str(), a.model_dir);
        else if (o == "--max-tokens") { need(o.c_str(), v); a.max_tokens = std::stoi(v); }
        else if (o == "--temperature") { need(o.c_str(), v); a.sample.temperature = std::stof(v); }
        else if (o == "--top-k") { need(o.c_str(), v); a.sample.top_k = std::stoi(v); }
        else if (o == "--top-p") { need(o.c_str(), v); a.sample.top_p = std::stof(v); }
        else if (o == "--seed") { need(o.c_str(), v); a.sample.seed = std::stoull(v); }
        else { std::cerr << "unknown option: " << o << "\n"; return false; }
    }
    return a.cmd == "prompt" || a.cmd == "tokenize";
}

int cmd_tokenize(const Args& a) {
    tinyinfer::Tokenizer tok;
    tok.load(a.model_dir + "/tokenizer.json");
    auto ids = tok.encode(a.text, false);
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i) std::cout << ' ';
        std::cout << ids[i];
    }
    std::cout << "\n";
    return 0;
}

int cmd_prompt(const Args& a) {
    tinyinfer::LlamaModel model;
    model.load(a.model_dir + "/model.safetensors", a.model_dir + "/config.json");
    tinyinfer::Tokenizer tok;
    tok.load(a.model_dir + "/tokenizer.json");

    std::vector<int> ids = tok.encode(a.text, true); // + bos
    const int max_seq = model.config().max_seq_len;
    if (static_cast<int>(ids.size()) + a.max_tokens > max_seq) {
        std::cerr << "error: prompt (" << ids.size() << " tokens) + max-tokens ("
                  << a.max_tokens << ") exceeds max_seq_len (" << max_seq << ")\n";
        return 1;
    }

    std::vector<float> logits(model.config().vocab_size);
    std::mt19937_64 rng(a.sample.seed);

    // Echo the prompt (decode strips the leading space from the prepended ▁).
    std::cout << tok.decode(ids) << std::flush;

    model.reset();
    auto t0 = std::chrono::steady_clock::now();
    int generated = 0, next = 0;
    for (int pos = 0; pos < static_cast<int>(ids.size()) + a.max_tokens; ++pos) {
        int cur = (pos < static_cast<int>(ids.size())) ? ids[pos] : next;
        model.forward(cur, pos, logits.data());
        if (pos < static_cast<int>(ids.size()) - 1) continue; // still in prompt
        next = tinyinfer::sample_token(logits.data(), model.config().vocab_size,
                                       a.sample, rng);
        if (next == tok.eos_id()) break;
        ids.push_back(next);
        ++generated;
        // Stream the new token: raw bytes, with ▁ rendered as a space.
        std::string piece = tok.decode_token(next);
        for (size_t i = 0; (i = piece.find("\xE2\x96\x81", i)) != std::string::npos;)
            piece.replace(i, 3, " ");
        std::cout << piece << std::flush;
        if (generated >= a.max_tokens) break;
    }
    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "\n";
    std::cerr << "[tinyinfer] " << generated << " tokens in " << secs << "s ("
              << (secs > 0 ? generated / secs : 0) << " tok/s)\n";
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    Args a;
    if (!parse_args(argc, argv, a)) {
        print_usage(argv[0]);
        return 1;
    }
    try {
        if (a.cmd == "tokenize") return cmd_tokenize(a);
        return cmd_prompt(a);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
