// Phase 5: tinyinfer CLI.
//
//   tinyinfer prompt "Once upon a time" [--model DIR] [--max-tokens N]
//             [--temperature T] [--top-k K] [--top-p P] [--seed S]
//   tinyinfer tokenize "some text" [--model DIR]   # debug: print token ids

#include <chrono>
#include <fstream>
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
              << "  " << prog << " bench \"text\" [--model DIR] [--max-tokens N]\n"
              << "  " << prog << " perplexity --data FILE [--model DIR] [--limit N]\n"
              << "             [--quant f32|i8]\n"
              << "\nDefaults: --model models/tinyllama-15m --max-tokens 50\n"
              << "          --temperature 0.8 --top-k 40 --top-p 0.9 --seed 42\n"
              << "temperature <= 0 means greedy.\n"
              << "bench: greedy generation with vs without KV cache; reports tok/s.\n";
}

struct Args {
    std::string cmd;
    std::string text;
    std::string model_dir = "models/tinyllama-15m";
    std::string data_file;          // perplexity --data (raw text)
    std::string ids_file;           // perplexity --ids (pre-tokenized ids)
    int max_tokens = 50;
    int limit = 20000;              // perplexity: max tokens to score
    std::string quant = "f32";      // f32 | i8
    bool skip_naive = false;        // bench: skip the naive recompute baseline
    tinyinfer::SampleConfig sample;
};

bool parse_args(int argc, char** argv, Args& a) {
    if (argc < 2) return false;
    a.cmd = argv[1];
    int i = 2;
    if (a.cmd != "perplexity") {
        if (argc < 3) return false;
        a.text = argv[2];
        i = 3;
    }
    for (; i < argc; ++i) {
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
        else if (o == "--data") need(o.c_str(), a.data_file);
        else if (o == "--ids") need(o.c_str(), a.ids_file);
        else if (o == "--quant") need(o.c_str(), a.quant);
        else if (o == "--max-tokens") { need(o.c_str(), v); a.max_tokens = std::stoi(v); }
        else if (o == "--skip-naive") a.skip_naive = true;
        else if (o == "--limit") { need(o.c_str(), v); a.limit = std::stoi(v); }
        else if (o == "--temperature") { need(o.c_str(), v); a.sample.temperature = std::stof(v); }
        else if (o == "--top-k") { need(o.c_str(), v); a.sample.top_k = std::stoi(v); }
        else if (o == "--top-p") { need(o.c_str(), v); a.sample.top_p = std::stof(v); }
        else if (o == "--seed") { need(o.c_str(), v); a.sample.seed = std::stoull(v); }
        else { std::cerr << "unknown option: " << o << "\n"; return false; }
    }
    return a.cmd == "prompt" || a.cmd == "tokenize" || a.cmd == "bench" ||
           a.cmd == "perplexity";
}

// Applies --quant to a freshly created model. Returns false on bad flag value.
bool apply_quant(const Args& a, tinyinfer::LlamaModel& model) {
    if (a.quant == "i8")
        model.set_quant_mode(tinyinfer::QuantMode::I8);
    else if (a.quant != "f32") {
        std::cerr << "error: --quant must be f32 or i8\n";
        return false;
    }
    return true;
}

int cmd_tokenize(const Args& a) {    tinyinfer::Tokenizer tok;
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
    if (!apply_quant(a, model)) return 1;
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

// Phase 6: benchmark cached (incremental forward) vs naive (forward_full per
// token, recomputing all K/V) generation. Greedy in both modes; the token
// streams must be identical, so this also stress-tests the KV cache.
int cmd_bench(const Args& a) {
    tinyinfer::LlamaModel model;
    if (!apply_quant(a, model)) return 1;
    model.load(a.model_dir + "/model.safetensors", a.model_dir + "/config.json");
    tinyinfer::Tokenizer tok;
    tok.load(a.model_dir + "/tokenizer.json");
    const int vocab = model.config().vocab_size;

    std::vector<int> prompt = tok.encode(a.text, true); // + bos
    const int max_seq = model.config().max_seq_len;
    if (static_cast<int>(prompt.size()) + a.max_tokens > max_seq) {
        std::cerr << "error: prompt (" << prompt.size() << " tokens) + max-tokens ("
                  << a.max_tokens << ") exceeds max_seq_len (" << max_seq << ")\n";
        return 1;
    }

    tinyinfer::SampleConfig greedy; // temperature 0 -> argmax
    greedy.temperature = 0.0f;
    std::vector<float> logits(vocab);

    struct Run {
        std::vector<int> gen;
        double ttft = 0;   // seconds to first sampled token (incl. prompt)
        double decode = 0; // seconds from first to last sampled token
    };
    auto run = [&](bool use_cache) {
        Run r;
        std::mt19937_64 rng(1234); // same draws both modes (greedy: unused)
        std::vector<int> ids = prompt;
        auto t0 = std::chrono::steady_clock::now();
        if (use_cache) {
            model.reset();
            for (int p = 0; p < static_cast<int>(ids.size()); ++p)
                model.forward(ids[p], p, logits.data());
        } else {
            model.forward_full(ids.data(), static_cast<int>(ids.size()), logits.data());
        }
        for (int step = 0; step < a.max_tokens; ++step) {
            int next =
                tinyinfer::sample_token(logits.data(), vocab, greedy, rng);
            auto t_now = std::chrono::steady_clock::now();
            if (step == 0)
                r.ttft = std::chrono::duration<double>(t_now - t0).count();
            if (next == tok.eos_id()) break;
            r.gen.push_back(next);
            ids.push_back(next);
            const int pos = static_cast<int>(ids.size()) - 1;
            if (use_cache)
                model.forward(next, pos, logits.data());
            else
                model.forward_full(ids.data(), static_cast<int>(ids.size()),
                                   logits.data());
        }
        auto t1 = std::chrono::steady_clock::now();
        r.decode = std::chrono::duration<double>(t1 - t0).count() - r.ttft;
        return r;
    };

    Run cached = run(true);

    auto tps = [](const Run& r) {
        return r.decode > 0 ? r.gen.size() / r.decode : 0.0;
    };
    std::cout << "[tinyinfer bench] prompt_tokens=" << prompt.size()
              << " max_tokens=" << a.max_tokens << " greedy\n";
    std::cout << "mode    generated  ttft(s)   decode(s)  tok/s\n";
    std::cout << "cached  " << cached.gen.size() << "         " << cached.ttft << "  "
              << cached.decode << "  " << tps(cached) << "\n";
    bool identical = true;
    if (!a.skip_naive) {
        Run naive = run(false);
        identical = cached.gen == naive.gen;
        std::cout << "naive   " << naive.gen.size() << "         " << naive.ttft
                  << "  " << naive.decode << "  " << tps(naive) << "\n";
        std::cout << "cache speedup: " << tps(cached) / tps(naive)
                  << "x; token streams identical: "
                  << (identical ? "yes" : "NO") << "\n";
    }
    std::cout << "sample: " << tok.decode(cached.gen).substr(0, 120) << "\n";
    return identical ? 0 : 1;
}

// Phase 7: perplexity = exp(mean negative log-likelihood) over a text file.
// Scores with one incremental pass (KV cache makes it O(n)); resets the cache
// at max_seq_len chunk boundaries (non-overlapping chunks).
int cmd_perplexity(const Args& a) {
    if (a.data_file.empty() && a.ids_file.empty()) {
        std::cerr << "error: perplexity needs --data FILE or --ids FILE\n";
        return 1;
    }
    tinyinfer::LlamaModel model;
    if (!apply_quant(a, model)) return 1;
    model.load(a.model_dir + "/model.safetensors", a.model_dir + "/config.json");
    tinyinfer::Tokenizer tok;
    tok.load(a.model_dir + "/tokenizer.json");
    const int vocab = model.config().vocab_size;
    const int max_seq = model.config().max_seq_len;

    std::vector<int> ids;
    if (!a.ids_file.empty()) {
        // Pre-tokenized ids (e.g. via Python tokenizers for long eval texts;
        // our BPE is correct but slow on megabyte inputs — see design log).
        std::ifstream f(a.ids_file);
        if (!f) {
            std::cerr << "error: cannot open " << a.ids_file << "\n";
            return 1;
        }
        int id;
        while (f >> id) ids.push_back(id);
    } else {
        std::ifstream f(a.data_file, std::ios::binary);
        if (!f) {
            std::cerr << "error: cannot open " << a.data_file << "\n";
            return 1;
        }
        std::string text((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
        ids = tok.encode(text, true); // + bos
    }
    if (a.limit > 0 && static_cast<int>(ids.size()) > a.limit)
        ids.resize(a.limit);

    std::vector<float> logits(vocab);
    double nll = 0.0;
    long scored = 0;
    auto t0 = std::chrono::steady_clock::now();
    model.reset();
    int pos = 0;
    for (int p = 0; p + 1 < static_cast<int>(ids.size()); ++p) {
        model.forward(ids[p], pos, logits.data());
        const int target = ids[p + 1];
        // log p(target) = l_t - logsumexp(l), in double precision
        float mx = logits[0];
        for (int i = 1; i < vocab; ++i)
            if (logits[i] > mx) mx = logits[i];
        double sum = 0.0;
        for (int i = 0; i < vocab; ++i) sum += std::exp((double)logits[i] - mx);
        nll -= (double)logits[target] - mx - std::log(sum);
        ++scored;
        if (++pos == max_seq - 1) {
            model.reset();
            pos = 0;
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "[tinyinfer perplexity] quant=" << a.quant << " tokens=" << scored
              << " nll/token=" << nll / scored << " ppl=" << std::exp(nll / scored)
              << " (" << secs << "s)\n";
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
        if (a.cmd == "bench") return cmd_bench(a);
        if (a.cmd == "perplexity") return cmd_perplexity(a);
        return cmd_prompt(a);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
