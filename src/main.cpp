#include <iostream>
#include <string>

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " --model <name> --prompt <text> --tokens <n>\n"
              << "  Phase 0 scaffold — full CLI lands in Phase 5.\n";
}

int main(int argc, char** argv) {
    std::string model = "tinyllama";
    std::string prompt = "Once upon a time";
    int tokens = 200;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) model = argv[++i];
        else if (a == "--prompt" && i + 1 < argc) prompt = argv[++i];
        else if (a == "--tokens" && i + 1 < argc) tokens = std::stoi(argv[++i]);
        else if (a == "--help" || a == "-h") { print_usage(argv[0]); return 0; }
    }

    std::cout << "[tinyinfer] Phase 0 scaffold\n"
              << "  model: " << model << "\n"
              << "  prompt: " << prompt << "\n"
              << "  tokens: " << tokens << "\n"
              << "  TODO: implement loader (Phase 2), ops (Phase 3), forward (Phase 4),\n"
              << "        tokenizer/sampler (Phase 5), KV-cache (Phase 6).\n";
    return 0;
}
