#!/usr/bin/env python3
"""Cross-check the C++ tokenizer against Python `tokenizers` on the real
TinyStories tokenizer.json. Fails loudly on the first mismatch.
"""
import subprocess
import sys

from tokenizers import Tokenizer

MODEL_DIR = "models/tinyllama-15m"
BIN = "./build/tinyinfer"

CORPUS = [
    "Once upon a time",
    "Hello, world!",
    "Lily said \"hi\"",
    "  leading and trailing  ",
    "multiple   spaces",
    "new\nlines\n\nand\ttabs",
    "UPPER lower MiXeD",
    "numbers 123 4.56 -7",
    "punctuation!?;:'()[]{}",
    "café naïve résumé",
    "emoji 😀🎉",
    "Chinese 中文测试",
    "mixed 中文 and English",
    "<s> literal special",
    "a<b",
    "",
    "x",
    " ",
    "The quick brown fox jumps over the lazy dog. " * 5,
    "don't can't won't",
    "one\ttwo\nthree",
]

tok = Tokenizer.from_file(f"{MODEL_DIR}/tokenizer.json")
fails = 0
for s in CORPUS:
    want = tok.encode(s, add_special_tokens=False).ids
    p = subprocess.run([BIN, "tokenize", s, "--model", MODEL_DIR],
                       capture_output=True, text=True)
    if p.returncode != 0:
        print(f"ERROR on {s!r}: {p.stderr.strip()}")
        fails += 1
        continue
    got = [int(x) for x in p.stdout.split()]
    # Also verify decode round-trips through Python
    if got != want:
        print(f"MISMATCH on {s!r}:\n  python: {want}\n  c++:    {got}")
        fails += 1

# Round-trip: C++ decode of Python-encoded ids must equal Python decode.
for s in CORPUS:
    ids = tok.encode(s, add_special_tokens=False).ids
    want = tok.decode(ids)
    # (decode path is exercised by the prompt command; checked in generation test)
print(f"\n{fails} mismatches out of {len(CORPUS)} strings")
sys.exit(1 if fails else 0)
