#!/usr/bin/env python3
"""Generate a minimal BPE tokenizer.json for the C++ tokenizer unit tests.

Small enough to eyeball; exercises merges-by-rank, byte fallback, added
special tokens, and round-tripping. Expected values are hardcoded in
tests/test_tokenizer.cpp — keep in sync.
"""
import json
import os

out = os.path.join(os.path.dirname(__file__), "..", "tests", "test_data",
                   "mini_tokenizer.json")
out_raw = os.path.join(os.path.dirname(__file__), "..", "tests", "test_data",
                       "mini_tokenizer_raw.json")

vocab = {
    "<unk>": 0, "<s>": 1, "</s>": 2,
    "a": 3, "b": 4, "c": 5,
    "ab": 6, "bc": 7, "abc": 8,
    "▁": 9, "▁a": 10, "▁ab": 11,
    "<0xC3>": 12, "<0xA9>": 13,
}
# ranks: "a b" < "ab c" < "▁ a" < "b c"
merges = ["a b", "ab c", "▁ a", "b c"]

tok = {
    "version": "1.0",
    "truncation": None,
    "padding": None,
    "added_tokens": [
        {"id": 0, "content": "<unk>", "special": True, "single_word": False,
         "lstrip": False, "rstrip": False, "normalized": True},
        {"id": 1, "content": "<s>", "special": True, "single_word": False,
         "lstrip": False, "rstrip": False, "normalized": True},
        {"id": 2, "content": "</s>", "special": True, "single_word": False,
         "lstrip": False, "rstrip": False, "normalized": True},
    ],
    "normalizer": {"type": "Sequence", "normalizers": [
        {"type": "Prepend", "prepend": "▁"},
        {"type": "Replace", "pattern": {"String": " "}, "content": "▁"},
    ]},
    "pre_tokenizer": None,
    "post_processor": None,
    "decoder": {"type": "Sequence", "decoders": [
        {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
        {"type": "ByteFallback"},
        {"type": "Fuse"},
        {"type": "Strip", "content": " ", "start": 1, "stop": 0},
    ]},
    "model": {
        "type": "BPE",
        "dropout": None,
        "unk_token": "<unk>",
        "continuing_subword_prefix": None,
        "end_of_word_suffix": None,
        "fuse_unk": True,
        "byte_fallback": True,
        "vocab": vocab,
        "merges": merges,
    },
}

with open(out, "w") as f:
    json.dump(tok, f, ensure_ascii=False, indent=1)
print(f"Wrote {out}")

# Variant with normalized=false added tokens (matched on raw text, each piece
# normalized independently). Exercises the other encode path.
import copy
tok_raw = copy.deepcopy(tok)
for a in tok_raw["added_tokens"]:
    a["normalized"] = False
with open(out_raw, "w") as f:
    json.dump(tok_raw, f, ensure_ascii=False, indent=1)
print(f"Wrote {out_raw}")

# Sanity: show what the Python `tokenizers` lib produces for our test strings.
from tokenizers import Tokenizer
t = Tokenizer.from_file(out)
for s in ["abc", "ab", "a<s>bc", "<s>bc", "<s> bc", "é", "a b"]:
    enc = t.encode(s, add_special_tokens=False)
    print(f"  normalized  {s!r} -> {enc.ids} -> {t.decode(enc.ids, skip_special_tokens=True)!r}")
tr = Tokenizer.from_file(out_raw)
for s in ["a<s>bc", "<s>bc"]:
    enc = tr.encode(s, add_special_tokens=False)
    print(f"  raw         {s!r} -> {enc.ids} -> {tr.decode(enc.ids, skip_special_tokens=True)!r}")
