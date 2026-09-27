#!/usr/bin/env python3
"""Verify C++ generation matches PyTorch greedy decoding exactly.

Runs the HF model (F32, same weights the C++ engine loads) greedily for a few
tokens and compares against `tinyinfer prompt --temperature 0` output text.
"""
import subprocess
import sys

import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer
from transformers import LlamaForCausalLM

MODEL_DIR = "models/tinyllama-15m"
PROMPT = "Once upon a time"
N = 8

tok = Tokenizer.from_file(f"{MODEL_DIR}/tokenizer.json")
ids = [1] + tok.encode(PROMPT, add_special_tokens=False).ids  # + bos

model = LlamaForCausalLM.from_pretrained(
    MODEL_DIR, torch_dtype=torch.float32, low_cpu_mem_usage=True
)
model.eval()

gen = list(ids)
with torch.no_grad():
    for _ in range(N):
        x = torch.tensor([gen])
        logits = model(x).logits[0, -1]
        gen.append(int(logits.argmax()))

want_ids = gen
print("pytorch greedy ids:", want_ids[-N:])

p = subprocess.run(
    ["./build/tinyinfer", "prompt", PROMPT, "--model", MODEL_DIR,
     "--max-tokens", str(N), "--temperature", "0"],
    capture_output=True, text=True,
)
if p.returncode != 0:
    print("C++ FAILED:\n", p.stderr)
    sys.exit(1)
# stdout is prompt-echo + completion + trailing newline; stderr has tok/s line
got_text = p.stdout.rstrip("\n")
print("c++     greedy:", repr(got_text))
print("tok/s line:", p.stderr.strip().splitlines()[-1])

# Compare on token ids: re-encode the C++ output text (its prompt echo has no
# <s>, which Python's decode keeps — a decode quirk, not an encode mismatch).
got_ids = [1] + tok.encode(got_text, add_special_tokens=False).ids
print("c++     greedy ids:", got_ids[-N:])
match = got_ids == want_ids
if match:
    print("MATCH: C++ greedy generation is token-identical to PyTorch")
else:
    print("MISMATCH")
    sys.exit(1)
