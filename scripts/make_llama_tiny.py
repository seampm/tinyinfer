#!/usr/bin/env python3
"""Phase 4 fixture: tiny random Llama + reference outputs for the C++ forward test.

Builds a small LlamaForCausalLM (fixed seed), saves weights (safetensors) and
config.json, then runs a fixed token-id prompt through it and exports every
layer's output hidden state plus logits as raw float32 .bin files with a
manifest.json. The C++ test (tests/test_forward.cpp) loads the same files and
must match within 1e-4 per layer.

No tokenizer needed — token ids are fed directly.
"""
import json
import os

import torch
from transformers import LlamaConfig, LlamaForCausalLM

OUT = os.path.join(os.path.dirname(__file__), "..", "tests", "test_data", "llama_tiny")
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(0)
cfg = LlamaConfig(
    vocab_size=64,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    rms_norm_eps=1e-6,
    rope_theta=10000.0,
    max_position_embeddings=128,
    tie_word_embeddings=False,
)
model = LlamaForCausalLM(cfg)
model.save_pretrained(OUT)  # config.json + model.safetensors
model.eval()

input_ids = [1, 7, 13, 42]
inputs = torch.tensor([input_ids])

layer_outs = {}
hooks = []


def make_hook(name):
    def hook(module, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        layer_outs[name] = t.detach().float().cpu().squeeze(0)  # [seq, dim]
    return hook


for i, layer in enumerate(model.model.layers):
    hooks.append(layer.register_forward_hook(make_hook(f"layer_{i}")))

with torch.no_grad():
    logits = model(inputs).logits.squeeze(0).float().cpu()  # [seq, vocab]

for h in hooks:
    h.remove()

manifest = {
    "input_ids": input_ids,
    "tolerance": 1e-4,
    "tensors": {},
}

for name, t in layer_outs.items():
    path = f"{name}.bin"
    t.numpy().astype("<f4").tofile(os.path.join(OUT, path))
    manifest["tensors"][name] = {"shape": list(t.shape), "file": path}

logits.numpy().astype("<f4").tofile(os.path.join(OUT, "logits.bin"))
manifest["tensors"]["logits"] = {"shape": list(logits.shape), "file": "logits.bin"}

with open(os.path.join(OUT, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

total = sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT))
print(f"Wrote fixture to {OUT}/ ({total} bytes)")
print(f"  layers: {len(layer_outs)}, seq: {len(input_ids)}, "
      f"logits shape: {list(logits.shape)}")
