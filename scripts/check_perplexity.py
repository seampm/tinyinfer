#!/usr/bin/env python3
"""Verify tinyinfer's perplexity against HF transformers (fp32).

Same tokenization (same tokenizer.json), same chunking (non-overlapping
max_seq_len chunks with reset), same token limit.
"""
import math
import sys

import torch
from tokenizers import Tokenizer
from transformers import LlamaForCausalLM

MODEL_DIR = "models/tinyllama-15m"
DATA = "data/wiki.test.raw"
LIMIT = 20000

tok = Tokenizer.from_file(f"{MODEL_DIR}/tokenizer.json")
text = open(DATA, encoding="utf-8").read()
ids = [1] + tok.encode(text, add_special_tokens=False).ids  # + bos
ids = ids[:LIMIT]
print(f"tokens: {len(ids)}")

model = LlamaForCausalLM.from_pretrained(
    MODEL_DIR, torch_dtype=torch.float32, low_cpu_mem_usage=True
)
model.eval()

max_seq = model.config.max_position_embeddings
nll, scored = 0.0, 0
with torch.no_grad():
    for start in range(0, len(ids) - 1, max_seq - 1):
        chunk = ids[start : start + max_seq]
        if len(chunk) < 2:
            break
        x = torch.tensor([chunk])
        logits = model(x).logits[0]  # [len, vocab]
        # position p predicts chunk[p+1]
        lp = torch.log_softmax(logits[:-1].float(), dim=-1)
        tgt = torch.tensor(chunk[1:])
        nll += -lp[range(len(tgt)), tgt].sum().item()
        scored += len(tgt)

print(f"pytorch fp32: tokens={scored} nll/token={nll/scored:.6f} ppl={math.exp(nll/scored):.4f}")
