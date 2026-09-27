#!/usr/bin/env python3
"""Phase 8: benchmark charts from bench/results.json -> bench/charts/."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "charts")
os.makedirs(OUT, exist_ok=True)
r = json.load(open(os.path.join(HERE, "results.json")))

plt.rcParams.update({"font.size": 11})

# 1. Head-to-head decode throughput
labels = ["tinyinfer\nfp32", "tinyinfer\nint8", "llama.cpp\nfp32"]
v15 = [r["tinyinfer_15m_f32"], r["tinyinfer_15m_i8"], r["llamacpp_15m_f32"]]
v11 = [r["tinyinfer_1.1b_f32"], float("nan"), r["llamacpp_1.1b_f32"]]
x = range(len(labels))
w = 0.35
fig, ax = plt.subplots(figsize=(8, 4.5))
b1 = ax.bar([i - w/2 for i in x], v15, w, label="TinyStories-15M")
b2 = ax.bar([i + w/2 for i in x], v11, w, label="TinyLlama-1.1B")
for b in list(b1) + list(b2):
    h = b.get_height()
    if h == h:  # not nan
        ax.text(b.get_x() + b.get_width()/2, h, f"{h:.0f}", ha="center", va="bottom", fontsize=10)
ax.set_xticks(list(x)); ax.set_xticklabels(labels)
ax.set_ylabel("decode tok/s (greedy, 128 tokens)")
ax.set_title("tinyinfer vs llama.cpp — decode throughput")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "throughput.png"), dpi=120)
print("wrote throughput.png")

# 2. tinyinfer's own optimization journey on 15M (from DESIGN_LOG numbers)
stages = ["scalar\nfp32", "+ AVX2\nfp32", "+ int8"]
vals = [67.0, r["tinyinfer_15m_f32"], r["tinyinfer_15m_i8"]]
fig, ax = plt.subplots(figsize=(7, 4))
b = ax.bar(stages, vals, color=["#888888", "#4c78a8", "#f58518"])
for bar, v in zip(b, vals):
    ax.text(bar.get_x() + bar.get_width()/2, v, f"{v:.0f}", ha="center", va="bottom")
ax.set_ylabel("decode tok/s")
ax.set_title("tinyinfer 15M: optimization journey (Phase 6 -> 7)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "journey.png"), dpi=120)
print("wrote journey.png")
