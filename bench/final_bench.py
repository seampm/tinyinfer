#!/usr/bin/env python3
"""Phase 8: final benchmark. Runs each config 3x (quiet machine assumed),
takes the median, writes bench/results.json and prints a markdown table."""
import json
import os
import re
import statistics
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
LLAMA = os.path.expanduser("~/workspace/llama.cpp/build/bin")
PROMPT = "Once upon a time, there was a little girl named Lily."
N = 128
REPS = 3


def ti_bench(model, quant):
    vals = []
    for _ in range(REPS):
        cmd = [f"{REPO}/build/tinyinfer", "bench", PROMPT,
               "--max-tokens", str(N), "--skip-naive",
               "--model", f"models/{model}"]
        if quant == "i8":
            cmd += ["--quant", "i8"]
        out = subprocess.run(cmd, capture_output=True, text=True,
                             cwd=REPO).stdout.splitlines()
        # cached line: "cached  128  <ttft>  <decode>  <tps>"
        vals.append(float(out[2].split()[-1]))
    return statistics.median(vals)


def lc_bench(gguf):
    vals = []
    for _ in range(REPS):
        out = subprocess.run(
            [f"{LLAMA}/llama-bench", "-m", f"{REPO}/models/gguf/{gguf}.gguf",
             "-p", "15", "-n", str(N)],
            capture_output=True, text=True).stdout.splitlines()
        for line in out:
            cols = [c.strip() for c in line.split("|")]
            if len(cols) > 7 and cols[6].startswith("tg"):
                m = re.search(r"([\d.]+)\s*±", cols[7])
                vals.append(float(m.group(1)))
    return statistics.median(vals)


r = {
    "prompt": PROMPT, "prompt_tokens": 15, "gen_tokens": N, "reps": REPS,
    "tinyinfer_15m_f32": ti_bench("tinyllama-15m", "f32"),
    "tinyinfer_15m_i8": ti_bench("tinyllama-15m", "i8"),
    "llamacpp_15m_f32": lc_bench("tinyllama-15m-f32"),
    "tinyinfer_1.1b_f32": ti_bench("tinyllama-1.1b", "f32"),
    "llamacpp_1.1b_f32": lc_bench("tinyllama-1.1b-f32"),
}
with open(f"{HERE}/results.json", "w") as f:
    json.dump(r, f, indent=2)

print(f"| {'model':<16} | {'engine':<10} | {'prec':<5} | {'tok/s':>7} |")
print(f"| {'-'*16} | {'-'*10} | {'-'*5} | {'-'*7} |")
for label, eng, prec, key in [
    ("TinyStories-15M", "tinyinfer", "fp32", "tinyinfer_15m_f32"),
    ("TinyStories-15M", "tinyinfer", "int8", "tinyinfer_15m_i8"),
    ("TinyStories-15M", "llama.cpp", "fp32", "llamacpp_15m_f32"),
    ("TinyLlama-1.1B", "tinyinfer", "fp32", "tinyinfer_1.1b_f32"),
    ("TinyLlama-1.1B", "llama.cpp", "fp32", "llamacpp_1.1b_f32"),
]:
    print(f"| {label:<16} | {eng:<10} | {prec:<5} | {r[key]:7.1f} |")
print(f"\n15M: tinyinfer fp32 is {r['tinyinfer_15m_f32']/r['llamacpp_15m_f32']:.0%} of llama.cpp; "
      f"int8 {r['tinyinfer_15m_i8']/r['tinyinfer_15m_f32']:.2f}x over fp32")
print(f"1.1B: tinyinfer fp32 is {r['tinyinfer_1.1b_f32']/r['llamacpp_1.1b_f32']:.0%} of llama.cpp")
