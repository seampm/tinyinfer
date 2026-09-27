"""
Phase 1: Python reference for Llama forward pass.

Runs a fixed prompt through Hugging Face transformers and saves per-layer
outputs to disk. This is the answer key for the C++ engine:
when C++ output disagrees, diff layer-by-layer to find the bug.

Usage:
    pip install -r scripts/requirements.txt
    python scripts/reference.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --prompt "Once upon a time" --out reference_outputs/

Quick validation without downloading 2GB (tiny random Llama):
    python scripts/reference.py --model hf-internal-testing/tiny-random-LlamaForCausalLM \
        --prompt "Once upon a time" --out /tmp/ref_test
"""
import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--out", default="reference_outputs")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.float32, low_cpu_mem_usage=True
        )
    except TypeError:
        # older transformers: dtype was torch_dtype
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float32, low_cpu_mem_usage=True
        )
    model.eval()

    inputs = tok(args.prompt, return_tensors="pt")
    layer_outputs = {}

    hooks = []

    def make_hook(name):
        def hook(module, inp, out):
            # out may be tuple; take first tensor
            t = out[0] if isinstance(out, tuple) else out
            layer_outputs[name] = t.detach().float().cpu()
        return hook

    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.register_forward_hook(make_hook(f"layer_{i}")))

    with torch.no_grad():
        outputs = model(**inputs)

    for h in hooks:
        h.remove()

    # Save logits + per-layer outputs
    torch.save(outputs.logits.cpu(), os.path.join(args.out, "logits.pt"))
    for name, t in layer_outputs.items():
        torch.save(t, os.path.join(args.out, f"{name}.pt"))

    # Save input ids for the C++ side
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(
            {
                "model": args.model,
                "prompt": args.prompt,
                "input_ids": inputs["input_ids"][0].tolist(),
                "tolerance": 1e-4,
            },
            f,
            indent=2,
        )

    print(f"Saved {len(layer_outputs)} layer outputs + logits to {args.out}/")
    print("C++ check: each layer should match within ~1e-4.")


if __name__ == "__main__":
    main()
