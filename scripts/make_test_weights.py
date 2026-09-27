#!/usr/bin/env python3
"""Generate a tiny safetensors fixture for the C++ loader unit tests.

Values are hardcoded in tests/test_loader.cpp — keep them in sync.
Writes to tests/test_data/tiny.safetensors (a few KB; the .gitignore
exception for tests/test_data/ allows committing it).
"""
import os
import torch
from safetensors.torch import save_file

out = os.path.join(os.path.dirname(__file__), "..", "tests", "test_data", "tiny.safetensors")
os.makedirs(os.path.dirname(out), exist_ok=True)

tensors = {
    "weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),  # 0..5
    "bias": torch.tensor([0.5, -1.5, 2.25], dtype=torch.float32),
    "scalar": torch.tensor(3.14, dtype=torch.float32),  # shape []
    "ids": torch.tensor([1, 2, 3, 4], dtype=torch.int32),
}
save_file(tensors, out, metadata={"format": "tinyinfer-test"})
print(f"Wrote {out} ({os.path.getsize(out)} bytes)")
for k, v in tensors.items():
    print(f"  {k}: dtype={v.dtype} shape={list(v.shape)}")
