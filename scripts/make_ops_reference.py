#!/usr/bin/env python3
"""NumPy reference values for Phase 3 op unit tests.

Prints expected outputs as C++ float literals. Paste into tests/test_ops.cpp;
keep this script as the source of truth (see DESIGN_LOG).
"""
import numpy as np

np.set_printoptions(precision=9, suppress=False)


def fmt(arr, name):
    vals = ", ".join(f"{v:.9g}f" for v in np.asarray(arr, dtype=np.float64).ravel())
    print(f"// {name}\nconst float k{name}[] = {{{vals}}};\n")


# --- matvec: W (3x4 row-major) @ x (4) ---
W = np.array([[1, 2, 3, 4], [0.5, -1, 2, 0], [1, 1, 1, 1]], dtype=np.float64)
x = np.array([1, 2, -1, 0.5], dtype=np.float64)
fmt(W @ x, "Matvec")

# --- rmsnorm: eps=1e-6 ---
xr = np.array([1.0, -2.0, 3.0, -4.0])
w = np.array([0.5, 1.0, 1.5, 2.0])
rms = xr / np.sqrt(np.mean(xr**2) + 1e-6) * w
fmt(rms, "Rmsnorm")

# --- softmax (stable) ---
xs = np.array([1001.0, 1002.0, 1003.0, 1004.0])  # large values: stability matters
e = np.exp(xs - xs.max())
fmt(e / e.sum(), "Softmax")

# --- rope: NeoX style, num_heads=2, head_dim=4, pos=3, theta=10000 ---
def rope_neox(v, pos, theta=10000.0):
    v = v.astype(np.float64)
    hd = v.shape[-1]
    half = hd // 2
    i = np.arange(half)
    ang = pos / (theta ** (2 * i / hd))
    c, s = np.cos(ang), np.sin(ang)
    out = np.empty_like(v)
    a, b = v[..., :half], v[..., half:]
    out[..., :half] = a * c - b * s
    out[..., half:] = a * s + b * c
    return out

q = np.array([[1, 2, 3, 4], [0.5, -0.5, 1.5, -1.5]])
k = np.array([[2, 0, -1, 3], [1, 1, 1, 1]])
fmt(rope_neox(q, 3), "RopeQ")
fmt(rope_neox(k, 3), "RopeK")

# --- silu ---
xsilu = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
fmt(xsily := xsilu / (1 + np.exp(-xsilu)), "Silu")

# --- vec_mul ---
a = np.array([1.5, -2.0, 0.0, 3.25])
b = np.array([2.0, 0.5, -1.0, 4.0])
fmt(a * b, "VecMul")
