# Triton for ML Ops

## Overview

Triton lets you write GPU kernels in Python at the *block* level: you program what a block of the tensor does, and the compiler handles threads, coalescing, and shared memory. This lesson covers the Triton programming model, an elementwise kernel, a small GEMM, and guidance on when to use Triton vs hand-written CUDA.

## Learning objectives

- Explain Triton's block-level programming model and how it differs from CUDA's thread-level model.
- Write a Triton elementwise kernel with masking and correct grid launch.
- Write a blocked GEMM in Triton using `tl.dot` and accumulators.
- Benchmark Triton kernels against PyTorch built-ins fairly (warm-up, synchronization).
- Decide when to prototype in Triton vs port to CUDA.

## Prerequisites and setup

```bash
pip install torch==2.2.0 triton==2.2.0
python -c "import triton, torch; print(triton.__version__, torch.cuda.is_available())"
```

## Environment
```CUDA 12.1, driver 535.x, Python 3.11, PyTorch 2.2.0, Triton 2.2.0 (bundled with PyTorch 2.2 wheels).```    

### Concepts and theory

**CUDA vs Triton mental model.**

| CUDA | Triton |
| Unit you program | one thread | one program (≈ one block) |
| Data | scalars per thread | block-sized tensors (tl.load a vector of BLOCK elems) |
| Shared memory / coalescing | manual | compiler-managed |
| Language | C++ | Python (@triton.jit) |


A Triton kernel instance ("program") is identified by `tl.program_id(axis)`; it loads a block of offsets, computes on them vectorized, and stores with a mask for edge blocks:

```
grid = (cdiv(N, BLOCK),)
pid 0: offsets [0..BLOCK)      ┐
pid 1: offsets [BLOCK..2B)     ├─ mask = offsets < N handles the ragged tail
pid k: ...                     ┘
```

**GEMM in Triton.** Each program computes a `BLOCK_M × BLOCK_N` output tile, looping over K in `BLOCK_K` chunks with `tl.dot(a, b, acc)` — the same tiling idea as lesson 3's shared-memory matmul, expressed in ~30 lines. `@triton.autotune` searches block-size configs automatically.

`When Triton vs CUDA.` Prototype in Triton for fused elementwise ops, softmax/norm variants, attention-style kernels — you get 80–100% of hand-tuned performance in a fraction of the time. Drop to CUDA when you need warp-level control, exotic data layouts, cross-block cooperation, or targets/features Triton doesn't expose.

### Hands-on exercises

**Exercise 1 — Elementwise add in Triton**

**Task:** Implement `c = a + b` for arbitrary-length float32 tensors; verify against `a + b` in PyTorch and compare runtime with the CUDA version from lesson 2.

**Input / expected behavior:** `torch.allclose` passes; bandwidth within ~10% of `torch.add`.

**Hints:**

1. `offsets = pid * BLOCK + tl.arange(0, BLOCK)`; mask with `offsets < n`.
2. Grid: `lambda meta: (triton.cdiv(n, meta['BLOCK']),)`.
3. BLOCK must be a power of 2 (`tl.arange` requirement).

**Acceptance tests:**
```
python add_triton.py    # prints "PASS" and GB/s for triton vs torch
```


**Exercise 2 — Small GEMM in Triton**
**Task:** Blocked FP16-input/FP32-accumulate matmul for M=N=K in {256, 512, 1024}; verify vs `torch.matmul` (rtol 1e-2 for fp16) and benchmark both.

**Input / expected behavior:** Correctness passes; Triton within ~0.5–1× of cuBLAS at these sizes (cuBLAS usually wins — that's expected).

**Hints:**

1. Per program: 2D tile id from `program_id(0)`; loop `for k in range(0, K, BLOCK_K)`.
2. Build pointer blocks from strides: `a_ptrs = A + offs_m[:,None]*stride_am + offs_k[None,:]*stride_ak`.
3. Accumulate in `tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)`; `tl.dot` maps to tensor cores for fp16 inputs.
4. Use `triton.testing.do_bench` for robust timing.

**Acceptance tests:**

```
python gemm_triton.py   # "PASS" for all sizes + TFLOPS table
```

**Exercise 3 — Fusion payoff (stretch)**
**Task:** Fuse `y = relu(x + b)` into one Triton kernel; compare against the two-kernel PyTorch sequence.

**Input / expected behavior:** Fused kernel faster (one DRAM round-trip instead of two).

**Hints:**

1. Reuse Exercise 1's structure; apply `tl.maximum(v, 0.0)` before the store.
2. Benchmark with large N (≥ 2^24) so launch overhead is negligible.

**Acceptance tests:**

```
python fused_relu.py    # fused time < unfused time, allclose PASS
```

### Performance checklist / common pitfalls
- [] Always mask loads/stores for ragged tails.
- [] `BLOCK` sizes are powers of 2; `tl.dot` needs blocks ≥ 16 per dim.
- [] Benchmark with `triton.testing.do_bench` (handles warm-up + sync).
- [] Pass strides, don't assume contiguity; call `.contiguous()` at the boundary if needed.
- Pitfall: first call includes JIT compilation — never time it.
- Pitfall: comparing fp16 results with fp32 tolerances (use rtol ≈ 1e-2).
- Pitfall: expecting to beat cuBLAS on plain large GEMM — Triton's win is fused/custom ops.

### Notes: prototype in Triton, port to CUDA when...
- You need warp-level primitives, inline PTX, or fine shared-memory control.
- The op is a plain GEMM/conv already served by cuBLAS/cuDNN — don't rewrite it.
- Triton's compiler falls short for your shape/layout and you've verified via profiling.
- Otherwise stay in Triton: iteration speed and autotuning usually dominate.

### Further reading & resources

- `Triton` official tutorials (triton-lang.org) — vector add, softmax, matmul walkthroughs.
- Tillet et al., "`Triton`: An Intermediate Language and Compiler for Tiled Neural Network Computations" (MAPL 2019).
- OpenAI "`Triton`" announcement blog post — motivation and design.
- FlashAttention (Dao et al.) — the flagship fused-kernel case study; `Triton` implementations exist.
- PyTorch `torch.compile` / TorchInductor docs — how Triton is generated automatically.

### Checkpoint quiz

1. In Triton you program one ___ per grid element, not one thread.
2. What does the mask argument to tl.load/tl.store handle?
3. tl.dot on fp16 inputs typically uses: (a) CUDA cores (b) tensor cores (c) CPU fallback.
4. Why is the first invocation of a Triton kernel slow?
5. Name two situations where hand-written CUDA beats Triton.

**Answers:**
1. Program (block). 
2. Out-of-bounds elements in edge blocks (ragged tails). 
3. (b). 
4. JIT compilation happens on first call. 
5. Warp-level/inline-PTX control; exotic layouts or cross-block cooperation Triton doesn't expose (also: plain GEMM is best left to cuBLAS/cuDNN).