# Hands-On Kernels

## Overview

You will implement the canonical GPU kernels — vector add, naive and tiled matrix multiply, and parallel reduction — verifying correctness against NumPy and measuring GFLOPS. These kernels teach the two most important optimization ideas: data reuse via shared memory and tree-structured communication.

## Learning objectives

- Implement and verify a naive matrix multiply kernel.
- Implement a tiled (shared-memory) matmul and explain why it is faster.
- Implement a tree-based parallel reduction with warp shuffle intrinsics.
- Compute GFLOPS from problem size and runtime.
- Validate GPU results against NumPy with appropriate float tolerances.
- Reason about tile size, block dimensions, and shared-memory usage.

## Prerequisites and setup

```bash
nvcc --version                 # CUDA 12.1
python3 -c "import numpy"      # NumPy for verification

# Set the target architecture for all nvcc commands below
export ARCH=sm_$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')
```

### Concepts and theory

Arithmetic intensity. Naive matmul reads 2N floats per output element for 2N flops → intensity ~0.25 flop/byte: memory-bound. Tiling loads each `TILE×TILE` chunk once into shared memory and reuses it `TILE` times, raising intensity by a factor of `TILE`.

```
Tiled matmul (TILE=16):
      B tile (16x16) ─┐
A tile (16x16) ───────┼──► each thread accumulates C[row][col]
   loaded once,       │    over K/16 tile iterations
   reused 16 times ───┘    __syncthreads() between load & compute
```


**Parallel reduction.** Summing N values with one thread is O(N); a tree does it in O(log N) steps:

```
step: [a b c d e f g h]
  1:  [a+e b+f c+g d+h . . . .]
  2:  [ae+cg bf+dh . . ]
  3:  [total . ]
```


The last 32 elements can be reduced without shared memory using warp shuffles (__shfl_down_sync), which exchange registers within a warp.


**GFLOPS.** For an N×N matmul: GFLOPS = 2N<sup>3</sup> / (t sec ⋅ 10<sup>9</sup>). Example: N = 1024, t = 3 ms → GFLOPS = 2⋅1024<sup>3</sup> / (0.003⋅10<sup>9</sup>) ≈ 716 GFLOPS.

**Hands-on exercises**

**Exercise 1 — Naive matrix multiply**
**Task:** One thread per output element, `C[i][j] = Σ A[i][k]·B[k][j]`, N = 1024 square float32. Verify against a NumPy reference (write matrices to .npy-compatible raw files or generate deterministically on both sides).

**Input / expected behavior:** Max relative error < 1e-3 vs CPU; print GFLOPS.

**Hints:**

1. 2D blocks: `dim3 block(16,16)`, grid = `ceil(N/16)` in each dim.
2. `row = blockIdx.y*16 + threadIdx.y`, `col = blockIdx.x*16 + threadIdx.x`.
3. Generate A, B deterministically (e.g., `a[i]=sin(i)`) so Python can reproduce them.

**Acceptance tests:**

```bash
nvcc -O3 -arch=$ARCH matmul.cu -o matmul && ./matmul naive
python3 verify_matmul.py    # "PASS"
```

**Exercise 2 — Tiled matrix multiply**
**Task:** Add a shared-memory tiled version (TILE=16 or 32) to the same binary; verify and compare GFLOPS to naive.

**Input / expected behavior:** Same numerical result; typically 2–6× faster than naive.

**Hints:**

1. `__shared__ float As[TILE][TILE], Bs[TILE][TILE];`
2. Loop over `K/TILE` phases: load one tile of A and B, `__syncthreads()`, multiply-accumulate, `__syncthreads()` again before the next load.
3. Guard loads when N is not a multiple of TILE (or require it, and assert).

**Acceptance tests:**

```bash
./matmul tiled    # error <1e-3, prints GFLOPS > naive
```
**Exercise 3 — Parallel reduction with warp intrinsics**
**Task:** Sum 1<<26 floats: block-level shared-memory tree, then warp-shuffle finish, then atomicAdd of per-block results. Verify vs CPU double-precision sum.

**Input / expected behavior:** Relative error < 1e-5; report GB/s.

**Hints:**

1. Each thread first grid-stride-loops to accumulate several elements into a register.
2. Shared-memory tree down to 32 elements, then `__shfl_down_sync(0xffffffff, v, offset)` for offsets 16,8,4,2,1.
3. Only lane 0 of warp 0 does the `atomicAdd`.

**Acceptance tests:**

```bash
nvcc -O3 -arch=$ARCH reduce.cu -o reduce && ./reduce   # "PASS rel_err=..."
```

### Performance checklist / common pitfalls
- [] Both __syncthreads() calls present in the tile loop (before and after compute).
- [] Shared-memory loads coalesced along threadIdx.x.
- [] Warm-up launch before timing; average over ≥10 iterations.
- [] Float32 verification uses relative, not absolute, tolerance.
- Pitfall: missing the second __syncthreads() — a fast thread overwrites a tile others still read (intermittent wrong results).
- Pitfall: shared-memory bank conflicts with TILE=32 column access; pad arrays ([TILE][TILE+1]) if profiling shows conflicts.
- Pitfall: single-precision reduction of large N accumulates rounding error — compare against a double-precision reference.

### Further reading & resources
- NVIDIA "Optimizing Parallel Reduction in CUDA" (Mark Harris) — the classic 7-step reduction deck.
- NVIDIA CUDA C++ Best Practices Guide — shared memory & tiling chapter.
- Programming Massively Parallel Processors — matmul tiling walkthrough.
- CUTLASS (GitHub: NVIDIA/cutlass) — production-grade GEMM building blocks to study next.
- Simon Boehm, "How to Optimize a CUDA Matmul Kernel" — step-by-step to near-cuBLAS performance.

### Checkpoint quiz

1. Why does tiling speed up matmul? (short answer)
2. GFLOPS formula for square matmul of size N?
3. What does __shfl_down_sync operate on? (a) shared memory (b) registers within a warp (c) global memory.
4. Why two __syncthreads() per tile iteration?
5. A naive matmul kernel is limited primarily by: (a) flops (b) global memory bandwidth (c) kernel launch overhead.

#### Answers:

1. Each global-memory tile is loaded once into shared memory and reused TILE times, raising arithmetic intensity. 
2. GFLOPS = 2 * N^3 / (time_in_seconds * 1e9)
3. (b). 
4. First ensures tiles are fully loaded before compute; second prevents overwriting tiles still being read.
5. (b).

