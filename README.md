# GPU Kernel Engineering Tutorial

A hands-on, three-part course that takes you from parallel programming fundamentals on the CPU to writing, verifying, and benchmarking real CUDA kernels on the GPU. Each lesson pairs a written guide (theory, pitfalls, further reading, and a checkpoint quiz) with small, self-contained exercise programs you compile and run yourself.

## Curriculum

| Lesson | Topic | Exercises |
|--------|-------|-----------|
| [1 — Basics of Parallel Programming](1-basics-of-parallel-programming/basics-of-parallel-programming.md) | Threads vs processes, data vs task parallelism, SIMD vs SIMT, latency vs throughput, Amdahl's & Gustafson's Laws, synchronization primitives | Multithreaded vector add, map-reduce sum, Amdahl's Law analysis |
| [2 — CUDA Fundamentals](2-cuda-fundamentals/cuda-fundamentals.md) | Grid/block/thread/warp hierarchy, GPU memory spaces, coalescing, warp divergence, occupancy, profiling with `nsys`/`ncu` | Hello kernel, GPU vector add, coalescing microbenchmark |
| [3 — Hands-On Kernels](3-hands-on-kernel/hands-on-kernel.md) | Arithmetic intensity, shared-memory tiling, tree-based reduction, warp shuffle intrinsics, GFLOPS/bandwidth math, NumPy verification | Naive & tiled matmul, parallel reduction |

## Repository layout

```
1-basics-of-parallel-programming/
  basics-of-parallel-programming.md   # lesson text
  1_vector_add_cpu/vector_add_cpu.cpp # std::thread vector add + speedup timing
  2_reduce_sum/reduce_sum.cpp         # per-thread partial sums vs mutex contention
  3_amdahl_law_analysis/amdahl.py     # Amdahl & Gustafson speedup tables

2-cuda-fundamentals/
  cuda-fundamentals.md                # lesson text
  1-hello/hello.cu                    # device printf, thread indexing
  2-vector-add-kernel/vector-add.cu   # GPU vector add with bounds guard
  3-coalesce-memory/coalesce.cu       # coalesced vs strided bandwidth benchmark

3-hands-on-kernel/
  hands-on-kernel.md                  # lesson text
  1-matmul/matmul.cu                  # naive + tiled matmul (selectable via CLI)
  1-matmul/verify_matmul.py           # NumPy reference check
  3-reduce/reduce.cu                  # grid-stride + shared-memory tree + warp shuffle reduction
```

## Prerequisites

**Lesson 1 (CPU only):**

- C++17 compiler (g++ ≥ 11 or clang ≥ 14)
- Python 3.11+

**Lessons 2–3 (GPU):**

- NVIDIA GPU with Compute Capability 7.0+, driver 535.x+
- CUDA Toolkit 12.1+ (`nvcc`)
- NumPy (for matmul verification)

```bash
# Verify your setup
g++ --version
nvidia-smi
nvcc --version
python3 -c "import numpy"
```

A convenient way to target your GPU's architecture in every `nvcc` invocation:

```bash
export ARCH=sm_$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')
```

## Building and running

### Lesson 1 — CPU parallelism

```bash
cd 1-basics-of-parallel-programming

g++ -O2 -std=c++17 -pthread 1_vector_add_cpu/vector_add_cpu.cpp -o vector_add_cpu
./vector_add_cpu 8        # expect speedup > 1.5x and "max error: 0"

g++ -O2 -std=c++17 -pthread 2_reduce_sum/reduce_sum.cpp -o reduce_sum
./reduce_sum 8            # relative error vs sequential < 1e-6

python3 3_amdahl_law_analysis/amdahl.py   # S(inf) prints 10.0
```

### Lesson 2 — CUDA fundamentals

```bash
cd 2-cuda-fundamentals

nvcc -arch=$ARCH 1-hello/hello.cu -o hello && ./hello
# 8 lines: "block bx thread tx -> global i" (order may vary)

nvcc -arch=$ARCH 2-vector-add-kernel/vector-add.cu -o vector-add && ./vector-add
# prints "max error = 0"

nvcc -arch=$ARCH 3-coalesce-memory/coalesce.cu -o coalesce && ./coalesce
# coalesced GB/s should be >3x strided GB/s
```

### Lesson 3 — Hands-on kernels

```bash
cd 3-hands-on-kernel/1-matmul
nvcc -O3 -arch=$ARCH matmul.cu -o matmul
./matmul naive            # one thread per output element; prints ms + GFLOPS
./matmul tiled            # 16x16 shared-memory tiles; typically 2-6x faster
python3 verify_matmul.py  # "PASS" (max relative error < 1e-3 vs NumPy)

cd ../3-reduce
nvcc -O3 -arch=$ARCH reduce.cu -o reduce
./reduce                  # sums 1<<26 floats; "PASS rel_err=..." and GB/s
```

The matmul exercise generates A and B deterministically (`sin(i*0.001)` / `cos(i*0.001)`) so the CUDA binary and [verify_matmul.py](3-hands-on-kernel/1-matmul/verify_matmul.py) reproduce identical inputs; the GPU result is written to `C_gpu.bin` for comparison.

## Key ideas covered

- **Amdahl's Law** — the serial fraction bounds speedup: $S(N) = \frac{1}{(1-p) + p/N}$
- **Gustafson's Law** — scaled workloads recover parallelism: $S(N) = (1-p) + pN$
- **SIMT execution** — warps of 32 threads execute in lockstep; divergence serializes branches
- **Memory coalescing** — contiguous per-warp access turns 32 transactions into a few
- **Shared-memory tiling** — reusing each tile `TILE` times raises arithmetic intensity from ~0.25 flop/byte
- **Tree reduction** — O(log N) steps, finished with `__shfl_down_sync` warp shuffles and a single `atomicAdd` per block

## Profiling

```bash
nsys profile -o run ./vector-add   # timeline of kernels and memcpys
nsys stats run.nsys-rep            # summary tables
ncu --set basic ./vector-add       # per-kernel hardware metrics
```

## Common pitfalls

- Timing kernels without `cudaDeviceSynchronize()` — kernel launches are asynchronous.
- Forgetting the `if (i < n)` guard — grids overshoot the array size.
- Using `clock()` for CPU timing — it sums CPU time across threads; use `std::chrono::steady_clock`.
- Compiling without `-O2`/`-O3` — makes performance comparisons meaningless.
- Locks in hot loops — prefer per-thread accumulators, combine at the end.

## Further reading

- *CUDA C++ Programming Guide* and *Best Practices Guide* (NVIDIA)
- *Programming Massively Parallel Processors* (Hwu, Kirk, Hajj)
- *Computer Architecture: A Quantitative Approach* (Hennessy & Patterson)
- *C++ Concurrency in Action* (Anthony Williams)
- Amdahl (1967) and Gustafson (1988) original papers
