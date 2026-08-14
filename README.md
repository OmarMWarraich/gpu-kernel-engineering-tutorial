# GPU Kernel Engineering Tutorial

A hands-on, five-part course that takes you from parallel programming fundamentals on the CPU to writing, verifying, profiling, and optimizing real CUDA kernels on the GPU, and finally to productive ML-focused GPU programming with Triton. Each lesson pairs a written guide (theory, pitfalls, further reading, and a checkpoint quiz) with small, self-contained exercise programs you compile and run yourself.

## Curriculum

| Lesson | Topic | Exercises | Difficulty |
|--------|-------|-----------|------------|
| [1 — Basics of Parallel Programming](1-basics-of-parallel-programming/basics-of-parallel-programming.md) | Threads vs processes, data vs task parallelism, SIMD vs SIMT, latency vs throughput, Amdahl's & Gustafson's Laws, synchronization primitives | Multithreaded vector add, map-reduce sum, Amdahl's Law analysis | ★★☆☆ |
| [2 — CUDA Fundamentals](2-cuda-fundamentals/cuda-fundamentals.md) | Grid/block/thread/warp hierarchy, GPU memory spaces, coalescing, warp divergence, occupancy, profiling with `nsys`/`ncu` | Hello kernel, GPU vector add, coalescing microbenchmark | ★★★☆ |
| [3 — Hands-On Kernels](3-hands-on-kernel/hands-on-kernel.md) | Arithmetic intensity, shared-memory tiling, tree-based reduction, warp shuffle intrinsics, GFLOPS/bandwidth math, NumPy verification | Naive & tiled matmul, parallel reduction | ★★★☆ |
| [4 — Profiling and Optimization](4-profiling-and-optimization/profiling-and-optimization.md) | Nsight Systems/Compute workflows, roofline model, Speed-of-Light analysis, memory- vs compute-bound diagnosis, tiling payoff, unrolling vs register pressure, occupancy tuning | Profile & diagnose naive matmul, tiled DRAM comparison, unroll vs register pressure | ★★★★ |
| [5 — Triton for ML Ops](5-triton-for-ml-ops/triton-for-ml-ops.md) | Triton programming model, tile-based scheduling, autotuning, kernel fusion for elementwise and GEMM-like layers, memory-bandwidth vs compute trade-offs in PyTorch workflows | Elementwise add in Triton, small GEMM task, fused ReLU/activation payoff | ★★★★★ |

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

4-profiling-and-optimization/
  profiling-and-optimization.md       # lesson text
  1-profile-and-diagnose/
    naive_report.ncu-rep              # Nsight Compute full report of the naive matmul

5-triton-for-ml-ops/
  triton-for-ml-ops.md                # lesson text
  1-elementwise-add-in-triton/        # Triton vector add
  2-small-gemm-triton-task/           # Triton matmul task
  3-fused-payoff/                     # fused activation + elementwise payoff
```

## Prerequisites

**Lesson 1 (CPU only):**

- C++17 compiler (g++ ≥ 11 or clang ≥ 14)
- Python 3.11+

**Lessons 2–5 (GPU):**

- NVIDIA GPU with Compute Capability 7.0+, driver 535.x+
- CUDA Toolkit 12.1+ (`nvcc`)
- NumPy (for matmul verification)
- Nsight Systems (`nsys`) and Nsight Compute (`ncu`) — lesson 4
- PyTorch and Triton — lesson 5

```bash
# Verify your setup
g++ --version
nvidia-smi
nvcc --version
python3 -c "import numpy"
python3 -c "import torch, triton"   # lesson 5
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

### Lesson 4 — Profiling and optimization

Uses the matmul binary from lesson 3.

```bash
cd 4-profiling-and-optimization/1-profile-and-diagnose

# Exercise 1: full profile of the naive matmul, then read the details page
ncu --set full -o naive_report ../../3-hands-on-kernel/1-matmul/matmul naive
ncu --import naive_report.ncu-rep --page details

# Exercise 2: compare DRAM traffic and runtime, naive vs tiled
ncu --metrics dram__bytes.sum,gpu__time_duration.sum ../../3-hands-on-kernel/1-matmul/matmul naive
ncu --metrics dram__bytes.sum,gpu__time_duration.sum ../../3-hands-on-kernel/1-matmul/matmul tiled

# Exercise 3: registers/spills when adding #pragma unroll to the tiled kernel
nvcc -O3 -arch=$ARCH -Xptxas -v ../../3-hands-on-kernel/1-matmul/matmul.cu -o matmul 2>&1 | grep -E "registers|spill"
```

### Lesson 5 — Triton for ML Ops

```bash
cd 5-triton-for-ml-ops

python3 1-elementwise-add-in-triton/*.py        # compare PyTorch vs Triton vector add
python3 2-small-gemm-triton-task/*.py           # Triton matmul exercise
python3 3-fused-payoff/fused_relu.py            # fused kernel vs separate ops
```

> `ncu` may need profiling permission: set `options nvidia NVreg_RestrictProfilingToAdminUsers=0` in `/etc/modprobe.d/` and reboot.
>
> Note: at small `N` the whole working set fits in L2, so `dram__bytes.sum` barely differs between naive and tiled — use `N ≥ 4096` to see the ~TILE-fold DRAM reduction.

The matmul exercise generates A and B deterministically (`sin(i*0.001)` / `cos(i*0.001)`) so the CUDA binary and [verify_matmul.py](3-hands-on-kernel/1-matmul/verify_matmul.py) reproduce identical inputs; the GPU result is written to `C_gpu.bin` for comparison.

## Key ideas covered

- **Amdahl's Law** — the serial fraction bounds speedup: $S(N) = \frac{1}{(1-p) + p/N}$
- **Gustafson's Law** — scaled workloads recover parallelism: $S(N) = (1-p) + pN$
- **SIMT execution** — warps of 32 threads execute in lockstep; divergence serializes branches
- **Memory coalescing** — contiguous per-warp access turns 32 transactions into a few
- **Shared-memory tiling** — reusing each tile `TILE` times raises arithmetic intensity from ~0.25 flop/byte
- **Tree reduction** — O(log N) steps, finished with `__shfl_down_sync` warp shuffles and a single `atomicAdd` per block
- **Roofline model** — attainable performance is bounded by $P = \min(P_{peak}, I \times BW_{mem})$; below the ridge point ($I_{ridge} = P_{peak}/BW_{mem}$) a kernel is memory-bound
- **Speed-of-Light analysis** — compare memory % vs compute % of peak first to classify the bottleneck
- **Occupancy vs runtime** — extra warps beyond latency-hiding add nothing; register spills from chasing occupancy can hurt
- **Unrolling & ILP** — `#pragma unroll` boosts instruction-level parallelism at the cost of register pressure
- **Triton tile programming** — express kernels as block-level operations over tiles and let the compiler handle thread/warp details
- **Kernel fusion** — combine elementwise, activation, and reduction ops into one kernel to avoid round-trips through DRAM
- **Autotuning** — search over tile sizes and pipeline stages to maximize occupancy and bandwidth on a target GPU

## Profiling

Covered in depth in [Lesson 4](4-profiling-and-optimization/profiling-and-optimization.md) and reused in [Lesson 5](5-triton-for-ml-ops/triton-for-ml-ops.md). Quick reference:

```bash
nsys profile -o run ./vector-add   # timeline of kernels and memcpys
nsys stats run.nsys-rep            # summary tables
ncu --set basic ./vector-add       # per-kernel hardware metrics
ncu --set full -o report ./matmul naive   # full report saved to report.ncu-rep
ncu --import report.ncu-rep --page details
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
