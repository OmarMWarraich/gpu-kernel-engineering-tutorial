# Profiling and Optimization

## Overview

Optimization without measurement is guesswork. This lesson teaches you to profile kernels with Nsight Systems (timeline) and Nsight Compute (per-kernel metrics), interpret them via the roofline model, and apply the standard remedies: tiling, loop unrolling, and occupancy tuning.

## Learning objectives

- Capture and read `nsys` timelines and `ncu` metric reports.
- Classify a kernel as memory-bound or compute-bound using the roofline model.
- Identify the top bottlenecks from Speed-of-Light and memory-workload sections.
- Apply tiling and quantify the improvement.
- Measure the effect of loop unrolling on register pressure and occupancy.
- Know the ordered checklist of what to try for each bottleneck class.

## Prerequisites and setup

```bash
nsys --version    # from CUDA 12.1 toolkit or standalone
ncu --version     # Nsight Compute 2023.x
# ncu may need permission: add to /etc/modprobe.d:
#   options nvidia NVreg_RestrictProfilingToAdminUsers=0  (then reboot)
```
```
CUDA 12.1, driver 535.x, Nsight Systems 2023.2, Nsight Compute 2023.1, Ubuntu 22.04.
```

# Set the target architecture for all nvcc commands below
```
export ARCH=sm_$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')
```

## Concepts and theory

Roofline model. Attainable performance is bounded by:

P = min(P<sub>peak</sub>, I × BW<sub>mem</sub>)

where I is arithmetic intensity (flops/byte), peak is the device's peak compute throughput, and BW<sub>mem</sub> is the memory bandwidth. The ridge point occurs at I<sub>ridge</sub> = P<sub>peak</sub> / BW<sub>mem</sub>. Kernels below the slanted roof are memory-bound; those below the horizontal roof are compute-bound. mem stands for the memory subsystem (L2/DRAM). The roofline is a useful first-order model, but it does not capture all bottlenecks (e.g., divergence, instruction fetch, shared-memory bank conflicts).

```
GFLOPS
 peak ─────────────────────────  ← compute roof
      /
     /   ← memory roof (slope = bandwidth)
    /
   /__●_________________________ intensity (flop/byte)
      kernel below the slanted roof = memory-bound
```

**Example:** GPU with 900 GB/s and 20 TFLOPS peak → ridge point at ~22 flop/byte. Vector add (I ≈ 0.08) is hopelessly memory-bound; large tiled GEMM (I > 30) can be compute-bound.

### Key Nsight Compute sections.

| Section | Tells you |
|---------|------------|
Speed of Light | % of peak compute & memory achieved — first look
Memory Workload Analysis | transactions per request (coalescing), L2/DRAM hit rates
Occupancy | achieved vs theoretical, and the limiter (regs/smem/blocks)
Scheduler / Warp State | stall reasons (long scoreboard = memory latency)

**Occupancy vs throughput.** More resident warps hide latency, but past the point where latency is hidden, extra occupancy adds nothing — and cutting registers to raise occupancy can slow a kernel by causing spills. Optimize the metric that matters: runtime.

**Unrolling & ILP.** `#pragma unroll` removes loop overhead and lets the compiler schedule independent instructions, increasing instruction-level parallelism — at the cost of registers.

### Hands-on exercises

**Exercise 1 — Profile and diagnose**

**Task:** Profile the naive matmul from lesson 3 with Nsight Compute and identify the top 3 bottlenecks from the report.

**Input / expected behavior:** A short written diagnosis citing SOL %, coalescing stats, and stall reasons.

**Hints:**

1. `ncu --set full -o naive_report ./matmul naive` then `ncu --import naive_report.ncu-rep --page details`.
2. Look at "Memory [%]" vs "Compute (SM) [%]" first.
3. Check L2 transactions per request: >4 for 32-bit loads means poor coalescing of B accesses.

### Acceptance tests:

```bash
ncu --set full -o naive_report ./matmul naive   # report file produced
```

**Exercise 2 — Apply tiling, measure improvement**

**Task:** Re-profile the tiled version; compare SOL memory %, DRAM bytes, and runtime against naive.

**Input / expected behavior:** DRAM traffic drops ~TILE-fold for the reused operand; runtime improves 2–6×.

**Hints:**

1. `ncu --metrics dram__bytes.sum,gpu__time_duration.sum ./matmul naive` and again for tiled.
2. Compute effective flop/byte for both and place them on your GPU's roofline.

### Acceptance tests:
```
ncu --metrics dram__bytes.sum ./matmul naive
ncu --metrics dram__bytes.sum ./matmul tiled   # dram bytes significantly lower
```

**Exercise 3 — Unrolling vs register pressure**
**Task:** Add #pragma unroll 8 (then 16) to the inner loop of the tiled matmul. Record registers/thread, achieved occupancy, and runtime for each variant.

**Input / expected behavior:** A 3-row table (no unroll / 8 / 16). Registers rise with unrolling; runtime may improve then regress if occupancy drops or spills appear.

**Hints:**

1. `nvcc -Xptxas -v` prints registers per thread at compile time.
2. `ncu --metrics launch__registers_per_thread,sm__warps_active.avg.pct_of_peak_sustained_active ./matmul tiled`.
3. Look for "spill stores" in the `-Xptxas -v` output — spills are a red flag.

### Acceptance tests:

```bash
nvcc -O3 -arch=$ARCH -Xptxas -v matmul.cu -o matmul 2>&1 | grep -E "registers|spill"
```


### Performance checklist / common pitfalls

If memory-bound, try in order:

- [] Fix coalescing (sectors/request ≤ 4 for 32-bit loads).
- [] Add reuse: shared-memory tiling, register blocking.
- [] Use vectorized loads (float4) and read-only cache (__ldg/const __restrict__).
- [] Fuse kernels to avoid round-trips through DRAM.

If compute-bound, try in order:

- [] Use faster math (FMA, -use_fast_math where tolerable, tensor cores).
- [] Increase ILP: unroll, multiple accumulators.
- [] Reduce divergence and redundant computation.
- Pitfalls: profiling debug builds; trusting occupancy over runtime; measuring a single launch (include warm-up); ncu serializes and replays kernels — use nsys for realistic end-to-end timing.

### Further reading & resources
- Nsight Compute Documentation & "Kernel Profiling Guide" (NVIDIA) — metric definitions.
- Nsight Systems User Guide — timeline analysis workflows.
- Williams, Waterman, Patterson — "Roofline: An Insightful Visual Performance Model" (CACM 2009).
- GTC talks: "CUDA Performance Optimization" series — practical, updated yearly.
- CUDA C++ Best Practices Guide — the optimization priority order chapter.


### Checkpoint quiz

1. A kernel at 95% memory SOL and 20% compute SOL is: memory- or compute-bound?
2. Roofline ridge point formula?
3. "Stall long scoreboard" indicates waiting on: (a) __syncthreads (b) global memory loads (c) instruction fetch.
4. Name two costs of aggressive loop unrolling.
5. Which tool gives per-kernel hardware counters: nsys or ncu?

**Answers:**
 
1. Memory-bound.

2. P = min(P<sub>peak</sub>, I × BW<sub>mem</sub>)
3. (b). 
4. Higher register pressure (lower occupancy) and possible register spills to local memory. 
5. ncu (Nsight Compute); nsys is the system-level timeline.