# CUDA Fundamentals

This lesson introduces the CUDA programming model: how work is organized into grids, blocks, threads, and warps; the GPU memory hierarchy; and synchronization. You will write, compile, and run your first kernels and observe memory coalescing effects with a microbenchmark.

## Learning Objectives

- Describe the grid/block/thread/warp hierarchy and compute global thread indices.
- Enumerate CUDA memory spaces (global, shared, local/register, constant, texture) and their tradeoffs.
- Write, compile (`nvcc`), and launch a kernel with correct grid/block sizing.
- Use `__syncthreads()` and atomics correctly.
- Explain occupancy, warp divergence, memory coalescing, and register pressure.
- Capture a basic profile with `nsys` and `nvprof`.

## Prerequisites and setup

- NVIDIA GPU with CUDA Compute Capability 7.0 or higher, driver version 535.x+, CUDA Toolkit 12.1+.


```bash
# Verify
nvidia-smi
nvcc --version

# Set the target architecture for all nvcc commands below
export ARCH=sm_$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')
```

### Concepts and theory

Execution hierarchy.

```
Grid --> Blocks (up to 2^31-1) --> Threads (<= 1024 per block) --> Warps (32 threads)
Global index: int i = blockIdx.x * blockDim.x + threadIdx.x
```

Blocks are scheduled independently onto Streaming Multiprocessors (SMs); threads within a block can cooperate via shared memory and `__syncthreads()`. Threads across blocks cannot synchronize within a kernel (barring cooperative groups in CUDA 9+). Each thread has a unique global index, which is used to access data in global memory.

Memory hierarchy.

| Space     | Scope        | Latency       | Size (typical) | Notes                                  |
|-----------|--------------|---------------|----------------|----------------------------------------|
| Registers | thread       | ~1 cycle      | 64K 32-bit/SM  | fastest; spills go to local mem        |
| Shared    | block        | ~20-30 cycles | 48-228 KB/SM   | programmer-managed cache               |
| Global    | device       | 400-800 cycles| GBs            | coalescing critical                    |
| Constant  | device (RO)  | cached        | 64 KB          | fast when all threads read same address|
| Texture   | device (RO)  | cached        | —              | 2D locality, filtering                 |
| Local     | thread       | = global      | —              | register spill target — avoid          |


**Coalescing**. A warp's 32 loads are combined into as few 32-byte transactions as possible when addresses are contiguous. Strided or scattered access multiplies memory traffic.

```
Coalesced: t0->a[0], t1->a[1] ... t31->a[31]  (1-4 transactions)
Strided:   t0->a[0], t1->a[32] ... t31->a[992]  (32 transactions)
```

**Warp divergence**. If threads in a warp take different branches, both paths execute serially with inactive lanes masked. Divergence within a warp hurts; divergence between warps is free.

**Occupancy**. Ratio of active warps per SM to the hardware maximum. Limited by register/thread, shared memory/block, and threads/block. Higher occupancy can hide latency, but is not always better.

**Hands-on exercises**

**Exercise 1 - Hello kernel**

**Task**: Write a kernel that prints `block bx thread tx -> global i` using device-side `printf`. Launch with 2 blocks x 4 threads.

**Input/expected behavior**: 8 lines (interleaving order may vary).

Hints:

1. Device printf works in any kernel; call cudaDeviceSynchronize() after launch to flush.
2. Compute i = blockIdx.x * blockDim.x + threadIdx.x.

Acceptance tests:

```bash
nvcc -arch=$ARCH hello.cu -o hello && ./hello # 8 lines of output
```

**Exercise 2 - Vector add kernel**

**Task**: GPU vector add for N = 1<<24 with a guard against out-of-range threads; verify against CPU result.

**Input/expected behavior**: prints `max error = 0` if correct.

**Hints**:

1. Grid size: `(N + block - 1) / block` with `block = 256`.
2. Always check `if (i < N)` - the grid overshoots.
3. Check every CUDA call's return code.

**Acceptance tests**:

```bash
nvcc -arch=$ARCH vector_add.cu -o vector_add && ./vector_add # prints max error = 0
```

**Exercise 3 - Coalescing benchmark**

**Task**: Copy an array with stride-1 access vs stride-32 access; time both with CUDA events and report effective bandwidth (GB/s).

**Input/expected behavior**: Coalesced version several times faster (often 5-15x).

**Hints**:

1. **Bandwidth** = bytes moved / time = `2 * N * sizeof(float) / seconds`
2. Use `cudaEventRecord`/`cudaEventElapsedTime`.
3. Warm up with one untimed branch.

**Acceptance tests**:

```bash
nvcc -arch=$ARCH coalesce.cu -o coalesce && ./coalesce # coalesced GB/s should be >3x strided GB/s
```

### Performance checklist / common pitfalls
- [] Guard if (i < n) in every grid-strided or overshooting launch.
- [] Check errors after every API call and kernel launch (cudaGetLastError).
- [] Block size a multiple of 32; 128–256 is a good default.
- [] Contiguous (coalesced) global memory access per warp.
- Pitfall: timing kernels without synchronizing — launches are asynchronous.
- Pitfall: forgetting cudaDeviceSynchronize() before reading device printf output.

### Profiling hints

```bash
nsys profile -o run ./vector_add          # timeline: kernels, memcpys
nsys stats run.nsys-rep                   # summary tables
ncu --set basic ./vector_add              # per-kernel hardware metrics
```

### Further reading & resources
- CUDA C++ Programming Guide (NVIDIA) — the authoritative reference.
- CUDA C++ Best Practices Guide (NVIDIA) — coalescing, occupancy guidance.
- Programming Massively Parallel Processors (Hwu, Kirk, Hajj) — the standard textbook.
- NVIDIA Developer Blog "CUDA Refresher" series — concise model overview.
- `deviceQuery` sample in cuda-samples (GitHub: NVqIDIA/cuda-samples) — inspect your GPU's limits.

### Checkpoint quiz
1. How many threads are in a warp?
2. Which memory is shared by all threads in a block? (a) registers (b) shared (c) constant.
3. Formula for the 1D global thread index?
4. Why can strided global access be ~32× slower per warp?
5. __syncthreads() synchronizes threads across: (a) the grid (b) a block (c) a warp only.