# Basics of Parallel Programming

## Overview

Before writing GPU kernels, you must understand the basics of parallel programming. This lesson covers threads vs processes, memory models, data
vs task parallelism, and the two laws that govern parallel speedup: Amdahl's Law and Gustafson's Law. You will build multi-threaded CPU programs to develop intuition you'll reuse on GPU's.

## Learning Objectives

- Distinguish threads from processes and shared from distributed memory.
- Explain data parallelism vs task parallelism with examples.
- Compute theoretical speedups using Amdahl's Law and Gustafson's Law.
- Constrast SIMD and SIMT execution models.
- Explain latency vs throughput orientation of CPUs vs GPUs.
- Use `std::thread`, mutexes, and barriers to write correct multi-threaded C++.
- Measure and interpret parallel speedup on real hardware.

## Prerequisites and setup

- A C++17 compiler (g++ ≥ 11 or clang ≥ 14), Python 3.11 for verification.

```bash
# Ubuntu
sudo apt-get update && sudo apt-get install -y g++ python3
g++ --version
```

### Concepts and theory

**Threads and processes**

> Threads are the smallest unit of execution within a process. Multiple threads within the same process share the same memory space, which allows for efficient communication but requires careful synchronization to avoid race conditions.

>Processes are independent execution units with their own memory space. Communication between processes is more expensive, typically involving inter-process communication (IPC) mechanisms.

> A process has its own address sapce; threads within a process share one address space (shared memory). Sharing memory makes communication cheap but introduces race conditions. Distributed-memory parallelism (e.g. MPI across nodes) communicates via explicit messsages.

**Data parallelism vs task parallelism**

> Data parallelism: The same operation is applied concurrently to multiple data elements. This is common in numerical computations, image processing, and scientific simulations. (c[i] = a[i] + b[i]). This is what GPUs are designed for.

> Task parallelism: Different operations are performed concurrently on different data elements. This is common in applications with independent tasks, such as web servers handling multiple requests or pipelines processing different stages of data. (e.g., one thread parses, one compresses, one encrypts).

**SIMD vs SIMT**

> SIMD (Single Instruction, Multiple Data) CPU vector unit: is a parallel computing model where a single instruction operates on multiple data points simultaneously. This is commonly used in vector processors and certain CPU architectures.

- one instruction --> [lane0..lane7]
- lanes are rigid, no per-lane PC

> SIMT (Single Instruction, Multiple Threads) GPU Warp: a group of threads that execute the same instruction simultaneously, is a parallel computing model used by GPUs, where a single instruction is executed by multiple threads concurrently. Each thread operates on its own data, allowing for massive parallelism.

- one instruction --> [thread0..thread31]
- threads have own registers/PC illusion; divergence serializes branches

**Latency vs throughput**

> CPUs minimize the latency the latency of one thread (big caches, branch prediction, out-of-order execution). GPUs maximize throughput of many threads (many cores, many threads, small caches, no branch prediction) and hide memory latency with massive parallelism by switching between threads.

**Amdahl's Law**

> If a fraction `p` of a program can be parallelized, the maximum speedup `S` achievable with `N` processors is given by Amdahl's Law:

```
S(N) = 1 / ((1 - p) + (p / N))
```

> The serial fraction (1 - p) limits the overall speedup, meaning that even with an infinite number of processors, the speedup is bounded by the serial portion of the program: with p = 0.95, max speedup is 20x, with p = 0.99, max speedup is 100x.

**Gustafson's Law**

> Gustafson's Law states that as the problem size increases, the parallel portion of the program can also increase, allowing for greater speedup. It is expressed as:

```
S(N) = (1 - p) + (p * N)
```

This is why GPU workloads are often larger than CPU workloads: the more data you have, the more parallelism you can exploit.

**Synchronization primitives**

> Synchronization primitives are mechanisms to coordinate the execution of concurrent threads and ensure correct access to shared resources. Common primitives include:

| Primitive | Purpose | GPU analogue |
|-----------|---------|--------------|
| Mutex/lock | Exclusive access to shared data | atomics |
| Barrier | All threads wait until everyone arrives | `__syncthreads()` |
| Atomic op | Indivisible read-modify-write | `atomicAdd` |
| Condition variables | Block threads until a certain condition is met | - |

## Hands-on exercises

Exercise 1 -- Multithreaded vector add

Task: Implement c[i] = a[i] + b[i] for N = 10^9 floats, single-threaded and with T threads (std::thread), each handling a contiguous chunk. Time both.

Input/expected behavior: Program prints both runtimes and a max-abs-error check (must be 0).

Hints:

1. Split the range [0, N) into T chunks: thread t handles [t*N/T, (t+1)*N/T).
2. Use `std::chrono::steady_clock` for timing.
3. No locks needed - each thread writes disjoint indices.

Acceptance tests:

```bash
g++ -O2 -std=c++17 -pthread vector_add_cpu.cpp -o vector_add_cpu
./vector_add_cpu 8   # expect speedup > 1.5x and "max error: 0"
```

Exercise 2 -- Map-reduce sum with threads

Task: Sum 100M floats: each thread computes a partial sum (map), then the main thread combines partials (reduce). Compare with a single mutex-protected accumulator version to see contention cost.

Input/expected behavior: Both versions produce the same sum (within float tolerance); partial-sum version is much faster.

Hints:

1. Give each thread its own local accumulator; only combine at the end.
2. For the slow version, lock a mutex on every addition and observe the slowdown.
3. Use `double` accumulators to reduce rounding error.

Acceptance tests:

```bash
g++ -O2 -std=c++17 -pthread reduce_sum.cpp -o reduce_sum
./reduce_sum 8   # relative error vs sequential < 1e-6
```

Exercise 3 -- Amdahl's Law analysis

Task: A workload spends 10%. of time in serial I/O and 90% in parallelizable loop. Compute theoretical speedup for N = 2, 8, 32, 1024, ∞ workers. Then compute Gustafon scaled speedup for the same N.

Input/expected behavior: A small script of table of speedups.

Hints: 

1. Plug p = 0.9 into both formulas.
2. Note the asymptote: 1/(1 - p) = 10.

Acceptance tests:

```bash
python3 amdahl.py   # S(inf) prints 10.0
```

### Performance checklist/ common pitfalls

- [ ] Thread works on disjoint data - no false sharing (pad hot per-thread data to cache-line size).
- [ ] Avoid locks in hot loops; use per-thread accumulators.
- [ ] Vector add is memory-bound; don't expect linear speedup with core count. 

- Pitfall: timing with clock() measures CPU time across all threads - use wall-clock (steady_clock) instead.
- Pitfall: forgetting -02 makes comparison meanigless.

Further reading & resource:

- Computer Architecture: A Quantitative Approach (Hennessy & Patterson) — chapters on TLP/DLP.
- C++ Concurrency in Action (Anthony Williams) — definitive std::thread reference.
- "The Free Lunch Is Over" (Herb Sutter) — why parallelism became mandatory.
- LLNL Parallel Computing Tutorial — classic free introduction (hpc-tutorials.llnl.gov).
- Amdahl (1967) and Gustafson (1988) original papers — short, readable.

### Checkpoint quiz

1. What limits maximum speedup under Amdahl's Law?

- The serial fraction (1 - p).

2. Threads within a process share: 
   a) stack
   b) address space
   c) program counter

  - b) address space

3. SIMT differs from from SIMD mainly because:
   a) SIMT has no vector units
   b) SIMT threads can diverge on branches
   c) SIMD is GPU-only

  - b) SIMT threads can diverge on branches

4. With p = 0.95 and infinite workers, Amdahl speedup?

  - 1 / (1 - 0.95) = 20

5. Why is a per-thread partial sum faster than a mutex-protected shared sum?

  - Because each thread works on its own local data without contention, avoiding the overhead of locking a shared resource.