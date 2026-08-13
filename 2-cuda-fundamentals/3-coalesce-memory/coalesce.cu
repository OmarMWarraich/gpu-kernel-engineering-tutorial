#include <cstdio>                       // Include standard I/O for printf

// Kernel: copies N floats from in to out using a strided/permuted index.
// stride=1: consecutive threads read consecutive floats -> coalesced memory access.
// stride=32: consecutive threads are separated by 32 floats (128 bytes),
//             so each warp lane touches a different 128B cache segment.
__global__ void copy_strided(const float* in, float* out, int n, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // Global thread ID
    long j = (long)i * stride % n;                  // Permuted index, keeps same total memory traffic
    out[j] = in[j];                                 // Single float load + store at permuted index
}

// Benchmark helper: runs the kernel once for warm-up, then 10 timed iterations.
float bench(const float* in, float* out, int n, int stride) {
    int block = 256;                       // Threads per CUDA block
    int grid = (n + block - 1) / block;    // Enough blocks to cover all n threads (round up)

    copy_strided<<<grid, block>>>(in, out, n, stride);   // Warm-up launch (not timed)

    cudaEvent_t t0, t1;                    // CUDA events for GPU-side timing
    cudaEventCreate(&t0);                  // Create start event
    cudaEventCreate(&t1);                  // Create stop event

    cudaEventRecord(t0);                   // Record start timestamp
    for (int r = 0; r < 10; ++r)           // Run 10 iterations for stable timing
        copy_strided<<<grid, block>>>(in, out, n, stride);
    cudaEventRecord(t1);                   // Record stop timestamp
    cudaEventSynchronize(t1);              // Wait until stop event completes on GPU

    float ms;                              // Elapsed time in milliseconds
    cudaEventElapsedTime(&ms, t0, t1);     // Compute time between t0 and t1

    // Effective bandwidth:
    //   bytes moved = 2 (read + write) * n * sizeof(float) * 10 iterations
    //   GB/s = bytes / (milliseconds * 1e6 bytes/GB)
    return 2.0f * n * sizeof(float) * 10 / (ms * 1e6f);
}

int main() {
    const int N = 1 << 26;                 // Allocate ~256 MB of floats (67,108,736 floats)
    float *in, *out;                       // Device pointers for input and output arrays
    cudaMalloc(&in, N * sizeof(float));    // Allocate input array on GPU
    cudaMalloc(&out, N * sizeof(float));   // Allocate output array on GPU

    // Run and print bandwidth for coalesced (stride 1) access pattern
    printf("coalesced (stride 1):  %.1f GB/s\n", bench(in, out, N, 1));
    // Run and print bandwidth for strided (stride 32) access pattern
    printf("strided   (stride 32): %.1f GB/s\n", bench(in, out, N, 32));

    return 0;
}