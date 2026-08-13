#include <cmath>   // Provides fabsf/fmaxf for error checking
#include <cstdio>  // Provides printf/fprintf for console output
#include <vector>  // Provides std::vector for host-side arrays

// Macro: run a CUDA API call, print message and exit if it fails
#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    exit(1); } } while (0)

// CUDA kernel: each thread computes one element of c = a + b
__global__ void vadd(const float* a, const float* b, float* c, int n) {
    // Compute global thread index from block/grid dimensions
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // Guard: only process elements inside the array bounds
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    // Number of elements: 2^24 (~16.7 million) floats
    const int N = 1 << 24;
    // Host vectors (CPU memory) for inputs a, b and output c
    std::vector<float> ha(N), hb(N), hc(N);
    // Initialize host inputs with simple arithmetic sequences
    for (int i = 0; i < N; ++i) { ha[i] = i * 0.5f; hb[i] = i * 0.25f; }

    // Device pointers (GPU memory) for a, b, c
    float *da, *db, *dc;
    // Allocate GPU memory for each array
    CHECK(cudaMalloc(&da, N * sizeof(float)));
    CHECK(cudaMalloc(&db, N * sizeof(float)));
    CHECK(cudaMalloc(&dc, N * sizeof(float)));
    // Copy input data from host (CPU) to device (GPU)
    CHECK(cudaMemcpy(da, ha.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(db, hb.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // Configure kernel launch: 256 threads per block, enough blocks to cover N
    int block = 256, grid = (N + block - 1) / block;
    // Launch the vector-add kernel on the GPU
    vadd<<<grid, block>>>(da, db, dc, N);
    // Check for any launch/runtime errors from the kernel
    CHECK(cudaGetLastError());
    // Copy the result back from GPU to CPU memory
    CHECK(cudaMemcpy(hc.data(), dc, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify result by finding the maximum absolute difference from expected
    float err = 0;
    for (int i = 0; i < N; ++i) err = fmaxf(err, fabsf(hc[i] - (ha[i] + hb[i])));
    // Print the largest deviation seen
    printf("max error: %g\n", err);
    // Free allocated GPU memory
    cudaFree(da); cudaFree(db); cudaFree(dc);
    // Return 0 if exact match, otherwise 1
    return err == 0 ? 0 : 1;
}