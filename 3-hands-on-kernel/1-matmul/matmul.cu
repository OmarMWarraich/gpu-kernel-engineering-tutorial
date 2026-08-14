#include <cmath>                  // Math helpers like sinf/cosf
#include <cstdio>                 // C stdio for printf and FILE
#include <cstring>                // strcmp for parsing CLI arguments
#include <vector>                 // std::vector for host-side matrices

constexpr int N = 8192;          // Matrix dimension: 8192 x 8192
constexpr int TILE = 16;         // Tile size used by the tiled kernel; 16x16 threads per block

__global__ void matmul_naive(const float* A, const float* B, float* C, int n) {
    // Compute the global row and column indices for this thread.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Ignore threads that map outside the valid matrix bounds.
    if (row >= n || col >= n) return;

    // Accumulator for the dot product of row A[row, :] and column B[:, col].
    float acc = 0.f;

    // Naive matrix multiplication: each thread computes one output element.
    // For every k, multiply A[row, k] by B[k, col] and accumulate.
    for (int k = 0; k < n; ++k) acc += A[row * n + k] * B[k * n + col];

    // Store the computed dot product in the output matrix C.
    C[row * n + col] = acc;
}

__global__ void matmul_tiled(const float* A, const float* B, float* C, int n) {
    // Shared memory tiles for A and B; each tile holds a 16x16 block.
    __shared__ float As[TILE][TILE], Bs[TILE][TILE];

    // Local thread indices inside the current 16x16 block.
    int ty = threadIdx.y, tx = threadIdx.x;

    // Global indices of this thread's output element in C.
    int row = blockIdx.y * TILE + ty, col = blockIdx.x * TILE + tx;

    // Accumulator for the final dot product of one output element.
    float acc = 0.f;

    // Process matrix in tiles of size TILE; each iteration loads one tile from A and B.
    for (int t = 0; t < n / TILE; ++t) {
        // Load the A tile: A[row, t*TILE + tx] for the current row.
        // This is coalesced along the x dimension (tx), which is good for memory access.
        As[ty][tx] = A[row * n + t * TILE + tx];

        // Load the B tile: B[(t*TILE + ty), col] for the current column.
        // This loads a transposed-like layout from B so the thread accesses contiguous data.
        Bs[ty][tx] = B[(t * TILE + ty) * n + col];

        // Wait until all threads in the block have loaded their shared-memory values.
        __syncthreads();

        // Multiply the shared tile entries and accumulate the partial dot product.
        // Each thread computes TILE multiply-adds in the inner loop.
        #pragma unroll
        for (int k = 0; k < TILE; ++k) acc += As[ty][k] * Bs[k][tx];

        // Wait before reusing shared memory in the next tile iteration.
        // This ensures all threads have finished reading As/Bs before new tiles are written.
        __syncthreads();
    }

    // Write the final accumulated value for this output element.
    C[row * n + col] = acc;
}

int main(int argc, char** argv) {
    // Choose which kernel to run: "tiled" for optimized version, otherwise naive.
    bool tiled = argc > 1 && !strcmp(argv[1], "tiled");

    // Allocate host-side matrices A, B, and C.
    std::vector<float> hA(N * N), hB(N * N), hC(N * N);

    // Initialize A and B with smooth sine/cosine values to avoid trivial zeros.
    for (int i = 0; i < N * N; ++i) { hA[i] = sinf(i * 0.001f); hB[i] = cosf(i * 0.001f); }

    // Device pointers for A, B, and C.
    float *dA, *dB, *dC;

    // Allocate GPU memory for the matrices.
    cudaMalloc(&dA, N*N*4); cudaMalloc(&dB, N*N*4); cudaMalloc(&dC, N*N*4);

    // Copy input matrices from host RAM to device VRAM.
    cudaMemcpy(dA, hA.data(), N*N*4, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB.data(), N*N*4, cudaMemcpyHostToDevice);

    // Define a 16x16 thread block and a grid of (N/TILE)x(N/TILE) blocks.
    dim3 block(TILE, TILE), grid(N / TILE, N / TILE);

    // Lambda that launches the selected kernel with the configured grid and block sizes.
    auto launch = [&] { tiled ? matmul_tiled<<<grid, block>>>(dA, dB, dC, N)
                              : matmul_naive<<<grid, block>>>(dA, dB, dC, N); };

    // Run once as a warm-up to initialize the GPU and avoid timing startup overhead.
    launch(); cudaDeviceSynchronize();

    // Create CUDA events to measure execution time accurately.
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);

    // Start timing before the repeated benchmark loop.
    cudaEventRecord(t0);

    // Run the kernel 10 times to get a stable average runtime.
    for (int r = 0; r < 10; ++r) launch();

    // Stop timing and synchronize the device before reading the result.
    cudaEventRecord(t1); cudaEventSynchronize(t1);

    // Compute average runtime in milliseconds over the 10 launches.
    float ms; cudaEventElapsedTime(&ms, t0, t1); ms /= 10;

    // Copy the device output matrix back to host memory.
    cudaMemcpy(hC.data(), dC, N*N*4, cudaMemcpyDeviceToHost);

    // Write the computed result to a binary file named C_gpu.bin.
    FILE* f = fopen("C_gpu.bin", "wb"); fwrite(hC.data(), 4, N*N, f); fclose(f);

    // Print the kernel name, runtime, and effective throughput in GFLOPS.
    printf("%s: %.2f ms, %.1f GFLOPS\n", tiled ? "tiled" : "naive",
           ms, 2.0 * N * N * N / (ms * 1e6));

    // Return success code 0.
    return 0;
}