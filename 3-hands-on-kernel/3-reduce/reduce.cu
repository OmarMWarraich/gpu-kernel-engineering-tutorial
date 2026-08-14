#include <cstdio>   // C-style I/O: printf
#include <vector>   // std::vector for host-side array storage

// CUDA kernel: parallel reduction (sum) of an array of n floats
__global__ void reduce_sum(const float* in, float* out, long n) {
    // shared memory: one float per thread in this 256-thread block
    __shared__ float smem[256];

    // global index assigned to this thread: block offset + lane within block
    long i = blockIdx.x * blockDim.x + threadIdx.x;

    // total number of threads in the grid (stride for grid-stride loop)
    long stride = (long)gridDim.x * blockDim.x;

    // local accumulator for this thread
    float v = 0.f;

    // grid-stride loop: each thread sums a subset of elements spaced 'stride' apart
    for (; i < n; i += stride) v += in[i];

    // write this thread's partial sum into shared memory
    smem[threadIdx.x] = v;

    // synchronize to ensure all threads have stored their partial sums
    __syncthreads();

    // tree reduction in shared memory, halving the active range each iteration
    // stop at warp size (32) because warp-level shuffles handle the rest
    for (int s = blockDim.x / 2; s >= 32; s >>= 1) {
        // active threads add the value from the partner slot 's' positions away
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        // wait for all active threads before next reduction step
        __syncthreads();
    }

    // first warp now holds the block's 32 remaining partial sums in smem[0..31]
    if (threadIdx.x < 32) {
        // reload this warp lane's value into a register
        v = smem[threadIdx.x];

        // warp-level butterfly reduction using shuffle instructions
        // each iteration adds the value from the thread 'off' lanes to the right
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_down_sync(0xffffffff, v, off);

        // lane 0 now has the total sum for this block; atomically accumulate into global result
        if (threadIdx.x == 0) atomicAdd(out, v);
    }
}

int main() {
    // number of elements: 2^26 = ~67 million floats
    const long N = 1L << 26;

    // host array to hold input data
    std::vector<float> h(N);

    // reference sum computed on the CPU in double precision
    double ref = 0;

    // fill host array with a repeating pattern and accumulate the reference sum
    for (long i = 0; i < N; ++i) { h[i] = 0.001f * (i % 1000); ref += h[i]; }

    // device pointers for input and output
    float *din, *dout;

    // allocate GPU memory for the input array (N floats) and the single output float
    cudaMalloc(&din, N * 4); cudaMalloc(&dout, 4);

    // copy the host input array to the device
    cudaMemcpy(din, h.data(), N * 4, cudaMemcpyHostToDevice);

    // initialize the device output (sum) to zero
    cudaMemset(dout, 0, 4);

    // launch kernel with 1024 blocks and 256 threads per block
    reduce_sum<<<1024, 256>>>(din, dout, N);

    // host variable to receive the final GPU sum
    float sum;

    // copy the reduced result back to the host
    cudaMemcpy(&sum, dout, 4, cudaMemcpyDeviceToHost);

    // compute relative error against the CPU reference
    double rel = fabs(sum - ref) / ref;

    // print PASS/FAIL with relative error and both sums
    printf("%s rel_err=%g (gpu=%f ref=%f)\n", rel < 1e-5 ? "PASS" : "FAIL", rel, sum, ref);

    // return 0 if the result is within tolerance, otherwise 1
    return rel < 1e-5 ? 0 : 1;
}