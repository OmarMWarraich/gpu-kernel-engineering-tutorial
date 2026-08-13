#include <cstdio>   // bring in printf

// #define cudaCheck(ans) { cudaAssert((ans), __FILE__, __LINE__); }
// inline void cudaAssert(cudaError_t code, const char *file, int line) {
//     if (code != cudaSuccess) {
//         fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(code), file, line);
//         exit(code);
//     }
// }

__global__ void hello() {   // kernel callable from host, runs on device
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // global thread index
    printf("block %d, thread %d -> global %d\n", blockIdx.x, threadIdx.x, i);   // print per-thread info
}

int main() {                // host entry point
    hello<<<2, 4>>>();      // launch kernel with 2 blocks of 4 threads
    // cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());    // flush device printf
    return 0;               // exit
}