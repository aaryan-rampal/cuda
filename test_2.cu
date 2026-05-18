#include <iostream>
#define N 10000000

// Wrapper macro for CUDA error handling
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

__global__ void vec_add_bad(float *out, float *a, float *b, int n) {
    for(int i = 0; i < n; i++){
        out[i] = a[i] + b[i];
    }
}

__global__ void vec_add(float *out, float *a, float *b, int n) {
    int workIndex = threadIdx.x + blockDim.x * blockIdx.x;

    if (workIndex >= n)
        return;

    out[workIndex] = a[workIndex] + b[workIndex];
}

int main(){
    float *a = nullptr;
    float *b = nullptr;
    float *out = nullptr;

    // Allocate memory
    cudaMallocManaged(&a, N * sizeof(float));
    cudaMallocManaged(&b, N * sizeof(float));
    cudaMallocManaged(&out, N * sizeof(float));

    // Initialize array
    for(int i = 0; i < N; i++){
        a[i] = 1.0f; b[i] = 2.0f;
        out[i] = 0;
    }

    // Main function
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    vec_add_bad<<<1, 1>>>(out, a, b, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << "Kernel time: " << ms << " ms\n";

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    int cnt = 0;
    for (int i = 0; i < N; i++) {
        if (out[i] != a[i] + b[i]) {
            std::cout << cnt << " nuh uh\n";
            std::cout << a[i] << " + " << b[i] << " != " << out[i];
            return -1;
        }
        cnt += 1;
    }

    std::cout << "oh shit";

    cudaFree(a);
    cudaFree(b);
    cudaFree(out);
}
