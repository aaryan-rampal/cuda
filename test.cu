#include <iostream>
#define N 10000000

void vector_add(float *out, float *a, float *b, int n) {
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
    // vector_add(out, a, b, N);
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vec_add<<<blocksPerGrid, threadsPerBlock>>>(out, a, b, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << "\n";
        return -1;
    }

    cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Device synchronize error: " << cudaGetErrorString(err) << "\n";
        return -1;
    }

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
