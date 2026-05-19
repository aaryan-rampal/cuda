#include <iostream>
#include <vector>

// Wrapper macro for CUDA error handling
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

bool verify(const std::vector<int>& A,
            const std::vector<int>& B,
            const std::vector<int>& C,
            int n) {
    for (int i = 0; i < n; i++) {
        int expected = A[i] + B[i];

        if (C[i] != expected) {
            std::cerr << "Verification failed at index " << i << "\n"
                      << "A[" << i << "] = " << A[i] << "\n"
                      << "B[" << i << "] = " << B[i] << "\n"
                      << "C[" << i << "] = " << C[i] << "\n"
                      << "Expected C[" << i << "] = " << expected << "\n";

            return false;
        }
    }

    return true;
}

__global__ void mat_add_cuda_reversed(int *A, int *B, int *C, int width, int height, int n) {
    // this is similar to mat_add_cuda EXCEPT
    // we use threadIdx.x (fastest moving index) for row (slowest moving iterator)
    // notice the difference in the kernel times

    int row = threadIdx.x + blockDim.x * blockIdx.x;
    int col = threadIdx.y + blockDim.y * blockIdx.y;
    if (row >= height || col >= width)
        return;

    int i = row * width + col;

    C[i] = A[i] + B[i];
}

__global__ void mat_add_cuda(int *A, int *B, int *C, int width, int height, int n) {
    // threadIdx.x moves fastest
    // for a 2D array A[i][j], we increment cols first then rows
    // i.e. threadIdx.x -> col, threadIdx.y -> row
    // for (row in threadIdx.y) {
    //   for (col in threadIdx.x) {
    //   }
    // }
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    if (row >= height || col >= width)
        return;

    int i = row * width + col;

    C[i] = A[i] + B[i];
}

float call_cuda(bool reversed){
    int width = 5000, height = 7500;
    int n = height * width;
    std::vector<int> A(n), B(n), C(n, 0);

    for (int i = 0; i < n; i++) {
        A[i] = rand() % 1000 + 1;
        B[i] = rand() % 1000 + 1;
        C[i] = 0;
    }

    // mat_add(A, B, C, n);
    int *d_A, *d_B, *d_C;
    size_t size = A.size() * sizeof(int);
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    cudaMemcpy(d_A, A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C.data(), size, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid;
    if (reversed) {
        grid = dim3(height / 16 + 1, width / 16 + 1);
    } else {
        grid = dim3(width / 16 + 1, height / 16 + 1);
    }


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    if (reversed) {
        mat_add_cuda_reversed<<<grid, block>>>(d_A, d_B, d_C, width, height, n);
    } else {
        mat_add_cuda<<<grid, block>>>(d_A, d_B, d_C, width, height, n);
    }


    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(A.data(), d_A, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(B.data(), d_B, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(C.data(), d_C, size, cudaMemcpyDeviceToHost);

    if (!verify(A, B, C, n)) {
        std::cout << "Not valid";
        exit(EXIT_FAILURE);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return ms;
}

/*
   Run on NVIDIA GeForce RTX 3060
   Normal kernel time: 1.54429
   Reversed kernel time: 2.72595
*/
int main() {
    float ms_normal = call_cuda(false);
    float ms_reversed = call_cuda(true);

    std::cout << "Normal kernel time: " << ms_normal << std::endl;
    std::cout << "Reversed kernel time: " << ms_reversed << std::endl;
}
