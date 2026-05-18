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

void mat_add(std::vector<int> &A, std::vector<int> &B, std::vector<int> &C, int n) {
    for (int i = 0; i < n; i++) {
        C[i] = A[i] + B[i];
    }
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

__global__ void mat_add_cuda(int *A, int *B, int *C, int width, int height, int n) {
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int i = row * width + col;

    if (i >= n)
        return;

    C[i] = A[i] + B[i];
}

int main(){
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

    dim3 grid(width / 8 + 1, height / 8 + 1);
    dim3 block(8,8);
    mat_add_cuda<<<grid, block>>>(d_A, d_B, d_C, width, height, n);

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
    return 0;
}
