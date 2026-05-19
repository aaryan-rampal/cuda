#include <cassert>
#include <iostream>
#include <vector>

// Wrapper macro for CUDA error handling
#define CHECK_CUDA(call)                                                             \
    {                                                                                \
        cudaError_t err = call;                                                      \
        if (err != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                        \
            exit(EXIT_FAILURE);                                                      \
        }                                                                            \
    }

bool verify(const std::vector<std::vector<int>> &A, const std::vector<std::vector<int>> &B,
            const std::vector<std::vector<int>> &C) {
    assert(C.size() > 0);

    int a_1 = A.size(), a_2 = B.size(), a_3 = C[0].size();

    assert(a_1 > 0);
    assert(a_2 > 0);
    assert(a_3 > 0);

    for (int i = 0; i < a_1; ++i) {
        for (int j = 0; j < a_3; ++j) {
            int expected = 0;
            for (int k = 0; k < a_2; ++k) {
                expected += A[i][k] * B[k][j];
            }
            if (C[i][j] != expected) {
                std::cerr << "Verification failed at (" << i << ", " << j << ")\n";
                return false;
            }
        }
    }

    return true;
}

bool verify(int *A, int *B, int *C, int a_1, int a_2, int a_3) {
    for (int i = 0; i < a_1; ++i) {
        for (int j = 0; j < a_3; ++j) {
            int expected = 0;
            for (int k = 0; k < a_2; ++k) {
                expected += A[i * a_2 + k] * B[k * a_3 + j];
            }
            if (C[i * a_3 + j] != expected) {
                std::cerr << "Verification failed at (" << i << ", " << j << "): "
                          << "expected " << expected << ", got " << C[i * a_3 + j] << "\n";
                return false;
            }
        }
    }
    return true;
}

void mat_mul(std::vector<std::vector<int>> &A, std::vector<std::vector<int>> &B,
             std::vector<std::vector<int>> &C) {
    assert(C.size() > 0);

    int a_1 = A.size(), a_2 = B.size(), a_3 = C[0].size();

    assert(a_1 > 0);
    assert(a_2 > 0);
    assert(a_3 > 0);

    for (int i = 0; i < a_1; ++i) {
        for (int j = 0; j < a_3; ++j) {
            C[i][j] = 0;
            for (int k = 0; k < a_2; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void fill_with_random(std::vector<std::vector<int>> &vec) {
    for (int i = 0; i < vec.size(); i++) {
        for (int j = 0; j < vec[i].size(); j++) {
            vec[i][j] = rand() % 1000 + 1;
        }
    }
}

void fill_with_random(int *vec, int size) {
    for (int i = 0; i < size; i++) {
        vec[i] = rand() % 1000 + 1;
    }
}

float call_cpu() {
    int a_1 = 50, a_2 = 75, a_3 = 80;
    std::vector<std::vector<int>> A(a_1, std::vector<int>(a_2, 0));
    std::vector<std::vector<int>> B(a_2, std::vector<int>(a_3, 0));
    std::vector<std::vector<int>> C(a_1, std::vector<int>(a_3, 0));

    fill_with_random(A);
    fill_with_random(B);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    mat_mul(A, B, C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    if (!verify(A, B, C)) {
        std::cout << "Not valid";
        exit(EXIT_FAILURE);
    }

    return ms;
}

__global__ void mat_mul(int *A, int *B, int *C, int a_1, int a_2, int a_3) {
    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= a_1 || j >= a_3 || k >= a_2)
        return;

    int idx_A = i * a_2 + k;
    int idx_B = k * a_3 + i;
    int idx_C = i * a_3 + j;
    C[idx_C] = A[idx_A] + B[idx_B];
}

float call_gpu() {
    int a_1 = 500, a_2 = 750, a_3 = 800;
    size_t size_A = a_1 * a_3, size_B = a_2 * a_3, size_C = a_1 * a_3;
    size_t mal_A = size_A * sizeof(int);
    size_t mal_B = size_B * sizeof(int);
    size_t mal_C = size_C * sizeof(int);

    int *A = (int *)malloc(mal_A);
    int *B = (int *)malloc(mal_B);
    int *C = (int *)malloc(mal_C);
    if (!A || !B || !C) {
        perror("malloc");
        exit(EXIT_FAILURE);
    }

    fill_with_random(A, a_1 * a_3);
    fill_with_random(B, a_2 * a_3);

    int *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, mal_A));
    CHECK_CUDA(cudaMalloc(&d_B, mal_B));
    CHECK_CUDA(cudaMalloc(&d_C, mal_C));
    CHECK_CUDA(cudaMemcpy(d_A, A, mal_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, B, mal_B, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C, C, mal_C, cudaMemcpyHostToDevice));

    dim3 grid(a_1 / 8 + 1, a_3 / 8 + 1, a_2 / 8 + 1);
    dim3 block(8, 8, 8);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    mat_mul<<<grid, block>>>(d_A, d_B, d_C, a_1, a_2, a_3);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    CHECK_CUDA(cudaMemcpy(A, d_A, mal_A, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(B, d_B, mal_B, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(C, d_C, mal_C, cudaMemcpyDeviceToHost));

    if (!verify(A, B, C, a_1, a_2, a_3)) {
        std::cout << "Not valid";
        exit(EXIT_FAILURE);
    }

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return ms;
}

int main() {
    float ms = call_cpu();
    std::cout << "CPU execution time: " << ms << " ms" << std::endl;
    float ms_gpu = call_gpu();
    std::cout << "GPU execution time: " << ms_gpu << " ms" << std::endl;
}
