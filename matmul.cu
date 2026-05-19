#include <iostream>
#include <cassert>
#include <vector>

// Wrapper macro for CUDA error handling
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

bool verify(const std::vector<std::vector<int>>& A,
            const std::vector<std::vector<int>>& B,
            const std::vector<std::vector<int>>& C
    ) {
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

void mat_mul(std::vector<std::vector<int>> &A, std::vector<std::vector<int>> &B, std::vector<std::vector<int>> &C) {
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

float call_cuda(){
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

int main() {
    float ms = call_cuda();
    std::cout << "CPU execution time: " << ms << " ms" << std::endl;
}
