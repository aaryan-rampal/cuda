# %% [markdown]
# ## ⬇ Your work — exercises & tests

# %% [markdown]
# ### Ex 06 warm-up — float4 + bank arithmetic

# %%
def float4_bytes():
    return None  # TODO: bytes moved by one float4 load

def bank_of(byte_address):
    return None  # TODO: shared-memory bank index = (addr/4) % 32

# %%
check("06 float4 bytes", float4_bytes(), 16)
check_fn("06 bank_of", bank_of, lambda a: (a // 4) % 32, [0, 4, 128, 4*32, 4*33, 1020])

# %% [markdown]
# ### Ex 06 — vectorized + transposed kernel (autograded, advanced)
# This is the hard one. Below is a complete reference-shaped skeleton with the
# `As` stored TRANSPOSED (`As[BK][BM]`) and `Bs` loaded with `float4`. The TODOs
# are the global-load indices. Get it green, then `bench_cuda` it against your
# module-05 kernel. (If you'd rather, start from your own sgemm_2d and add one
# change at a time — that's the recommended discipline.)

# %%
sgemm_vec_src = r"""
#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8
__global__ void sgemm_vec(const float* A, const float* B, float* C,
                          int M, int N, int K) {
    __shared__ float As[BK * BM];   // transposed: As[k][m]
    __shared__ float Bs[BK * BN];
    int cRow = blockIdx.y, cCol = blockIdx.x;
    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    int threadRow = threadIdx.y, threadCol = threadIdx.x;  // 0..15
    int tid = threadRow * blockDim.x + threadCol;          // 0..255

    // each of 256 threads loads one float4 (=4 floats) of A and of B per chunk
    int innerRowA = tid / (BK / 4);        // BK/4 = 2 float4 columns of A
    int innerColA = tid % (BK / 4);        // 0..1
    int innerRowB = tid / (BN / 4);        // BN/4 = 32 float4 columns of B
    int innerColB = tid % (BN / 4);        // 0..31

    float acc[TM][TN] = {0.0f};
    float regM[TM], regN[TN];

    for (int k0 = 0; k0 < K; k0 += BK) {
        // --- load A as float4, scatter into transposed As ---
        float4 a = reinterpret_cast<const float4*>(
            &A[ /* TODO A[innerRowA][k0 + innerColA*4] */ 0 ])[0];
        As[(innerColA*4 + 0) * BM + innerRowA] = a.x;
        As[(innerColA*4 + 1) * BM + innerRowA] = a.y;
        As[(innerColA*4 + 2) * BM + innerRowA] = a.z;
        As[(innerColA*4 + 3) * BM + innerRowA] = a.w;
        // --- load B as float4 straight into Bs ---
        reinterpret_cast<float4*>(&Bs[innerRowB * BN + innerColB*4])[0] =
            reinterpret_cast<const float4*>(
                &B[ /* TODO B[k0+innerRowB][innerColB*4] */ 0 ])[0];
        __syncthreads();
        A += BK;
        B += BK * N;

        for (int k = 0; k < BK; k++) {
            for (int i = 0; i < TM; i += 4)
                reinterpret_cast<float4*>(&regM[i])[0] =
                    reinterpret_cast<float4*>(&As[k*BM + threadRow*TM + i])[0];
            for (int j = 0; j < TN; j += 4)
                reinterpret_cast<float4*>(&regN[j])[0] =
                    reinterpret_cast<float4*>(&Bs[k*BN + threadCol*TN + j])[0];
            for (int i = 0; i < TM; i++)
                for (int j = 0; j < TN; j++)
                    acc[i][j] += regM[i] * regN[j];
        }
        __syncthreads();
    }
    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j++)
            C[(threadRow*TM + i)*N + (threadCol*TN + j)] = acc[i][j];
}
"""

# %%
launch_vec = ("dim3 block(16, 16); "
              "dim3 grid(N/128, M/128); "
              "sgemm_vec<<<grid, block>>>(dA, dB, dC, M, N, K);")
check_cuda("06 vectorized+transposed correct", sgemm_vec_src, launch_vec, M=256, N=256, K=256)

# %%
bench_cuda(sgemm_vec_src, launch_vec, sizes=(1024, 2048, 4096))
