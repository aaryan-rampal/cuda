# %% [markdown]
# ## ⬇ Your work — exercises & tests

# %% [markdown]
# ### Ex 05.1 / warm-up — register-blocking arithmetic

# %%
def reuse_ratio_2d(TM, TN):
    return None  # TODO: FMAs per shared-load = TM*TN / (TM+TN)

def threads_per_block_2d(BM, BN, TM, TN):
    return None  # TODO

def accumulator_regs(TM, TN):
    return None  # TODO: floats for acc[TM][TN] + regM[TM] + regN[TN]

# %%
check_fn("05.1 reuse_ratio_2d", reuse_ratio_2d,
         lambda TM, TN: TM*TN/(TM+TN), [(8,8), (4,4), (8,4)])
check_fn("05.1 threads_per_block_2d", threads_per_block_2d,
         lambda BM,BN,TM,TN: BM*BN//(TM*TN), [(128,128,8,8), (64,64,4,4)])
check("05.1 (128,128,8,8) -> 256 threads", threads_per_block_2d(128,128,8,8), 256)
check("05.1 8x8 tile accumulator regs", accumulator_regs(8,8), 80)

# %% [markdown]
# ### Ex 05.3 — write the 2D blocktiling kernel (autograded)
# The workhorse: each thread computes a TM×TN tile via an outer product. Fill
# the two load indices.

# %%
sgemm_2d_src = r"""
#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8
__global__ void sgemm_2d(const float* A, const float* B, float* C,
                         int M, int N, int K) {
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];
    int cRow = blockIdx.y, cCol = blockIdx.x;
    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;
    int threadRow = threadIdx.y, threadCol = threadIdx.x;   // 0..15
    int tid = threadRow * blockDim.x + threadCol;           // 0..255
    int innerRowA = tid / BK,  innerColA = tid % BK;
    int innerRowB = tid / BN,  innerColB = tid % BN;
    const int strideA = (blockDim.x*blockDim.y) / BK;
    const int strideB = (blockDim.x*blockDim.y) / BN;
    float acc[TM][TN] = {0.0f};
    float regM[TM], regN[TN];
    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int o = 0; o < BM; o += strideA)
            As[(innerRowA + o) * BK + innerColA] = A[ /* TODO A[innerRowA+o][k0+innerColA] */ 0 ];
        for (int o = 0; o < BK; o += strideB)
            Bs[(innerRowB + o) * BN + innerColB] = B[ /* TODO B[k0+innerRowB+o][innerColB] */ 0 ];
        __syncthreads();
        A += BK;
        B += BK * N;
        for (int k = 0; k < BK; k++) {
            for (int i = 0; i < TM; i++) regM[i] = As[(threadRow*TM + i)*BK + k];
            for (int j = 0; j < TN; j++) regN[j] = Bs[k*BN + (threadCol*TN + j)];
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
launch_2d = ("dim3 block(16, 16); "               # (BN/TN, BM/TM)
             "dim3 grid(N/128, M/128); "
             "sgemm_2d<<<grid, block>>>(dA, dB, dC, M, N, K);")
# Use a multiple of 128 so the grid divides evenly:
check_cuda("05.3 2D blocktiling correct", sgemm_2d_src, launch_2d, M=256, N=256, K=256)

# %% [markdown]
# Now the performance-engineering loop: sweep sizes, and try editing TM/TN/BM/BN.
# Also compile with `--ptxas-options=-v` in your shell to read registers/thread.

# %%
bench_cuda(sgemm_2d_src, launch_2d, sizes=(512, 1024, 2048))
