# %% [markdown]
# ## ⬇ Your work — exercises & tests

# %% [markdown]
# ### Ex 04.1 / 04.2 — coarsening arithmetic

# %%
def fmas_per_b_load(TM):
    return None  # TODO: FLOPs produced per single shared-memory load of B

def threads_per_block_1d(BM, BN, TM):
    return None  # TODO

# %%
check_fn("04.1 fmas_per_b_load", fmas_per_b_load, lambda TM: 2 * TM, [8, 4, 16])
check_fn("04.2 threads_per_block_1d", threads_per_block_1d,
         lambda BM, BN, TM: BM * BN // TM, [(64,64,8), (128,128,8)])
check("04.2 (64,64,TM=8) -> 512 threads", threads_per_block_1d(64,64,8), 512)

# %% [markdown]
# ### Ex 04.3 — write the 1D blocktiling kernel (autograded)
# Each thread owns a column of TM results in registers. Fill the 3 blanks.

# %%
sgemm_1d_src = r"""
#define BM 64
#define BN 64
#define BK 8
#define TM 8
__global__ void sgemm_1d(const float* A, const float* B, float* C,
                         int M, int N, int K) {
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];
    int cRow = blockIdx.y, cCol = blockIdx.x;
    int threadCol = threadIdx.x;
    int threadRow = threadIdx.y;
    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;
    int tid = threadRow * blockDim.x + threadCol;
    int aRow = tid / BK, aCol = tid % BK;
    int bRow = tid / BN, bCol = tid % BN;
    float threadResults[TM] = {0.0f};
    for (int k0 = 0; k0 < K; k0 += BK) {
        As[aRow * BK + aCol] = A[ /* TODO A[aRow][k0+aCol] */ 0 ];
        Bs[bRow * BN + bCol] = B[ /* TODO B[k0+bRow][bCol] */ 0 ];
        __syncthreads();
        A += BK;
        B += BK * N;
        for (int k = 0; k < BK; k++) {
            float tmpB = Bs[k * BN + threadCol];
            for (int r = 0; r < TM; r++)
                threadResults[r] += As[ /* TODO As[threadRow*TM+r][k] */ 0 ] * tmpB;
        }
        __syncthreads();
    }
    for (int r = 0; r < TM; r++)
        C[(threadRow * TM + r) * N + threadCol] = threadResults[r];
}
"""

# %%
launch_1d = ("dim3 block(64, 8); "                 # (BN, BM/TM)
             "dim3 grid(N/64, M/64); "
             "sgemm_1d<<<grid, block>>>(dA, dB, dC, M, N, K);")
# N=256 is a multiple of BM=BN=64, so the grid divides evenly.
check_cuda("04.3 1D blocktiling correct", sgemm_1d_src, launch_1d, M=256, N=256, K=256)

# %%
bench_cuda(sgemm_1d_src, launch_1d, sizes=(512, 1024, 2048))
