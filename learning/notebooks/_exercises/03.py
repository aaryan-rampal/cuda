# %% [markdown]
# ## ⬇ Your work — exercises & tests

# %% [markdown]
# ### Ex 03.1 — DRAM traffic and intensity of tiled matmul
# Square NxN, square tiles of side BK.

# %%
def dram_floats_tiled(N, BK):
    return None  # TODO: total floats read from DRAM across all blocks

def intensity_tiled(BK):
    return None  # TODO: FLOP/byte (float = 4 bytes), in terms of BK

# %%
check_fn("03.1 dram_floats_tiled", dram_floats_tiled,
         lambda N, BK: 2 * N**3 / BK, [(1024, 32), (512, 16), (2048, 32)])
check_fn("03.1 intensity_tiled", intensity_tiled,
         lambda BK: BK / 4, [8, 16, 32])
check("03.1 BK=32 intensity", intensity_tiled(32), 8.0, tol=1e-9)

# %% [markdown]
# ### Ex 03.3 — write the shared-memory tiled kernel (autograded)
# Keep `BK` == block side. Test uses N=256 (multiple of 32), so no bounds-pad
# needed. Fill the two load indices and the inner product.

# %%
sgemm_tiled_src = r"""
#define BK 32
__global__ void sgemm_tiled(const float* A, const float* B, float* C,
                            int M, int N, int K) {
    __shared__ float As[BK][BK];
    __shared__ float Bs[BK][BK];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * BK + ty;
    int col = blockIdx.x * BK + tx;
    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += BK) {
        As[ty][tx] = A[ /* TODO A[row][k0+tx] */ 0 ];
        Bs[ty][tx] = B[ /* TODO B[k0+ty][col] */ 0 ];
        __syncthreads();
        for (int k = 0; k < BK; k++)
            acc += /* TODO As[ty][k] * Bs[k][tx] */ 0.0f;
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}
"""

# %%
launch_tiled = ("dim3 block(32,32); "
                "dim3 grid((N+31)/32, (M+31)/32); "
                "sgemm_tiled<<<grid, block>>>(dA, dB, dC, M, N, K);")
check_cuda("03.3 tiled SGEMM correct", sgemm_tiled_src, launch_tiled, M=256, N=256, K=256)

# %%
# Compare to naive — and try editing BK to 16 (and block to 16x16) for occupancy.
bench_cuda(sgemm_tiled_src, launch_tiled, sizes=(512, 1024, 2048))
