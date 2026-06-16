# %% [markdown]
# ## ⬇ Your work — exercises & tests
#
# First the coalescing math, then your **first autograded CUDA kernel**: write
# the kernel as a Python string, run the test cell, and the harness compiles it
# with `nvcc`, runs it on your 3060, verifies the result, and prints GFLOP/s.

# %% [markdown]
# ### Ex 02.1 — address stride between consecutive threads of a warp
# Fast kernel: consecutive `threadIdx.x` → consecutive `col`. Slow ("reversed")
# kernel: consecutive `threadIdx.x` → consecutive `row` (stride = width).

# %%
def stride_fast():
    return None  # TODO: address stride between t and t+1 in the FAST kernel

def stride_slow(width):
    return None  # TODO: ... in the SLOW (reversed) kernel, in terms of width

# %%
check("02.1 fast stride", stride_fast(), 1)
check_fn("02.1 slow stride", stride_slow, lambda w: w, [5000, 1024, 1])

# %% [markdown]
# ### Ex 02.3 — write the naive float SGEMM kernel (autograded)
# Fill the blanks in the kernel string. `threadIdx.x` MUST map to the column.

# %%
sgemm_naive_src = r"""
__global__ void sgemm_naive(const float* A, const float* B, float* C,
                            int M, int N, int K) {
    int row = /* TODO map threadIdx.y */ 0;
    int col = /* TODO map threadIdx.x */ 0;
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; k++)
        acc += /* TODO A[row][k] */ 0.0f * /* TODO B[k][col] */ 0.0f;
    C[row * N + col] = acc;
}
"""

# %%
launch_naive = ("dim3 block(32,32); "
                "dim3 grid((N+31)/32, (M+31)/32); "
                "sgemm_naive<<<grid, block>>>(dA, dB, dC, M, N, K);")
check_cuda("02.3 naive SGEMM correct", sgemm_naive_src, launch_naive, M=256, N=256, K=256)

# %% [markdown]
# Once it's green, see how it scales (and how little of the GPU it uses):

# %%
bench_cuda(sgemm_naive_src, launch_naive, sizes=(256, 512, 1024))
