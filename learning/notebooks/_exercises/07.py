# %% [markdown]
# ## ⬇ Your work — exercises & tests

# %% [markdown]
# ### Ex 07 — GFLOP/s and the column-major swap

# %%
def gflops(M, N, K, ms):
    return None  # TODO: 2*M*N*K FLOPs done in ms milliseconds, as GFLOP/s

# %%
check_fn("07 gflops", gflops,
         lambda M,N,K,ms: (2.0*M*N*K)/(ms*1e-3)/1e9,
         [(1024,1024,1024,2.0), (2048,2048,2048,5.0)])

# %%
# cuBLAS is column-major. To get row-major C = A·B you call it with A and B
# SWAPPED and pass N, M, K. Which argument goes first?
cublas_first_matrix = None  # TODO: "A" or "B" ?
check("07 column-major swap (first arg to cublasSgemm)", cublas_first_matrix, "B")

# %% [markdown]
# ### Capstone — autograde YOUR final kernel
# Paste your best kernel (from module 05 or 06, or written fresh from a blank
# cell) and its launch config. Get it green, then sweep sizes. This is your
# headline number.

# %%
my_final_src = r"""
// TODO: paste your best __global__ kernel here (e.g. sgemm_2d or sgemm_vec)
"""
my_final_launch = "/* TODO: your launch, e.g. dim3 block(...); dim3 grid(...); my_kernel<<<grid,block>>>(dA,dB,dC,M,N,K); */"

# %%
check_cuda("CAPSTONE — my kernel correct", my_final_src, my_final_launch, M=512, N=512, K=512)

# %%
bench_cuda(my_final_src, my_final_launch, sizes=(512, 1024, 2048, 4096))

# %% [markdown]
# ### cuBLAS head-to-head (run in your shell, not this cell)
# The harness here verifies correctness + measures *your* kernel. For the true
# ceiling, build the `bench.cu` from the markdown (module 07, Ex 07.1) with
# `nvcc ... -lcublas` and compare. Aim for 80–95% of cuBLAS — a kernel you
# understand line by line.
