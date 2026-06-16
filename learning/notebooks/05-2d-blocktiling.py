# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# > **Generated notebook.** Source of truth is the matching `.md` (edit that,
# > then re-run `_build.py`). Run cells in VS Code or JupyterLab; markdown +
# > Mermaid render there. Fill the `TODO`s in the code cells, run the test cell
# > below each, and iterate until you see ✅.

# %%
# bootstrap: make the autograder (checks.py) importable, wherever you launched from
import os, sys
for _c in [os.getcwd(),
           os.path.join(os.getcwd(), "learning", "notebooks"),
           os.path.dirname(os.path.abspath("__file__"))]:
    if os.path.exists(os.path.join(_c, "checks.py")) and _c not in sys.path:
        sys.path.insert(0, _c)
from checks import *  # check, approx, check_true, check_fn, check_cuda, bench_cuda
print("autograder loaded. nvcc available:", have_nvcc())

# %% [markdown]
# # Module 05 — 2D block tiling (the workhorse kernel)

# %% [markdown]
# ## Where this fits
#
# Module 04 had each thread compute a `TM×1` column. This module has each thread
# compute a `TM×TN` **rectangle** of outputs held entirely in registers. This is
# *the* kernel — the one that gets a hand-written SGEMM into the 60–80%-of-cuBLAS
# range on your 3060. Everything in module 06 is polish on top of this skeleton.

# %% [markdown]
# ## The idea
#
# Per `k` in the inner loop, load `TM` values of `A` and `TN` values of `B` from
# shared memory into registers, then do the full **outer product**: `TM×TN` FMAs.
#
# ```
# for (k = 0; k < BK; k++) {
#     for (i=0;i<TM;i++) regM[i] = As[...];     // TM shared loads
#     for (j=0;j<TN;j++) regN[j] = Bs[...];     // TN shared loads
#     for (i=0;i<TM;i++)
#       for (j=0;j<TN;j++)
#         acc[i][j] += regM[i] * regN[j];        // TM*TN FMAs
# }
# ```
#
# Count it: `TM + TN` shared loads produce `TM·TN` FMAs. The ratio of compute to
# shared-memory traffic is now `TM·TN / (TM+TN)`. For `TM=TN=8` that's `64/16 = 4`
# — each shared load feeds 4 FMAs, vs 1 in plain tiling. And by enlarging the
# block tile (`BM=BN=128`) relative to `BK=8`, you finally push **DRAM** intensity
# up toward the ridge point too.
#
# This is the outer-product / register-blocking formulation of GEMM that every
# high-performance library uses (including cuBLAS, just with more levels).
#
# ```mermaid
# flowchart TB
#     rm["regM[0..TM-1]<br/>TM loads from As"]
#     rn["regN[0..TN-1]<br/>TN loads from Bs"]
#     subgraph op["outer product → TM×TN FMAs per k (all in registers)"]
#         direction TB
#         grid["acc[i][j] += regM[i] · regN[j]<br/><br/>regN →  n0  n1  n2  ...  n7<br/>regM↓ m0 ▒▒ ▒▒ ▒▒ ... ▒▒<br/>      m1 ▒▒ ▒▒ ▒▒ ... ▒▒<br/>      ..  ▒▒ ▒▒ ▒▒ ... ▒▒<br/>      m7 ▒▒ ▒▒ ▒▒ ... ▒▒"]
#     end
#     rm --> grid
#     rn --> grid
# ```
#
# `TM+TN = 16` shared loads produce `TM·TN = 64` FMAs — that ratio (4) is the
# intensity win, and it's why the reused operands live in registers.

# %% [markdown]
# ## Read this
#
# - Simon Boehm worklog, **"Kernel 5: 2D Blocktiling"** + its register-blocking
#   diagram. This module is a guided re-derivation of that kernel.
# - (Optional, deeper) CUTLASS docs, **"Efficient GEMM in CUDA"** —
#   https://github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md —
#   the industrial version of the same hierarchy (block tile → warp tile → thread
#   tile). Skim for the picture.

# %% [markdown]
# ## Warm-up
#
# 1. With `BM=BN=128, BK=8, TM=TN=8`: outputs per block? threads per block
#    (`= BM·BN/(TM·TN)`)? warps? Under the 1024/block cap?
# 2. Each thread now holds `acc[TM][TN]` plus `regM[TM]` and `regN[TN]` in
#    registers. For `TM=TN=8`, that's how many float registers *just for those*?
#    Why is this the thing that will start to limit occupancy (recall module 00
#    Ex 00.3)?
# 3. Shared memory: `As` is `BM×BK`, `Bs` is `BK×BN`, both `128×8`. Total bytes?
#    Within 48 KB? How many blocks/SM from shared memory alone?
# 4. Compute-to-shared-load ratio for `TM=TN=8` vs `TM=TN=4`. Which has higher
#    arithmetic intensity? What's the cost of going bigger (hint: question 2)?
#
# <details>
# <summary>Solutions</summary>
#
# 1. `128×128 = 16384` outputs/block; `16384/64 = 256` threads = 8 warps; well
#    under 1024. (256-thread blocks → up to `1536/256 = 6` blocks/SM by threads —
#    *if* registers allow.)
# 2. `acc[8][8] = 64` + `regM[8]` + `regN[8]` = **80 registers** for accumulators
#    and operands alone (the compiler adds more for indices/addresses — you'll
#    measure ~100–130 total with `-v`). At ~128 regs/thread, `65536/(128·256) ≈ 2`
#    blocks/SM — occupancy drops. That's the central tradeoff of this kernel:
#    bigger `TM,TN` = more reuse but more registers = less occupancy. You tune it.
# 3. `128×8×4 × 2 = 8192 B = 8 KB`. Within 48 KB; `48/8 = 6` blocks/SM from shared
#    memory — so **registers, not shared memory, are the binding constraint here.**
# 4. `TM=TN=8 → 64/16 = 4`; `TM=TN=4 → 16/8 = 2`. The `8×8` thread tile has 2×
#    the compute-per-shared-load, but uses ~4× the accumulator registers, cutting
#    occupancy. The sweet spot is empirical (you'll sweep it in 05.4).
#
# </details>

# %% [markdown]
# ## By hand
#
# **Exercise 05.1 — the three reuse levels.** Fill the table (FMAs fed per load at
# each level), to cement why each module existed:
#
# | Kernel | shared loads → FMAs ratio | who holds the reused operand |
# |--------|---------------------------|------------------------------|
# | 03 tiled | 1 → 1 | shared memory |
# | 04 1D | 1 B-load → `TM` | register (`tmpB`) |
# | 05 2D | `TM+TN` loads → `TM·TN` | registers (`regM[]`, `regN[]`) |
#
# State the trend in one sentence.
#
# **Exercise 05.2 — loading a 128×8 tile with 256 threads.** Each thread must load
# `128·8/256 = 4` elements of `As` (and 4 of `Bs`). Design the load so it's
# **coalesced from DRAM** (consecutive threads read consecutive global addresses).
# A standard trick: index `As` load as `innerRow = tid / BK`, `innerCol = tid %
# BK` won't coalesce well for A (stride BK). Instead transpose `As` on load — but
# that's module 06. For now use the simplest correct mapping and note in a comment
# that the loads aren't yet optimally coalesced. (Boehm's kernel 5 loads `As`
# non-transposed; kernel 6 transposes + vectorizes.)
#
# **Exercise 05.3 — write the kernel.** This is the big one. Fill the blanks.
#
# ```c
# #define BM 128
# #define BN 128
# #define BK 8
# #define TM 8
# #define TN 8
# // launch: block = (BN/TN, BM/TM) = (16,16) = 256 threads; grid = (N/BN, M/BM)
#
# __global__ void sgemm_2d(const float* A, const float* B, float* C,
#                          int M, int N, int K) {
#     __shared__ float As[BM * BK];
#     __shared__ float Bs[BK * BN];
#
#     int cRow = blockIdx.y, cCol = blockIdx.x;
#     A += cRow * BM * K;
#     B += cCol * BN;
#     C += cRow * BM * N + cCol * BN;
#
#     // thread's tile within the 128x128 output block
#     int threadRow = threadIdx.y;   // 0..15  -> owns rows threadRow*TM .. +TM-1
#     int threadCol = threadIdx.x;   // 0..15  -> owns cols threadCol*TN .. +TN-1
#
#     // cooperative-load indices (each of 256 threads loads 4 of each tile)
#     int tid = threadRow * blockDim.x + threadCol;       // 0..255
#     int innerRowA = tid / BK,  innerColA = tid % BK;     // for As (128x8): 256 threads cover 256 of 1024 -> stride loop x4
#     int innerRowB = tid / BN,  innerColB = tid % BN;     // for Bs (8x128): 256 threads cover 256 of 1024 -> stride loop x4
#     const int strideA = (blockDim.x*blockDim.y) / BK;    // rows of As loaded per pass
#     const int strideB = (blockDim.x*blockDim.y) / BN;    // rows of Bs loaded per pass
#
#     float acc[TM][TN] = {0.0f};
#     float regM[TM], regN[TN];
#
#     for (int k0 = 0; k0 < K; k0 += BK) {
#         // load As (128x8) and Bs (8x128), 4 rows each per thread via stride loop
#         for (int o = 0; o < BM; o += strideA)
#             As[(innerRowA + o) * BK + innerColA] = A[ ____________ ];   // A[innerRowA+o][k0+innerColA]
#         for (int o = 0; o < BK; o += strideB)
#             Bs[(innerRowB + o) * BN + innerColB] = B[ ____________ ];   // B[k0+innerRowB+o][innerColB]
#         __syncthreads();
#         A += BK;
#         B += BK * N;
#
#         for (int k = 0; k < BK; k++) {
#             for (int i = 0; i < TM; i++) regM[i] = As[(threadRow*TM + i)*BK + k];
#             for (int j = 0; j < TN; j++) regN[j] = Bs[k*BN + (threadCol*TN + j)];
#             for (int i = 0; i < TM; i++)
#                 for (int j = 0; j < TN; j++)
#                     acc[i][j] += regM[i] * regN[j];        // <-- the outer product
#         }
#         __syncthreads();
#     }
#
#     for (int i = 0; i < TM; i++)
#         for (int j = 0; j < TN; j++)
#             C[(threadRow*TM + i)*N + (threadCol*TN + j)] = acc[i][j];
# }
# ```
#
# **Exercise 05.4 — measure and sweep.** Run at `N=1024` (and `N=2048` — bigger
# matrices amortize launch overhead and show the asymptotic GFLOPS better, since
# intensity arguments are about steady state). Then **sweep the tile params**:
# `(TM,TN) ∈ {(4,4),(8,8)}` and `(BM,BN) ∈ {(64,64),(128,128)}`. For each, also
# read the register count:
#
# ```bash
# nvcc -O3 -arch=native --ptxas-options=-v learning/kernels/sgemm.cu -o bin/sgemm
# # look for "Used XXX registers" per kernel in the compile output
# ```
#
# Record GFLOPS and registers/thread for each config. Find your 3060's sweet spot.
# You're doing real performance engineering now — the same loop a library author
# runs.
#
# <details>
# <summary>Solutions</summary>
#
# **05.1:** Loads: `A[(innerRowA+o)*K + (k0+innerColA)]`;
# `B[(k0+innerRowB+o)*N + innerColB]`. Trend: each level moves the reused operand
# into faster storage and raises FMAs-per-load — from shared (1:1) to registers
# (`TM`:1) to register outer-product (`TM·TN`:`TM+TN`).
#
# **05.4:** On a 3060, a clean 2D blocktiled kernel typically reaches **~8–11
# TFLOP/s** at N=2048 — i.e. **60–85% of the 12.7 TFLOP/s peak**, and often
# 60–80% of cuBLAS (which you'll confirm in module 07). `(8,8)` thread tiles with
# `128×128` block tiles is the usual winner *if* registers don't spill; if `-v`
# shows spills ("XXX bytes stack frame" / "spill stores"), back off to `(8,8)`
# with `64×64` or `(4,4)`. Watching register count gate occupancy gate GFLOPS is
# the whole craft.
#
# </details>

# %% [markdown]
# ## Check yourself
#
# You can derive the `TM·TN/(TM+TN)` reuse ratio, write the outer-product inner
# loop from memory, read register usage from `ptxas -v`, and explain the
# register↔occupancy↔intensity three-way tradeoff. This kernel alone is a strong
# interview-grade artifact.
#
# → Next: [Module 06 — Vectorize & finish](06-vectorize-and-finish.md): `float4`
# loads, transposed `As`, bank conflicts, occupancy — the last ~20% to cuBLAS.

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
