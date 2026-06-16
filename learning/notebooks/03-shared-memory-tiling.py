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
# # Module 03 — Shared-memory tiling (the first big jump)

# %% [markdown]
# ## Where this fits
#
# Module 01 proved the problem: naive matmul has arithmetic intensity 0.25,
# ~140× below your ridge point of 35, because it re-reads the same rows of `A` and
# columns of `B` from DRAM over and over. This module fixes it with the single
# most important optimization in all of GPU computing: **stage a block of data in
# on-chip shared memory once, then reuse it many times.** You'll roughly multiply
# your intensity — and your GFLOPS — by the tile size.

# %% [markdown]
# ## The idea
#
# Shared memory is a small (your 3060: 48 KB/block), fast, **software-managed
# cache** that all threads in a block share. It lives on the SM, ~100× lower
# latency than DRAM. The plan:
#
# 1. Chop the output `C` into `BK × BK` tiles. One block computes one output tile.
# 2. March along the `K` dimension in chunks of `BK`. At each step:
#    - **Cooperatively load** a `BK×BK` tile of `A` and a `BK×BK` tile of `B` from
#      global memory into shared memory (each thread loads a few elements).
#    - `__syncthreads()` so the whole tile is present.
#    - Each thread computes partial dot-products **using only shared memory**.
#    - `__syncthreads()` before overwriting the shared tiles next step.
# 3. After all `K`-chunks, each thread writes its `C` element once.
#
# Why it wins: in naive matmul, each element of `A` is read from DRAM `N` times
# (once per output column). In tiled matmul, each element is read from DRAM
# **once per tile-row it participates in**, i.e. `N/BK` times — a factor of `BK`
# fewer DRAM reads. Reuse moved from DRAM to shared memory.
#
# One output tile is the sum of products of tile-pairs marched along `K`:
#
# ```mermaid
# flowchart LR
#     subgraph A["A (rows of this block)"]
#         a0["A-tile @k0"]:::t
#         a1["A-tile @k1"]:::t
#         a2["..."]
#     end
#     subgraph B["B (cols of this block)"]
#         b0["B-tile @k0"]:::t
#         b1["B-tile @k1"]:::t
#         b2["..."]
#     end
#     a0 & b0 --> p0["As·Bs"]
#     a1 & b1 --> p1["As·Bs"]
#     p0 --> sum["C-tile += (accumulate over all K-chunks)"]
#     p1 --> sum
#     classDef t fill:#dde6ff
# ```
#
# And the per-chunk loop every block runs — the two barriers are not optional:
#
# ```mermaid
# flowchart TB
#     start(["for k0 in 0..K step BK"]) --> load["each thread loads 1 elem of As, 1 of Bs<br/>from DRAM → shared memory"]
#     load --> s1{{"__syncthreads()<br/>(tile fully loaded?)"}}
#     s1 --> comp["inner loop: acc += As[ty][k]·Bs[k][tx]<br/>(reads only shared memory)"]
#     comp --> s2{{"__syncthreads()<br/>(safe to overwrite tile?)"}}
#     s2 --> start
#     start -->|done| write["write acc → C once"]
# ```

# %% [markdown]
# ## Read this
#
# - NVIDIA blog, **"Using Shared Memory in CUDA C/C++"** —
#   https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/
# - PMPP 4th ed., **Ch. 5 "Memory architecture and data locality"** — the tiled
#   matmul is the worked example of the chapter. Read it; this module mirrors it.
# - Simon Boehm worklog, **"Kernel 3: Shared Memory Cache-Blocking."**

# %% [markdown]
# ## Warm-up
#
# 1. What two things does `__syncthreads()` guarantee, and what goes wrong if you
#    forget the *second* one (before overwriting the shared tile)?
# 2. Shared memory is per-**block**. Two different blocks computing two different
#    output tiles — do they share the same physical shared memory? Can block 5
#    read block 6's tile?
# 3. A `32×32` float tile is how many bytes? You need two of them (A-tile and
#    B-tile) resident. Does that fit in your 3060's 48 KB/block? How many such
#    blocks could fit per SM from the shared-memory limit alone?
#
# <details>
# <summary>Solutions</summary>
#
# 1. It's a **barrier**: (i) all threads in the block reach this point before any
#    proceed, and (ii) all shared-memory writes before it are visible to all
#    threads after it. Forget the barrier *before overwriting* the tile and a fast
#    thread starts loading the next `K`-chunk into shared memory while a slow
#    thread is still reading the previous chunk → **race**, wrong results.
# 2. No. Each block gets its **own** shared-memory allocation; blocks cannot see
#    each other's shared memory. That's why every block must re-load the tiles it
#    needs.
# 3. `32×32×4 = 4096` bytes per tile, two tiles = **8 KB**. Fits easily in 48 KB.
#    From shared memory alone: `48/8 = 6` blocks/SM — which happens to match the
#    thread-cap of 6 blocks (1536/256... wait, 32×32=1024 threads → only
#    `1536/1024 = 1` block/SM by threads!). Note the tension: 32×32 blocks give
#    great tiles but only 1 block/SM → low occupancy. Hold that thought for Ex
#    03.4.
#
# </details>

# %% [markdown]
# ## By hand — derive the intensity gain, then write the kernel
#
# **Exercise 03.1 — count the DRAM reads.** For `N×N` matmul with square tiles of
# side `BK`:
#
# - (a) How many output tiles are there? How many `K`-chunks does each block march
#   through?
# - (b) Per `K`-chunk, a block loads how many floats of `A` and `B` into shared
#   memory?
# - (c) Total floats read from DRAM across all blocks? Compare to naive's `2N³`.
#   By what factor did DRAM traffic drop?
# - (d) New arithmetic intensity `I = 2N³ FLOPs / (DRAM bytes)`? Express in terms
#   of `BK`. For `BK = 32`, what's `I`, and how does it compare to the ridge
#   point 35?
#
# **Exercise 03.2 — the index gymnastics (paper first).** This is the part people
# get wrong. A block computes output tile at block-row `cRow = blockIdx.y`,
# block-col `cCol = blockIdx.x`. Thread has local `(tx, ty)` in `0..BK-1`. For the
# `K`-chunk starting at `k0`:
#
# - Which **global** element of `A` does thread `(tx,ty)` load into `As[ty][tx]`?
#   (A-tile spans rows `cRow*BK .. cRow*BK+BK-1`, cols `k0 .. k0+BK-1`.)
# - Which **global** element of `B` does it load into `Bs[ty][tx]`? (B-tile spans
#   rows `k0 .. k0+BK-1`, cols `cCol*BK .. cCol*BK+BK-1`.)
# - Write the two flat indices into `A` and `B`.
#
# **Exercise 03.3 — write the kernel.** Add to `learning/kernels/sgemm.cu`. Fill
# the blanks yourself before peeking:
#
# ```c
# #define BK 32   // tile side == block side
#
# __global__ void sgemm_tiled(const float* A, const float* B, float* C,
#                             int M, int N, int K) {
#     __shared__ float As[BK][BK];
#     __shared__ float Bs[BK][BK];
#
#     int tx = threadIdx.x, ty = threadIdx.y;
#     int row = blockIdx.y * BK + ty;   // global output row
#     int col = blockIdx.x * BK + tx;   // global output col
#
#     float acc = 0.0f;
#     for (int k0 = 0; k0 < K; k0 += BK) {
#         // cooperative load: each thread brings ONE element of each tile
#         As[ty][tx] = A[ __________________ ];   // A[row][k0+tx]
#         Bs[ty][tx] = B[ __________________ ];   // B[k0+ty][col]
#         __syncthreads();
#
#         for (int k = 0; k < BK; k++)
#             acc += As[ty][k] * Bs[k][tx];       // dot-product within the tile
#         __syncthreads();                         // before reloading tiles
#     }
#     if (row < M && col < N)
#         C[row * N + col] = acc;
# }
# ```
#
# Two subtleties to handle and explain:
# - **Bounds for non-multiples of BK.** The load lines can read out of bounds when
#   `M,N,K` aren't multiples of `BK`. For now keep `M=N=K=1024` (multiple of 32)
#   so it's exact; in a comment, note how you'd guard (load 0.0f when the global
#   index is out of range). Real libraries pad.
# - **Why is `acc += As[ty][k] * Bs[k][tx]` coalescing-friendly in shared
#   memory?** Shared memory has no "coalescing," but it has **banks** (32 of
#   them). `Bs[k][tx]` across a warp (varying `tx`) hits 32 different banks → no
#   conflict. `As[ty][k]` is the same address for all `tx` in a row → broadcast.
#   Both good. (Module 06 returns to bank conflicts when the access pattern gets
#   trickier.)
#
# **Exercise 03.4 — measure, and confront occupancy.** Wire `sgemm_tiled` into
# your harness (warm-up + timed launch + GFLOPS print). Run at `N=1024`. You
# should see a solid multi-× speedup over naive.
#
# Then the twist from the warm-up: `BK=32` ⇒ `32×32=1024` threads/block ⇒ only
# **one block per SM** (1536 thread cap). Try `BK=16` (256 threads, up to 6
# blocks/SM) and compare. Which is faster on your 3060? Write down the GFLOPS for
# both. (There's no universal answer — it's the tile-size-vs-occupancy tradeoff,
# and measuring it *is* the lesson. Modules 04–05 break this tension by letting
# each thread do more work, so you get big tiles *and* small blocks.)
#
# <details>
# <summary>Solutions</summary>
#
# **03.1:**
# - (a) `(N/BK)²` output tiles; each block marches `N/BK` chunks.
# - (b) `BK²` floats of A + `BK²` of B = `2·BK²` per chunk.
# - (c) Total = `(N/BK)² tiles × (N/BK) chunks × 2BK² = 2N³/BK` floats. Naive was
#   `2N³`. **DRAM traffic dropped by a factor of `BK`.**
# - (d) `I = 2N³ FLOPs / (2N³/BK · 4 bytes) = BK/4 FLOP/byte`. For `BK=32`,
#   `I = 8`. Still below the ridge (35), but **32× better than naive's 0.25** —
#   you went from <1% toward double digits %. That's why modules 04–06 push `I`
#   further (you need ~35, and register tiling gets you past it).
#
# **03.2 / 03.3:** `As[ty][tx] = A[(row)*K + (k0+tx)] = A[(blockIdx.y*BK+ty)*K +
# k0+tx]`. `Bs[ty][tx] = B[(k0+ty)*N + col] = B[(k0+ty)*N + blockIdx.x*BK+tx]`.
#
# **03.4:** On a 3060, tiled at N=1024 typically lands roughly ~1.5–4 TFLOP/s
# depending on `BK` and occupancy — call it a 5–15× jump over naive. `BK=16`
# (more blocks/SM, better latency hiding) often *beats* `BK=32` here even though
# its intensity is lower, because 1 block/SM can't hide latency. That surprise is
# the point: intensity isn't the only lever — occupancy (latency hiding from
# module 00) matters too, and the next modules let you stop choosing between them.
#
# </details>

# %% [markdown]
# ## Check yourself
#
# You can (1) derive that tiling cuts DRAM traffic by `BK×` and raises intensity to
# `BK/4`, (2) write the cooperative-load + `__syncthreads` + inner-product pattern
# from memory, and (3) explain why a bigger tile isn't automatically faster
# (occupancy). That's the core of GPU performance engineering.
#
# → Next: [Module 04 — 1D block tiling](04-1d-blocktiling.md), where each thread
# computes *several* outputs so you raise intensity without shrinking occupancy.

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
