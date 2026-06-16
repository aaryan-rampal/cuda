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
# # Module 00 — A precise mental model (grid / block / warp / SM)

# %% [markdown]
# ## Where this fits
#
# You've launched three kernels already: `vec_add` (1D), `mat_add_cuda` (2D), and
# your naive `mat_mul`. They work. But "it works" and "I know exactly which
# hardware does what" are different things, and every optimization from here is
# about exploiting the second. This module makes the picture precise, using your
# own RTX 3060 numbers. No new code — just nailing the model.

# %% [markdown]
# ## The idea
#
# There are two hierarchies, and you must hold both in your head at once:
#
# **Software hierarchy** (what you write): `grid` → `block` → `thread`.
# You choose these in `<<<grid, block>>>`.
#
# **Hardware hierarchy** (what runs): `GPU` → `SM` (Streaming Multiprocessor) →
# `warp` (32 threads) → `CUDA core`.
#
# ```mermaid
# flowchart LR
#     subgraph SW["What you write (software)"]
#         G["grid"] --> B["block"] --> T["thread"]
#     end
#     subgraph HW["What runs (hardware) — your RTX 3060"]
#         GPU["GPU<br/>28 SMs"] --> SM["SM<br/>≤1536 threads<br/>≤16 blocks"]
#         SM --> W["warp<br/>32 threads, lockstep"]
#         W --> C["CUDA core"]
#     end
#     B -. "a block is placed on exactly one SM<br/>(an SM holds many blocks)" .-> SM
#     T -. "32 threads = 1 warp" .-> W
# ```
#
# The mapping between them is the source of all performance intuition:
#
# - A **block** is assigned to **one SM** and stays there until it finishes. A
#   block never spans two SMs.
# - An SM runs **many blocks at once** if it has the resources (registers, shared
#   memory, thread slots). Your 3060: max 1536 threads/SM, 48 KB shared/block,
#   65536 registers/SM, up to 16 blocks/SM.
# - Inside an SM, threads run in **warps of 32**, in lockstep (SIMT). `threadIdx`
#   0–31 are warp 0, 32–63 are warp 1, etc. (using the linearized thread index,
#   with `x` fastest).
# - The SM hides memory latency by **switching between warps**: while one warp
#   waits on a memory load, another computes. This is why you launch *way* more
#   threads than cores — latency hiding, not just parallelism. "Occupancy" =
#   how many warps the SM has resident to switch between.
#
# That last bullet is the deepest idea on the page. A GPU is not fast because it
# does one thing quickly; it's fast because it always has another warp ready to
# run while others wait on memory.

# %% [markdown]
# ## Read this
#
# - PMPP 4th ed., **Ch. 4 "Compute architecture and scheduling"** — the
#   authoritative version of the above. If you only read one thing, read §4.1–4.5.
# - NVIDIA Programming Guide, **§ "Hardware Implementation" (SIMT Architecture)**
#   and **§ "Thread Hierarchy"** —
#   https://docs.nvidia.com/cuda/cuda-c-programming-guide/

# %% [markdown]
# ## Warm-up
#
# Answer out loud or on paper before the spoiler.
#
# 1. Your naive matmul launches with `dim3 block(16, 16)`. How many threads is
#    that per block? How many **warps** is that per block?
# 2. How are those 256 threads split into warps — i.e. which `(threadIdx.x,
#    threadIdx.y)` pairs share a warp? (Hint: the linear index is
#    `threadIdx.y * blockDim.x + threadIdx.x`.)
# 3. Your 3060 has 28 SMs and allows 1536 threads/SM. What's the max number of
#    threads that can be *physically resident* across the whole GPU at once?
# 4. With `16×16 = 256`-thread blocks, how many such blocks fit on one SM if the
#    only limit were the 1536-threads/SM cap? Does that beat the "16 blocks/SM"
#    hardware cap?
#
# <details>
# <summary>Solutions</summary>
#
# 1. `16 × 16 = 256` threads = `256 / 32 = 8` warps per block.
# 2. With linear index `y*16 + x`: warp 0 is `y=0, x=0..15` **plus** `y=1,
#    x=0..15` (indices 0–31). Warp 1 is `y=2..3`, all x. So a warp spans **two
#    rows** of the 16-wide block. This matters later: a warp's 32 threads cover
#    columns 0–15 of two adjacent output rows.
# 3. `28 × 1536 = 43,008` threads resident at once. (Your 6-digit "max grid"
#    numbers are about *addressable* blocks, which is far larger — but only
#    ~43 k threads actually execute at any instant; the rest are queued.)
# 4. `1536 / 256 = 6` blocks per SM by the thread cap. That's below the 16
#    blocks/SM hardware cap, so **threads are the binding constraint** here, not
#    block count. (Registers/shared memory can lower it further — module 06.)
#
# </details>

# %% [markdown]
# ## By hand
#
# **Exercise 00.1 — index arithmetic without running it.** A block is
# `dim3 block(16,16)`, grid is `dim3 grid(4, 4)`. For the thread with
# `blockIdx=(2,1)`, `threadIdx=(3,5)`, compute the global `(row, col)` it handles
# using *your own* convention from `matmul.cu`:
# ```
# int i /*row*/ = threadIdx.y + blockIdx.y * blockDim.y;
# int j /*col*/ = threadIdx.x + blockIdx.x * blockDim.x;
# ```
#
# **Exercise 00.2 — draw the warp.** On paper, draw the 16×16 block as a grid of
# cells. Shade the 32 threads that form **warp 0**. Then answer: do the 32 threads
# of warp 0 access *consecutive* `col` values, or do they jump? You'll use this
# exact picture in module 02. Here's the layout to reproduce and reason about
# (linear id `= y*16 + x`, so warp 0 = ids 0–31 = rows y=0 and y=1):
#
# ```mermaid
# flowchart TB
#     subgraph block["16×16 block — each row is 16 threads (x = 0..15)"]
#         direction TB
#         r0["y=0 :  x=0..15   → linear ids 0..15  ─┐ warp 0"]
#         r1["y=1 :  x=0..15   → linear ids 16..31 ─┘ (32 threads)"]
#         r2["y=2 :  x=0..15   → linear ids 32..47  ─┐ warp 1"]
#         r3["y=3 :  x=0..15   → linear ids 48..63  ─┘"]
#         rdots["... y=4..15 → warps 2..7"]
#     end
#     r0 --- r1 --- r2 --- r3 --- rdots
# ```
#
# Key takeaway from the picture: within each row, `x` (and therefore `col`) is
# **consecutive** — that's the seed of memory coalescing in module 02.
#
# **Exercise 00.3 — predict occupancy.** Each thread in your naive kernel uses
# maybe ~30 registers (you'll measure the real number in module 06 with
# `nvcc --ptxas-options=-v`). With 65536 registers/SM and 256-thread blocks, how
# many blocks can reside per SM *from the register limit alone*? Is it above or
# below the 6 from the thread cap?
#
# <details>
# <summary>Solutions</summary>
#
# **00.1:** `i = 5 + 1*16 = 21`, `j = 3 + 2*16 = 35`. Output element `C[21][35]`.
#
# **00.2:** Warp 0 = linear indices 0–31 = `(x=0..15, y=0)` and `(x=0..15, y=1)`.
# Within each row the `col = x` values are consecutive (0..15). So the warp covers
# two runs of 16 consecutive columns. Consecutive-`x` ⇒ consecutive-`col` ⇒
# consecutive memory in a row-major matrix. Hold that thought.
#
# **00.3:** Registers per block = `256 threads × 30 regs = 7680`.
# `65536 / 7680 ≈ 8.5 → 8` blocks/SM from registers. That's *above* the 6 from
# the thread cap, so registers aren't binding yet — the thread cap wins, 6
# blocks = 1536 threads = full occupancy. Good. When you push registers to ~120
# in module 05, recompute this and watch occupancy fall (that's the tradeoff).
#
# </details>

# %% [markdown]
# ## Check yourself before moving on
#
# You should be able to answer, instantly: *"A block lives on one ___; threads run
# in groups of ___ called ___; the GPU hides memory latency by ___."* If any blank
# is fuzzy, reread PMPP Ch. 4.
#
# → Next: [Module 01 — The roofline](01-roofline.md), where you compute the single
# number that explains why your naive matmul leaves ~98% of the GPU idle.

# %% [markdown]
# ## ⬇ Your work — exercises & tests
#
# Fill in each `TODO`, then run the test cell right below it. ✅ = on track,
# ❌ = the test tells you what's off. Iterate until green. (Don't peek at the
# `<details>` solutions above until your test passes.)

# %% [markdown]
# ### Ex 00.1 — global (row, col) from block/thread indices
# Using *your* convention from `matmul.cu` (`threadIdx.x → col`).

# %%
def global_row_col(bx, by, tx, ty, bdx, bdy):
    """Return (row, col) for a thread. bd* = blockDim."""
    row = None  # TODO
    col = None  # TODO
    return row, col

# %%
_ref_rc = lambda bx, by, tx, ty, bdx, bdy: (ty + by*bdy, tx + bx*bdx)
check_fn("00.1 global_row_col", global_row_col, _ref_rc,
         [(2,1,3,5,16,16), (0,0,0,0,16,16), (4,4,15,15,16,16), (3,2,1,7,32,8)])

# %% [markdown]
# ### Ex 00.2 — which warp does a thread belong to?
# Linear id `= ty*blockDim.x + tx`; a warp is 32 consecutive linear ids.

# %%
def warp_id(tx, ty, bdx):
    linear = None  # TODO: linear id
    return None    # TODO: which warp (0,1,2,...)?

# %%
_ref_warp = lambda tx, ty, bdx: (ty*bdx + tx) // 32
check_fn("00.2 warp_id", warp_id, _ref_warp,
         [(0,0,16), (15,1,16), (0,2,16), (5,3,16), (31,0,32)])

# %% [markdown]
# ### Ex 00.3 — occupancy from the thread cap
# How many blocks fit on one SM if threads/SM is the only limit?

# %%
def blocks_per_sm_by_threads(threads_per_block, max_threads_per_sm=1536):
    return None  # TODO

# %%
_ref_occ = lambda t, m=1536: m // t
check_fn("00.3 blocks_per_sm", blocks_per_sm_by_threads, _ref_occ,
         [(256,), (1024,), (512,), (128,)])
check("00.3 your 16x16 block (256 threads) -> blocks/SM",
      blocks_per_sm_by_threads(256), 6)
