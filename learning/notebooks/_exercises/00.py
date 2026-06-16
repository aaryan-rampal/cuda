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
