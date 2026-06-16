# %% [markdown]
# ## ⬇ Your work — exercises & tests
#
# Compute each quantity from first principles, then run the test. These are the
# numbers that define your 3060's roofline. Numbers come from your `info.txt`.

# %% [markdown]
# ### Ex 01.1 — peak compute π (TFLOP/s)
# `sms` SMs × `cores_per_sm` FP32 cores × clock (GHz) × 2 (FMA = 2 FLOPs/cycle).

# %%
def peak_tflops(sms, cores_per_sm, ghz):
    return None  # TODO

# %%
_ref_pi = lambda s, c, g: s * c * g * 2 / 1000.0
check_fn("01.1 peak_tflops", peak_tflops, _ref_pi,
         [(28, 128, 1.78), (28, 128, 1.0), (1, 128, 1.0)])
approx("01.1 your 3060 peak (TFLOP/s)", peak_tflops(28, 128, 1.78), 12.7, rel=0.05)

# %% [markdown]
# ### Ex 01.2 — peak bandwidth β (GB/s)
# `mem_clock_MHz × 2 (DDR) × (bus_bits / 8)`, converted to GB/s.

# %%
def peak_bandwidth_GBs(mem_clock_MHz, bus_bits):
    return None  # TODO

# %%
_ref_bw = lambda f, b: f * 1e6 * 2 * (b / 8) / 1e9
check_fn("01.2 peak_bandwidth_GBs", peak_bandwidth_GBs, _ref_bw,
         [(7501, 192), (7000, 256), (1000, 128)])
approx("01.2 your 3060 bandwidth (GB/s)", peak_bandwidth_GBs(7501, 192), 360, rel=0.05)

# %% [markdown]
# ### Ex 01.3 — the ridge point I* = π / β  (FLOP/byte)

# %%
pi_flops  = peak_tflops(28, 128, 1.78) * 1e12      # FLOP/s
beta_byts = peak_bandwidth_GBs(7501, 192) * 1e9    # byte/s
ridge = None  # TODO: I* = pi / beta

# %%
approx("01.3 ridge point ≈ 35", ridge, 35, rel=0.1)

# %% [markdown]
# ### Ex 01.4 — arithmetic intensity of NAIVE matmul (FLOP/byte)
# It must NOT depend on N. The test feeds many N and checks both the value AND
# N-independence.

# %%
def arithmetic_intensity_naive(N):
    flops = None  # TODO: total FLOPs for NxN matmul
    bytes_dram = None  # TODO: total DRAM bytes read (float = 4 bytes)
    return flops / bytes_dram

# %%
check_fn("01.4 naive intensity (and N-independent)",
         arithmetic_intensity_naive, lambda N: 0.25, [64, 256, 1024, 4096])

# %% [markdown]
# ### Ex 01.5 — the verdict
# Best achievable GFLOP/s for naive = I × β. As a fraction of peak?

# %%
best_naive_gflops = None  # TODO: arithmetic_intensity_naive(1024) * beta (in GFLOP/s)
frac_of_peak = None       # TODO: best_naive_gflops / (pi in GFLOP/s)

# %%
approx("01.5 naive best GFLOP/s ≈ 90", best_naive_gflops, 90, rel=0.1)
check_true("01.5 naive uses < 1% of the GPU", frac_of_peak is not None and frac_of_peak < 0.01,
           "should be ~0.007 (0.7%)")
