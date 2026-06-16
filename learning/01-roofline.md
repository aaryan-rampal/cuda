# Module 01 — The roofline: why naive matmul is doomed

## Where this fits

This is the "statistical bias in estimators" of CUDA — the first-principles
foundation that everything else is a reaction to. Before optimizing anything,
you compute *one ratio* that tells you whether your kernel is starved for
arithmetic or starved for memory. For naive matmul the answer is brutal, and it
explains every single optimization in modules 03–06. Skip this and the rest is
cargo-culting.

## The idea

Every kernel does two things: **arithmetic** (FLOPs) and **memory traffic**
(bytes moved to/from global memory / DRAM). The hardware has a ceiling on each:

- **Peak compute** `π` (FLOP/s) — how fast the cores can do FMAs.
- **Peak bandwidth** `β` (byte/s) — how fast DRAM can feed them.

A kernel's **arithmetic intensity** is

```
I = (FLOPs performed) / (bytes moved from DRAM)      [FLOP/byte]
```

The **roofline model** says your achievable performance is bounded by:

```
P ≤ min( π ,  I × β )
```

- If `I` is small → `I×β` is the binding term → **memory-bound**. You're
  limited by how fast DRAM feeds you; the cores idle.
- If `I` is large → `π` is the binding term → **compute-bound**. The cores are
  the bottleneck; you're using the machine well.

The crossover is the **ridge point** `I* = π / β`. To use a GPU well you need
`I ≥ I*`. **All of matmul optimization is one move: raise `I` until you cross
the ridge** — and you raise it by reusing each byte you load many times instead
of re-fetching it from DRAM.

## Read this

- **Williams, Waterman, Patterson (2009), "Roofline: an insightful visual
  performance model for multicore architectures."** The original paper; read
  the first 3 pages. (Search the title — it's a free PDF everywhere.)
- Simon Boehm's worklog intro, the "lower bounding the fastest possible runtime"
  section — https://siboehm.com/articles/22/CUDA-MMM

## Warm-up

1. In one sentence: what does it mean for a kernel to be "memory-bound"?
2. If kernel A has `I = 0.25` and kernel B has `I = 40`, and your ridge point is
   `I* = 35`, which one is leaving the GPU mostly idle?
3. True/false: making the arithmetic faster (e.g. fancier FMAs) helps a
   memory-bound kernel. Why or why not?

<details>
<summary>Solutions</summary>

1. DRAM can't deliver bytes fast enough to keep the cores busy; performance is
   set by `I×β`, not by `π`.
2. A (`0.25 ≪ 35`) is deeply memory-bound and idles the cores. B (`40 > 35`) is
   compute-bound and uses the machine.
3. **False.** If you're memory-bound, the cores already wait on memory; making
   them faster widens the wait. You must raise `I` (reuse data) or `β`
   (can't — it's fixed hardware). This is *the* reason tiling exists.

</details>

## By hand — compute your 3060's roofline

Use the real numbers from your `info.txt` and the GA106 (RTX 3060) spec.

**Exercise 01.1 — peak compute `π`.** Your 3060 has 28 SMs, each with 128 FP32
cores, boosting to ~1.78 GHz (your `info.txt` clock rate is 1,777,000 kHz). Each
core does a fused multiply-add = 2 FLOPs per cycle. Compute `π` in TFLOP/s.

**Exercise 01.2 — peak bandwidth `β`.** Your `info.txt`: memory clock
7501 MHz, bus width 192 bits. GDDR6 transfers 2 bits per pin per clock cycle
(DDR). Bandwidth = `mem_clock × 2 × (bus_bits / 8)`. Compute `β` in GB/s.

**Exercise 01.3 — the ridge point.** `I* = π / β`. Compute it (FLOP/byte). This
is the arithmetic intensity you must reach to stop being memory-bound.

**Exercise 01.4 — arithmetic intensity of NAIVE matmul.** For `C = A·B` with all
matrices `N×N`, in your naive kernel each output element loops `k = 0..N-1`
reading one element of `A` and one of `B` from global memory each iteration, and
does one multiply + one add.

- (a) Total FLOPs for the whole matmul?
- (b) Total *element* reads from global memory (assume no cache reuse — the
  pessimistic naive model)? Convert to bytes (`float` = 4 bytes).
- (c) `I = FLOPs / bytes`. Notice it does **not** depend on `N`. What number do
  you get?

**Exercise 01.5 — the verdict.** Compare your `I` from 01.4 to `I*` from 01.3.
By what factor is naive matmul below the ridge? Then estimate the best speed
naive matmul can reach: `P ≤ I × β`. As a fraction of peak `π`, how much of your
GPU is naive matmul able to use, *at best*?

<details>
<summary>Solutions</summary>

**01.1:** `π = 28 × 128 × 1.78e9 × 2 = 1.275e13 ≈ 12.7 TFLOP/s`. (Matches the
3060's advertised ~12.7 TFLOPS FP32.)

**01.2:** `β = 7.501e9 × 2 × (192/8) = 7.501e9 × 2 × 24 = 3.60e11 ≈ 360 GB/s`.
(Matches the 3060's advertised 360 GB/s.)

**01.3:** `I* = 12.7e12 / 360e9 ≈ 35 FLOP/byte`. You must reach ~35 FLOP/byte to
become compute-bound on this card. Remember this number.

**01.4:**
- (a) Each output does `N` mults + `N` adds = `2N` FLOPs; there are `N²` outputs
  → **`2N³` FLOPs**.
- (b) Each output reads `2N` elements (one A, one B per `k`) → `2N³` element
  reads → `8N³` bytes.
- (c) `I = 2N³ / 8N³ = 0.25 FLOP/byte`. **N cancels** — naive matmul has a
  *constant, tiny* intensity no matter the size.

**01.5:** `0.25` vs `35` → naive is **~140× below the ridge**. Best case
`P ≤ I×β = 0.25 × 360e9 = 90 GFLOP/s`. As a fraction of peak:
`90 / 12700 ≈ 0.7%`. **Your naive matmul can use, at most, under 1% of the
GPU.** That's not a tuning problem — it's an arithmetic-intensity problem, and
the only fix is reuse. That is what modules 03–06 do.

</details>

## The float caveat (promised in the README)

Notice 01.4 assumed `float`. Your current `matmul.cu` uses `int` and verifies
with `==`. Once you go float, `A[i][k]*B[k][j]` summed over `k` accumulates
rounding error, so two *correct* implementations can differ in the last digits.
That's why the README said to verify with a *relative tolerance*
(`fabs(c-expected) < 1e-2f*fabs(expected)`), not `==`. Keep this in your harness
from module 02 onward.

## Check yourself

You can now state the thesis of this whole workbook in one sentence: *"Naive
matmul reaches <1% of peak because its arithmetic intensity (0.25) is ~140×
below the ridge point (35); every optimization that follows raises arithmetic
intensity by reusing loaded data."* If you can say that cold, go on.

→ Next: [Module 02 — Coalescing](02-coalescing.md), where you prove your naive
kernel already got *one* thing right — and connect it to the experiment you ran
in `matrix_add.cu`.
