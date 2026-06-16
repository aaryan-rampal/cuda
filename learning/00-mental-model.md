# Module 00 — A precise mental model (grid / block / warp / SM)

## Where this fits

You've launched three kernels already: `vec_add` (1D), `mat_add_cuda` (2D), and
your naive `mat_mul`. They work. But "it works" and "I know exactly which
hardware does what" are different things, and every optimization from here is
about exploiting the second. This module makes the picture precise, using your
own RTX 3060 numbers. No new code — just nailing the model.

## The idea

There are two hierarchies, and you must hold both in your head at once:

**Software hierarchy** (what you write): `grid` → `block` → `thread`.
You choose these in `<<<grid, block>>>`.

**Hardware hierarchy** (what runs): `GPU` → `SM` (Streaming Multiprocessor) →
`warp` (32 threads) → `CUDA core`.

The mapping between them is the source of all performance intuition:

- A **block** is assigned to **one SM** and stays there until it finishes. A
  block never spans two SMs.
- An SM runs **many blocks at once** if it has the resources (registers, shared
  memory, thread slots). Your 3060: max 1536 threads/SM, 48 KB shared/block,
  65536 registers/SM, up to 16 blocks/SM.
- Inside an SM, threads run in **warps of 32**, in lockstep (SIMT). `threadIdx`
  0–31 are warp 0, 32–63 are warp 1, etc. (using the linearized thread index,
  with `x` fastest).
- The SM hides memory latency by **switching between warps**: while one warp
  waits on a memory load, another computes. This is why you launch *way* more
  threads than cores — latency hiding, not just parallelism. "Occupancy" =
  how many warps the SM has resident to switch between.

That last bullet is the deepest idea on the page. A GPU is not fast because it
does one thing quickly; it's fast because it always has another warp ready to
run while others wait on memory.

## Read this

- PMPP 4th ed., **Ch. 4 "Compute architecture and scheduling"** — the
  authoritative version of the above. If you only read one thing, read §4.1–4.5.
- NVIDIA Programming Guide, **§ "Hardware Implementation" (SIMT Architecture)**
  and **§ "Thread Hierarchy"** —
  https://docs.nvidia.com/cuda/cuda-c-programming-guide/

## Warm-up

Answer out loud or on paper before the spoiler.

1. Your naive matmul launches with `dim3 block(16, 16)`. How many threads is
   that per block? How many **warps** is that per block?
2. How are those 256 threads split into warps — i.e. which `(threadIdx.x,
   threadIdx.y)` pairs share a warp? (Hint: the linear index is
   `threadIdx.y * blockDim.x + threadIdx.x`.)
3. Your 3060 has 28 SMs and allows 1536 threads/SM. What's the max number of
   threads that can be *physically resident* across the whole GPU at once?
4. With `16×16 = 256`-thread blocks, how many such blocks fit on one SM if the
   only limit were the 1536-threads/SM cap? Does that beat the "16 blocks/SM"
   hardware cap?

<details>
<summary>Solutions</summary>

1. `16 × 16 = 256` threads = `256 / 32 = 8` warps per block.
2. With linear index `y*16 + x`: warp 0 is `y=0, x=0..15` **plus** `y=1,
   x=0..15` (indices 0–31). Warp 1 is `y=2..3`, all x. So a warp spans **two
   rows** of the 16-wide block. This matters later: a warp's 32 threads cover
   columns 0–15 of two adjacent output rows.
3. `28 × 1536 = 43,008` threads resident at once. (Your 6-digit "max grid"
   numbers are about *addressable* blocks, which is far larger — but only
   ~43 k threads actually execute at any instant; the rest are queued.)
4. `1536 / 256 = 6` blocks per SM by the thread cap. That's below the 16
   blocks/SM hardware cap, so **threads are the binding constraint** here, not
   block count. (Registers/shared memory can lower it further — module 06.)

</details>

## By hand

**Exercise 00.1 — index arithmetic without running it.** A block is
`dim3 block(16,16)`, grid is `dim3 grid(4, 4)`. For the thread with
`blockIdx=(2,1)`, `threadIdx=(3,5)`, compute the global `(row, col)` it handles
using *your own* convention from `matmul.cu`:
```
int i /*row*/ = threadIdx.y + blockIdx.y * blockDim.y;
int j /*col*/ = threadIdx.x + blockIdx.x * blockDim.x;
```

**Exercise 00.2 — draw the warp.** On paper, draw the 16×16 block as a grid of
cells. Shade the 32 threads that form **warp 0**. Then answer: do the 32 threads
of warp 0 access *consecutive* `col` values, or do they jump? You'll use this
exact picture in module 02.

**Exercise 00.3 — predict occupancy.** Each thread in your naive kernel uses
maybe ~30 registers (you'll measure the real number in module 06 with
`nvcc --ptxas-options=-v`). With 65536 registers/SM and 256-thread blocks, how
many blocks can reside per SM *from the register limit alone*? Is it above or
below the 6 from the thread cap?

<details>
<summary>Solutions</summary>

**00.1:** `i = 5 + 1*16 = 21`, `j = 3 + 2*16 = 35`. Output element `C[21][35]`.

**00.2:** Warp 0 = linear indices 0–31 = `(x=0..15, y=0)` and `(x=0..15, y=1)`.
Within each row the `col = x` values are consecutive (0..15). So the warp covers
two runs of 16 consecutive columns. Consecutive-`x` ⇒ consecutive-`col` ⇒
consecutive memory in a row-major matrix. Hold that thought.

**00.3:** Registers per block = `256 threads × 30 regs = 7680`.
`65536 / 7680 ≈ 8.5 → 8` blocks/SM from registers. That's *above* the 6 from
the thread cap, so registers aren't binding yet — the thread cap wins, 6
blocks = 1536 threads = full occupancy. Good. When you push registers to ~120
in module 05, recompute this and watch occupancy fall (that's the tradeoff).

</details>

## Check yourself before moving on

You should be able to answer, instantly: *"A block lives on one ___; threads run
in groups of ___ called ___; the GPU hides memory latency by ___."* If any blank
is fuzzy, reread PMPP Ch. 4.

→ Next: [Module 01 — The roofline](01-roofline.md), where you compute the single
number that explains why your naive matmul leaves ~98% of the GPU idle.
