# CUDA Matmul, From First Principles — A Self-Paced Workbook

This is a learning workbook, not a tutorial you skim. It takes you from the
**naive matmul you already wrote** (`matmul.cu`) all the way to a
**register/warp-tiled SGEMM kernel** that lands in the same ballpark as cuBLAS
on your RTX 3060.

Every module is grounded in *your* code and *your* hardware (pulled from your
`info.txt`: NVIDIA GeForce RTX 3060, compute capability 8.6, 28 SMs, 1024
threads/block, 48 KB shared memory/block, warp size 32). The numbers you
compute will be the actual numbers for your GPU.

## How to use this (read this part — it's the whole point)

Each module has the same shape:

1. **Where this fits** — one paragraph connecting it to what you already built.
2. **The idea** — the concept, short.
3. **Read this** — one or two sources. Read them *before* the exercises.
4. **Warm-up** — quick questions to check you understood the reading.
5. **By hand** — derive the math and write the code yourself. Pen and paper
   for the math, your editor for the code. This is where the learning happens.
6. **Build & measure** — compile it, run it, look at the timing.
7. **Solutions** — behind `<details>` spoilers.

Most modules include **Mermaid diagrams** (the ```` ```mermaid ```` blocks).
GitHub renders these inline when you view the file in the web UI — so read these
on GitHub (or any Mermaid-aware Markdown viewer / VS Code with a Mermaid
extension) to get the pictures, not just the code fences.

**The one rule:** write your answer *before* you open the spoiler. Every time.
You already proved this works — you discovered memory coalescing on your own in
`matrix_add.cu` (your "normal" 1.54 ms vs "reversed" 2.72 ms experiment).
That stuck because you found it by hand. Do the same here.

## The map

| #  | Module | What you'll build | You've already... |
|----|--------|-------------------|-------------------|
| 00 | [Mental model](00-mental-model.md) | A precise picture of grid/block/warp/SM on *your* 3060 | written 3 working kernels |
| 01 | [The roofline](01-roofline.md) | The number that proves naive matmul is doomed | — |
| 02 | [Coalescing](02-coalescing.md) | Proof your naive matmul is *already* coalesced (and why `matrix_add` reversed wasn't) | discovered this empirically |
| 03 | [Shared-memory tiling](03-shared-memory-tiling.md) | Kernel that reuses data from on-chip memory | — |
| 04 | [1D block tiling](04-1d-blocktiling.md) | Each thread computes a *column* of results | — |
| 05 | [2D block tiling](05-2d-blocktiling.md) | Each thread computes a *tile* of results in registers | — |
| 06 | [Vectorize & finish](06-vectorize-and-finish.md) | `float4` loads, no bank conflicts, tuned occupancy | — |
| 07 | [Benchmark vs cuBLAS](07-benchmark-and-cublas.md) | A GFLOPS number next to cuBLAS's | wrote a timing/verify harness |

Work them in order. Each one stacks on the last — that's the whole design.

## One setup decision before you start: `int` → `float`

Your current `matmul.cu` works in `int`. For the optimization journey we switch
to `float` (this is "SGEMM" — Single-precision GEneral Matrix Multiply). Three
reasons, all of which you'll feel later:

- Performance is measured in **FLOPS** (floating-point ops/sec). That's the
  vocabulary of every source you'll read and of cuBLAS.
- The fast-load trick in module 06 (`float4`, a 128-bit load) is a float thing.
- cuBLAS's `cublasSgemm` (module 07) is float, so to compare we must match.

So: keep `matmul.cu` as your trophy (the naive `int` version that works). Each
module builds a new file `learning/kernels/sgemm.cu` that grows over time. The
first thing module 02 has you do is port your naive kernel to `float`.

One float caveat you'll need: never check `c == expected` with `==`. Use a
tolerance, e.g. `fabs(c - expected) < 1e-2f * fabs(expected)`. You'll see why
in module 01.

## Prereqs / build

You already have `nvcc` and a `justfile`. For these modules:

```bash
# compile one kernel file (your justfile already does -O3 -arch=native)
nvcc -O3 -arch=native learning/kernels/sgemm.cu -o bin/sgemm && ./bin/sgemm
# module 07 also needs cuBLAS:
nvcc -O3 -arch=native learning/kernels/bench.cu -o bin/bench -lcublas && ./bin/bench
```

Reference sources used throughout (bookmark these — they're the canon):

- **Simon Boehm, "How to Optimize a CUDA Matmul Kernel for cuBLAS-like
  Performance: a Worklog"** — https://siboehm.com/articles/22/CUDA-MMM
  (this workbook follows roughly the same ladder, but makes you derive it)
- **NVIDIA CUDA C++ Programming Guide** —
  https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- **NVIDIA CUDA C++ Best Practices Guide** —
  https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- **PMPP**: *Programming Massively Parallel Processors* (Hwu, Kirk, El Hajj),
  4th ed. — the textbook; chapters noted per module.

If a link ever rots, search the title — these are all stable, well-known docs.

Now go to [module 00](00-mental-model.md).
