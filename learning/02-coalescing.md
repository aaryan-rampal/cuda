# Module 02 — Memory coalescing (you already discovered this)

## Where this fits

You ran the experiment in `matrix_add.cu`: same kernel, two thread-to-data
mappings, and you measured **1.54 ms vs 2.72 ms** on your 3060. You wrote in the
comments "notice the difference in the kernel times." That difference *is*
memory coalescing. This module makes you explain *why*, then proves your naive
`matmul` already got the mapping right — and ports it to `float` so the rest of
the workbook has a clean starting kernel.

## The idea

When a warp (32 threads) issues a global-memory load, the hardware services it
in **memory transactions** of a fixed width (a 32-byte sector; a 128-byte cache
line is 4 sectors). The cost is *the number of transactions*, not the number of
threads.

- If the 32 threads of a warp read **32 consecutive `float`s** (128 bytes), the
  hardware coalesces them into **one** 128-byte transaction. Full efficiency.
- If those 32 threads read addresses that are **strided** (e.g. each thread one
  full row apart), the warp needs **up to 32 separate transactions**, each
  delivering one useful float and wasting the rest of the sector. Up to ~32×
  more memory traffic for the *same* data.

The rule: **make consecutive threads in a warp (consecutive `threadIdx.x`) touch
consecutive addresses.**

Now recall module 00: a warp is consecutive linear indices = consecutive
`threadIdx.x` (x is fastest). And row-major storage means address increases with
*column*. So: **`threadIdx.x` should map to the column**, so a warp sweeps a
contiguous row. That's the whole rule, and it's exactly what your `matrix_add`
experiment measured.

## Read this

- NVIDIA blog, **"How to Access Global Memory Efficiently in CUDA C/C++
  Kernels"** —
  https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/
- NVIDIA Best Practices Guide, **§"Coalesced Access to Global Memory."**

## Warm-up — explain your own experiment

Open `matrix_add.cu`. The two kernels differ only here:

```c
// mat_add_cuda (the FAST one, 1.54 ms):
int col = threadIdx.x + blockDim.x * blockIdx.x;
int row = threadIdx.y + blockDim.y * blockIdx.y;
int i = row * width + col;

// mat_add_cuda_reversed (the SLOW one, 2.72 ms):
int row = threadIdx.x + blockDim.x * blockIdx.x;   // x -> row
int col = threadIdx.y + blockDim.y * blockIdx.y;   // y -> col
int i = row * width + col;
```

1. In the **fast** kernel, two threads with `threadIdx.x` = 5 and 6 (same warp,
   same `y`) access indices `i` and `i+?`. What's the stride between them?
2. In the **slow** kernel, the same two threads access `i` and `i+?`. What's the
   stride now (in terms of `width`)?
3. `width = 5000`. So in the slow kernel, consecutive threads of a warp are how
   many floats apart? How many distinct 32-byte sectors does one warp touch?
4. Explain the 1.54 → 2.72 ms slowdown in one sentence using the word
   "transaction."

<details>
<summary>Solutions</summary>

1. Fast: `col` differs by 1, `i = row*width + col` differs by **1** → adjacent
   floats → one coalesced transaction per warp.
2. Slow: `row` differs by 1, so `i` differs by **`width`** → a 5000-float jump.
3. 5000 floats apart = 20,000 bytes apart. Each thread lands in its **own**
   32-byte sector → up to **32 transactions** for one warp's load (vs 1).
4. The reversed kernel makes each warp's load scatter across ~32 sectors instead
   of 1, so it pays many more memory transactions to move the same bytes — that
   extra traffic is the ~1.2 ms. (It's not a clean 32× because the L2 cache
   recovers some of the wasted sectors on later warps — but it's a big hit.)

</details>

## By hand — prove your matmul is already coalesced, then port it

**Exercise 02.1 — audit your naive matmul kernel.** Here's your kernel from
`matmul.cu`:

```c
int i /*row*/ = threadIdx.y + blockIdx.y * blockDim.y;
int j /*col*/ = threadIdx.x + blockIdx.x * blockDim.x;
for (int k = 0; k < a_2; k++)
    C[i*a_3 + j] += A[i*a_2 + k] * B[k*a_3 + j];
```

For a single warp (fix `k`, vary `threadIdx.x` ⇒ vary `j` by 1 across 32
threads), classify each of the three accesses as **coalesced**, **broadcast**
(all threads read the *same* address), or **strided/bad**:

- (a) `C[i*a_3 + j]`
- (b) `B[k*a_3 + j]`
- (c) `A[i*a_2 + k]`

Then state the verdict: is your naive matmul's memory pattern good or bad?

**Exercise 02.2 — break it on purpose (paper only).** What single change to the
kernel would turn it into the "reversed" disaster — i.e. make `j` depend on
`threadIdx.y` and `i` on `threadIdx.x`? Why would that be ~10–30× slower per
your 3060's sector width? (You don't have to run it; you already ran the
equivalent in `matrix_add.cu`.)

**Exercise 02.3 — port to float (write the code).** Create
`learning/kernels/sgemm.cu`. Port your naive matmul to `float`, with a clean
harness you'll reuse for every later kernel. Fill in the blanks yourself first:

```c
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); } \
} while (0)

// C[MxN] = A[MxK] * B[KxN], row-major, all float
__global__ void sgemm_naive(const float* A, const float* B, float* C,
                            int M, int N, int K) {
    int row = ____________________;   // map threadIdx.y
    int col = ____________________;   // map threadIdx.x  (which one must be x?)
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; k++)
        acc += ______________ * ______________;   // A[row][k] * B[k][col], flat-indexed
    C[row * N + col] = acc;            // why accumulate locally then write once? (see note)
}

static bool verify(const float* A, const float* B, const float* C,
                   int M, int N, int K) {
    for (int i = 0; i < M; i++)
      for (int j = 0; j < N; j++) {
        double expected = 0.0;
        for (int k = 0; k < K; k++) expected += (double)A[i*K+k] * B[k*N+j];
        float got = C[i*N+j];
        if (fabs(got - expected) > 1e-2 * fabs(expected) + 1e-3) {
            fprintf(stderr, "mismatch at (%d,%d): got %f want %f\n", i,j,got,expected);
            return false;
        }
      }
    return true;
}

int main() {
    int M = 1024, N = 1024, K = 1024;      // start square; tune later
    size_t bytesA = (size_t)M*K*sizeof(float);
    size_t bytesB = (size_t)K*N*sizeof(float);
    size_t bytesC = (size_t)M*N*sizeof(float);
    std::vector<float> A(M*K), B(K*N), C(M*N, 0.0f);
    for (auto& x : A) x = (float)(rand() % 10) - 4.5f;   // small spread, includes negatives
    for (auto& x : B) x = (float)(rand() % 10) - 4.5f;

    float *dA,*dB,*dC;
    CHECK_CUDA(cudaMalloc(&dA, bytesA));
    CHECK_CUDA(cudaMalloc(&dB, bytesB));
    CHECK_CUDA(cudaMalloc(&dC, bytesC));
    CHECK_CUDA(cudaMemcpy(dA, A.data(), bytesA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, B.data(), bytesB, cudaMemcpyHostToDevice));

    dim3 block(32, 32);
    dim3 grid((N + 31)/32, (M + 31)/32);   // note: grid.x covers N (cols), grid.y covers M (rows)

    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    // warm-up launch (first launch includes one-time init; never time it)
    sgemm_naive<<<grid, block>>>(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEventRecord(s);
    sgemm_naive<<<grid, block>>>(dA, dB, dC, M, N, K);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms = 0; cudaEventElapsedTime(&ms, s, e);

    CHECK_CUDA(cudaMemcpy(C.data(), dC, bytesC, cudaMemcpyDeviceToHost));
    double gflops = (2.0*M*N*K) / (ms*1e-3) / 1e9;
    printf("naive: %.3f ms   %.1f GFLOP/s   %s\n",
           ms, gflops, verify(A.data(),B.data(),C.data(),M,N,K) ? "OK" : "FAIL");

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}
```

Three things to notice and be able to explain (they recur every module):
- **Why `dim3 block(32,32)`?** 32 = warp width, so `threadIdx.x` 0–31 is exactly
  one warp sweeping one contiguous row. `32×32 = 1024` = your max threads/block.
- **Why accumulate into `acc` then write `C` once** instead of `C[...] +=` in the
  loop like your `int` kernel did? Your `int` kernel did `C[idx] += ...` *N
  times* — that's N read-modify-writes to global memory per output. The
  `acc`-in-register version writes once. (Also: your `int` kernel relied on `C`
  being pre-zeroed via `cudaMemcpy`; `acc=0` removes that dependency.)
- **Why the warm-up launch before timing?** The first kernel launch pays
  one-time context/JIT costs. Time the *second* one.

**Exercise 02.4 — run it.** Compile and run:
```bash
nvcc -O3 -arch=native learning/kernels/sgemm.cu -o bin/sgemm && ./bin/sgemm
```
Record your GFLOP/s. Compare to the **~90 GFLOP/s ceiling** you derived in module
01 (Ex 01.5). Are you near it? (You should be in the same order of magnitude —
naive is memory-bound and roughly hits its sad little roofline.)

<details>
<summary>Solutions</summary>

**02.1:**
- (a) `C[i*a_3 + j]`: across the warp `j` varies by 1 → addresses differ by 1 →
  **coalesced**. ✓
- (b) `B[k*a_3 + j]`: `j` varies by 1, `k` fixed → differ by 1 → **coalesced**. ✓
- (c) `A[i*a_2 + k]`: doesn't depend on `j` at all → all 32 threads read the
  **same** address → **broadcast** (served from cache/one transaction, cheap). ✓
- Verdict: **good**. Your naive matmul already maps `threadIdx.x → column`, so
  it's coalesced. You got the module-02 lesson right by instinct. (This is the
  difference between Boehm's "kernel 1 naive" and "kernel 2 coalesced" — you
  skipped straight to kernel 2's memory pattern.)

**02.2:** Swap so `int j = threadIdx.y + ...; int i = threadIdx.x + ...`. Then a
warp varies `i` (row) by 1 → `C[i*a_3+j]` and `A[i*a_2+k]` jump by a full row
(`a_3`/`a_2` floats) per thread → scattered → many transactions per warp. Same
failure you measured in `matrix_add.cu`.

**02.3:** `row = threadIdx.y + blockIdx.y*blockDim.y`,
`col = threadIdx.x + blockIdx.x*blockDim.x`, and
`acc += A[row*K + k] * B[k*N + col];`. `threadIdx.x` **must** be `col` for
coalescing.

**02.4:** Typical naive result on a 3060 is roughly ~100–300 GFLOP/s at
N=1024 (the L1/L2 caches rescue naive matmul somewhat, so you often beat the
0.25-intensity 90 GFLOP/s floor — caching gives real-world reuse the pessimistic
model ignored). Either way you're at **1–3% of the 12,700 GFLOP/s peak.** That
gap is the rest of this workbook.

</details>

## Check yourself

You can explain your own `matrix_add` benchmark in terms of memory transactions,
and you can point at each array access in a kernel and say "coalesced /
broadcast / strided." That's the skill. Onward.

→ Next: [Module 03 — Shared-memory tiling](03-shared-memory-tiling.md): the
first big intensity jump, where you stop re-reading the same rows and columns
from DRAM.
