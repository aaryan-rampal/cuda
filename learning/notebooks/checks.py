"""
checks.py — the autograder for the CUDA workbook notebooks.

Two kinds of tests, both runnable from a notebook cell:

1. PURE-PYTHON checks for the math/derivation exercises:
     check("name", got, want)          # exact / tolerant compare
     approx("name", got, want, rel=..) # relative-tolerance compare
     check_true("name", cond, hint)    # a boolean must hold
     check_fn("name", user_fn, ref_fn, cases)  # property test over inputs

2. A CUDA AUTOGRADER that compiles your kernel with nvcc, runs it on real
   inputs, verifies the result against a double-precision reference, and
   reports PASS/FAIL + GFLOP/s:
     check_cuda("name", kernel_src, launch)

   `kernel_src` is your __global__ kernel as a Python string.
   `launch` is the launch snippet (may use dA,dB,dC,M,N,K), e.g.
       "dim3 b(32,32), g((N+31)/32,(M+31)/32); my_kernel<<<g,b>>>(dA,dB,dC,M,N,K);"

No third-party packages required (numpy is used only if present).
nvcc + a CUDA GPU are required only for the check_cuda path — that's your 3060.
"""

from __future__ import annotations
import os
import re
import shutil
import subprocess
import tempfile
import textwrap

# ---------------------------------------------------------------- pretty print
PASS = "\033[32m✅ PASS\033[0m"
FAIL = "\033[31m❌ FAIL\033[0m"
INFO = "\033[36mℹ️ \033[0m"


def _ok(name, extra=""):
    print(f"{PASS}  {name}" + (f"   {extra}" if extra else ""))
    return True


def _no(name, extra=""):
    print(f"{FAIL}  {name}" + (f"\n        {extra}" if extra else ""))
    return False


# ----------------------------------------------------------- pure-python checks
def check(name, got, want, tol=1e-9):
    """Pass if got == want (numbers compared within absolute tol)."""
    try:
        if isinstance(got, (int, float)) and isinstance(want, (int, float)):
            good = abs(got - want) <= tol
        else:
            good = got == want
    except Exception as ex:  # noqa: BLE001
        return _no(name, f"comparison raised {ex!r}")
    if good:
        return _ok(name, f"got {got}")
    return _no(name, f"got {got!r}, expected {want!r}")


def approx(name, got, want, rel=1e-2):
    """Pass if got is within `rel` relative error of want."""
    if want == 0:
        good = abs(got) <= rel
    else:
        good = abs(got - want) / abs(want) <= rel
    if good:
        return _ok(name, f"got {got:.6g} (want ≈ {want:.6g})")
    return _no(name, f"got {got:.6g}, expected ≈ {want:.6g} (rel tol {rel})")


def check_true(name, cond, hint=""):
    return _ok(name) if cond else _no(name, hint)


def check_fn(name, user_fn, ref_fn, cases, rel=1e-6):
    """Property test: user_fn must match ref_fn on every input tuple in cases.

    `cases` is an iterable of argument tuples. This checks correctness
    WITHOUT revealing the closed form — you discover whether your formula is
    right by whether it matches the reference on many inputs.
    """
    if user_fn is None:
        return _no(name, "user_fn is None — did you fill in the TODO?")
    cases = list(cases)
    for args in cases:
        a = args if isinstance(args, tuple) else (args,)
        try:
            got = user_fn(*a)
        except Exception as ex:  # noqa: BLE001
            return _no(name, f"your function raised {ex!r} on input {a}")
        want = ref_fn(*a)
        if isinstance(want, (int, float)) and isinstance(got, (int, float)):
            bad = (abs(got - want) > rel * (abs(want) + 1e-12))
        else:
            bad = got != want
        if bad:
            return _no(name, f"on input {a}: got {got!r}, reference says {want!r}")
    return _ok(name, f"matched reference on {len(list(cases))} inputs")


# ------------------------------------------------------------- the CUDA grader
_PROGRAM_TEMPLATE = r"""
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do {                                              \
    cudaError_t _e = (call);                                               \
    if (_e != cudaSuccess) {                                               \
        fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__,             \
                cudaGetErrorString(_e));                                   \
        return 2;                                                          \
    }                                                                      \
} while (0)

/* ================= YOUR KERNEL ================= */
__KERNEL_SRC__
/* =============== END YOUR KERNEL =============== */

int main() {
    int M = __M__, N = __N__, K = __K__;
    size_t bA = (size_t)M*K*sizeof(float);
    size_t bB = (size_t)K*N*sizeof(float);
    size_t bC = (size_t)M*N*sizeof(float);
    std::vector<float> A(M*K), B(K*N), C(M*N);
    srand(1234);
    for (auto& x : A) x = (float)(rand()%9) - 4.0f;
    for (auto& x : B) x = (float)(rand()%9) - 4.0f;

    float *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, bA));
    CHECK_CUDA(cudaMalloc(&dB, bB));
    CHECK_CUDA(cudaMalloc(&dC, bC));
    CHECK_CUDA(cudaMemcpy(dA, A.data(), bA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, B.data(), bB, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC, 0, bC));

    /* warm-up (never timed) */
    { __LAUNCH__ }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemset(dC, 0, bC));

    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    { __LAUNCH__ }
    cudaEventRecord(e);
    CHECK_CUDA(cudaEventSynchronize(e));
    CHECK_CUDA(cudaGetLastError());
    float ms = 0; cudaEventElapsedTime(&ms, s, e);

    CHECK_CUDA(cudaMemcpy(C.data(), dC, bC, cudaMemcpyDeviceToHost));

    int ok = 1; double worst = 0.0;
    for (int i = 0; i < M && ok; i++) {
        for (int j = 0; j < N; j++) {
            double exp = 0.0;
            for (int k = 0; k < K; k++) exp += (double)A[i*K+k] * (double)B[k*N+j];
            double got = C[i*N+j];
            double err = fabs(got - exp) / (fabs(exp) + 1e-6);
            if (err > worst) worst = err;
            if (err > 1e-2) {
                printf("FIRSTBAD i=%d j=%d got=%f exp=%f\n", i, j, got, exp);
                ok = 0; break;
            }
        }
    }
    double gflops = (2.0*M*N*K) / (ms*1e-3) / 1e9;
    printf("RESULT ok=%d ms=%.4f gflops=%.2f worsterr=%.3e\n",
           ok, ms, gflops, worst);
    return ok ? 0 : 1;
}
"""


def have_nvcc():
    return shutil.which("nvcc") is not None


def run_cuda_kernel(kernel_src, launch, M=256, N=256, K=256,
                    arch="native", extra_nvcc=("-O3",), verbose=False):
    """Compile + run a kernel. Returns a dict with the outcome.

    keys: compiled(bool), ok(bool), ms, gflops, worsterr, log(str)
    """
    if not have_nvcc():
        return {"compiled": False, "ok": False, "ms": None, "gflops": None,
                "worsterr": None,
                "log": "nvcc not found on PATH. The CUDA tests need your "
                       "local machine with the RTX 3060 + CUDA toolkit."}

    src = (_PROGRAM_TEMPLATE
           .replace("__KERNEL_SRC__", kernel_src)
           .replace("__LAUNCH__", launch)
           .replace("__M__", str(M)).replace("__N__", str(N)).replace("__K__", str(K)))

    tmp = tempfile.mkdtemp(prefix="sgemm_")
    cu = os.path.join(tmp, "prog.cu")
    binp = os.path.join(tmp, "prog")
    with open(cu, "w") as f:
        f.write(src)

    cmd = ["nvcc", *extra_nvcc, f"-arch={arch}", cu, "-o", binp]
    comp = subprocess.run(cmd, capture_output=True, text=True)
    if comp.returncode != 0:
        return {"compiled": False, "ok": False, "ms": None, "gflops": None,
                "worsterr": None,
                "log": "nvcc compile error:\n" + comp.stderr.strip()}

    try:
        run = subprocess.run([binp], capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return {"compiled": True, "ok": False, "ms": None, "gflops": None,
                "worsterr": None, "log": "kernel timed out (>120s)"}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    out = run.stdout + "\n" + run.stderr
    m = re.search(r"RESULT ok=(\d+) ms=([\d.]+) gflops=([\d.]+) worsterr=([\d.eE+-]+)", out)
    if not m:
        return {"compiled": True, "ok": False, "ms": None, "gflops": None,
                "worsterr": None, "log": "no RESULT line. Output:\n" + out.strip()}
    ok = m.group(1) == "1"
    res = {"compiled": True, "ok": ok, "ms": float(m.group(2)),
           "gflops": float(m.group(3)), "worsterr": float(m.group(4)), "log": out.strip()}
    bad = re.search(r"FIRSTBAD .*", out)
    if bad:
        res["log"] = bad.group(0)
    return res


def check_cuda(name, kernel_src, launch, M=256, N=256, K=256, arch="native",
               peak_gflops=12700.0):
    """Autograde a CUDA kernel: compile, run, verify, report GFLOP/s."""
    r = run_cuda_kernel(kernel_src, launch, M=M, N=N, K=K, arch=arch)
    if not r["compiled"]:
        return _no(name, r["log"])
    if not r["ok"]:
        return _no(name, "kernel ran but produced WRONG results.\n        " + r["log"])
    pct = 100.0 * r["gflops"] / peak_gflops
    return _ok(name, f"{r['gflops']:.0f} GFLOP/s  ({pct:.1f}% of 3060 peak)  "
                     f"[{M}x{N}x{K}, {r['ms']:.3f} ms]")


def bench_cuda(kernel_src, launch, sizes=(256, 512, 1024), arch="native"):
    """Print a GFLOP/s sweep across square sizes (for the perf exercises)."""
    if not have_nvcc():
        print(INFO, "nvcc not found — run this on your 3060.")
        return
    print(f"{'N':>6}  {'ms':>10}  {'GFLOP/s':>10}  {'% peak':>7}  ok")
    for n in sizes:
        r = run_cuda_kernel(kernel_src, launch, M=n, N=n, K=n, arch=arch)
        if not r["compiled"]:
            print(r["log"]); return
        print(f"{n:>6}  {r['ms']:>10.3f}  {r['gflops']:>10.1f}  "
              f"{100*r['gflops']/12700:>6.1f}%  {'OK' if r['ok'] else 'FAIL'}")


if __name__ == "__main__":
    # self-test of the pure-python machinery (no GPU needed)
    check("demo exact", 2 + 2, 4)
    approx("demo ridge point", 12700 / 360, 35.0, rel=0.05)
    check_true("demo bool", 0.25 < 35, "naive intensity below ridge")
    check_fn("demo property", lambda n: 2 * n, lambda n: n + n, [1, 5, 100])
    print(INFO, "nvcc available:" , have_nvcc())
