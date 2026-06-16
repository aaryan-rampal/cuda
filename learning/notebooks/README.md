# Notebooks — the runnable, autograded version of the workbook

Same content as the `../NN-*.md` modules, but as **`# %%` notebooks** you can
run cell-by-cell, with **tests you run to check yourself** — like a course
autograder. Fill in a `TODO`, run the test cell, get ✅ or ❌ with a hint,
iterate.

## How to open them

These `.py` files use the **jupytext "percent" format** (`# %%` = code cell,
`# %% [markdown]` = markdown cell). Two easy ways to run them as notebooks:

- **VS Code** (no extra install): open any `NN-*.py`. VS Code shows "Run Cell"
  on every `# %%`. Markdown cells (incl. **Mermaid** diagrams) render in the
  notebook view. This is the smoothest path.
- **JupyterLab**: `pip install jupytext`, then right-click → *Open With →
  Notebook*, or convert: `jupytext --to notebook 03-shared-memory-tiling.py`.
  Recent JupyterLab renders Mermaid in markdown cells.

Run them **in order** (00 → 07); each builds on the last.

## What you need

- **For the math/derivation tests** (modules 00–01 and the warm-ups): just
  Python 3. No packages.
- **For the CUDA autograder** (`check_cuda` / `bench_cuda`, modules 02–07): your
  **local machine with the RTX 3060 + CUDA toolkit** (`nvcc` on PATH). The
  harness compiles your kernel string with `nvcc`, runs it, verifies against a
  double-precision reference, and prints GFLOP/s. If `nvcc` isn't found, those
  cells tell you so and the math cells still work.

## The autograder (`checks.py`)

The bootstrap cell at the top of every notebook does `from checks import *`,
giving you:

| call | use |
|------|-----|
| `check("name", got, want)` | exact / tolerant value compare |
| `approx("name", got, want, rel=0.05)` | relative-tolerance compare |
| `check_true("name", cond, "hint")` | a boolean must hold |
| `check_fn("name", your_fn, ref_fn, cases)` | property test — your formula must match a reference on many inputs (so it checks *correctness* without revealing the closed form) |
| `check_cuda("name", kernel_src, launch)` | compile + run + verify your CUDA kernel, report GFLOP/s |
| `bench_cuda(kernel_src, launch, sizes=(...))` | GFLOP/s sweep across sizes |

For the CUDA cells you write the kernel as a Python string and a one-line launch
snippet (it can use `dA, dB, dC, M, N, K`), e.g.:

```python
launch = "dim3 b(32,32), g((N+31)/32,(M+31)/32); sgemm_naive<<<g,b>>>(dA,dB,dC,M,N,K);"
check_cuda("naive", sgemm_naive_src, launch)
# -> ✅ PASS  naive   312 GFLOP/s  (2.5% of 3060 peak)  [256x256x256, 0.34 ms]
# or on a bug:
# -> ❌ FAIL  naive   kernel ran but produced WRONG results.  FIRSTBAD i=0 j=1 got=.. exp=..
# or on a typo:
# -> ❌ FAIL  naive   nvcc compile error: ... expected ';' before ...
```

That's the loop you wanted: *write it → run the test → "am I on track?" → fix →
rerun*, no need to ask an AI whether it's right.

## Files

- `00-*.py` … `07-*.py` — the notebooks (generated).
- `checks.py` — the autograder. Run `python3 checks.py` to self-test it.
- `_exercises/NN.py` — the hand-authored code+test cells per module.
- `_build.py` — regenerates the notebooks from `../NN-*.md` + `_exercises/`.
  Run `python3 _build.py` after editing any markdown or exercise file.

> Note: the `.md` modules in `../` are the **source of truth for the prose**
> (and render Mermaid on GitHub). The notebooks add the runnable exercises and
> tests. Edit prose in the `.md`, then re-run `_build.py`.
