#!/usr/bin/env python3
"""Generate jupytext '# %%' notebooks from the workbook markdown + exercises.

For each ../NN-*.md it writes ./NN-*.py containing:
  - a jupytext percent header,
  - a bootstrap cell that imports the autograder (checks.py),
  - the markdown prose as markdown cells (split on '## ' headings; mermaid kept),
  - the hand-authored exercise + test cells from ./_exercises/NN.py.

Re-run after editing any markdown or exercise file:  python3 _build.py
"""
import os
import re
import glob

HERE = os.path.dirname(os.path.abspath(__file__))
MD_DIR = os.path.dirname(HERE)              # learning/
EX_DIR = os.path.join(HERE, "_exercises")

HEADER = """# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---
"""

BOOTSTRAP = '''# %% [markdown]
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
'''


def md_to_markdown_cells(md_text):
    """Split markdown into '# %% [markdown]' cells on level-2 ('## ') headings."""
    lines = md_text.splitlines()
    chunks, cur = [], []
    for ln in lines:
        if ln.startswith("## ") and cur:
            chunks.append(cur)
            cur = [ln]
        else:
            cur.append(ln)
    if cur:
        chunks.append(cur)

    out = []
    for chunk in chunks:
        # trim leading/trailing blank lines in the chunk
        while chunk and chunk[0].strip() == "":
            chunk.pop(0)
        while chunk and chunk[-1].strip() == "":
            chunk.pop()
        if not chunk:
            continue
        out.append("# %% [markdown]")
        for ln in chunk:
            out.append("#" if ln.strip() == "" else "# " + ln)
        out.append("")
    return "\n".join(out)


def build_one(md_path):
    base = os.path.basename(md_path)[:-3]          # e.g. "01-roofline"
    num = base.split("-")[0]                        # "01"
    ex_path = os.path.join(EX_DIR, num + ".py")

    with open(md_path) as f:
        md = f.read()

    parts = [HEADER, BOOTSTRAP, md_to_markdown_cells(md)]
    if os.path.exists(ex_path):
        with open(ex_path) as f:
            parts.append(f.read().rstrip() + "\n")

    out_path = os.path.join(HERE, base + ".py")
    with open(out_path, "w") as f:
        f.write("\n".join(parts).rstrip() + "\n")
    return out_path


def main():
    mds = sorted(glob.glob(os.path.join(MD_DIR, "[0-9][0-9]-*.md")))
    if not mds:
        print("no module markdown found in", MD_DIR)
        return
    for md in mds:
        print("built", os.path.relpath(build_one(md), MD_DIR))


if __name__ == "__main__":
    main()
