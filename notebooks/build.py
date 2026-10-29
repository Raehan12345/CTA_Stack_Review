"""Turn a chapter source file into an executed notebook.

Each chapter is written as a plain Python file with cell markers, so it can
be read, diffed and reviewed like any other code:

    # %% [markdown]
    # # A heading
    # Some prose.

    # %%
    x = 1 + 1

    python notebooks/build.py notebooks/src/ch01_trend_sleeve.py [--no-exec]

writes notebooks/ch01_trend_sleeve.ipynb with its outputs. Needs the
licensed data (CTADATA_DIR) to execute.
"""
import argparse
import os
import re

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

HERE = os.path.dirname(os.path.abspath(__file__))


def parse(text: str):
    cells = []
    for block in re.split(r"^# %%", text, flags=re.M)[1:]:
        header, _, body = block.partition("\n")
        if "[markdown]" in header:
            lines = [re.sub(r"^# ?", "", ln) for ln in body.rstrip().splitlines()]
            cells.append(nbformat.v4.new_markdown_cell("\n".join(lines).strip()))
        else:
            src = body.strip("\n")
            if src.strip():
                cells.append(nbformat.v4.new_code_cell(src))
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--no-exec", action="store_true")
    ap.add_argument("--timeout", type=int, default=7200)
    a = ap.parse_args()
    nb = nbformat.v4.new_notebook()
    nb.cells = parse(open(a.src, encoding="utf-8").read())
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    out = os.path.join(HERE, os.path.basename(a.src).replace(".py", ".ipynb"))
    try:
        if not a.no_exec:
            ExecutePreprocessor(timeout=a.timeout, kernel_name="python3").preprocess(
                nb, {"metadata": {"path": HERE}})
    finally:
        # written even when a cell fails, so the failing cell can be inspected
        nbformat.write(nb, out)
        print("wrote", out)


if __name__ == "__main__":
    main()
