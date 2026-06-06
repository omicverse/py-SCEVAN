"""Build pyscevan example notebooks from jupytext-style percent-format .py sources.

Produces (next to this file):
  - tutorial_quickstart.ipynb           — end-to-end MVP demo on the bundled
    MGH106 frozen subset (200 cells), driven through `pyscevan.pipeline_cna`
    (DataFrame API) and the AnnData-native path.
  - tutorial_quickstart.executed.ipynb  — same, executed so the outputs render
    on GitHub without a local kernel.

Run from the repo root:

    uv run python examples/_build_notebooks.py tutorial_quickstart.py
    uv run python examples/_build_notebooks.py --execute tutorial_quickstart.py
    uv run python examples/_build_notebooks.py                 # build all .py here

Never hand-edit the .ipynb JSON — always re-run this builder.

Conversion rules (percent-format subset; intentionally small, no jupytext dep):

  - Lines starting with ``# %% [markdown]`` open a markdown cell; subsequent
    lines starting with ``# `` / ``#`` contribute its body.
  - Lines starting with ``# %%`` (optionally ``# %% [title]``) open a code cell;
    subsequent non-cell-marker lines are its source.
  - A leading shebang / module docstring before the first ``# %%`` marker is
    ignored (the script header).

Depends only on ``nbformat`` (JSON) + ``nbclient`` (``--execute``) — the same
two deps as the pyinfercnv / pycopykat builders.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

HERE = Path(__file__).parent.resolve()
REPO_ROOT = HERE.parent

DEFAULT_KERNEL = os.environ.get("PYSCEVAN_KERNEL", "python3")

CELL_MARK = re.compile(r"^#\s*%%(?:\s*\[(?P<kind>[a-zA-Z]+)\])?\s*(?P<title>.*)$")


def _strip_md(lines: list[str]) -> str:
    """Convert commented markdown body back to markdown text."""
    out: list[str] = []
    for ln in lines:
        if ln.startswith("# "):
            out.append(ln[2:])
        elif ln.startswith("#"):
            out.append(ln[1:])
        else:
            out.append(ln)
    return "\n".join(out).rstrip() + "\n"


def py_to_ipynb(src_path: Path) -> nbf.NotebookNode:
    """Parse a percent-format .py file into an nbformat notebook.

    The first block (up to the first ``# %%`` marker) is the file header and is
    discarded, so the source .py may open with a module docstring.
    """
    lines = src_path.read_text(encoding="utf-8").splitlines()

    i = 0
    while i < len(lines) and not CELL_MARK.match(lines[i]):
        i += 1
    if i == len(lines):
        raise ValueError(f"no `# %%` cell markers found in {src_path}")

    nb = nbf.v4.new_notebook()
    cells = nb.cells
    current_kind = "code"
    buf: list[str] = []

    def _flush() -> None:
        if not buf and current_kind == "code":
            return
        if current_kind == "markdown":
            cells.append(nbf.v4.new_markdown_cell(_strip_md(buf)))
        else:
            cells.append(nbf.v4.new_code_cell("\n".join(buf).rstrip()))

    while i < len(lines):
        m = CELL_MARK.match(lines[i])
        if m is not None:
            _flush()
            buf = []
            kind = (m.group("kind") or "code").lower()
            current_kind = "markdown" if kind == "markdown" else "code"
            i += 1
            continue
        buf.append(lines[i])
        i += 1
    _flush()

    while cells and cells[0]["cell_type"] == "code" and not cells[0]["source"].strip():
        cells.pop(0)

    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": DEFAULT_KERNEL,
    }
    nb.metadata["language_info"] = {
        "name": "python",
        "mimetype": "text/x-python",
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
    }
    return nb


def build(src_path: Path) -> Path:
    nb = py_to_ipynb(src_path)
    out = src_path.with_suffix(".ipynb")
    with open(out, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    nbf.validate(nb)
    return out


def execute(src_path: Path, *, kernel: str | None = None) -> Path:
    nb = py_to_ipynb(src_path)
    client = NotebookClient(
        nb,
        timeout=3600,
        kernel_name=kernel or DEFAULT_KERNEL,
        resources={"metadata": {"path": str(HERE)}},
    )
    client.execute()
    out = src_path.with_suffix(".executed.ipynb")
    with open(out, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    nbf.validate(nb)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("targets", nargs="*",
                    help="Source .py files under examples/. Default: every *.py not starting with '_'.")
    ap.add_argument("--execute", action="store_true", help="also produce <name>.executed.ipynb")
    ap.add_argument("--kernel", default=None, help="override kernel name (default: $PYSCEVAN_KERNEL or python3)")
    ap.add_argument("--no-build", action="store_true", help="skip plain .ipynb (only --execute)")
    args = ap.parse_args()

    if args.targets:
        srcs = [HERE / t if not os.path.isabs(t) else Path(t) for t in args.targets]
    else:
        srcs = sorted(p for p in HERE.glob("*.py") if not p.name.startswith("_"))

    if not srcs:
        print("no tutorial .py sources found", file=sys.stderr)
        return 1

    for src in srcs:
        if not src.exists():
            print(f"missing: {src}", file=sys.stderr)
            return 1
        if not args.no_build:
            print(f"built    {build(src).relative_to(REPO_ROOT)}")
        if args.execute:
            print(f"executed {execute(src, kernel=args.kernel).relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
