"""Build a .ipynb from a jupytext-percent .py source.

`jupytext` is not installed in this environment and this is the only thing we need from it.
The .py file is the reviewable artefact: notebooks do not diff, so the source of truth for
`09_map_timeseries.ipynb` is `09_map_timeseries.py` and the notebook is
regenerated from it.

`--check` executes every code cell in-process with a `display` shim and reports which ones
raise. Worth running before committing: a notebook that fails on cell 2 is worse than no
notebook, and the failure only shows up when someone opens it.

Usage:
    python notebooks/build_notebook.py notebooks/09_map_timeseries.py
    python notebooks/build_notebook.py --check notebooks/09_map_timeseries.py
"""
import json, sys, re

def convert(src, dst):
    lines = open(src).read().split("\n")
    # drop the leading shebang/header block up to the first cell marker
    try:
        start = next(i for i, l in enumerate(lines) if l.startswith("# %%"))
    except StopIteration:
        raise SystemExit("no '# %%' cell markers found")
    cells, cur, kind = [], [], None
    def flush():
        if kind is None:
            return
        body = "\n".join(cur).strip("\n")
        if not body.strip():
            return
        if kind == "markdown":
            text = "\n".join(re.sub(r"^# ?", "", l) for l in body.split("\n"))
            cells.append(dict(cell_type="markdown", metadata={}, source=text.split("\n")))
        else:
            cells.append(dict(cell_type="code", metadata={}, execution_count=None,
                              outputs=[], source=body.split("\n")))
    for line in lines[start:]:
        if line.startswith("# %%"):
            flush()
            kind = "markdown" if "[markdown]" in line else "code"
            cur = []
        else:
            cur.append(line)
    flush()
    for c in cells:
        c["source"] = [s + "\n" for s in c["source"][:-1]] + [c["source"][-1]]
    nb = dict(cells=cells, metadata=dict(
        kernelspec=dict(display_name="Python 3", language="python", name="python3"),
        language_info=dict(name="python", version="3.11")),
        nbformat=4, nbformat_minor=5)
    json.dump(nb, open(dst, "w"), indent=1)
    print(f"{dst}: {len(cells)} cells "
          f"({sum(c['cell_type']=='markdown' for c in cells)} markdown)")

def split_cells(src):
    lines = open(src).read().split("\n")
    start = next(i for i, l in enumerate(lines) if l.startswith("# %%"))
    cells, cur = [], []
    for line in lines[start:]:
        if line.startswith("# %%"):
            if cur:
                cells.append("\n".join(cur))
            cur = [] if "[markdown]" not in line else None
        elif cur is not None:
            cur.append(line)
    if cur:
        cells.append("\n".join(cur))
    return [c for c in cells if c.strip()]


def check(src):
    """Run every code cell; return the number that raised."""
    import builtins, contextlib, io, os, traceback
    import matplotlib
    matplotlib.use("Agg")
    builtins.display = lambda *a, **k: None        # noqa: A001 - notebook shim
    here = os.path.dirname(os.path.abspath(src)) or "."
    cwd = os.getcwd()
    os.chdir(here)
    g, bad = {"__name__": "__main__"}, 0
    try:
        for i, c in enumerate(split_cells(os.path.basename(src))):
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(compile(c, f"<cell {i}>", "exec"), g)
            except Exception:
                bad += 1
                print(f"=== cell {i} raised:")
                traceback.print_exc(limit=2)
    finally:
        os.chdir(cwd)
    print(f"{len(split_cells(src))} code cells, {bad} with errors")
    return bad


args = [a for a in sys.argv[1:] if not a.startswith("-")]
if "--check" in sys.argv:
    raise SystemExit(1 if check(args[0]) else 0)
src = args[0]
dst = args[1] if len(args) > 1 else src.rsplit(".", 1)[0] + ".ipynb"
convert(src, dst)
