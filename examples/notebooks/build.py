"""Build the notebooks in this folder from their `.py` sources.

Each `<name>.py` here is the source of truth for `<name>.ipynb`: cells are
separated by `# %%` lines (`# %% [markdown]` for text, whose lines start with
`# `), as in Jupytext's "percent" format. Two markers expand into cells shared
by every notebook:

    # %% [header] Title of the notebook     -> the title, the Colab badge and the Python note
    # %% [setup] llama                      -> the install cell, with examples/llama on the path
    # %% [bench]                            -> a code cell that only times things (tests skip it)

Usage:

    python examples/notebooks/build.py                 # write every .ipynb from its .py, without outputs
    python examples/notebooks/build.py --execute       # ... and run them, keeping the outputs
    python examples/notebooks/build.py --execute llama vit
    python examples/notebooks/build.py --check         # exit 1 if an .ipynb is out of date with its .py
    python examples/notebooks/build.py --script llama  # print the notebook's code as one script

`--execute` runs the notebooks against this checkout (no install from GitHub;
the setup cell finds the checkout above it) with nbclient, then keeps the
outputs. Rich displays that GitHub cannot render (the interactive diagram
and `expex.inspect` pages, which are inline HTML with scripts, and large) are
committed as PNG images instead, and the other expex displays as their text;
run the notebook to get the interactive versions. `--live` keeps the rich
HTML outputs as a run in Jupyter or Colab shows them (for checking how they
render; such notebooks are several MB, so they are not committed).
"""

import argparse
import copy
import json
import os
import pathlib
import re
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
BRANCH = "main"  # the branch the Colab links and the install cells point to
GITHUB = "davidesartor/expex"
NOTEBOOKS = ("quickstart", "llama", "mamba", "vit", "nerf", "ppo", "pgx", "mctx", "multidevice")


def colab_url(name: str, ref: str = BRANCH) -> str:
    return f"https://colab.research.google.com/github/{GITHUB}/blob/{ref}/examples/notebooks/{name}.ipynb"


BADGE = "https://colab.research.google.com/assets/colab-badge.svg"

HEADER = """# {title}

[![Open In Colab]({badge})]({url})

> **Python 3.14+ is needed.** expex requires Python 3.14, and Colab's hosted runtimes ship
> Python 3.12 (the 2026.01 runtime has 3.12.12), so on a hosted runtime the setup cell stops
> with a message. To run this notebook in Colab, start Jupyter on Python 3.14 on your machine
> and use *Connect → Connect to a local runtime*
> ([how](https://github.com/{github}/blob/{ref}/examples/notebooks/README.md#running-in-colab)),
> or open it in Jupyter directly. If *Runtime → Change runtime type* offers a runtime with
> Python 3.14, that works too. The outputs below are from a run on a CPU.
"""

SETUP = '''# Setup: install expex from GitHub (or use a local checkout) and fetch the examples.
EXPEX_REF = "{ref}"  # the branch or tag of github.com/{github} to install

import os, pathlib, subprocess, sys

if sys.version_info < (3, 14):
    raise RuntimeError(
        f"expex needs Python 3.14+, and this runtime has Python {{sys.version.split()[0]}}: "
        "connect to a local runtime with Python 3.14 (see the note at the top of the notebook)"
    )


def find_checkout() -> pathlib.Path | None:
    """A local checkout of expex: $EXPEX_REPO, or a folder above the working directory."""
    if os.environ.get("EXPEX_REPO"):
        return pathlib.Path(os.environ["EXPEX_REPO"]).resolve()
    for folder in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]:
        if (folder / "src" / "expex").is_dir() and (folder / "examples").is_dir():
            return folder
    return None


def run(*cmd: str) -> None:
    subprocess.run(cmd, check=True)


REPO = find_checkout()
if REPO is None:  # Colab, or a notebook opened on its own: install from GitHub
    REPO = pathlib.Path("expex").resolve()
    run(sys.executable, "-m", "pip", "install", "-q", f"git+https://github.com/{github}@{{EXPEX_REF}}"{extra})
    if not REPO.exists():  # the examples and the reference outputs of the originals
        run("git", "clone", "-q", "--depth", "1", "--branch", EXPEX_REF, "https://github.com/{github}", str(REPO))
else:
    try:
        import expex  # noqa: F401
    except ImportError:
        run(sys.executable, "-m", "pip", "install", "-q", "-e", str(REPO){extra})
sys.path.insert(0, str(REPO / "examples"{path}))
print("Python", sys.version.split()[0], "| examples from", REPO.name)'''


def cells(name: str) -> list[tuple[str, str]]:
    """The cells of notebook `name`: (kind, source) with kind "markdown" or "code"."""
    return [(kind if kind == "markdown" else "code", src) for kind, src in parse(name)]


def parse(name: str) -> list[tuple[str, str]]:
    """The cells of notebook `name`: (kind, source), kind "markdown", "code" or "bench" (a benchmark's code)."""
    text = (HERE / f"{name}.py").read_text(encoding="utf-8")
    out: list[tuple[str, list[str]]] = []
    for line in text.splitlines():
        m = re.match(r"# %%(?: \[(\w+)\])?\s*(.*)$", line)
        if m:
            kind, arg = m.group(1) or "code", m.group(2).strip()
            if kind == "header":
                out.append(("markdown", HEADER.format(title=arg, badge=BADGE, url=colab_url(name), github=GITHUB, ref=BRANCH).splitlines()))
            elif kind == "setup":
                port, *extra = arg.split()
                pip_extra = "".join(f', "{e}"' for e in extra)
                path = "".join(f' / "{p}"' for p in port.split("/") if p != ".")
                out.append(("code", SETUP.format(ref=BRANCH, github=GITHUB, extra=pip_extra, path=path).splitlines()))
            else:
                out.append((kind, []))
            continue
        if not out:
            continue  # the module docstring before the first cell
        kind, lines = out[-1]
        if kind == "markdown":
            lines.append(re.sub(r"^# ?", "", line))
        elif kind in ("code", "bench"):
            lines.append(line)
        else:
            raise SystemExit(f"{name}.py: lines after a [{kind}] marker")
    result = []
    for kind, lines in out:
        while lines and not lines[-1].strip():
            lines.pop()
        while lines and not lines[0].strip():
            lines.pop(0)
        result.append((kind, "\n".join(lines)))
    return result


def script(name: str, benchmarks: bool = True) -> str:
    """The notebook's code cells as one script (without the `# %% [bench]` cells if not `benchmarks`)."""
    return "\n\n".join(src for kind, src in parse(name) if kind == "code" or (kind == "bench" and benchmarks)) + "\n"


def notebook(name: str):
    import nbformat

    nb = nbformat.v4.new_notebook()
    for kind, src in parse(name):
        make = nbformat.v4.new_markdown_cell if kind == "markdown" else nbformat.v4.new_code_cell
        nb.cells.append(make(src))
        if kind == "bench":
            nb.cells[-1].metadata["tags"] = ["benchmark"]
    for i, cell in enumerate(nb.cells):
        cell.id = f"{name}-{i}"  # stable ids: rebuilding does not change the file
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"name": f"{name}.ipynb", "provenance": []},
    }
    return nb


def same_sources(a, b) -> bool:
    strip = lambda nb: [(c.cell_type, c.source) for c in nb.cells]  # noqa: E731
    return strip(a) == strip(b)


# ------------------------------------------------------------------ execution


def static_displays() -> None:
    """Run inside the kernel by `--execute`: expex's displays as GitHub can show them.

    GitHub renders a notebook's images and text, but not inline pages with
    scripts. The interactive diagram and inspect pages become PNG images
    (drawn by a headless browser), and every other display its text.
    """
    from IPython import get_ipython
    from IPython.display import Image, display

    import expex
    from expex._diagram import Diagram

    ip = get_ipython()
    fmt = ip.display_formatter.formatters
    fmt["text/html"].enabled = False  # every expex display has a text form

    def diagram_png(d: Diagram) -> bytes:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "d.png"
            d.save(path)
            return path.read_bytes()

    def page_png(page: str, width: int = 1200, height: int = 760) -> bytes:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(1) as pool:  # Playwright's sync API, outside the kernel's event loop
            return pool.submit(shoot, page, width, height).result()

    def shoot(page: str, width: int, height: int) -> bytes:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            try:
                browser = p.chromium.launch()
            except Exception:  # no browser of Playwright's own: a Chrome on the path
                browser = p.chromium.launch(executable_path=_chrome())
            tab = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=1.5)
            tab.set_content(page)
            tab.wait_for_timeout(800)  # the page unpacks its parts and lays them out
            png = tab.screenshot()
            browser.close()
            return png

    fmt["image/png"].for_type(Diagram, diagram_png)
    fmt["text/plain"].for_type(Diagram, lambda d, p, cycle: p.text(d.text))
    fmt["image/png"].for_type(expex.Inspect, lambda page: page_png(page.html()))
    fmt["text/plain"].for_type(
        expex.Inspect,
        lambda page, p, cycle: p.text(f"{page!r}  (an interactive page when the notebook runs; a screenshot here)"),
    )

    def error_display(e: BaseException) -> None:
        from expex.tools.display import error_text

        print(error_text(e))
        d = getattr(e, "diagram", None)
        if isinstance(d, Diagram):
            display(Image(diagram_png(d)))

    ip.display_formatter.ipython_display_formatter.for_type(expex.ShapeError, error_display)


def _chrome() -> str | None:
    import shutil

    for name in ("google-chrome", "chromium", "chromium-browser", "chrome"):
        if path := shutil.which(name):
            return path
    return None


def execute(name: str, live: bool = False, out: pathlib.Path | None = None) -> pathlib.Path:
    import nbclient
    import nbformat

    nb = notebook(name)
    prelude = nbformat.v4.new_code_cell(
        f"import sys; sys.path.insert(0, {str(HERE)!r}); import build; build.static_displays()" if not live else "pass"
    )
    nb.cells.insert(0, prelude)
    os.environ["EXPEX_REPO"] = str(ROOT)
    os.environ.setdefault("PYTHONWARNINGS", "ignore::RuntimeWarning:site")
    t = time.time()
    client = nbclient.NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(HERE)}})
    client.execute()
    nb.cells.pop(0)
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.execution_count = cell.execution_count - 1 if cell.execution_count else None
            cell.outputs = [clean(o) for o in cell.outputs]
        cell.metadata.pop("execution", None)
    nb.metadata.pop("widgets", None)
    path = out or HERE / f"{name}.ipynb"
    nbformat.write(nb, path)
    print(f"{name}: ran in {time.time() - t:.0f} s, {path.stat().st_size / 1e3:.0f} kB -> {path}")
    return path


def clean(output):
    """An output without this machine's paths: files in the checkout are named from its root."""
    output = copy.deepcopy(output)

    def fix(text: str) -> str:
        return text.replace(str(ROOT) + "/", "").replace(str(ROOT), ".")

    if output.get("output_type") == "stream":
        output["text"] = fix(output["text"])
    for mime, value in output.get("data", {}).items():
        if isinstance(value, str) and (mime.startswith("text/") or mime.endswith("json")):
            output["data"][mime] = fix(value)
    if output.get("output_type") == "error":
        output["traceback"] = [fix(line) for line in output["traceback"]]
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", help=f"notebooks to build (default: all of {', '.join(NOTEBOOKS)})")
    parser.add_argument("--execute", action="store_true", help="run them and keep the outputs")
    parser.add_argument("--live", action="store_true", help="with --execute: keep the interactive HTML outputs")
    parser.add_argument("--out", help="with --execute: a folder to write to instead of this one")
    parser.add_argument("--check", action="store_true", help="exit 1 if an .ipynb's cells differ from its .py")
    parser.add_argument("--script", action="store_true", help="print the code cells as one script")
    args = parser.parse_args()
    names = args.names or list(NOTEBOOKS)
    if args.script:
        for name in names:
            print(script(name))
        return
    import nbformat

    if args.check:
        stale = [n for n in names if not (HERE / f"{n}.ipynb").exists()
                 or not same_sources(nbformat.read(HERE / f"{n}.ipynb", as_version=4), notebook(n))]  # fmt: skip
        if stale:
            print("out of date (run python examples/notebooks/build.py --execute " + " ".join(stale) + "):", *stale)
            sys.exit(1)
        print(f"{len(names)} notebooks up to date")
        return
    for name in names:
        if args.execute:
            out = pathlib.Path(args.out) / f"{name}.ipynb" if args.out else None
            if out:
                out.parent.mkdir(parents=True, exist_ok=True)
            execute(name, live=args.live, out=out)
        else:
            nbformat.write(notebook(name), HERE / f"{name}.ipynb")
            print(f"{name}: written without outputs")


if __name__ == "__main__":
    main()
