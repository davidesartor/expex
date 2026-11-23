"""The notebooks in examples/notebooks/: each runs as a script, and each .ipynb matches its .py source.

The `.py` files are the source of truth (cells marked with `# %%`), and
examples/notebooks/build.py writes the `.ipynb` files from them. Here each
notebook's code cells run as one script, in a process of its own, against this
checkout (the setup cell finds it through $EXPEX_REPO instead of installing
from GitHub). None of this needs Jupyter: the notebooks are read as JSON.
"""

import importlib.util
import json
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).parent.parent
HERE = ROOT / "examples" / "notebooks"

_spec = importlib.util.spec_from_file_location("notebooks_build", HERE / "build.py")
assert _spec is not None and _spec.loader is not None
build = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build)

NEEDS = {"ppo": "optax", "multidevice": "optax"}


@pytest.mark.parametrize("name", build.NOTEBOOKS)
def test_notebook_runs_as_a_script(name, tmp_path):
    if name in NEEDS:
        pytest.importorskip(NEEDS[name])
    script = tmp_path / f"{name}.py"
    script.write_text(build.script(name), encoding="utf-8")
    env = dict(os.environ, EXPEX_REPO=str(ROOT), MPLBACKEND="Agg")
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), env.get("PYTHONPATH", "")])
    env.pop("XLA_FLAGS", None)
    p = subprocess.run([sys.executable, str(script)], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-5000:]
    assert "examples from" in p.stdout  # the setup cell used this checkout


@pytest.mark.parametrize("name", build.NOTEBOOKS)
def test_notebook_matches_its_source(name):
    path = HERE / f"{name}.ipynb"
    nb = json.loads(path.read_text(encoding="utf-8"))
    got = [(c["cell_type"], "".join(c["source"])) for c in nb["cells"]]
    assert got == build.cells(name), f"{path.name} is out of date: python examples/notebooks/build.py --execute {name}"


@pytest.mark.parametrize("name", build.NOTEBOOKS)
def test_notebook_is_committed_with_outputs_and_small(name):
    path = HERE / f"{name}.ipynb"
    assert path.stat().st_size < 2_000_000
    nb = json.loads(path.read_text(encoding="utf-8"))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert sum(bool(c["outputs"]) for c in code) > len(code) // 2, "run with build.py --execute"
    assert not [o for c in code for o in c["outputs"] if o["output_type"] == "error"]
    text = json.dumps(nb)
    assert str(ROOT) not in text  # no paths of the machine that ran it
    assert f"EXPEX_REF = \"{build.BRANCH}\"" in text and build.colab_url(name) in text


def test_readme_links_every_notebook():
    readme = (HERE / "README.md").read_text(encoding="utf-8")
    for name in build.NOTEBOOKS:
        assert build.colab_url(name) in readme, name
