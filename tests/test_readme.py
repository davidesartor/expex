"""Every Python block in the README and the guide pages must run."""

import pathlib
import re
import runpy

import jax.numpy as jnp
import jax.random as jr

ROOT = pathlib.Path(__file__).parent.parent


GUIDE = ["functions", "modules", "compiling", "cost-and-diagrams", "debugging", "reference"]


def run_blocks(doc: str | list[str], tmp_path: pathlib.Path) -> dict:
    docs = [doc] if isinstance(doc, str) else doc
    text = "\n".join((ROOT / d).read_text() for d in docs)
    blocks = re.findall(r"```python\n(.*?)```", text, flags=re.S)
    # written to a real file so that @fn can read the source it converts
    path = tmp_path / (pathlib.Path(docs[0]).stem + "_blocks.py")
    path.write_text("\n\n".join(blocks))
    return runpy.run_path(str(path))


def test_readme_blocks_run(tmp_path, capsys):
    ns = run_blocks("README.md", tmp_path)
    assert "B is 2 (from `q: f32[2, 3, 4]`), but axis 0 of `k` is 5." in capsys.readouterr().out


def test_readme_comparisons_agree(tmp_path):
    """Each JAX / expex pair on the front page computes the same thing."""
    text = (ROOT / "README.md").read_text()
    blocks = re.findall(r"```python\n(.*?)```", text, flags=re.S)
    jax_side, expex_side = blocks[1:-1:2], blocks[2:-1:2]
    src = blocks[0] + "".join(
        f"\n{j}\njax_{i} = dict(globals())\n{e}\n" for i, (j, e) in enumerate(zip(jax_side, expex_side))
    )
    path = tmp_path / "pairs.py"
    path.write_text(src)
    ns = runpy.run_path(str(path))
    w, xs = jr.normal(jr.key(0), (8, 8)) * 0.3, jr.normal(jr.key(1), (6, 8))
    q, k = jr.normal(jr.key(2), (2, 3, 4)), jr.normal(jr.key(3), (2, 5, 4))
    assert jnp.allclose(ns["jax_0"]["attend"](q, k), ns["attend"](q, k), atol=1e-5)
    assert jnp.allclose(ns["jax_1"]["rnn"](w, xs), ns["rnn"](w, xs), atol=1e-5)
    z = jnp.array([2.0, -1.0, 0.0])
    assert jnp.array_equal(ns["jax_2"]["safe_log"](z), ns["safe_log"](z))
    model, x = ns["model"], jr.normal(jr.key(4), (64,))
    params = {"w": model.blocks.w, "b": model.blocks.b} if not isinstance(model.blocks, list) else {
        "w": jnp.stack([b.w for b in model.blocks]), "b": jnp.stack([b.b for b in model.blocks])
    }  # fmt: skip
    assert jnp.allclose(ns["stack"](params, x), model(x), atol=1e-4)


def test_guide_lists_every_page():
    index = (ROOT / "docs/README.md").read_text()
    assert sorted(GUIDE) == sorted(p.stem for p in (ROOT / "docs").glob("*.md") if p.stem != "README")
    assert all(f"]({page}.md)" in index for page in GUIDE)


def test_guide_blocks_run(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)  # the examples save files
    ns = run_blocks([f"docs/{page}.md" for page in GUIDE], tmp_path)
    out = capsys.readouterr().out
    assert "B is 2 (from `q: f32[2, 3, 4]`), but axis 0 of `k` is 5." in out
    assert "axis 0 of `return` should be T, but is T - 1." in out
    assert "[  8  16 111]" in out
    assert "for -> lax.scan (carries h)" in out
    assert "NaN first appears at" in out
    assert ns["dropout"].random and ns["safe_log"].auto_vectorize
    assert "# scores: f32[B, T, T]" in out
    assert type(ns["grads"]).__name__ == "Stack" and ns["y"].shape == (4, 16)
