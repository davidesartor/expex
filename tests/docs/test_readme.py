"""Every Python block in the README and the guide pages must run."""

import importlib.util
import pathlib
import re
import runpy

import jax.numpy as jnp
import jax.random as jr
import pytest

import expex

ROOT = pathlib.Path(__file__).parents[2]


GUIDE = ["functions", "modules", "compiling", "cost-and-diagrams", "debugging", "tools", "porting", "errors", "interop", "design", "architecture"]


def run_blocks(doc: str | list[str], tmp_path: pathlib.Path) -> dict:
    docs = [doc] if isinstance(doc, str) else doc
    text = "\n".join((ROOT / d).read_text() for d in docs)
    # ```python equinox: a block that needs a library (top-level names: equinox, flax, orbax), skipped without it
    found = re.findall(r"```python([^\n]*)\n(.*?)```", text, flags=re.S)
    blocks = [code for needs, code in found if all(map(importlib.util.find_spec, needs.split()))]
    # written to a real file so that @fn can read the source it converts
    path = tmp_path / (pathlib.Path(docs[0]).stem + "_blocks.py")
    path.write_text("\n\n".join(blocks))
    return runpy.run_path(str(path))


def test_readme_images_exist():
    for doc in ["README.md", *(f"docs/{page}.md" for page in GUIDE)]:
        text = (ROOT / doc).read_text()
        for src in re.findall(r'(?:src|srcset)="([^"]+)"', text):
            assert (ROOT / doc).parent.joinpath(src).is_file(), (doc, src)


def test_readme_blocks_run(tmp_path, capsys):
    ns = run_blocks("README.md", tmp_path)
    out = capsys.readouterr().out
    assert "Stack.__call__: proved for all sizes" in out
    assert "flops        2·D²·L + 10·D·L" in out
    # the wrong call the README quotes, and the message it shows
    text = (ROOT / "README.md").read_text()
    assert "`attend(jnp.ones((2, 3, 4)), jnp.ones((5, 3, 4)))`" in text
    with pytest.raises(expex.ShapeError) as e:
        ns["attend"](jnp.ones((2, 3, 4)), jnp.ones((5, 3, 4)))
    assert e.value.code == "EX101"
    message = str(e.value)
    for line in ("declared  k: f32[B, S, D]", "received     f32[5, 3, 4]", "B is 2 (from `q: f32[2, 3, 4]`), but axis 0 of `k` is 5."):
        assert line in message and line in text


def test_readme_comparisons_agree(tmp_path):
    """Each JAX / expex pair on the front page computes the same thing."""
    text = (ROOT / "README.md").read_text()
    blocks = re.findall(r"```python\n(.*?)```", text, flags=re.S)
    pairs = [b.split("\n# expex\n") for b in blocks if "\n# expex\n" in b]
    jax_side, expex_side = [j for j, _ in pairs], [e for _, e in pairs]
    src = blocks[0] + "".join(
        f"\n{j}\njax_{i} = dict(globals())\n{e}\n" for i, (j, e) in enumerate(zip(jax_side, expex_side))
    )
    path = tmp_path / "pairs.py"
    path.write_text(src)
    ns = runpy.run_path(str(path))
    q, k = jr.normal(jr.key(2), (2, 3, 4)), jr.normal(jr.key(3), (2, 5, 4))
    assert jnp.allclose(ns["jax_0"]["attend"](q, k), ns["attend"](q, k), atol=1e-5)
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
    assert ns["dropout"].random and ns["safe_log"](jnp.ones((2, 3))).shape == (2, 3)
    assert "# scores: f32[B, T, T]" in out
    assert "def attend_jax[B, T, S, D, A](q: f32[B, T, D], k: f32[B, S, D], v: f32[B, S, A])" in out
    assert "lookup(tokens: Int[batch, seq], table: Float[vocab, d]) -> Float[batch, seq, d]" in out  # jaxtyping
    assert "EX101 expected dtype Int, got f32." in out
    assert type(ns["grads"]).__name__ == "Stack" and ns["y"].shape == (4, 16)
    # functions.md: the plain JAX versions compute the same as the expex ones
    w, xs = jr.normal(jr.key(0), (3, 3)) * 0.3, jr.normal(jr.key(1), (6, 3))
    assert jnp.allclose(ns["rnn_jax"](xs, w), ns["rnn"](xs, w), atol=1e-5)
    z = jnp.array([2.0, -1.0, 0.0])
    assert jnp.array_equal(ns["safe_log_jax"](z), ns["safe_log"](z))
    # design.md
    assert "flops        4·D·F + 9·F + D" in out and "MLP(up=True, bias=False, down=True)" in out
    assert "['f32[1024,64,8]', 'f32[8,3]']\n['f32[65536,8]', 'f32[8,3]']" in out
