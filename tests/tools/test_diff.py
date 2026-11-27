"""expex.diff: comparing models and checkpoints field by field."""

import dataclasses
import pathlib
import sys

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from expex.tools.diff import diff
from expex.io import save

ROOT = pathlib.Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "examples"))
from transformer import Transformer  # noqa: E402


def make(blocks: int = 2, heads: int = 4, F: int = 16) -> Transformer:
    return Transformer(jr.key(0), Vocab=11, D=8, F=F, blocks=blocks, heads=heads)


def edit(model, path, **changes):
    """``model`` with the module at ``path`` (``("blocks", 1, "attn")``) replaced field-wise."""
    if not path:
        return dataclasses.replace(model, **changes)
    head, *rest = path
    child = model[head] if isinstance(head, int) else getattr(model, head)
    new = edit(child, rest, **changes)
    if isinstance(head, int):
        return [new if i == head else m for i, m in enumerate(model)]
    return dataclasses.replace(model, **{head: new})


def test_identical_models_have_no_diff():
    d = diff(make(), make())
    assert not d and d.changes == [] and str(d) == f"{len(d.unchanged)} unchanged"
    assert {f.path for f in d.unchanged} >= {"embed", "blocks[1].mlp.w2", "blocks[0].attn.heads"}


def test_changed_values_are_ranked_by_relative_change():
    a = make()
    b = edit(a, (), embed=a.embed * 1.5, unembed=a.unembed + 1e-3)
    b = edit(b, ("blocks", 0, "mlp"), w1=a.blocks[0].mlp.w1.at[0, 0].add(1e-6))
    d = diff(a, b)
    assert bool(d) and [f.path for f in d.changed] == ["embed", "unembed", "blocks[0].mlp.w1"]
    embed = d.changed[0]
    assert embed.rel == pytest.approx(0.5 / 1.5, rel=1e-5) and embed.frac == 1.0  # relative to the larger side
    assert embed.max_abs == pytest.approx(float(jnp.abs(a.embed).max() * 0.5), rel=1e-5)
    assert d.changed[2].frac == pytest.approx(1 / a.blocks[0].mlp.w1.size)
    assert str(d).splitlines()[0] == f"3 changed, {len(d.unchanged)} unchanged"
    assert diff(a, b, atol=1e-5).changed[-1].path == "unembed"  # the 1e-6 change is within tolerance


def test_shape_change():
    a, b = make(F=16), make(F=20)
    d = diff(a, b)
    assert {f.path for f in d.reshaped} == {f"blocks[{i}].mlp.{w}" for i in (0, 1) for w in ("w1", "w2")}
    assert "blocks[1].mlp.w1  f32[8, 16] → f32[8, 20]" in str(d)
    assert d.changes[0].kind == "reshaped"


def test_added_and_removed_layers():
    d = diff(make(blocks=2), make(blocks=3))
    assert d.removed == [] and len(d.added) == 10 and {f.path.split(".")[0] for f in d.added} == {"blocks[2]"}
    paths = [f.path for f in d.fields]
    assert paths.index("blocks[2].mlp.w2") < paths.index("norm.scale")  # in model order
    back = diff(make(blocks=3), make(blocks=2))
    assert [f.path for f in back.removed] == [f.path for f in d.added] and back.removed[0].a == "f32[8]"


def test_static_field_change():
    a = make()
    d = diff(a, edit(a, ("blocks", 1, "attn"), heads=2))
    assert [(f.path, f.a, f.b) for f in d.static] == [("blocks[1].attn.heads", "4", "2")]
    assert "blocks[1].attn.heads  4 → 2" in str(d) and str(d).startswith("1 static changed,")


def test_model_against_saved_file(tmp_path):
    a = make()
    b = edit(a, (), embed=a.embed.at[3].set(0.0))
    save(b, tmp_path / "b.npz")
    d = diff(a, tmp_path / "b.npz")
    assert [f.path for f in d.changes] == ["embed"] and d.names == ("Transformer", "b.npz (Transformer)")
    assert not diff(str(tmp_path / "b.npz"), b)
    assert d.changed[0].frac == pytest.approx(8 / a.embed.size)


def test_plain_pytrees_and_nans():
    a = {"w": jnp.array([1.0, jnp.nan]), "b": jnp.zeros(2)}
    assert not diff(a, a)
    d = diff(a, {"w": jnp.array([1.0, 2.0]), "b": jnp.ones(2)})
    w, b = d.changed  # a NaN that became a number changed infinitely; zeros that moved, by 1
    assert (w.path, w.max_abs, w.frac) == ("['w']", float("inf"), 0.5) and (b.path, b.rel) == ("['b']", 1.0)


def test_html_report(tmp_path):
    a = make()
    b = jax.tree.map(lambda x: x * 1.01, edit(a, ("blocks", 1, "attn"), heads=2))
    d = diff(a, b)
    page = d._repr_html_()
    assert page.count('class="txdiff"') == 1 and "<script>" in page and "http" not in page
    assert "k-static" in page and "k-changed" in page and "blocks" in page
    d.save(tmp_path / "d.html")
    text = (tmp_path / "d.html").read_text()
    assert text.startswith("<!doctype html>") and "prefers-color-scheme:dark" in text
    shapes = diff(make(blocks=2), make(blocks=3, F=20))._repr_html_()
    assert "k-added" in shapes and "k-reshaped" in shapes and "f32[8, 16] → f32[8, 20]" in shapes
