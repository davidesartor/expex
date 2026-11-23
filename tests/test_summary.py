"""`expex.summary`: one row per module, lists collapsed, parameters and FLOPs as numbers and formulas."""

import importlib.util
import math
import pathlib
import re
import sys

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import expex
from expex import f32
from expex.tools.cost import Poly
from expex.tools.summary import Summary, summary

ROOT = pathlib.Path(__file__).resolve().parent.parent


def load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tf():
    return load("summary_transformer", "examples/transformer.py")


@pytest.fixture(scope="module")
def tf_model(tf):
    return tf.Transformer.init(jr.key(0), Vocab=50, D=32, F=64, blocks=4)


@pytest.fixture(scope="module")
def tf_summary(tf_model) -> Summary:
    return summary(tf_model, T=128)


def n_params(tree) -> int:
    return sum(math.prod(x.shape) for x in jax.tree.leaves(tree))


def by_path(s: Summary):
    return {r.path: r for r in s.rows}


def check_totals(s: Summary, model) -> None:
    """The total equals the arrays' sizes, the rows add up to it, and the formula agrees."""
    assert s.total.params == n_params(model)
    assert s.total.nbytes == sum(x.size * x.dtype.itemsize for x in jax.tree.leaves(model))
    assert sum(r.params * r.count for r in s.rows if r.parent < 0) == s.total.params
    assert s.total.formula.subs(s.sizes).value == s.total.params
    for r in s.rows:
        assert r.formula.subs(s.sizes).value == r.params, r.path
        children = [c for c in s.rows if c.parent == s.rows.index(r)]
        if children:
            assert sum(c.params * c.count for c in children) == r.params, r.path


def test_transformer_rows(tf_summary, tf_model):
    s, rows = tf_summary, by_path(tf_summary)
    assert list(rows) == [
        "embed", "blocks[*]",
        "blocks[*].norm1", "blocks[*].norm1.scale", "blocks[*].norm1.bias",
        "blocks[*].attn", "blocks[*].attn.wq", "blocks[*].attn.wk", "blocks[*].attn.wv", "blocks[*].attn.wo",
        "blocks[*].norm2", "blocks[*].norm2.scale", "blocks[*].norm2.bias",
        "blocks[*].mlp", "blocks[*].mlp.w1", "blocks[*].mlp.w2",
        "norm", "norm.scale", "norm.bias", "unembed",
    ]  # fmt: skip
    assert [r.depth for r in s.rows][:4] == [0, 0, 1, 2]
    assert rows["blocks[*]"].count == 4 and rows["blocks[*].attn"].count == 1
    assert rows["blocks[*].attn"].weight == 4
    assert rows["blocks[*].attn"].label == "Attention[D=32]"
    assert rows["blocks[*]"].label == "Block[D=32, F=64]"
    assert rows["embed"].label == "f32[Vocab, D]" and rows["embed"].params == 1600
    assert rows["blocks[*].attn"].formula == Poly.of("D") ** 2 * 4
    assert rows["blocks[*].attn"].formula.format() == "4·D²"
    assert rows["blocks[*].mlp"].formula == Poly.of("D") * Poly.of("F") * 2
    assert s.sizes == {"T": 128, "D": 32, "F": 64, "Vocab": 50, "L": 4}
    check_totals(s, tf_model)
    assert s.total.params == 36544


def test_transformer_flops(tf, tf_summary, tf_model):
    rows = by_path(tf_summary)
    attn = rows["blocks[*].attn"]
    assert attn.flops == expex.cost(tf.Attention.__call__).flops
    assert attn.flops_value == expex.cost(tf_model.blocks[0].attn.__call__, T=128).flops.value
    assert tf_summary.total.flops_value == expex.cost(tf_model.__call__, T=128).flops.value
    assert tf_summary.total.flops == expex.cost(tf.Transformer.__call__).flops
    assert rows["blocks[*].norm1"].flops_value == 9 * 32 + 8  # one call: per vector, no T
    assert rows["embed"].flops is None and rows["embed"].flops_value is None


def test_class_and_sizes(tf, tf_summary, tf_model):
    s = summary(tf.Transformer, Vocab=50, D=32, F=64, blocks=4)
    assert [r.path for r in s.rows] == [r.path for r in tf_summary.rows]
    assert [r.params for r in s.rows] == [r.params for r in tf_summary.rows]
    assert s.total.params == n_params(tf_model)
    assert s.total.flops == tf_summary.total.flops and s.total.flops_value is None  # no T: formula only
    assert by_path(s)["blocks[*].norm1"].flops_value == 296
    with_t = summary(tf.Transformer, Vocab=50, D=32, F=64, blocks=4, T=128)
    assert with_t.total.flops_value == tf_summary.total.flops_value


def test_static_fields_differ_from_defaults(tf):
    model = tf.Transformer.init(jr.key(0), Vocab=50, D=32, F=64, blocks=2, heads=8)
    s = summary(model, T=16)
    attn = by_path(s)["blocks[*].attn"]
    assert attn.flops == expex.cost(model.blocks[0].attn.__call__).flops  # the instance's, in T
    assert attn.flops_value == expex.cost(model.blocks[0].attn.__call__, T=16).flops.value
    assert "blocks[*].attn" in s.notes[-1] and "static fields" in s.notes[-1]


def test_unknown_size(tf_model):
    with pytest.raises(TypeError, match="unknown size Q"):
        summary(tf_model, Q=3)


def test_mamba():
    mamba = load("summary_mamba", "examples/mamba/mamba.py")
    model = mamba.Mamba.create(jr.key(0), vocab=100, d=64, layers=3)
    s = summary(model, T=24)
    rows = by_path(s)
    assert [p for p in rows if "[" in p and "[*]" not in p] == []  # no per-layer rows
    assert rows["layers[*]"].count == 3
    assert rows["layers[*].mixer"].label == "Mixer[D=64, E=128, N=16, R=4, K=4]"
    assert rows["layers[*].mixer.x_proj"].label == "f32[R + 2 * N, E]"
    assert rows["layers[*].mixer.x_proj"].formula.format() == "2·E·N + E·R"
    check_totals(s, model)
    assert s.total.flops_value == expex.cost(model.__call__, T=24).flops.value


@expex.struct
class Linear[I, O]:
    w: f32[I, O] = jax.nn.initializers.normal()
    b: f32[O] = 0.0

    def __call__(self, x: f32[I]) -> f32[O]:
        return x @ self.w + self.b


@expex.struct
class Net[D, F]:
    up: Linear[D, F]
    down: Linear[F, D]
    gains: list[f32[D]]
    extra: Linear[D, D] | None = None
    scale: float = 1.0


def test_renamed_sizes_and_plain_lists():
    model = Net.init(jr.key(0), D=3, F=5, gains=[jnp.ones(3), jnp.ones(3)])
    s = summary(model)
    rows = by_path(s)
    assert list(rows) == ["up", "up.w", "up.b", "down", "down.w", "down.b", "gains[*]"]  # extra is None
    assert rows["up"].label == "Linear[I=3, O=5]"
    assert rows["down"].formula.format(s.order) == "D·F + D"  # Linear's I, O in Net's names
    assert rows["down"].flops.format(s.order) == "2·D·F + D"  # x @ w + b with I=F, O=D
    assert rows["down"].flops_value == 2 * 15 + 3
    assert rows["gains[*]"].count == 2 and rows["gains[*]"].label == "f32[D]"
    assert s.total.flops is None  # Net has no __call__
    check_totals(s, model)


def test_text(tf_summary):
    text = str(tf_summary)
    lines = text.splitlines()
    assert lines[0] == "Transformer   (T=128, D=32, F=64, Vocab=50, L=4)"
    assert re.search(r"^blocks\[\*\]\s+Block\[D=32, F=64\]\s+× 4\s+8,320\s+4·D² \+ 2·D·F \+ 4·D", text, re.M)
    assert re.search(r"^  attn\s+Attention\[D=32\]\s+4,096\s+4·D²\s+16.4 kB\s+3.65 M\s+8·T·D²", text, re.M)
    assert re.search(r"^embed\s+f32\[Vocab, D\]\s+1,600\s+D·Vocab", text, re.M)
    assert re.search(r"^total\s+Transformer\s+36,544\s+4·D²·L", text, re.M)
    assert "blocks[0]" not in text and lines[-1] == "note: L = len(blocks)"


def test_html(tf_summary, tmp_path):
    page = tf_summary._repr_html_()
    assert page.count("<tr ") == len(tf_summary.rows) + 1
    assert "prefers-color-scheme:dark" in page and "<script>" in page
    assert not re.search(r"(src|href)=[\"']?https?:", page)  # self-contained
    assert 'title="8·T·D² + 4·T²·D' in page  # the formula on hover
    assert 'data-k="params"' in page and 'data-k="flops"' in page
    assert page != tf_summary._repr_html_()  # each table has its own id
    path = tmp_path / "summary.html"
    tf_summary.save(path)
    text = path.read_text()
    assert text.startswith("<!doctype html>") and "<title>Transformer summary</title>" in text
