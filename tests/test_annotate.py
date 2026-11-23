"""expex.annotate: annotations proposed for plain JAX code, from a run on examples."""

import importlib.util
import pathlib
import sys
import textwrap
import jax
import jax.numpy as jnp
import pytest

import expex

z = jnp.zeros

# The plain JAX code being annotated, written to a file so that annotate can read
# (and apply() can edit) its source.
PLAIN = '''
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp


def attend(q, k):
    return jnp.einsum("btd,bsd->bts", q, k)


def rnn(w, xs):
    def step(h, x):
        h = jnp.tanh(w @ h + x)
        return h, h

    h0 = jnp.zeros(xs.shape[1])
    _, hs = jax.lax.scan(step, h0, xs)
    return hs


def pad(x):
    return jnp.pad(x, ((0, 0), (0, 1)))


def with_bos(x):
    bos = jnp.zeros((x.shape[0], 1, x.shape[2]))
    return jnp.concatenate([bos, x], axis=1)


def split_heads(x, heads):
    T, D = x.shape
    return x.reshape(T, heads, D // heads).transpose(1, 0, 2)


def merge_heads(x):
    return x.transpose(1, 0, 2).reshape(x.shape[1], -1)


def layer(params, x):
    return x + jnp.tanh(x @ params["w"] + params["b"])


def stats(x):
    return x.mean(-1), x.std(-1)


def scale(x, alpha, steps):
    for _ in range(steps):
        x = x * alpha
    return x


def init_state(x, w):
    return jnp.zeros((x.shape[0], w.shape[1]))


def self_attention(x, wq, wk):
    q = x @ wq
    k = x @ wk
    scores = jnp.einsum("btd,bsd->bts", q, k)
    weights = jax.nn.softmax(scores, axis=-1)
    return weights @ x


def embed(tokens, table):
    return table[tokens]


class Linear(NamedTuple):
    w: jax.Array
    b: jax.Array


def linear(p: Linear, x):
    return x @ p.w + p.b


def summarize(x):
    return {"mean": x.mean(0), "total": x.sum()}


@partial(jax.jit, static_argnames=("eps",))
def norm(x, *, eps):
    return x / jnp.sqrt((x * x).mean(-1, keepdims=True) + eps)


def tree_size(params):
    return sum(p.size for p in jax.tree.leaves(params))
'''


@pytest.fixture
def plain(tmp_path):
    """The module above, imported from a fresh file."""
    path = tmp_path / "plain_model.py"
    path.write_text(PLAIN)
    return load(path)


def load(path: pathlib.Path):
    name = f"_annotate_{path.stem}_{id(path)}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ sizes and names


def test_attention(plain):
    a = expex.annotate(plain.attend, z((2, 3, 4)), z((2, 5, 4)))
    assert a.signature == "def attend[B, T, S, D](q: f32[B, T, D], k: f32[B, S, D]) -> f32[B, T, S]:"
    assert a.sizes == {"B": 2, "T": 3, "S": 5, "D": 4}
    assert not a.warnings and not a.notes


def test_attention_with_equal_lengths(plain):
    """T == S in the example: the dataflow still keeps them apart, and says the example cannot confirm it."""
    a = expex.annotate(plain.attend, z((2, 3, 4)), z((2, 3, 4)))
    assert a.signature == "def attend[B, T, S, D](q: f32[B, T, D], k: f32[B, S, D]) -> f32[B, T, S]:"
    assert any("T and S are both 3" in n and "vary=True" in n for n in a.notes)
    v = expex.annotate(plain.attend, z((2, 3, 4)), z((2, 3, 4)), vary=True)
    assert v.signature == a.signature and not v.notes and not v.warnings


def test_rnn_with_scan(plain):
    a = expex.annotate(plain.rnn, z((4, 4)), z((6, 4)))
    assert a.signature == "def rnn[T, D](w: f32[D, D], xs: f32[T, D]) -> f32[T, D]:"


def test_pad_and_concatenate_add_one(plain):
    assert expex.annotate(plain.pad, z((2, 3))).signature == "def pad[B, T](x: f32[B, T]) -> f32[B, T + 1]:"
    a = expex.annotate(plain.with_bos, z((2, 5, 4)))
    assert a.signature == "def with_bos[B, T, D](x: f32[B, T, D]) -> f32[B, T + 1, D]:"


def test_split_into_heads(plain):
    """D split into (heads, D // heads): the size from the static argument, the rest by division, and its constraint."""
    a = expex.annotate(plain.split_heads, z((5, 8)), 2)
    assert a.signature == (
        "def split_heads[T, D](x: f32[T, D], heads: int) -> f32[heads, T, D // heads]:\n"
        "    assert D % heads == 0"
    )  # T and D: the names the body gives them (`T, D = x.shape`)


def test_merge_heads(plain):
    a = expex.annotate(plain.merge_heads, z((3, 5, 4)))
    assert a.signature == "def merge_heads[T, H, K](x: f32[H, T, K]) -> f32[T, H * K]:"


def test_names_are_fixed_and_propagate(plain):
    a = expex.annotate(plain.attend, z((2, 3, 4)), z((2, 5, 4)), names={"q": "N Q E"})
    assert a.signature == "def attend[N, Q, S, E](q: f32[N, Q, E], k: f32[N, S, E]) -> f32[N, Q, S]:"
    with pytest.raises(ValueError, match="3 names for 2 axes"):
        expex.annotate(plain.pad, z((2, 3)), names={"x": "B T D"})


def test_names_join_sizes(plain):
    """Two axes given one name are one size, even where the dataflow does not join them."""
    a = expex.annotate(plain.init_state, z((2, 3)), z((3, 2)), names={"x": "B D", "w": "D B"})
    assert a.signature == "def init_state[B, D](x: f32[B, D], w: f32[D, B]) -> f32[B, B]:"


def test_embedding_vocabulary(plain):
    a = expex.annotate(plain.embed, z((2, 5), jnp.int32), z((10, 4)))
    assert a.signature == "def embed[B, T, V, D](tokens: Int[B, T], table: f32[V, D]) -> f32[B, T, D]:"


def test_python_numbers_and_static_configuration(plain):
    a = expex.annotate(plain.scale, z((2, 3)), 0.5, 3)
    assert a.signature == "def scale[B, D](x: f32[B, D], alpha: float, steps: int) -> f32[B, D]:"
    n = expex.annotate(plain.norm, z((2, 3)), eps=1e-6)  # a jitted function: the function inside
    assert n.signature == "def norm[B, D](x: f32[B, D], *, eps: float) -> f32[B, D]:"


# ------------------------------------------------------------------ coincidences


def test_coinciding_sizes_warn(plain):
    """A result built from shapes alone: with B == C in the example, nothing says which is which."""
    a = expex.annotate(plain.init_state, z((2, 3)), z((3, 2)))
    assert a.params["x"] == "f32[B, D]" and a.params["w"] == "f32[A, C]"
    assert a.returns == "f32[B, B]"  # a guess
    assert len(a.warnings) == 2 and all("could be B or C" in w and "vary=True" in w for w in a.warnings)
    v = expex.annotate(plain.init_state, z((2, 3)), z((3, 2)), vary=True)
    assert v.returns == "f32[B, C]" and not v.warnings
    d = expex.annotate(plain.init_state, z((2, 3)), z((5, 7)))  # distinct sizes: no doubt
    assert d.returns == "f32[B, C]" and not d.warnings and "by value only" in d.notes[0]


def test_vary_checks_derived_sizes(plain):
    a = expex.annotate(plain.split_heads, z((5, 8)), 2, vary=True)
    assert "f32[heads, T, D // heads]" in a.signature and not a.warnings


# ------------------------------------------------------------------ structs and results


def test_params_dict_suggests_a_struct(plain):
    a = expex.annotate(plain.layer, {"w": z((4, 4)), "b": z(4)}, z((2, 4)))
    assert str(a).startswith(
        "@expex.struct\n"
        "class Params[D]:\n"
        "    w: f32[D, D]\n"
        "    b: f32[D]\n\n"
        "def layer[B, D](params: Params[D], x: f32[B, D]) -> f32[B, D]:"
    )
    assert any("callers pass Params(w=..., b=...)" in n for n in a.notes)


def test_dict_used_as_a_whole_is_left_alone(plain):
    a = expex.annotate(plain.tree_size, {"w": z((4, 3)), "b": z(3)})
    assert a.signature == "def tree_size(params) -> Int:"
    assert "cannot become the struct Params" in a.warnings[0]


def test_output_tuple_and_dict(plain):
    assert expex.annotate(plain.stats, z((2, 3))).returns == "tuple[f32[B], f32[B]]"
    a = expex.annotate(plain.summarize, z((5, 3)))
    assert a.returns == "SummarizeOutput[D]"
    assert a.structs[0].source() == "@expex.struct\nclass SummarizeOutput[D]:\n    mean: f32[D]\n    total: f32"


def test_namedtuple_is_made_generic(plain):
    a = expex.annotate(plain.linear, plain.Linear(z((4, 3)), z(3)), z((2, 4)))
    assert a.signature == "def linear[B, D, A](p: Linear[D, A], x: f32[B, D]) -> f32[B, A]:"
    assert a.structs[0].source() == "class Linear[D, A](NamedTuple):\n    w: f32[D, A]\n    b: f32[A]"


# ------------------------------------------------------------------ locals


def test_locals_with_a_size_on_two_axes(plain):
    a = expex.annotate(plain.self_attention, z((2, 5, 4)), z((4, 3)), z((4, 3)), locals=True)
    assert [(loc.name, loc.annotation) for loc in a.locals] == [
        ("scores", "f32[B, T, T]"),
        ("weights", "f32[B, T, T]"),
    ]
    assert '    scores: f32[B, T, T] = jnp.einsum("btd,bsd->bts", q, k)' in str(a)
    every = expex.annotate(plain.self_attention, z((2, 5, 4)), z((4, 3)), z((4, 3)), locals="all")
    assert [loc.name for loc in every.locals] == ["q", "k", "scores", "weights"]
    assert every.locals[0].annotation == "f32[B, T, A]"


# ------------------------------------------------------------------ diff, apply, prove


CASES = {
    "attend": lambda m: ((z((2, 3, 4)), z((2, 3, 4))), {"vary": True}),
    "rnn": lambda m: ((z((4, 4)), z((6, 4))), {}),
    "pad": lambda m: ((z((2, 3)),), {}),
    "with_bos": lambda m: ((z((2, 5, 4)),), {}),
    "split_heads": lambda m: ((z((5, 8)), 2), {}),
    "merge_heads": lambda m: ((z((3, 5, 4)),), {}),
    "layer": lambda m: (({"w": z((4, 4)), "b": z(4)}, z((2, 4))), {}),
    "stats": lambda m: ((z((2, 3)),), {}),
    "scale": lambda m: ((z((2, 3)), 0.5, 3), {}),
    "init_state": lambda m: ((z((2, 3)), z((3, 2))), {"vary": True}),
    "self_attention": lambda m: ((z((2, 5, 4)), z((4, 3)), z((4, 3))), {"locals": True}),
    "embed": lambda m: ((z((2, 5), jnp.int32), z((10, 4))), {}),
    "linear": lambda m: ((m.Linear(z((4, 3)), z(3)), z((2, 4))), {}),
    "summarize": lambda m: ((z((5, 3)),), {}),
    "norm": lambda m: ((z((2, 3)),), {"eps": 1e-6}),
}


def test_diff_does_not_write(plain):
    path = pathlib.Path(plain.__file__)
    a = expex.annotate(plain.pad, z((2, 3)))
    d = a.diff()
    assert "-def pad(x):\n+def pad[B, T](x: f32[B, T]) -> f32[B, T + 1]:\n" in d
    assert "+from expex import f32\n" in d
    assert path.read_text() == PLAIN


def test_applied_annotations_are_proved(plain):
    """Every example annotated and written back; the file then imports, and each function, under @expex.fn, is proved."""
    for name, case in CASES.items():
        args, kw = case(plain)
        expex.annotate(getattr(plain, name), *args, **kw).apply()
    path = pathlib.Path(plain.__file__)
    text = path.read_text()
    assert text.count("from expex import") == 1
    assert "class Linear[D, A](NamedTuple):\n    w: f32[D, A]\n    b: f32[A]" in text
    assert "    assert D % heads == 0\n    T, D = x.shape" in text
    assert "return x + jnp.tanh(x @ params.w + params.b)" in text
    assert 'return SummarizeOutput(mean=x.mean(0), total=x.sum())' in text
    assert '    scores: f32[B, T, T] = jnp.einsum("btd,bsd->bts", q, k)' in text
    annotated = load(path)
    for name in CASES:
        report = expex.prove(expex.fn(getattr(annotated, name)))
        assert report.status == "proved", (name, str(report))
    # and they still compute the same: the struct replaces the dict at the call site
    x, w = jax.random.normal(jax.random.key(0), (2, 4)), jax.random.normal(jax.random.key(1), (4, 4))
    layer = expex.fn(annotated.layer)
    assert jnp.allclose(layer(annotated.Params(w=w, b=w[0]), x), plain.layer({"w": w, "b": w[0]}, x))
    q, k = z((2, 3, 4)), z((2, 6, 4))
    assert expex.fn(annotated.attend)(q, k).shape == (2, 3, 6)


def test_apply_refuses_a_changed_file(plain):
    a = expex.annotate(plain.pad, z((2, 3)))
    path = pathlib.Path(plain.__file__)
    path.write_text(PLAIN + "\n# edited\n")
    with pytest.raises(RuntimeError, match="changed since annotate read it"):
        a.apply()


def test_without_source_it_only_prints():
    ns: dict = {}
    exec(textwrap.dedent("def f(x):\n    return x * 2\n"), {"jnp": jnp}, ns)
    a = expex.annotate(ns["f"], z((2, 3)))
    assert a.signature == "def f[B, D](x: f32[B, D]) -> f32[B, D]:"
    with pytest.raises(ValueError, match="no source file"):
        a.diff()


def test_inline_functions_have_source():
    """Functions defined in a test module (not written to a file by the test) work as well."""

    def mlp(params, x):
        h = jax.nn.relu(x @ params["w1"])
        return h @ params["w2"]

    a = expex.annotate(mlp, {"w1": z((4, 7)), "w2": z((7, 3))}, z((2, 4)))
    assert a.signature == "def mlp[B, D, A, C](params: Params[D, A, C], x: f32[B, D]) -> f32[B, C]:"
    assert a.structs[0].source() == "@expex.struct\nclass Params[D, A, C]:\n    w1: f32[D, A]\n    w2: f32[A, C]"
